"""In-process rate limiting for the endpoints that can be abused.

Scope and honesty: the app runs one worker (the model artifacts sit in process memory), so a shared
external store would be new infrastructure for no benefit yet. The counter lives in this process,
which means the limits are per worker and reset when it restarts — stated in the README rather than
implied. If the app is ever scaled out, this module is the thing to swap for Redis, and its
interface is deliberately small enough to make that a single change.

Two dimensions are limited, because they stop different attacks:

- the **client** (IP), which bounds one machine hammering signup, login or the reset endpoint;
- the **identity** (the email being attempted), which bounds a distributed attack on one account and
  stops the reset endpoint being used to flood somebody's inbox.

The client address comes from ``request.client`` unless ``TRUST_FORWARDED_FOR`` is on, in which case
the first hop of ``X-Forwarded-For`` is used. That header is trivially forged, so trusting it by
default would turn the limiter into decoration; a deployment behind a proxy opts in explicitly.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from backend.app.config import get_settings

# key -> timestamps of the requests still inside their window.
_buckets: dict[tuple[str, str], deque[float]] = {}
_lock = threading.Lock()

# Injectable so tests can move time without sleeping.
_clock = time.monotonic

# A hard ceiling on tracked keys. Reached only by a flood of distinct addresses; evicting the
# oldest entries keeps memory bounded without letting an attacker grow the process.
MAX_TRACKED_KEYS = 20_000


@dataclass(frozen=True)
class Rule:
    """``limit`` requests per ``window_seconds``, within one named ``scope``."""

    scope: str
    limit: int
    window_seconds: float

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and self.window_seconds > 0


def parse_rule(text: str, scope: str) -> Rule:
    """Read ``"10/60"`` as ten requests per sixty seconds.

    A malformed value disables that rule rather than failing the request: a typo in an environment
    variable should not take the login page down, and the alternative — crashing at startup — is
    worse than running unlimited with a value the operator can see in the settings.
    """
    try:
        limit_text, window_text = str(text).split("/", 1)
        return Rule(scope=scope, limit=int(limit_text), window_seconds=float(window_text))
    except (ValueError, TypeError):
        return Rule(scope=scope, limit=0, window_seconds=0)


def reset() -> None:
    """Forget every counter. Used by tests, and by anything that needs a clean window."""
    with _lock:
        _buckets.clear()


def set_clock(clock) -> None:
    """Replace the time source. Tests pass a controllable one; nothing else should call this."""
    global _clock
    _clock = clock


def _prune(bucket: deque[float], cutoff: float) -> None:
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()


def check(
    key: tuple[str, str], rule: Rule, *, now: float | None = None, record: bool = True
) -> float | None:
    """Return ``None`` when the caller may proceed, or the seconds to wait when not.

    A rejected call is never recorded, so a client that backs off recovers on schedule instead of
    extending its own ban by retrying. ``record=False`` asks the question without spending a slot,
    which is how a failed-attempt budget stays separate from successful use.
    """
    if not rule.enabled:
        return None

    moment = _clock() if now is None else now
    cutoff = moment - rule.window_seconds

    with _lock:
        if key not in _buckets and len(_buckets) >= MAX_TRACKED_KEYS:
            _evict_locked(cutoff)

        bucket = _buckets.setdefault(key, deque())
        _prune(bucket, cutoff)

        if len(bucket) >= rule.limit:
            # The oldest hit inside the window is what has to age out before one more is allowed.
            return max(bucket[0] + rule.window_seconds - moment, 0.0)

        if record:
            bucket.append(moment)
        return None


def _evict_locked(cutoff: float) -> None:
    """Drop empty or expired buckets; fall back to the oldest entries if that is not enough."""
    for key in [k for k, v in _buckets.items() if not v or v[-1] <= cutoff]:
        del _buckets[key]
    while len(_buckets) >= MAX_TRACKED_KEYS:
        oldest = min(_buckets, key=lambda k: _buckets[k][-1])
        del _buckets[oldest]


def client_identifier(request: Request) -> str:
    """The caller's address, honouring a trusted proxy's forwarded header."""
    settings = get_settings()
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    return client.host if client is not None else "unknown"


def _reject(wait_seconds: float) -> HTTPException:
    seconds = max(1, math.ceil(wait_seconds))
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Too many attempts. Try again in {seconds} "
            f"second{'' if seconds == 1 else 's'}."
        ),
        headers={"Retry-After": str(seconds)},
    )


def enforce(key: tuple[str, str], rule: Rule) -> None:
    """Raise a 429 when ``key`` has spent its budget, otherwise count the request."""
    wait = check(key, rule)
    if wait is not None:
        raise _reject(wait)


def client_limiter(scope: str, setting: str):
    """A FastAPI dependency limiting the caller's address for one endpoint."""

    def dependency(request: Request) -> None:
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return
        rule = parse_rule(getattr(settings, setting, "0/0"), scope)
        enforce((scope, client_identifier(request)), rule)

    return dependency


def identity_limiter(scope: str, setting: str, identity: str) -> None:
    """Limit an identity such as the email being attempted, once the body has been parsed."""
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return
    rule = parse_rule(getattr(settings, setting, "0/0"), scope)
    enforce((scope, identity.strip().lower()), rule)


def note_failed_attempt(scope: str, setting: str, identity: str) -> None:
    """Record one rejected credential and refuse further guesses once the budget is spent.

    This is called *after* verification, and only when it failed, which is what keeps a per-account
    limit from becoming a denial of service: whoever knows the password never touches this counter,
    so a stranger guessing at their address cannot lock them out. An over-budget attempt still ran
    bcrypt once, so the limit bounds how many guesses reach the hash rather than saving work — the
    per-client limit is what bounds total cost.
    """
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return
    rule = parse_rule(getattr(settings, setting, "0/0"), scope)
    wait = check((scope, identity.strip().lower()), rule)
    if wait is not None:
        raise _reject(wait)

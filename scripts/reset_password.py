"""Mint a password reset link, or set a password, for an instance with no mail server.

A deployment that has no SMTP settings cannot send the reset email, and the API deliberately never
returns the token over HTTP — that would let anyone reset any account. The operator's way in is this
script, which runs on the server with database access:

    uv run python scripts/reset_password.py --email ada@example.com
    uv run python scripts/reset_password.py --email ada@example.com --set-password

The link form prints a URL for the user to open; ``--set-password`` prompts for a new password and
never echoes it. Both paths bump the account's session version, so any open session elsewhere ends.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from sqlalchemy import select

# Allow `python scripts/reset_password.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.account import issue_reset_token, reset_link, set_password  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
from backend.app.db import get_session_factory, init_database  # noqa: E402
from backend.app.models import User  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True, help="address of the account to help")
    parser.add_argument(
        "--set-password",
        action="store_true",
        help="prompt for a new password and apply it now instead of printing a link",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    init_database()

    with get_session_factory()() as session:
        user = session.scalar(select(User).where(User.email == args.email.strip().lower()))
        if user is None:
            print(f"No account for {args.email}.", file=sys.stderr)
            return 1
        if user.is_seeded:
            print(
                "That address belongs to a seeded MovieLens demo user, which has no usable "
                "password by design.",
                file=sys.stderr,
            )
            return 1

        if args.set_password:
            first = getpass.getpass("New password: ")
            if len(first) < 8:
                print("Password must be at least 8 characters.", file=sys.stderr)
                return 1
            if first != getpass.getpass("Repeat password: "):
                print("Passwords did not match.", file=sys.stderr)
                return 1
            set_password(session, user, first)
            session.commit()
            print(f"Password updated for {user.email}. Existing sessions are signed out.")
            return 0

        raw = issue_reset_token(
            session, user, ttl_minutes=settings.password_reset_ttl_minutes
        )
        session.commit()
        print(f"Reset link for {user.email} (valid {settings.password_reset_ttl_minutes} minutes):")
        print(reset_link(settings, raw))
        if "localhost" in settings.app_base_url:
            print(
                "APP_BASE_URL points at localhost; set it to the public origin before sending "
                "this to a real user.",
                file=sys.stderr,
            )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

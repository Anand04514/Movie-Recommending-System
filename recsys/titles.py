"""Normalize MovieLens titles.

MovieLens stores titles like ``Matrix, The (1999)``: the release year is inside the title and
English articles are moved to the end. Search needs ``the matrix`` and ``matrix`` to both match,
so we keep three forms:

- ``title``: the title as MovieLens wrote it, minus the year (``Matrix, The``)
- ``display_title``: human-readable, article moved to the front (``The Matrix``)
- ``search_title``: lowercase ASCII-ish, punctuation removed, article at the front
  (``the matrix``) — safe to store in a trigram-indexed column

The functions here are pure: no pandas, web, or database imports.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Articles MovieLens moves to the end of a title. MovieLens is predominantly English but
# contains a number of foreign-language films using the same "Title, Article" convention.
ARTICLES = frozenset(
    {
        "a",
        "an",
        "the",
        "der",
        "die",
        "das",
        "det",
        "el",
        "il",
        "l",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
    }
)

_YEAR_RE = re.compile(r"\(\s*(\d{4})\s*\)\s*$")
# "..., The" / "..., The." — the article must be the final word after the last comma.
# An optional trailing parenthetical covers titles such as
# "Battle of Algiers, The (La battaglia di Algeri)".
_TRAILING_ARTICLE_RE = re.compile(
    r"^(?P<body>.*\S)\s*,\s*(?P<article>[^\W\d_]+)\.?(?P<suffix>\s*\([^()]*\))?\s*$",
    re.UNICODE,
)
_APOSTROPHES_RE = re.compile(r"['’`´]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedTitle:
    """The three title forms plus the release year extracted from a raw MovieLens title."""

    raw: str
    title: str
    display_title: str
    search_title: str
    release_year: int | None


def _move_article_to_front(text: str) -> str:
    """Turn ``Matrix, The`` into ``The Matrix``; leave other titles untouched."""
    match = _TRAILING_ARTICLE_RE.match(text)
    if not match:
        return text
    article, body = match.group("article"), match.group("body").strip()
    suffix = (match.group("suffix") or "").strip()
    if article.lower() not in ARTICLES or not body:
        return text
    # "the" -> "The", "les" -> "Les"; keep the article as written otherwise.
    canonical = article[:1].upper() + article[1:] if article.islower() else article
    moved = f"{canonical} {body}"
    return f"{moved} {suffix}" if suffix else moved


def _strip_marks(text: str) -> str:
    """Remove accents so ``Amélie`` is searchable as ``amelie``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_for_search(text: str) -> str:
    """Normalize a title or a user query to the same lowercase search form.

    Both sides of a search go through this function, so ``Matrix, The`` and ``The Matrix``
    and ``matrix`` all normalize consistently.
    """
    text = _strip_marks(text).casefold()
    text = text.replace("&", " and ")
    text = _APOSTROPHES_RE.sub("", text)
    text = _move_article_to_front(text).casefold()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def parse_title(raw: str) -> ParsedTitle:
    """Split a raw MovieLens title into title, display title, search form, and year.

    Raises ``ValueError`` only for an empty title. Year-only titles such as ``" (2019)"``
    exist in ml-32m and are preserved rather than rejected.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("title must be a non-empty string")
    raw = raw.strip()

    year: int | None = None
    without_year = raw
    year_match = _YEAR_RE.search(raw)
    if year_match:
        year = int(year_match.group(1))
        without_year = raw[: year_match.start()].strip()

    title = _SPACES_RE.sub(" ", without_year).strip()
    if not title:
        # A few catalog rows carry only a year, e.g. " (2019)". Keep them searchable by year
        # instead of rejecting an otherwise valid catalog entry.
        title = raw

    display_title = _move_article_to_front(title)
    return ParsedTitle(
        raw=raw,
        title=title,
        display_title=display_title,
        search_title=normalize_for_search(display_title),
        release_year=year,
    )


def split_genres(genres: str | None) -> list[str]:
    """Split MovieLens' pipe-separated genres, dropping the ``(no genres listed)`` placeholder."""
    if not isinstance(genres, str) or not genres.strip():
        return []
    items = [part.strip() for part in genres.split("|") if part.strip()]
    return [item for item in items if item != "(no genres listed)"]

"""Phase 1.1: title parsing and search normalization."""

from __future__ import annotations

import pytest

from recsys.titles import normalize_for_search, parse_title, split_genres

# raw title, expected display title, expected year
PARSE_CASES = [
    ("Matrix, The (1999)", "The Matrix", 1999),
    ("Fish Called Wanda, A (1988)", "A Fish Called Wanda", 1988),
    (
        "Lord of the Rings: The Fellowship of the Ring, The (2001)",
        "The Lord of the Rings: The Fellowship of the Ring",
        2001,
    ),
    (
        "Battle of Algiers, The (La battaglia di Algeri) (1966)",
        "The Battle of Algiers (La battaglia di Algeri)",
        1966,
    ),
    ("2001: A Space Odyssey (1968)", "2001: A Space Odyssey", 1968),
    ("1984 (1984)", "1984", 1984),
    ("Star Wars: Episode IV - A New Hope (1977)", "Star Wars: Episode IV - A New Hope", 1977),
    ("Amélie (2001)", "Amélie", 2001),
    ("Pokémon: The First Movie (1998)", "Pokémon: The First Movie", 1998),
    ("Fahrenheit 9/11 (2004)", "Fahrenheit 9/11", 2004),
    ("Naked Gun 33 1/3: The Final Insult (1994)", "Naked Gun 33 1/3: The Final Insult", 1994),
    ("Dead Pool, The (1995)", "The Dead Pool", 1995),
    ("Se7en (1995)", "Se7en", 1995),
    ("M*A*S*H (1970)", "M*A*S*H", 1970),
    ("WALL·E (2008)", "WALL·E", 2008),
    ("A.I. Artificial Intelligence (2001)", "A.I. Artificial Intelligence", 2001),
    ("Les Misérables (2012)", "Les Misérables", 2012),
    ("Fauteuils d'orchestre, Les (2006)", "Les Fauteuils d'orchestre", 2006),
    ("Hurt Locker, The (2008)", "The Hurt Locker", 2008),
    ("Kill Bill: Vol. 1 (2003)", "Kill Bill: Vol. 1", 2003),
    ("  Spaced Out   (1999)  ", "Spaced Out", 1999),
    ("No Year Here", "No Year Here", None),
    ("Schindler's List (1993)", "Schindler's List", 1993),
    ("Star Trek: The Motion Picture (1979)", "Star Trek: The Motion Picture", 1979),
    ("The Matrix (1999)", "The Matrix", 1999),
    ("Matrix,The (1999)", "The Matrix", 1999),
    ("Der Untergang (2004)", "Der Untergang", 2004),
    ("Straight Story, The (1999)", "The Straight Story", 1999),
    ("Wall Street (1987)", "Wall Street", 1987),
    ("Matrix, The", "The Matrix", None),
]


@pytest.mark.parametrize(("raw", "display", "year"), PARSE_CASES)
def test_parse_title(raw: str, display: str, year: int | None) -> None:
    parsed = parse_title(raw)
    assert parsed.raw == raw.strip()
    assert parsed.display_title == display
    assert parsed.release_year == year
    assert parsed.title != ""
    assert parsed.search_title == normalize_for_search(display)


@pytest.mark.parametrize(
    ("raw", "search"),
    [
        ("Matrix, The (1999)", "the matrix"),
        ("2001: A Space Odyssey (1968)", "2001 a space odyssey"),
        ("Amélie (2001)", "amelie"),
        ("Pokémon: The First Movie (1998)", "pokemon the first movie"),
        ("Fahrenheit 9/11 (2004)", "fahrenheit 9 11"),
        ("M*A*S*H (1970)", "m a s h"),
        ("WALL·E (2008)", "wall e"),
        ("A.I. Artificial Intelligence (2001)", "a i artificial intelligence"),
        ("Fauteuils d'orchestre, Les (2006)", "les fauteuils dorchestre"),
        ("Schindler's List (1993)", "schindlers list"),
        ("Kill Bill: Vol. 1 (2003)", "kill bill vol 1"),
        ("Les Misérables (2012)", "les miserables"),
        ("Matrix, The", "the matrix"),
    ],
)
def test_search_title_is_lowercase_and_punctuation_free(raw: str, search: str) -> None:
    assert parse_title(raw).search_title == search


@pytest.mark.parametrize(
    ("query", "normalized"),
    [
        ("The Matrix", "the matrix"),
        ("MATRIX, THE", "the matrix"),
        ("matrix", "matrix"),
        ("Amélie", "amelie"),
        ("M*A*S*H", "m a s h"),
        ("Cheech & Chong", "cheech and chong"),
        ("Schindler's List", "schindlers list"),
        ("  the   matrix  ", "the matrix"),
        ("Stranger Things: Season 1", "stranger things season 1"),
        ("", ""),
    ],
)
def test_normalize_for_search_on_queries(query: str, normalized: str) -> None:
    assert normalize_for_search(query) == normalized


def test_the_matrix_is_searchable_as_matrix_and_the_matrix() -> None:
    stored = parse_title("Matrix, The (1999)").search_title
    assert "matrix" in normalize_for_search("matrix")
    assert normalize_for_search("matrix") in stored
    assert normalize_for_search("The Matrix") == stored


def test_leading_articles_are_not_stripped_from_titles() -> None:
    parsed = parse_title("The Lord of the Rings: The Two Towers (2002)")
    assert parsed.display_title == "The Lord of the Rings: The Two Towers"
    assert parsed.search_title == "the lord of the rings the two towers"


def test_trailing_article_requires_a_known_article() -> None:
    # "Interview" is a word, not an article: leave the title alone.
    parsed = parse_title("Frost/Nixon, The Interview (2008)")
    assert parsed.display_title == "Frost/Nixon, The Interview"


def test_inner_parentheses_are_preserved() -> None:
    parsed = parse_title("Fanny and Alexander (Fanny och Alexander) (1982)")
    assert parsed.display_title == "Fanny and Alexander (Fanny och Alexander)"
    assert parsed.release_year == 1982


@pytest.mark.parametrize("raw", ["", "   "])
def test_invalid_titles_raise(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_title(raw)


def test_year_only_title_is_preserved() -> None:
    """MovieLens ml-32m has one real row whose title is just " (2019)"."""
    parsed = parse_title(" (2019)")
    assert parsed.display_title == "(2019)"
    assert parsed.release_year == 2019
    assert parsed.search_title == "2019"


@pytest.mark.parametrize(
    ("genres", "expected"),
    [
        ("Action|Sci-Fi", ["Action", "Sci-Fi"]),
        ("Drama", ["Drama"]),
        ("(no genres listed)", []),
        ("", []),
        (None, []),
        ("Comedy|Romance|IMAX", ["Comedy", "Romance", "IMAX"]),
    ],
)
def test_split_genres(genres: str | None, expected: list[str]) -> None:
    assert split_genres(genres) == expected

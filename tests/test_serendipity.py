"""Discovery picks and favourite-genre extraction."""

from __future__ import annotations

import pytest

from recsys.serendipity import (
    DiscoveryCandidate,
    discovery_slots,
    favourite_genres,
    genre_overlap,
    pick_discovery,
)


def test_genre_overlap_is_a_share_of_the_candidates_genres() -> None:
    assert genre_overlap(["Drama", "Romance"], {"drama"}) == 0.5
    assert genre_overlap(["Drama"], {"Drama"}) == 1.0
    assert genre_overlap(["Comedy"], {"Drama"}) == 0.0
    assert genre_overlap([], {"Drama"}) == 0.0


def test_discovery_slots_only_apply_to_full_lists_from_engaged_users() -> None:
    assert discovery_slots(12, ratings_count=10) == 2
    assert discovery_slots(8, ratings_count=5) == 1
    assert discovery_slots(12, ratings_count=4) == 0
    assert discovery_slots(5, ratings_count=20) == 0


def test_picks_skip_mainstream_genres_and_keep_rank_order() -> None:
    candidates = [
        DiscoveryCandidate(1, 0.9, ("Drama", "Romance")),
        DiscoveryCandidate(2, 0.8, ("Horror",)),
        DiscoveryCandidate(3, 0.7, ("Documentary",)),
        DiscoveryCandidate(4, 0.6, ("Comedy", "Drama")),
    ]
    picks = pick_discovery(candidates, favourite_genres={"Drama"}, slots=2)
    assert [pick.movie_id for pick in picks] == [2, 3]


def test_no_favourites_or_no_outsiders_means_no_discovery() -> None:
    candidates = [DiscoveryCandidate(1, 1.0, ("Drama",))]
    assert pick_discovery(candidates, favourite_genres=set(), slots=1) == []
    assert pick_discovery(candidates, favourite_genres={"Drama"}, slots=1) == []
    assert pick_discovery([], favourite_genres={"Drama"}, slots=1) == []
    assert pick_discovery(candidates, favourite_genres={"Drama"}, slots=0) == []


def test_discovery_respects_the_overlap_threshold() -> None:
    candidate = DiscoveryCandidate(1, 1.0, ("Drama", "Comedy"))
    assert pick_discovery([candidate], favourite_genres={"Drama"}, slots=1, max_overlap=0.5)
    assert pick_discovery([candidate], favourite_genres={"Drama"}, slots=1, max_overlap=0.4) == []
    with pytest.raises(ValueError, match="between 0 and 1"):
        pick_discovery([candidate], favourite_genres={"Drama"}, slots=1, max_overlap=2.0)


def test_favourite_genres_ignore_disliked_shelves() -> None:
    rated = [
        (("Action", "Sci-Fi"), 5.0),
        (("Action",), 4.0),
        (("Horror",), 1.0),
        (("Drama",), 4.5),
        (("Comedy",), 3.0),
    ]
    favourites = favourite_genres(rated, limit=2)
    assert favourites == {"action", "sci-fi"}
    assert "horror" not in favourites

    assert favourite_genres([], limit=2) == set()
    with pytest.raises(ValueError, match="positive"):
        favourite_genres(rated, limit=0)

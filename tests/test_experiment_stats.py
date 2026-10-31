"""The A/B report's uncertainty arithmetic.

The engagement table used to name a leader on percentages alone, which turns 3-of-10 against 2-of-10
into a "10 point win". These tests pin the interval that stops it: a report may only call a winner
when the whole interval sits on one side of zero.
"""

from __future__ import annotations

import pytest

from backend.app.experiments import (
    ArmComparison,
    ExperimentReport,
    VariantReport,
    compare_arms,
    wilson_interval,
)


def arm(variant: str, likes: int, impressions: int, **counts: int) -> VariantReport:
    return VariantReport(
        variant=variant,
        users=counts.get("users", 1),
        impressions=impressions,
        likes=likes,
        rejects=counts.get("rejects", 0),
        clicks=counts.get("clicks", 0),
        rated_after=counts.get("rated_after", 0),
    )


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes() -> None:
    """The normal approximation gives a zero-width interval at 0% and 100%, which is a lie."""
    empty = wilson_interval(0, 10)
    assert empty is not None
    assert empty[0] == 0.0
    assert empty[1] == pytest.approx(0.2776, abs=1e-4)

    full = wilson_interval(10, 10)
    assert full is not None
    assert full[1] == pytest.approx(1.0, abs=1e-9)
    assert full[0] == pytest.approx(0.7224, abs=1e-4)


def test_wilson_interval_needs_traffic() -> None:
    assert wilson_interval(0, 0) is None


def test_comparison_matches_newcombes_worked_example() -> None:
    """10% against 20% on 100 picks each: a 10-point gap whose interval just clears zero."""
    comparison = compare_arms(
        arm("a", likes=10, impressions=100), arm("b", likes=20, impressions=100)
    )

    assert isinstance(comparison, ArmComparison)
    assert comparison.difference == pytest.approx(0.10)
    assert comparison.low == pytest.approx(0.0002, abs=2e-3)
    assert comparison.high == pytest.approx(0.1998, abs=2e-3)
    assert comparison.confidence == 0.95
    assert comparison.significant is True
    assert comparison.a_variant == "a"
    assert comparison.b_variant == "b"


def test_identical_arms_are_not_significant() -> None:
    comparison = compare_arms(
        arm("a", likes=10, impressions=100), arm("b", likes=10, impressions=100)
    )

    assert comparison is not None
    assert comparison.difference == 0.0
    assert comparison.low < 0 < comparison.high
    assert comparison.significant is False


def test_a_small_sample_cannot_crown_a_winner() -> None:
    """Two of three against one of three is not a result, whatever the percentage says."""
    comparison = compare_arms(arm("a", likes=1, impressions=3), arm("b", likes=2, impressions=3))

    assert comparison is not None
    assert comparison.low < 0 < comparison.high
    assert comparison.significant is False


def test_more_traffic_narrows_the_interval() -> None:
    small = compare_arms(arm("a", likes=1, impressions=10), arm("b", likes=2, impressions=10))
    large = compare_arms(
        arm("a", likes=100, impressions=1000), arm("b", likes=200, impressions=1000)
    )

    assert small is not None and large is not None
    assert (large.high - large.low) < (small.high - small.low)
    assert large.significant is True
    assert small.significant is False


def test_comparison_counts_every_engagement_not_just_likes() -> None:
    """Clicks and follow-up ratings are engagement too; leaving them out would understate an arm."""
    quiet = arm("a", likes=5, impressions=100)
    busy = arm("b", likes=5, impressions=100, clicks=10, rated_after=5)

    comparison = compare_arms(quiet, busy)

    assert comparison is not None
    assert comparison.difference == pytest.approx(0.15)


def test_comparison_needs_traffic_on_both_arms() -> None:
    assert compare_arms(arm("a", likes=0, impressions=0), arm("b", likes=3, impressions=10)) is None


def test_report_compares_the_first_two_arms_only() -> None:
    report = ExperimentReport(
        name="x",
        description="d",
        variants=[arm("a", 10, 100), arm("b", 30, 100)],
    )
    assert report.comparison is not None
    assert report.comparison.significant is True

    three = ExperimentReport(
        name="x", description="d", variants=[arm("a", 10, 100), arm("b", 30, 100), arm("c", 5, 100)]
    )
    assert three.comparison is None
    assert (
        ExperimentReport(name="x", description="d", variants=[arm("a", 1, 10)]).comparison is None
    )

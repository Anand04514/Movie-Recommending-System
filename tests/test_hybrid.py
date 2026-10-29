"""Score normalisation, hybrid blending and tuned-weight persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recsys.hybrid import (
    DEFAULT_COLLABORATIVE_WEIGHT,
    blend_rows,
    combine_predictions,
    load_weights,
    normalise_row,
    save_weights,
)


def test_normalise_row_scales_to_the_best_positive_score() -> None:
    row = normalise_row(np.array([4.0, 2.0, -1.0, 0.0]))
    assert row.tolist() == [1.0, 0.5, 0.0, 0.0]


def test_normalise_row_keeps_blocked_items_at_zero_and_handles_no_signal() -> None:
    assert normalise_row(np.array([-np.inf, 3.0])).tolist() == [0.0, 1.0]
    assert normalise_row(np.array([-np.inf, -np.inf])).tolist() == [0.0, 0.0]
    assert normalise_row(np.array([-2.0, 0.0])).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="one-dimensional"):
        normalise_row(np.zeros((2, 2)))


def test_weight_one_reproduces_collaborative_ranking_and_zero_reproduces_content() -> None:
    collaborative = np.array([2.0, 8.0, -np.inf])
    content = np.array([0.9, 0.1, -np.inf])

    pure_cf = blend_rows(collaborative, content, weight=1.0)
    assert pure_cf.tolist() == [0.25, 1.0, -np.inf]

    pure_content = blend_rows(collaborative, content, weight=0.0)
    assert pure_content.tolist() == [1.0, pytest.approx(1 / 9), -np.inf]

    halfway = blend_rows(collaborative, content, weight=0.5)
    assert halfway[1] == pytest.approx(0.5 * 1.0 + 0.5 * (0.1 / 0.9))
    assert halfway[0] == pytest.approx(0.5 * 0.25 + 0.5 * 1.0)
    assert np.isneginf(halfway[2])


def test_blend_validates_shapes_and_weight() -> None:
    with pytest.raises(ValueError, match="same shape"):
        blend_rows(np.zeros(3), np.zeros(2), weight=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_rows(np.zeros(3), np.zeros(3), weight=1.5)


def test_combine_predictions_falls_back_to_the_available_model() -> None:
    cf = np.array([4.0, 3.0, 2.5])
    content = np.array([3.0, 5.0, 1.0])
    blended = combine_predictions(
        cf,
        content,
        weight=0.75,
        available_collaborative=np.array([True, False, True]),
        available_content=np.array([True, True, False]),
    )
    assert blended[0] == pytest.approx(3.75)
    assert blended[1] == 5.0
    assert blended[2] == 2.5
    assert np.all((blended >= 0.5) & (blended <= 5.0))


def test_tuned_weight_round_trips_with_its_validation_evidence(tmp_path: Path) -> None:
    path = tmp_path / "hybrid_weights.json"
    assert load_weights(path) == DEFAULT_COLLABORATIVE_WEIGHT

    save_weights(path, weight=0.7, validation={"ndcg_at_k": 0.113, "users": 1500})
    assert load_weights(path) == pytest.approx(0.7)
    assert '"ndcg_at_k": 0.113' in path.read_text(encoding="utf-8")


def test_a_broken_or_out_of_range_weight_file_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "hybrid_weights.json"
    path.write_text("not json", encoding="utf-8")
    assert load_weights(path) == DEFAULT_COLLABORATIVE_WEIGHT

    path.write_text('{"collaborative_weight": 4.2}', encoding="utf-8")
    assert load_weights(path) == DEFAULT_COLLABORATIVE_WEIGHT

    with pytest.raises(ValueError, match="between 0 and 1"):
        save_weights(path, weight=-0.1)

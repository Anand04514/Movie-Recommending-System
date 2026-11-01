"""Weighted blending of a collaborative score and a content score.

Scores from the two models are on completely different scales: item-CF produces an unbounded sum
of similarities weighted by how far a rating sits from the user's mean, while content produces a
cosine between 0 and 1. Adding them directly would let the collaborative score drown the content
model. Each row is therefore scaled by its own best positive score before the blend, which keeps
the *ranking evidence* from both models and lets one tuned weight decide the mixture.

The weight itself is tuned offline on a validation slice of the training data (never on test) by
``scripts/evaluate_models.py`` and saved next to the other artifacts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_COLLABORATIVE_WEIGHT = 0.65
DEFAULT_COLLABORATIVE_BASE = "user_cf"
COLLABORATIVE_BASES = ("item_cf", "user_cf")


def blend_rows(
    collaborative: np.ndarray,
    content: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend two score rows for the same candidate set.

    ``weight`` is the collaborative share in ``[0, 1]``. A candidate that neither model can score
    stays ``-inf`` so it can never be picked by accident.
    """
    collaborative = np.asarray(collaborative, dtype="float64")
    content = np.asarray(content, dtype="float64")
    if collaborative.ndim != 1:
        raise ValueError("both score rows must be one-dimensional")
    return blend_matrix(collaborative[None, :], content[None, :], weight=weight)[0]


def blend_matrix(
    collaborative: np.ndarray,
    content: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend a block of score rows, normalising each row on its own.

    The evaluation harness scores a block of users at a time, and each user has their own score
    scale; normalising across the whole block would let the most active user set the scale for
    everyone else.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    collaborative = np.asarray(collaborative, dtype="float64")
    content = np.asarray(content, dtype="float64")
    if collaborative.shape != content.shape:
        raise ValueError("both score blocks must have the same shape")
    if collaborative.ndim != 2:
        raise ValueError("score blocks must be two-dimensional")

    def scaled(block: np.ndarray) -> np.ndarray:
        positive = np.where(np.isfinite(block), np.maximum(block, 0.0), 0.0)
        best = positive.max(axis=1, keepdims=True)
        return np.divide(
            positive,
            best,
            out=np.zeros_like(positive),
            where=best > 0,
        )

    blended = weight * scaled(collaborative) + (1.0 - weight) * scaled(content)
    neither = ~np.isfinite(collaborative) & ~np.isfinite(content)
    return np.where(neither, -np.inf, blended)


def combine_predictions(
    collaborative: np.ndarray | float,
    content: np.ndarray | float,
    *,
    weight: float,
    available_collaborative: np.ndarray | bool,
    available_content: np.ndarray | bool,
) -> np.ndarray:
    """Blend predicted ratings, falling back to whichever model covers the pair.

    Rating predictions are already in the same 0.5-5.0 space, so they are averaged with the same
    weight as the ranking without rescaling.
    """
    collaborative_array = np.asarray(collaborative, dtype="float64")
    content_array = np.asarray(content, dtype="float64")
    have_collaborative = np.asarray(available_collaborative, dtype=bool)
    have_content = np.asarray(available_content, dtype=bool)

    blended = np.where(
        have_collaborative & have_content,
        weight * collaborative_array + (1.0 - weight) * content_array,
        np.where(have_collaborative, collaborative_array, content_array),
    )
    return np.clip(blended, 0.5, 5.0)


def save_weights(
    path: Path,
    *,
    weight: float,
    collaborative: str = DEFAULT_COLLABORATIVE_BASE,
    validation: dict[str, object] | None = None,
) -> None:
    """Record the tuned collaborative share and base with the evidence that chose them."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    if collaborative not in COLLABORATIVE_BASES:
        raise ValueError(f"collaborative must be one of {list(COLLABORATIVE_BASES)}")
    payload = {
        "collaborative_weight": round(float(weight), 4),
        "collaborative": collaborative,
        "tuned_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "validation": validation or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_weights(path: Path) -> float:
    """Read the tuned weight, falling back to the default when the file is absent or broken.

    A missing tuning run must not stop the API from serving: the default is the value the app was
    developed with, and it is labelled as untuned in the artifact when the script writes one.
    """
    path = Path(path)
    if not path.is_file():
        return DEFAULT_COLLABORATIVE_WEIGHT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        weight = float(payload["collaborative_weight"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return DEFAULT_COLLABORATIVE_WEIGHT
    return weight if 0.0 <= weight <= 1.0 else DEFAULT_COLLABORATIVE_WEIGHT


def load_collaborative_base(path: Path) -> str:
    """Which collaborative model the tuned hybrid uses; user-CF by default."""
    path = Path(path)
    if not path.is_file():
        return DEFAULT_COLLABORATIVE_BASE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = str(payload.get("collaborative", DEFAULT_COLLABORATIVE_BASE))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return DEFAULT_COLLABORATIVE_BASE
    return base if base in COLLABORATIVE_BASES else DEFAULT_COLLABORATIVE_BASE

"""The real-time feedback loop: like, reject or click a recommendation.

The effect is immediate — liked and rejected films leave the next recommendation list — while the
models are retrained only on a schedule (``scripts/train_and_precompute.py``), never on a click.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.deps import get_current_user
from backend.app.models import Feedback, Movie, User, utcnow
from backend.app.schemas import FeedbackCreate, FeedbackOut

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

def _to_out(row: Feedback) -> FeedbackOut:
    return FeedbackOut(
        movie_id=row.movie_id,
        action=row.action,
        source_algorithm=row.source_algorithm,
        created_at=row.created_at,
    )


@router.get("", response_model=list[FeedbackOut])
def list_feedback(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[FeedbackOut]:
    rows = db.scalars(
        select(Feedback)
        .where(Feedback.user_id == user.id)
        .order_by(Feedback.created_at.desc(), Feedback.id.desc())
    ).all()
    return [_to_out(row) for row in rows]


@router.post("", response_model=FeedbackOut)
def record_feedback(
    payload: FeedbackCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FeedbackOut:
    movie = db.get(Movie, payload.movie_id)
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")

    row = db.scalar(
        select(Feedback).where(
            Feedback.user_id == user.id,
            Feedback.movie_id == payload.movie_id,
            Feedback.action == payload.action,
        )
    )
    if row is None:
        row = Feedback(
            user_id=user.id,
            movie_id=payload.movie_id,
            action=payload.action,
            source_algorithm=payload.source_algorithm,
        )
        db.add(row)
    else:
        row.source_algorithm = payload.source_algorithm
        row.created_at = utcnow()

    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.delete("/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_feedback(
    movie_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Undo every reaction to one film, putting it back into the recommendation pool."""
    result = db.execute(
        delete(Feedback).where(
            Feedback.user_id == user.id,
            Feedback.movie_id == movie_id,
        )
    )
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No feedback found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

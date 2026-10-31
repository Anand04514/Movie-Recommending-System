"""The evaluation report can populate ``model_runs`` without re-running the evaluation.

A deployed container ships ``evaluation.json`` as an artifact but starts on a database that has
never run the evaluation. This import is what keeps the accuracy dashboard from being empty on a
fresh deploy, so it has to be exact, idempotent and safe when the report is missing or broken.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app.db import get_session_factory
from backend.app.models import ModelRun
from scripts.evaluate_models import main

REPORT: dict[str, Any] = {
    "generated_at": "2026-10-12T09:31:07.123456+00:00",
    "dataset": "MovieLens ml-32m",
    "protocol": {
        "split": "chronological per user, most recent 20% held out",
        "relevance_threshold": 4.0,
        "k": 10,
        "sample_users": 10_000,
        "seed": 42,
    },
    "results": [
        {
            "method": "item_cf",
            "params": {"k_neighbors": 50, "damping": 10.0},
            "rmse": 0.8741,
            "mae": 0.6712,
            "rated_pairs": 412_345,
            "fallback_rate": 0.0123,
            "precision_at_k": 0.0812,
            "recall_at_k": 0.0431,
            "ndcg_at_k": 0.1174,
            "catalogue_coverage": 0.2141,
            "evaluated_users": 9_982,
            "skipped_users": 18,
            "k": 10,
            "runtime_seconds": 41.25,
            "peak_memory_mb": 612.5,
        },
        {
            "method": "hybrid",
            "params": {"base": "svd", "weight": 0.6},
            "rmse": 0.8602,
            "mae": 0.6588,
            "rated_pairs": 412_345,
            "fallback_rate": 0.0044,
            "precision_at_k": 0.0933,
            "recall_at_k": 0.0502,
            "ndcg_at_k": 0.1298,
            "catalogue_coverage": 0.3002,
            "evaluated_users": 9_982,
            "skipped_users": 18,
            "k": 10,
            "runtime_seconds": 55.75,
            "peak_memory_mb": 688.0,
        },
    ],
}


def write_report(tmp_path: Path, payload: Any = REPORT) -> Path:
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def rows() -> list[ModelRun]:
    with get_session_factory()() as session:
        return list(session.query(ModelRun).order_by(ModelRun.algorithm).all())


def as_utc(value: datetime) -> datetime:
    """SQLite drops the zone on a ``DateTime(timezone=True)`` column; PostgreSQL keeps it."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def test_import_records_every_method_from_the_report(tmp_path: Path) -> None:
    path = write_report(tmp_path)

    code = main(["--import-only", "--json", str(path)])

    assert code == 0
    stored = rows()
    assert [row.algorithm for row in stored] == ["hybrid", "item_cf"]
    hybrid = stored[0]
    assert hybrid.rmse == 0.8602
    assert hybrid.precision_at_k == 0.0933
    assert hybrid.ndcg_at_k == 0.1298
    # The moment the numbers were measured travels with them, so the dashboard can date them.
    assert as_utc(hybrid.trained_at) == datetime(2026, 10, 12, 9, 31, 7, 123456, tzinfo=UTC)
    params = json.loads(hybrid.params)
    assert params["sample_users"] == 10_000
    assert params["imported_from"] == "evaluation.json"
    assert params["coverage"] == 0.3002
    assert params["runtime_seconds"] == 55.75


def test_import_needs_no_processed_artifacts(tmp_path: Path) -> None:
    """The whole point is skipping the evaluation, so nothing under the processed dir is read."""
    path = write_report(tmp_path)

    code = main(["--import-only", "--json", str(path), "--processed-dir", str(tmp_path / "empty")])

    assert code == 0
    assert len(rows()) == 2


def test_import_is_idempotent_and_replaces_older_rows(tmp_path: Path) -> None:
    with get_session_factory()() as session:
        session.add(
            ModelRun(
                algorithm="item_cf",
                params=json.dumps({"stale": True}),
                rmse=9.99,
                mae=9.99,
                precision_at_k=0.0,
                recall_at_k=0.0,
                ndcg_at_k=0.0,
                trained_at=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
        session.commit()

    path = write_report(tmp_path)
    assert main(["--import-only", "--json", str(path)]) == 0
    assert main(["--import-only", "--json", str(path)]) == 0

    stored = rows()
    assert len(stored) == 2, "a restart must refresh rows, not duplicate them"
    item_cf = next(row for row in stored if row.algorithm == "item_cf")
    assert item_cf.rmse == 0.8741
    assert "stale" not in json.loads(item_cf.params)


def test_imported_runs_reach_the_accuracy_endpoint(client: TestClient, tmp_path: Path) -> None:
    path = write_report(tmp_path)
    assert main(["--import-only", "--json", str(path)]) == 0

    response = client.get("/api/model-runs")

    assert response.status_code == 200
    body = response.json()
    assert {run["algorithm"] for run in body} == {"item_cf", "hybrid"}
    item_cf = next(run for run in body if run["algorithm"] == "item_cf")
    assert item_cf["rmse"] == 0.8741
    assert item_cf["ndcg_at_k"] == 0.1174
    assert item_cf["trained_at"].startswith("2026-10-12")


def test_missing_report_is_an_error_not_a_crash(tmp_path: Path) -> None:
    code = main(["--import-only", "--json", str(tmp_path / "absent.json")])

    assert code == 1
    assert rows() == []


def test_broken_report_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    path.write_text("{not json", encoding="utf-8")

    assert main(["--import-only", "--json", str(path)]) == 1

    no_results = write_report(tmp_path, {"generated_at": REPORT["generated_at"], "results": []})
    assert main(["--import-only", "--json", str(no_results)]) == 1
    missing_timestamp = write_report(tmp_path, {"results": REPORT["results"]})
    assert main(["--import-only", "--json", str(missing_timestamp)]) == 1
    assert rows() == []


def test_naive_timestamps_are_treated_as_utc(tmp_path: Path) -> None:
    payload = {**REPORT, "generated_at": "2026-10-12T09:31:07"}
    path = write_report(tmp_path, payload)

    assert main(["--import-only", "--json", str(path)]) == 0

    # A report without an offset must be read as UTC, never as the server's local time.
    stamps = {as_utc(row.trained_at) for row in rows()}
    assert stamps == {datetime(2026, 10, 12, 9, 31, 7, tzinfo=UTC)}

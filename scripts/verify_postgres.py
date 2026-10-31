#!/usr/bin/env python
"""End-to-end verification of the PostgreSQL path, against a throwaway database.

The backend suite runs on SQLite, so nothing in it exercises `pg_trgm`, PL/pgSQL-less extension
probing, or the planner's opinion of a trigram index. This script does: it migrates a real
PostgreSQL server (up, down, up again), seeds the 87,585-row catalog, imports the evaluation
report, and then drives the API through signup, search, rating and recommendations.

It writes to the database it is pointed at, so it refuses a database that already has the schema
unless `--force` is passed.

    # Zero setup: boot a temporary bundled server (needs `uv run --with pgserver`)
    uv run --with pgserver python scripts/verify_postgres.py --bundled

    # Or point it at a scratch database you own
    uv run python scripts/verify_postgres.py --database-url postgresql://postgres:@127.0.0.1:5432/scratch

Expected `pg_trgm`: with it, search answers misspellings through word similarity; without it, search
must still answer, ranked by word-start and popularity. Both runs are part of the check — a build
without contrib is a real deployment case, and the fallback is the reason startup does not fail.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Allow `python scripts/verify_postgres.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPORT_NAME = "evaluation.json"


class Checks:
    """Collects pass/fail lines and the exit status they imply."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        print(
            f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True
        )
        return bool(ok)

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [result for result in self.results if not result[1]]

    def summary(self) -> str:
        return f"{len(self.results) - len(self.failed)}/{len(self.results)} checks passed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the PostgreSQL path end to end.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--database-url", help="A throwaway PostgreSQL database to write to.")
    target.add_argument(
        "--bundled",
        action="store_true",
        help="Boot a temporary server with `pgserver` instead of using an existing one.",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing to a database that already holds this schema (it will be dropped).",
    )
    parser.add_argument("--limit", type=int, default=10, help="Recommendations to request.")
    return parser


def start_bundled_server() -> tuple[str, object]:
    import pgserver

    data = os.path.join(tempfile.gettempdir(), f"verify-pg-{int(time.time())}")
    os.makedirs(data)
    server = pgserver.get_server(data, cleanup_mode=None)
    bin_dir = Path(pgserver.__file__).resolve().parent / "pginstall" / "bin"

    def stop() -> None:
        # ``cleanup()`` only covers the interpreter-exit path, so the cluster is stopped here and
        # its directory removed; a verification run must not leave a server listening.
        server.cleanup()
        for name in ("pg_ctl.exe", "pg_ctl"):
            pg_ctl = bin_dir / name
            if pg_ctl.is_file():
                subprocess.run(
                    [str(pg_ctl), "-D", data, "-m", "fast", "-W", "stop"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                )
                break
        shutil.rmtree(data, ignore_errors=True)

    return server.get_uri(), stop


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    stop_server = lambda: None  # noqa: E731 - a no-op keeps the exit paths identical
    if args.bundled:
        uri, cleanup = start_bundled_server()
        stop_server = cleanup
    else:
        uri = args.database_url
    # The app reads its configuration from the environment, so this must precede every import of
    # backend.app.config.
    os.environ["DATABASE_URL"] = uri
    os.environ["PROCESSED_DIR"] = str(args.processed_dir)
    os.environ["SERVE_FRONTEND"] = "0"
    os.environ.pop("ARTIFACTS_URL", None)

    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    check = Checks()
    engine = sa.create_engine(uri)
    try:
        with engine.connect() as connection:
            version = connection.execute(sa.text("select version()")).scalar_one_or_none()
        if not version or not str(version).startswith("PostgreSQL"):
            print(f"error: {uri} is not a PostgreSQL server", file=sys.stderr)
            return 2
        print(f"server: {str(version).split(' on ')[0]}", flush=True)

        if sa.inspect(engine).has_table("movies") and not args.force:
            print(
                "error: this database already holds the schema; use a throwaway database or "
                "--force (the check runs migrate down to base)",
                file=sys.stderr,
            )
            return 2

        root = Path(__file__).resolve().parents[1]
        config = Config(str(root / "alembic.ini"))
        # Absolute, so the script works from any working directory.
        config.set_main_option("script_location", str(root / "migrations"))

        started = time.perf_counter()
        command.upgrade(config, "head")
        check("alembic upgrade head", True, f"{time.perf_counter() - started:.1f}s")
        command.downgrade(config, "base")
        check("alembic downgrade base", True)
        command.upgrade(config, "head")
        check("alembic upgrade head again", True)

        with engine.connect() as connection:
            extensions = connection.execute(sa.text("select extname from pg_extension")).scalars()
            extensions = set(extensions)
            indexes = set(
                connection.execute(
                    sa.text("select indexname from pg_indexes where tablename = 'movies'")
                ).scalars()
            )
            tables = set(sa.inspect(connection).get_table_names())

        has_trgm = "pg_trgm" in extensions
        print(
            f"pg_trgm: {'available' if has_trgm else 'not available (fallback path)'}", flush=True
        )
        check(
            "the schema matches the models",
            {
                "users",
                "movies",
                "ratings",
                "feedback",
                "recommendation_logs",
                "model_runs",
                "experiment_assignments",
                "precomputed_recs",
            }
            <= tables,
            ", ".join(sorted(tables)),
        )
        check(
            "the trigram index follows the extension",
            ("ix_movies_title_search_trgm" in indexes) is has_trgm,
            "indexed" if has_trgm else "no index without pg_trgm",
        )

        # --- catalog and evaluation report ------------------------------------------------
        from scripts import seed_db

        seed_args = [
            "--processed-dir",
            str(args.processed_dir),
            "--demo-users",
            "0",
            "--skip-if-seeded",
        ]
        check("catalog seeded", seed_db.main(seed_args) == 0)
        skipped = seed_db.main(seed_args)
        check("a second seed is a no-op", skipped == 0)

        from scripts.evaluate_models import import_report

        report = args.processed_dir / REPORT_NAME
        if report.is_file():
            check("evaluation report imported", import_report(report) == 0)

        # --- the API against this server --------------------------------------------------
        from fastapi.testclient import TestClient

        from backend.app.db import init_database, trgm_available
        from backend.app.main import app

        init_database()
        check("startup probe agrees with the server", trgm_available() is has_trgm)

        with TestClient(app) as client:
            health = client.get("/api/health")
            check(
                "health reports the postgres dialect",
                health.status_code == 200
                and health.json().get("database", {}).get("dialect") == "postgresql",
                health.text[:120],
            )

            email = f"verify-{uuid.uuid4().hex[:8]}@example.com"
            signup = client.post(
                "/api/auth/signup",
                json={
                    "email": email,
                    "password": "correct-horse-battery",
                    "display_name": "Verify",
                },
            )
            check("signup", signup.status_code == 201, signup.text[:120])
            headers = {"Authorization": f"Bearer {signup.json()['access_token']}"}

            def search(term: str, limit: int = 5) -> list[dict]:
                started = time.perf_counter()
                response = client.get("/api/movies/search", params={"q": term, "limit": limit})
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    print(f"search {term!r}: HTTP {response.status_code} {response.text[:120]}")
                    return []
                hits = response.json()["results"]
                print(
                    f"search {term!r}: {len(hits)} hits in {elapsed:.0f}ms -> "
                    f"{[hit['title'] for hit in hits]}",
                    flush=True,
                )
                return hits

            top = search("the matrix")
            check("'the matrix' finds The Matrix", bool(top) and top[0]["id"] == 2571)
            plain = search("matrix")
            check("'matrix' finds it too", bool(plain) and plain[0]["id"] == 2571)
            godfather = search("godfather")
            check(
                "popularity outranks a namesake",
                bool(godfather) and godfather[0]["title"] == "The Godfather",
                str([hit["title"] for hit in godfather[:3]]),
            )
            inception = search("inception")
            check(
                "an exact hit is not padded with lookalikes",
                len(inception) == len([h for h in inception if "inception" in h["title"].lower()]),
                str([hit["title"] for hit in inception]),
            )
            if has_trgm:
                typo = search("matrx")
                check(
                    "a misspelling is recovered",
                    bool(typo) and typo[0]["id"] == 2571,
                    str([hit["title"] for hit in typo[:3]]),
                )
                partial = search("godfath")
                check(
                    "a partial word is recovered",
                    bool(partial) and partial[0]["id"] == 858,
                    str([hit["title"] for hit in partial[:3]]),
                )
            else:
                # The fallback is allowed to miss; it is not allowed to fail the request.
                answer = client.get("/api/movies/search", params={"q": "matrx", "limit": 5})
                check(
                    "search answers without pg_trgm",
                    answer.status_code == 200,
                    f"HTTP {answer.status_code}",
                )
            check(
                "LIKE wildcards are escaped",
                client.get("/api/movies/search", params={"q": "%", "limit": 3}).status_code == 200,
            )

            rated: set[int] = set()
            for term, score in (("the matrix", 5.0), ("godfather", 4.5), ("inception", 4.0)):
                movie_id = search(term, limit=1)[0]["id"]
                response = client.post(
                    "/api/ratings", json={"movie_id": movie_id, "rating": score}, headers=headers
                )
                check(
                    f"rate {term} {score}", response.status_code in (200, 201), response.text[:100]
                )
                rated.add(movie_id)

            started = time.perf_counter()
            recs = client.get("/api/recommend", params={"limit": args.limit}, headers=headers)
            elapsed = (time.perf_counter() - started) * 1000
            picks = recs.json().get("recommendations", [])
            check(
                "recommendations come back with a reason",
                recs.status_code == 200
                and bool(picks)
                and all(pick.get("reason") for pick in picks),
                f"{len(picks)} picks in {elapsed:.0f}ms",
            )
            check(
                "rated films are excluded",
                not ({pick["movie"]["id"] for pick in picks} & rated),
            )
            for pick in picks[:4]:
                print(
                    f"  {pick['score']:.2f}  {pick['movie']['title']} — {pick['reason']}",
                    flush=True,
                )

            filtered = client.get(
                "/api/recommend",
                params={"limit": 5, "genres": "Drama", "year_min": 1970},
                headers=headers,
            )
            check("filtered recommendations", filtered.status_code == 200, filtered.text[:120])
            compare = client.get("/api/compare", params={"limit": 5}, headers=headers)
            check("algorithm comparison", compare.status_code == 200, compare.text[:120])

            if report.is_file():
                runs = client.get("/api/model-runs")
                check(
                    "model_runs serve the imported report",
                    runs.status_code == 200 and len(runs.json()) > 0,
                    f"{len(runs.json()) if runs.status_code == 200 else runs.text[:80]} rows",
                )
            check(
                "the new-release report answers",
                client.get("/api/new-releases/report").status_code == 200,
            )
            check(
                "recommendations stay behind auth",
                client.get("/api/recommend").status_code in (401, 403),
            )
    finally:
        engine.dispose()
        stop_server()

    print(f"\n=== {check.summary()} ===", flush=True)
    for name, _ok, detail in check.failed:
        print(f"FAILED: {name} [{detail}]", flush=True)
    return 0 if not check.failed else 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

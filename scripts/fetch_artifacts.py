#!/usr/bin/env python
"""Download and unpack the deployment artifacts bundle at container start.

Usage:
    uv run python scripts/fetch_artifacts.py --url "$ARTIFACTS_URL" --sha256 "$ARTIFACTS_SHA256"
    uv run python scripts/fetch_artifacts.py --url ... --force     # re-download anyway

Exit codes: 0 when the artifacts are ready (downloaded or already present), 1 on any failure.
Nothing is left half-written: a failed or unverifiable download removes its temporary file, and
extraction only accepts flat, regular files from a manifest-checked archive.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.artifact_bundle import (  # noqa: E402
    BundleError,
    download,
    is_complete,
    safe_extract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch the deployment artifacts bundle.")
    parser.add_argument("--url", required=True, help="Public URL of the artifacts archive.")
    parser.add_argument(
        "--sha256",
        default=None,
        help="Expected archive digest. Strongly recommended: without it a truncated download is "
        "only caught later, when an artifact fails to load.",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even when every required artifact is already present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.force and is_complete(args.processed_dir):
        print(f"Artifacts already present in {args.processed_dir}; nothing to fetch")
        return 0

    try:
        with tempfile.TemporaryDirectory(prefix="artifacts-") as staging:
            archive = download(
                args.url,
                Path(staging) / "artifacts.tar.gz",
                sha256=args.sha256,
            )
            written = safe_extract(archive, args.processed_dir)
    except (BundleError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - httpx and tarfile both raise many types
        print(f"error: could not fetch artifacts: {exc}", file=sys.stderr)
        return 1

    print(f"Extracted {len(written)} files into {args.processed_dir}")
    if not is_complete(args.processed_dir):
        print("error: the archive did not contain every required artifact", file=sys.stderr)
        return 1
    # The download itself is not kept: the extracted files are what the app reads.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

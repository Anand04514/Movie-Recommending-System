#!/usr/bin/env python
"""Package the serving artifacts into one checksummed archive for deployment.

Usage:
    uv run python scripts/build_artifact_bundle.py
    uv run python scripts/build_artifact_bundle.py --output dist/artifacts.tar.gz

The archive holds only what the API needs at request time plus the catalog it seeds from; the
chronological split and the evaluation report stay local. Publish the archive and its ``.sha256``
somewhere the host can reach (a release asset, S3/R2, a CDN) and point ``ARTIFACTS_URL`` and
``ARTIFACTS_SHA256`` at them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/build_artifact_bundle.py` from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.artifact_bundle import (  # noqa: E402
    OPTIONAL_FILES,
    REQUIRED_FILES,
    BundleError,
    build_bundle,
    sha256_of,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the deployment artifacts bundle.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/artifacts.tar.gz"),
        help="Archive to write. A sibling <name>.sha256 file is written too.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_bundle(args.processed_dir, args.output)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    digest = sha256_of(args.output)
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")

    total = sum(entry["bytes"] for entry in manifest["files"].values())
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(manifest['files'])} artifacts, {total / 1e6:.1f} MB before compression")
    print(f"  sha256 {digest}")
    print(f"  checksum file {checksum_path}")
    print()
    print("Publish the archive, then set on the host:")
    print(f"  ARTIFACTS_URL=<public url of {args.output.name}>")
    print(f"  ARTIFACTS_SHA256={digest}")
    print(f"  (optional artifacts: {', '.join(OPTIONAL_FILES)})")
    print(f"  (required artifacts: {', '.join(REQUIRED_FILES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deployment artifacts: build one download, verify it, unpack it safely (Phase 6).

`data/processed/` is derived and never committed, but a deployed API cannot serve a single
recommendation without it. Rather than baking ~25 MB of Parquet and sparse matrices into the
image, the build output is published once as a checksummed tarball and fetched at container
start. The archive is treated as untrusted input: only flat, regular files are accepted, so a
tampered bundle cannot write outside the processed directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1

# Exactly what the API loads at request time, plus the catalog it seeds from. Derived training
# files (the chronological split, the evaluation report) are deliberately left out.
REQUIRED_FILES = (
    "item_neighbors.npz",
    "svd_model.npz",
    "content_features.npz",
    "movie_id_map.parquet",
    "train_matrix.npz",
    "catalog.parquet",
)
OPTIONAL_FILES = ("hybrid_weights.json", "content_features.json")


class BundleError(RuntimeError):
    """The bundle is missing, corrupt, or not the thing it claimed to be."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(directory: str | Path) -> bool:
    """True when every required artifact is already on disk."""
    target = Path(directory)
    return all((target / name).is_file() for name in REQUIRED_FILES)


def build_bundle(
    processed_dir: str | Path,
    output: str | Path,
    *,
    extra: tuple[str, ...] = (),
) -> dict:
    """Write ``output`` (tar.gz) holding the serving artifacts and their hashes.

    Returns the manifest so a caller can print or publish the checksum of the archive itself.
    """
    source = Path(processed_dir)
    names = [*REQUIRED_FILES, *OPTIONAL_FILES, *extra]
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise BundleError(
            "cannot build a bundle without "
            + ", ".join(missing)
            + "; run scripts/prepare_data.py and scripts/train_and_precompute.py first"
        )

    present = [name for name in names if (source / name).is_file()]
    manifest = {
        "format": FORMAT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {
            name: {"sha256": sha256_of(source / name), "bytes": (source / name).stat().st_size}
            for name in present
        },
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name in present:
            archive.add(source / name, arcname=name, recursive=False)
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mtime = int(datetime.now(UTC).timestamp())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    return manifest


def verify_archive(path: str | Path, expected_sha256: str) -> None:
    """Raise unless the archive's digest matches what the publisher announced."""
    actual = sha256_of(Path(path))
    if actual != expected_sha256.strip().lower():
        raise BundleError(
            f"checksum mismatch: expected {expected_sha256.strip().lower()}, got {actual}"
        )


def _safe_member(member: tarfile.TarInfo) -> bool:
    name = member.name
    if not name or name.startswith("/") or ".." in Path(name).parts:
        return False
    if member.issym() or member.islnk() or member.isdir():
        return False
    # Flat archives only: no nested paths, no leading "./", no absolute names.
    return member.isfile() and Path(name).name == name


def safe_extract(archive_path: str | Path, destination: str | Path) -> list[str]:
    """Extract a flat archive of regular files; return the names written."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not _safe_member(member):
                raise BundleError(f"refusing to extract unsafe archive member: {member.name!r}")
            extracted = archive.extractfile(member)
            if extracted is None:  # pragma: no cover - guarded by isfile()
                raise BundleError(f"could not read {member.name!r} from the archive")
            data = extracted.read()
            (target / member.name).write_bytes(data)
            written.append(member.name)
    if MANIFEST_NAME not in written:
        raise BundleError("archive has no manifest; it is not an artifacts bundle")

    # An outer checksum proves the download; this proves the files agree with their own manifest.
    manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    for name, entry in manifest.get("files", {}).items():
        path = target / name
        if not path.is_file() or sha256_of(path) != entry.get("sha256"):
            raise BundleError(f"{name} does not match its manifest hash after extraction")
    return written


def read_manifest(archive_path: str | Path) -> dict:
    """The manifest inside an archive, without extracting anything else."""
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(MANIFEST_NAME)
        except KeyError as exc:
            raise BundleError("archive has no manifest") from exc
        extracted = archive.extractfile(member)
        if extracted is None:  # pragma: no cover
            raise BundleError("manifest is unreadable")
        return json.loads(extracted.read().decode("utf-8"))


def verify_contents(archive_path: str | Path) -> None:
    """Check every extracted file against the hashes recorded in its manifest."""
    manifest = read_manifest(archive_path)
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise BundleError("manifest does not list file hashes")
    with tarfile.open(archive_path, "r:gz") as archive:
        for name, entry in expected.items():
            try:
                member = archive.getmember(name)
            except KeyError as exc:
                raise BundleError(f"manifest lists {name!r}, which the archive lacks") from exc
            extracted = archive.extractfile(member)
            data = extracted.read() if extracted is not None else b""
            actual = hashlib.sha256(data).hexdigest()
            if actual != entry.get("sha256"):
                raise BundleError(f"{name} does not match the manifest hash")


def download(
    url: str,
    destination: str | Path,
    *,
    sha256: str | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 300.0,
) -> Path:
    """Stream ``url`` to ``destination``, verifying the digest before returning."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(transport=transport, follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        if sha256:
            verify_archive(target, sha256)
    except BaseException:
        # A partial or unverified archive must never be mistaken for a usable one.
        target.unlink(missing_ok=True)
        raise
    return target

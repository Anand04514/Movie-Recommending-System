"""Phase 6: the deployment artifacts bundle is complete, verified and safe to unpack.

The archive is treated as untrusted input on purpose: it travels over the network from a bucket to
a container, and it writes files into the directory the API loads models from.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from scripts.artifact_bundle import (
    MANIFEST_NAME,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    BundleError,
    build_bundle,
    download,
    is_complete,
    read_manifest,
    safe_extract,
    sha256_of,
    verify_archive,
)
from scripts.build_artifact_bundle import main as build_main
from scripts.fetch_artifacts import main as fetch_main


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    directory = tmp_path / "processed"
    directory.mkdir()
    for name in REQUIRED_FILES:
        (directory / name).write_bytes(f"{name} payload".encode())
    (directory / "hybrid_weights.json").write_text('{"weight": 0.8}', encoding="utf-8")
    (directory / "content_features.json").write_text('{"features": 334}', encoding="utf-8")
    # Not part of the bundle: derived training data stays local.
    (directory / "ratings_train.parquet").write_bytes(b"huge")
    return directory


@pytest.fixture
def bundle(processed: Path, tmp_path: Path) -> Path:
    output = tmp_path / "artifacts.tar.gz"
    build_bundle(processed, output)
    return output


def _write_tar(path: Path, members: dict[str, bytes], *, link: str | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if link is not None:
            info = tarfile.TarInfo(link)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
    return path


# ------------------------------------------------------------------ building


def test_the_bundle_holds_every_serving_artifact_and_nothing_else(
    processed: Path, tmp_path: Path
) -> None:
    output = tmp_path / "artifacts.tar.gz"
    manifest = build_bundle(processed, output)

    expected = set(REQUIRED_FILES) | set(OPTIONAL_FILES)
    assert set(manifest["files"]) == expected
    assert "ratings_train.parquet" not in manifest["files"]
    for name in expected:
        assert manifest["files"][name]["sha256"] == sha256_of(processed / name)

    with tarfile.open(output, "r:gz") as archive:
        assert set(archive.getnames()) == expected | {MANIFEST_NAME}

    # The manifest inside the archive is the same one, and every payload matches its hash.
    assert read_manifest(output)["files"].keys() == expected


def test_a_missing_catalog_is_named_rather_than_silently_left_out(
    processed: Path, tmp_path: Path
) -> None:
    (processed / "catalog.parquet").unlink()
    with pytest.raises(BundleError, match="catalog.parquet"):
        build_bundle(processed, tmp_path / "artifacts.tar.gz")


def test_is_complete_only_accepts_the_whole_set(processed: Path) -> None:
    assert is_complete(processed) is True
    (processed / "svd_model.npz").unlink()
    assert is_complete(processed) is False


# ------------------------------------------------------------------ verification


def test_a_corrupted_download_is_rejected(bundle: Path) -> None:
    good = sha256_of(bundle)
    verify_archive(bundle, good.upper())  # case-insensitive on purpose
    with pytest.raises(BundleError, match="checksum mismatch"):
        verify_archive(bundle, "0" * 64)


def test_a_payload_that_disagrees_with_its_manifest_is_caught(tmp_path: Path) -> None:
    payload = b"real model bytes"
    manifest = {
        "format": 1,
        "files": {"item_neighbors.npz": {"sha256": "a" * 64, "bytes": len(payload)}},
    }
    archive = _write_tar(
        tmp_path / "tampered.tar.gz",
        {"item_neighbors.npz": payload, MANIFEST_NAME: json.dumps(manifest).encode()},
    )
    with pytest.raises(BundleError, match="does not match its manifest"):
        safe_extract(archive, tmp_path / "out")


# ------------------------------------------------------------------ extraction


def test_extraction_writes_the_artifacts_and_verifies_them(bundle: Path, tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    written = safe_extract(bundle, destination)

    assert set(written) == set(REQUIRED_FILES) | set(OPTIONAL_FILES) | {MANIFEST_NAME}
    assert is_complete(destination)
    assert (destination / "catalog.parquet").read_bytes() == b"catalog.parquet payload"


def test_a_path_traversal_member_is_refused(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "evil.tar.gz",
        {"../escaped.txt": b"nope", MANIFEST_NAME: b"{}"},
    )
    with pytest.raises(BundleError, match="unsafe archive member"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_a_nested_path_is_refused_even_without_traversal(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "nested.tar.gz",
        {"sub/item_neighbors.npz": b"payload", MANIFEST_NAME: b"{}"},
    )
    with pytest.raises(BundleError, match="unsafe archive member"):
        safe_extract(archive, tmp_path / "out")


def test_a_symlink_is_refused(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "link.tar.gz",
        {MANIFEST_NAME: b"{}"},
        link="catalog.parquet",
    )
    with pytest.raises(BundleError, match="unsafe archive member"):
        safe_extract(archive, tmp_path / "out")


def test_an_archive_without_a_manifest_is_not_a_bundle(tmp_path: Path) -> None:
    archive = _write_tar(tmp_path / "plain.tar.gz", {"catalog.parquet": b"payload"})
    with pytest.raises(BundleError, match="no manifest"):
        safe_extract(archive, tmp_path / "out")


# ------------------------------------------------------------------ download


def test_download_verifies_and_keeps_the_archive(bundle: Path, tmp_path: Path) -> None:
    payload = bundle.read_bytes()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=payload))

    target = download(
        "https://example.invalid/a.tar.gz", tmp_path / "dl.tar.gz", transport=transport
    )
    assert target.read_bytes() == payload
    download(
        "https://example.invalid/a.tar.gz",
        tmp_path / "checked.tar.gz",
        sha256=hashlib.sha256(payload).hexdigest(),
        transport=transport,
    )


def test_a_bad_download_leaves_nothing_behind(bundle: Path, tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"truncated"))
    target = tmp_path / "bad.tar.gz"
    with pytest.raises(BundleError, match="checksum mismatch"):
        download(
            "https://example.invalid/a.tar.gz",
            target,
            sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
            transport=transport,
        )
    assert not target.exists()


def test_a_server_error_is_not_mistaken_for_an_archive(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500, content=b"oops"))
    target = tmp_path / "error.tar.gz"
    with pytest.raises(httpx.HTTPStatusError):
        download("https://example.invalid/a.tar.gz", target, transport=transport)
    assert not target.exists()


# ------------------------------------------------------------------ command line


def test_the_build_command_writes_the_archive_and_its_checksum(
    processed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "dist" / "artifacts.tar.gz"
    assert build_main(["--processed-dir", str(processed), "--output", str(output)]) == 0

    checksum_path = output.parent / f"{output.name}.sha256"
    checksum = checksum_path.read_text(encoding="utf-8")
    assert checksum.split()[0] == sha256_of(output)
    assert "ARTIFACTS_SHA256=" in capsys.readouterr().out


def test_the_build_command_explains_a_missing_artifact(
    processed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (processed / "item_neighbors.npz").unlink()
    output = tmp_path / "artifacts.tar.gz"
    assert build_main(["--processed-dir", str(processed), "--output", str(output)]) == 1
    assert "item_neighbors.npz" in capsys.readouterr().err
    assert not output.exists()


def test_the_fetch_command_does_nothing_when_the_artifacts_are_present(
    processed: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The URL is never used: a warm container must not spend a request, let alone a download.
    code = fetch_main(
        ["--url", "https://example.invalid/a.tar.gz", "--processed-dir", str(processed)]
    )
    assert code == 0
    assert "nothing to fetch" in capsys.readouterr().out


def test_the_fetch_command_reports_a_failed_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(*_args, **_kwargs):
        raise BundleError("checksum mismatch: expected abc, got def")

    monkeypatch.setattr("scripts.fetch_artifacts.download", explode)
    code = fetch_main(
        ["--url", "https://example.invalid/a.tar.gz", "--processed-dir", str(tmp_path)]
    )
    assert code == 1
    assert "checksum mismatch" in capsys.readouterr().err

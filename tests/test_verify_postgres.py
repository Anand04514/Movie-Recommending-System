"""The PostgreSQL verification script's own contract.

The live checks need a server, so they are run by hand (see the README); what is testable without
one is the argument handling and the pass/fail bookkeeping the exit status depends on.
"""

from __future__ import annotations

import pytest

from scripts.verify_postgres import Checks, build_parser


def test_exactly_one_target_is_required() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--bundled", "--database-url", "postgresql://x/y"])

    assert parser.parse_args(["--bundled"]).bundled is True
    assert (
        parser.parse_args(["--database-url", "postgresql://x/y"]).database_url == "postgresql://x/y"
    )


def test_writing_to_an_existing_schema_needs_force() -> None:
    """The check drops and recreates the schema, so it must not be a default."""
    parser = build_parser()

    assert parser.parse_args(["--bundled"]).force is False
    assert parser.parse_args(["--bundled", "--force"]).force is True


def test_failures_are_collected_and_reported(capsys: pytest.CaptureFixture[str]) -> None:
    check = Checks()

    assert check("good", True) is True
    assert check("bad", 0, "detail") is False

    assert check.summary() == "1/2 checks passed"
    assert [name for name, _ok, _detail in check.failed] == ["bad"]
    out = capsys.readouterr().out
    assert "PASS  good" in out
    assert "FAIL  bad  [detail]" in out

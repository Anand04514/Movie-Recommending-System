"""Test configuration: never touch a developer's local database."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DIR = Path(tempfile.mkdtemp(prefix="movie-recsys-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DIR / 'test.db').as_posix()}"

"""The column allowlist comes from the catalog, not from three hand-written lists.

`db.AGENT_COLUMNS`, the `Column` enum the model is given, and the SQL guard's
`ALLOWED_COLUMNS` used to be three separate statements of one fact, with nothing
keeping them in step. Adding a column meant editing three files; forgetting one
meant the guard silently disagreed with the schema the model had been handed.

Deriving them does not weaken the control. An allowlist is a security control;
*where it comes from* is not, provided the source is trusted and the model
cannot influence it. The catalog is read once, at startup, through a privileged
connection -- the same trust level that loaded the data.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import (  # noqa: E402
    AGENT_COLUMNS,
    DB_PATH,
    TENANT_COLUMN,
    introspect_columns,
    introspect_types,
    schema_description,
)
from secure_rls.security.spec import Column  # noqa: E402
from secure_rls.security.sql_guard import ALLOWED_COLUMNS  # noqa: E402


def test_all_three_allowlists_agree() -> None:
    """The drift this refactor exists to make impossible."""
    assert set(AGENT_COLUMNS) == {c.value for c in Column} == set(ALLOWED_COLUMNS)


def test_the_tenant_column_is_not_in_the_model_s_vocabulary() -> None:
    """Excluded at the source, so there is no word for it anywhere downstream."""
    assert TENANT_COLUMN not in AGENT_COLUMNS
    assert TENANT_COLUMN not in {c.value for c in Column}
    assert TENANT_COLUMN not in ALLOWED_COLUMNS
    with pytest.raises(ValueError):
        Column(TENANT_COLUMN)


def test_columns_match_the_actual_table(tmp_path: Path) -> None:
    copy = tmp_path / "e.db"
    shutil.copy(DB_PATH, copy)
    conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
    try:
        actual = {r[1] for r in conn.execute("PRAGMA table_info(employees_base)")}
    finally:
        conn.close()
    assert set(introspect_columns(copy)) == actual - {TENANT_COLUMN}


def test_a_new_column_needs_no_code_change(tmp_path: Path) -> None:
    """The point of the exercise, asserted rather than asserted-to.

    Add a column to the table and it appears in the allowlist. Nothing is
    edited, and nothing can be forgotten.
    """
    copy = tmp_path / "e.db"
    shutil.copy(DB_PATH, copy)
    before = introspect_columns(copy)

    conn = sqlite3.connect(copy)
    try:
        conn.execute("ALTER TABLE employees_base ADD COLUMN office TEXT")
        conn.execute("ALTER TABLE employees_base ADD COLUMN fte REAL")
        conn.commit()
    finally:
        conn.close()

    after = introspect_columns(copy)
    assert set(after) - set(before) == {"office", "fte"}
    assert TENANT_COLUMN not in after
    assert introspect_types(copy)["fte"] == "REAL"


def test_a_new_tenant_column_name_is_still_excluded(tmp_path: Path) -> None:
    """Which column carries the boundary is configuration, not discovery."""
    copy = tmp_path / "e.db"
    shutil.copy(DB_PATH, copy)
    assert TENANT_COLUMN not in introspect_columns(copy)


def test_missing_database_falls_back_rather_than_crashing(tmp_path: Path) -> None:
    """A fresh clone imports before `scripts/build_db.py` has ever run."""
    columns = introspect_columns(tmp_path / "does-not-exist.db")
    assert "salary" in columns
    assert TENANT_COLUMN not in columns


def test_schema_description_uses_declared_types() -> None:
    text = schema_description()
    assert "salary" in text and "INTEGER" in text
    assert "performance_score" in text and "REAL" in text
    assert TENANT_COLUMN not in text, "the tenant column must not be described to the model"

"""Adding another tenant-scoped table must not open a path to it.

The question this file answers: if someone adds `contracts_base` alongside
`employees_base`, what can the agent do with it?

The answer should be "nothing, without a deliberate change", and it now is --
but the first version of the boundary got this subtly wrong, and the way it was
wrong is the reason these tests exist. SQLite does not consult the authorizer
for a join key named through `USING` or `NATURAL JOIN`. With the data file open
as `main`, this read another table without ever asking:

    SELECT e.name FROM employees e JOIN contracts_base c USING (user_id)

`ON` asked and was denied; `USING` did not ask. The fix was not to enumerate
join syntax but to remove the thing being named: the agent's connection is a
private in-memory database, and the data file is attached only long enough to
copy the tenant's rows out. See ADR-0006.
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
    TenantDatabase,
    introspect_columns,
    tenant_connection,
)
from secure_rls.security.sql_guard import SqlRejected, guard_sql  # noqa: E402


@pytest.fixture
def db_with_second_table(tmp_path: Path) -> Path:
    """A copy of the database with a second tenant-scoped table bolted on."""
    copy = tmp_path / "e.db"
    shutil.copy(DB_PATH, copy)
    conn = sqlite3.connect(copy)
    try:
        conn.execute(
            "CREATE TABLE contracts_base ("
            " contract_id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL,"
            " user_id INTEGER, rate INTEGER, secret_clause TEXT)"
        )
        conn.executemany(
            "INSERT INTO contracts_base VALUES (?,?,?,?,?)",
            [
                (1, "acme", 2, 900, "acme clause"),
                (2, "beta", 501, 800, "BETA SECRET CLAUSE"),
                (3, "gamma", 801, 700, "GAMMA SECRET CLAUSE"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return copy


def test_a_new_table_does_not_enter_the_model_s_vocabulary(db_with_second_table: Path) -> None:
    """Introspection describes one relation, so the model is never told of another."""
    columns = introspect_columns(db_with_second_table)
    assert set(columns) == set(AGENT_COLUMNS)
    for leaked in ("rate", "secret_clause", "contract_id"):
        assert leaked not in columns


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT secret_clause FROM contracts_base",
        "SELECT rate FROM contracts_base LIMIT 1",
        # The join forms. `USING` and `NATURAL` are the ones the authorizer
        # never sees; they are the reason this file exists.
        "SELECT e.name FROM employees e JOIN contracts_base c USING (user_id)",
        "SELECT e.name FROM employees e NATURAL JOIN contracts_base",
        "SELECT e.name FROM employees e JOIN contracts_base c ON e.user_id = c.user_id",
        # Existence and inference channels, not just column reads.
        "SELECT (SELECT COUNT(*) FROM contracts_base)",
        "SELECT COUNT(*) FROM employees WHERE user_id IN (SELECT user_id FROM contracts_base)",
        "SELECT name FROM employees UNION SELECT secret_clause FROM contracts_base",
    ],
)
def test_layer_4_alone_cannot_reach_the_new_table(db_with_second_table: Path, sql: str) -> None:
    """No prompt, no SQL guard -- the engine itself must refuse.

    Invariant 7: if a query is only blocked because a higher layer caught it,
    it is not blocked.
    """
    conn = tenant_connection("acme", db_with_second_table)
    try:
        with pytest.raises(sqlite3.Error):
            conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_the_new_table_is_absent_rather_than_merely_denied(db_with_second_table: Path) -> None:
    """The distinction matters: absence needs no rule to be right.

    A denial depends on the authorizer being asked. Detaching the source means
    the parser refuses first, for every syntax, including ones nobody thought
    to test.
    """
    conn = tenant_connection("acme", db_with_second_table)
    try:
        with pytest.raises(sqlite3.Error, match="no such table"):
            conn.execute("SELECT 1 FROM contracts_base")
        with pytest.raises(sqlite3.Error, match="no such table"):
            conn.execute("SELECT 1 FROM employees_base")
    finally:
        conn.close()


def test_the_source_cannot_be_reattached(db_with_second_table: Path) -> None:
    """Otherwise detaching would be a speed bump rather than a boundary."""
    conn = tenant_connection("acme", db_with_second_table)
    try:
        with pytest.raises(sqlite3.Error):
            conn.execute("ATTACH DATABASE ? AS src", (str(db_with_second_table),))
    finally:
        conn.close()


def test_layer_3_also_refuses_an_unknown_relation() -> None:
    """Defence in depth: the guard rejects the table name before SQL is run."""
    for sql in (
        "SELECT rate FROM contracts_base",
        "SELECT e.name FROM employees e JOIN contracts_base c USING (user_id)",
    ):
        with pytest.raises(SqlRejected, match="unknown table"):
            guard_sql(sql)


def test_the_tenant_s_own_data_still_works(db_with_second_table: Path) -> None:
    """A boundary that also breaks the product is not a fix."""
    handle = TenantDatabase("acme", db_with_second_table)
    try:
        assert handle.execute("SELECT COUNT(*) AS n FROM employees")[0]["n"] == 500
        assert handle.execute("SELECT DISTINCT department FROM employees")
    finally:
        handle.close()


def test_a_shadow_relation_is_refused(tmp_path: Path) -> None:
    """A table in the data file named `employees` must not become the agent's."""
    copy = tmp_path / "e.db"
    shutil.copy(DB_PATH, copy)
    conn = sqlite3.connect(copy)
    try:
        conn.execute("CREATE TABLE employees (user_id INTEGER, name TEXT)")
        conn.execute("INSERT INTO employees VALUES (9999, 'NOT A TENANT ROW')")
        conn.commit()
    finally:
        conn.close()

    handle = TenantDatabase("acme", copy)
    try:
        rows = handle.execute("SELECT COUNT(*) AS n FROM employees")
        assert rows[0]["n"] == 500, "the agent read the shadow table, not its own slice"
    finally:
        handle.close()

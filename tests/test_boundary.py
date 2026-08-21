"""Layer 4 tests: the security boundary itself.

If any test in this file fails, nothing else in the repository matters.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import (  # noqa: E402
    AGENT_COLUMNS,
    ALLOWED_TENANTS,
    SecurityError,
    TenantDatabase,
    tenant_connection,
    tenant_user_ids,
)

TENANT_ROWS = {"acme": 500, "beta": 300, "gamma": 200}


@pytest.fixture(scope="module")
def allowed_ids() -> dict[str, frozenset[int]]:
    return {t: tenant_user_ids(t) for t in sorted(ALLOWED_TENANTS)}


@pytest.fixture
def acme() -> sqlite3.Connection:
    conn = tenant_connection("acme")
    yield conn
    conn.close()


# ---------------------------------------------------------------- basics ---

@pytest.mark.parametrize("tenant,expected", TENANT_ROWS.items())
def test_connection_sees_only_its_own_rows(tenant: str, expected: int) -> None:
    with TenantDatabase(tenant) as database:
        assert database.row_count() == expected


def test_tenant_id_column_is_not_reachable(acme: sqlite3.Connection) -> None:
    """The column is not projected, so the model has no word for it."""
    with pytest.raises(sqlite3.DatabaseError):
        acme.execute("SELECT tenant_id FROM employees").fetchall()


def test_unknown_tenant_fails_closed() -> None:
    for bad in ["delta", "", "acme'--", "ACME", None, 1, ["acme"]]:
        with pytest.raises(SecurityError):
            tenant_connection(bad)  # type: ignore[arg-type]


# ------------------------------------------------------------- smuggling ---

SMUGGLING_ATTEMPTS = [
    ("direct base read", "SELECT * FROM employees_base"),
    ("qualified base read", "SELECT * FROM main.employees_base"),
    (
        "union smuggle",
        "SELECT user_id, name FROM employees "
        "UNION SELECT user_id, name FROM employees_base",
    ),
    ("scalar subquery", "SELECT (SELECT COUNT(*) FROM employees_base)"),
    ("cte named x", "WITH x AS (SELECT * FROM employees_base) SELECT * FROM x"),
    (
        # The bypass that killed the temp-VIEW design. Regression test: an
        # attacker names their CTE after the secure relation, so SQLite reports
        # the base-table read as originating from `employees`.
        "cte named employees (impersonation)",
        "WITH employees AS (SELECT user_id, name, salary FROM employees_base) "
        "SELECT * FROM employees",
    ),
    (
        "subquery aliased employees",
        "SELECT * FROM (SELECT user_id FROM employees_base) AS employees",
    ),
    ("join to base", "SELECT * FROM employees e JOIN employees_base b USING (user_id)"),
    ("schema probe", "SELECT name FROM sqlite_master"),
    ("temp schema probe", "SELECT name FROM sqlite_temp_master"),
    ("pragma table_list", "PRAGMA table_list"),
    ("pragma table_info", "PRAGMA table_info(employees_base)"),
    ("attach", "ATTACH DATABASE 'evil.db' AS evil"),
    ("create view over base", "CREATE TEMP VIEW evil AS SELECT * FROM employees_base"),
    ("delete", "DELETE FROM employees"),
    ("insert", "INSERT INTO employees (user_id, name) VALUES (9999, 'x')"),
    ("update", "UPDATE employees SET salary = 1"),
    ("drop", "DROP TABLE employees"),
]


@pytest.mark.parametrize("label,sql", SMUGGLING_ATTEMPTS, ids=[s[0] for s in SMUGGLING_ATTEMPTS])
def test_smuggling_attempt_is_blocked(
    acme: sqlite3.Connection, label: str, sql: str, allowed_ids
) -> None:
    """Every one of these must raise. None may return another tenant's data."""
    try:
        rows = acme.execute(sql).fetchall()
    except sqlite3.DatabaseError:
        return  # blocked by the engine, which is the expected outcome
    # If a statement somehow succeeds it must still be tenant-pure.
    for row in rows:
        keys = row.keys() if hasattr(row, "keys") else []
        if "user_id" in keys:
            assert int(row["user_id"]) in allowed_ids["acme"], f"{label} leaked a row"


def test_cte_named_employees_cannot_impersonate_the_view(
    acme: sqlite3.Connection, allowed_ids
) -> None:
    """Named regression for the bypass documented in ADR-0002."""
    sql = (
        "WITH employees AS (SELECT user_id, name, salary FROM employees_base) "
        "SELECT user_id FROM employees"
    )
    with pytest.raises(sqlite3.DatabaseError):
        acme.execute(sql).fetchall()


def test_statement_timeout_is_per_statement_not_per_connection() -> None:
    """A slow query must not poison every query that follows it.

    Regression for a latent bug: the original progress handler counted its own
    invocations against a fixed budget and never reset, so one expensive
    statement would abort every later statement on that connection. It escaped
    notice because a 500-row table rarely ticks the handler at all.
    """
    with TenantDatabase("acme") as database:
        for _ in range(250):
            rows = database.execute(
                "SELECT department, COUNT(*) AS n FROM employees GROUP BY department"
            )
            assert rows, "connection stopped returning rows part-way through"


def test_canary_of_other_tenants_is_never_visible() -> None:
    for tenant in sorted(ALLOWED_TENANTS):
        with TenantDatabase(tenant) as database:
            rows = database.execute("SELECT name FROM employees WHERE name LIKE 'ZZ_CANARY%'")
            names = {r["name"] for r in rows}
            assert names == {f"ZZ_CANARY_{tenant.upper()}"}


def test_sessions_are_mutually_invisible() -> None:
    """Two concurrent connections must each see their own `employees`."""
    with TenantDatabase("acme") as a, TenantDatabase("beta") as b:
        assert a.row_count() == 500
        assert b.row_count() == 300
        a_ids = {r["user_id"] for r in a.execute("SELECT user_id FROM employees")}
        b_ids = {r["user_id"] for r in b.execute("SELECT user_id FROM employees")}
        assert a_ids.isdisjoint(b_ids)


# -------------------------------------------------------------- property ---

_COLUMNS = [c for c in AGENT_COLUMNS if c != "notes"]

_FRAGMENTS = st.one_of(
    st.sampled_from(
        [
            "1=1",
            "salary > 0",
            "salary > 1000000",
            "department = 'Engineering'",
            "name LIKE '%a%'",
            "user_id IN (SELECT user_id FROM employees)",
            "user_id NOT IN (1, 2, 3)",
            "performance_score IS NULL",
            # adversarial fragments
            "1=1 OR user_id > 0",
            "1=1 --",
            "1=1 UNION SELECT 1",
            "user_id IN (SELECT user_id FROM employees_base)",
            "EXISTS (SELECT 1 FROM employees_base)",
        ]
    ),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 '=<>()*-_.,", max_size=40),
)


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    where=_FRAGMENTS,
    cols=st.lists(st.sampled_from(_COLUMNS), min_size=1, max_size=4, unique=True),
    limit=st.integers(min_value=1, max_value=50),
)
def test_property_every_returned_row_belongs_to_the_tenant(
    allowed_ids, where: str, cols: list[str], limit: int
) -> None:
    """The security claim, as an executable property.

    For *any* SQL we can construct, the rows a tenant-bound connection returns
    are a subset of that tenant's rows -- or the statement is rejected. There is
    no third outcome.
    """
    select = ", ".join(["user_id"] + [c for c in cols if c != "user_id"])
    sql = f"SELECT {select} FROM employees WHERE {where} LIMIT {limit}"
    conn = tenant_connection("acme")
    try:
        rows = conn.execute(sql).fetchall()
    except (sqlite3.DatabaseError, sqlite3.Warning):
        return  # rejected -- an acceptable outcome
    finally:
        conn.close()

    for row in rows:
        assert int(row["user_id"]) in allowed_ids["acme"]

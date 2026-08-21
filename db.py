"""Layer 4 -- the security boundary.

Everything else in this codebase is defence in depth. This module is the part
that makes the guarantee true: a connection handed out by `tenant_connection`
cannot read another tenant's rows, no matter what SQL is executed on it.

Design
------
SQLite has no GRANT, no roles and no row-level security policies, so we build
an equivalent out of two engine-level primitives:

1. A per-connection TEMP TABLE named ``employees`` holding *only* the calling
   tenant's rows, materialised before the connection is handed to any agent
   code. Temp tables are private to their connection, so two concurrent
   sessions genuinely see two different ``employees``.

2. An authorizer callback (``Connection.set_authorizer``). SQLite consults it
   during statement preparation for every table and column touched -- including
   inside subqueries, CTEs and set operations -- and it sits *below* any SQL our
   code or the model composes. It denies ``employees_base`` unconditionally,
   plus ``sqlite_master``, ATTACH, PRAGMA and all writes.

Why a temp TABLE and not a temp VIEW
------------------------------------
The obvious design is a temp VIEW over ``employees_base`` with the authorizer
allowing base-table reads only when SQLite reports the read as originating from
that view (the callback's ``source`` argument). That design is broken, and the
break is worth knowing about:

    WITH employees AS (SELECT * FROM employees_base) SELECT * FROM employees

SQLite sets ``source`` to the name of the *CTE* performing the read. An
attacker who names their CTE ``employees`` is indistinguishable from the secure
view, and the authorizer waves them through -- all three tenants returned. See
docs/adr/0002-sqlite-authorizer.md and the regression test in
tests/test_boundary.py::test_cte_named_employees_cannot_impersonate_the_view.

Materialising instead means ``employees_base`` is denied with no exceptions at
all, so there is no ``source`` value for an attacker to forge. The cost is a
per-session copy of the tenant's rows, which is the right trade at this scale
and which disappears entirely on a platform with native RLS (see
docs/adr/0004-postgres-parity.md).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "employees.db"
CSV_PATH = ROOT / "employees.csv"

#: The only tenants that exist. Anything else fails closed.
ALLOWED_TENANTS = frozenset({"acme", "beta", "gamma"})

#: The real table. The agent never sees this name work.
BASE_TABLE = "employees_base"

#: The per-connection relation the agent is allowed to query.
AGENT_TABLE = "employees"

#: Columns exposed to the agent. `tenant_id` is deliberately not among them:
#: inside a session there is only one tenant, so the column carries no
#: information and its absence removes a word from the model's vocabulary.
AGENT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "name",
    "department",
    "salary",
    "performance_score",
    "hire_date",
    "notes",
)

#: Hard ceiling on rows returned to the agent from any single statement.
MAX_ROWS = 500

#: Rough statement timeout, enforced via the VM-instruction progress handler.
_PROGRESS_INSTRUCTIONS = 100_000
_MAX_PROGRESS_CALLS = 200


class SecurityError(RuntimeError):
    """Raised when a request violates the security model. Never suppressed."""


# --------------------------------------------------------------------------
# Privileged operations. These run at startup, never on behalf of the agent.
# --------------------------------------------------------------------------

def build_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> int:
    """Load the CSV into SQLite. Privileged: called by scripts, never by tools."""
    import csv as _csv

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"""
            CREATE TABLE {BASE_TABLE} (
                user_id           INTEGER PRIMARY KEY,
                tenant_id         TEXT    NOT NULL,
                name              TEXT    NOT NULL,
                department        TEXT    NOT NULL,
                salary            INTEGER NOT NULL,
                performance_score REAL,
                hire_date         TEXT    NOT NULL,
                notes             TEXT
            )
            """
        )
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = [
                (
                    int(r["user_id"]),
                    r["tenant_id"],
                    r["name"],
                    r["department"],
                    int(r["salary"]),
                    float(r["performance_score"]) if r["performance_score"] else None,
                    r["hire_date"],
                    r["notes"] or None,
                )
                for r in _csv.DictReader(fh)
            ]
        conn.executemany(
            f"INSERT INTO {BASE_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        conn.execute(f"CREATE INDEX idx_tenant ON {BASE_TABLE}(tenant_id)")
        conn.execute(f"CREATE INDEX idx_tenant_dept ON {BASE_TABLE}(tenant_id, department)")
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def tenant_user_ids(tenant: str, db_path: Path = DB_PATH) -> frozenset[int]:
    """Every ``user_id`` legitimately belonging to `tenant`.

    Privileged, and read once at startup. The output guard (layer 5) uses this
    as an *independent* source of truth: it must not verify the tenant boundary
    using the same code path that enforced it, or a single bug would satisfy
    both the enforcement and its own audit.
    """
    _require_known_tenant(tenant)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT user_id FROM {BASE_TABLE} WHERE tenant_id = ?", (tenant,)
        ).fetchall()
        return frozenset(int(r[0]) for r in rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------

def _require_known_tenant(tenant: object) -> str:
    """Fail closed. A tenant value never reaches SQL without passing here."""
    if not isinstance(tenant, str) or tenant not in ALLOWED_TENANTS:
        raise SecurityError(f"unknown tenant: {tenant!r}")
    return tenant


# Actions the agent's connection may perform at all. Anything not listed is
# denied -- fail closed, so a future SQLite version adding a new action code
# cannot silently widen what the agent can do.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _assert_no_shadow_relation(conn: sqlite3.Connection) -> None:
    """The authorizer permits reads of `employees` without checking the schema
    name in one case (see below), which is only safe while the main database
    holds no relation of that name. Verify it instead of trusting it.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM main.sqlite_master WHERE name = ? AND type IN ('table','view')",
        (AGENT_TABLE,),
    ).fetchone()
    if row[0]:
        raise SecurityError(
            f"main database contains a relation named {AGENT_TABLE!r}, which would "
            f"shadow the tenant-scoped temp table"
        )


def _make_authorizer(tenant: str):
    """Build the callback SQLite consults for every table/column access."""

    def authorizer(action: int, arg1: Any, arg2: Any, dbname: Any, source: Any) -> int:
        if action == sqlite3.SQLITE_READ:
            # The tenant's own materialised table, and nothing else. Note there
            # is no exception for BASE_TABLE: not for views, not for CTEs, not
            # for any value of `source`. That is the whole point.
            #
            # `dbname` is "temp" for ordinary column reads but None for reads
            # that name no column -- COUNT(*) is the common case. Both are
            # accepted for AGENT_TABLE, which is safe because the main database
            # contains no relation of that name; `_assert_no_shadow_relation`
            # enforces that rather than assuming it.
            if arg1 == AGENT_TABLE and dbname in ("temp", None):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        if action in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK

        # ATTACH, DETACH, PRAGMA, every DDL and DML verb, everything unknown.
        return sqlite3.SQLITE_DENY

    return authorizer


def tenant_connection(tenant: str, db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a connection that can only ever see `tenant`'s rows.

    Ordering matters and is not incidental:
      1. validate the tenant against the allowlist (fail closed),
      2. open read-only,
      3. materialise the tenant's rows into a private temp table,
      4. *then* install the authorizer and shut the door.

    Installing the authorizer earlier would block step 3; installing it later
    would leave a window in which the base table is readable.
    """
    tenant = _require_known_tenant(tenant)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    _assert_no_shadow_relation(conn)

    columns = ", ".join(AGENT_COLUMNS)
    # `tenant` is bound as a parameter, never interpolated -- even though it has
    # already been checked against a frozenset of three literals. Defence in
    # depth means not relying on the check one line above.
    conn.execute(
        f"CREATE TEMP TABLE {AGENT_TABLE} AS "
        f"SELECT {columns} FROM {BASE_TABLE} WHERE tenant_id = ?",
        (tenant,),
    )
    conn.execute(f"CREATE INDEX temp.idx_dept ON {AGENT_TABLE}(department)")

    conn.set_authorizer(_make_authorizer(tenant))
    _install_timeout(conn)
    return conn


def _install_timeout(conn: sqlite3.Connection) -> None:
    """Abort runaway statements. A cartesian join is a denial-of-service too."""
    calls = {"n": 0}

    def handler() -> int:
        calls["n"] += 1
        return 1 if calls["n"] > _MAX_PROGRESS_CALLS else 0

    conn.set_progress_handler(handler, _PROGRESS_INSTRUCTIONS)


class TenantDatabase:
    """A tenant-bound handle. The only way agent tools reach data.

    Holds the connection so that no caller can pass a different tenant later:
    the binding happens once, in the constructor, from the session principal.
    """

    def __init__(self, tenant: str, db_path: Path = DB_PATH) -> None:
        self.tenant = _require_known_tenant(tenant)
        self._conn = tenant_connection(self.tenant, db_path)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> list[dict]:
        """Run a statement on the bound connection and return capped rows."""
        try:
            cursor = self._conn.execute(sql, tuple(params or ()))
        except sqlite3.DatabaseError as exc:
            # Sanitised: the raw message can name the base table, which tells an
            # attacker what to aim at next (layer 5, error-message leakage).
            raise SecurityError(f"query rejected by the database: {exc}") from exc
        rows = cursor.fetchmany(MAX_ROWS)
        return [dict(r) for r in rows]

    def columns(self) -> tuple[str, ...]:
        return AGENT_COLUMNS

    def row_count(self) -> int:
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {AGENT_TABLE}").fetchone()[0])

    def sample(self, n: int = 3) -> list[dict]:
        """A few of the tenant's own rows, for grounding the model's schema view."""
        rows = self._conn.execute(
            f"SELECT {', '.join(AGENT_COLUMNS)} FROM {AGENT_TABLE} LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TenantDatabase":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def schema_description() -> str:
    """The virtual schema shown to the model. Names only what it may query."""
    lines = [f"TABLE {AGENT_TABLE} (one row per employee in your organisation)"]
    types = {
        "user_id": "INTEGER, unique employee id",
        "name": "TEXT, full name",
        "department": "TEXT, e.g. Engineering, Sales, Marketing, Finance, Operations, Support",
        "salary": "INTEGER, annual gross in EUR",
        "performance_score": "REAL 1.0-5.0, may be NULL",
        "hire_date": "TEXT, ISO date YYYY-MM-DD",
        "notes": "TEXT free-form HR note, may be NULL",
    }
    lines += [f"  {col:18} {types[col]}" for col in AGENT_COLUMNS]
    return "\n".join(lines)


def iter_tenants() -> Iterable[str]:
    return sorted(ALLOWED_TENANTS)

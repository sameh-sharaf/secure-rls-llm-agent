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
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "employees.db"
CSV_PATH = ROOT / "employees.csv"

#: The only tenants that exist. Anything else fails closed.
ALLOWED_TENANTS = frozenset({"acme", "beta", "gamma"})

#: The real table. The agent never sees this name work.
BASE_TABLE = "employees_base"

#: The per-connection relation the agent is allowed to query.
AGENT_TABLE = "employees"

#: The column that identifies the tenant. Configuration, not discovery: which
#: column carries the boundary is the one thing a catalog cannot tell you.
TENANT_COLUMN = "tenant_id"

#: Fallback used only when the database has not been built yet -- a fresh clone
#: before `scripts/build_db.py`, or a test that never touches a real file.
_FALLBACK_COLUMNS: tuple[str, ...] = (
    "user_id", "name", "department", "salary", "performance_score", "hire_date", "notes",
)


def introspect_columns(db_path: Path = DB_PATH) -> tuple[str, ...]:
    """Columns the agent may see, read from the database catalog.

    The allowlist used to be written out by hand in three places -- here, the
    `Column` enum the model is given, and the SQL guard's own set. Three copies
    of one truth, each maintained separately, which is a drift waiting to
    happen: add a column and the guard silently disagrees with the schema.

    Deriving it from the catalog does not weaken anything. An allowlist is a
    security control; *where it comes from* is not, so long as the source is
    trusted and the model cannot influence it. This reads the catalog once, at
    startup, through a privileged connection -- the same trust level that loads
    the data in the first place -- and the model never touches it.

    `TENANT_COLUMN` is excluded here rather than filtered later. Inside a
    session there is exactly one tenant, so the column carries no information,
    and leaving it out removes the word from the model's vocabulary entirely.
    """
    if not db_path.exists():
        return _FALLBACK_COLUMNS
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"PRAGMA table_info({BASE_TABLE})").fetchall()
    except sqlite3.DatabaseError:
        return _FALLBACK_COLUMNS
    finally:
        conn.close()
    found = tuple(r[1] for r in rows if r[1] != TENANT_COLUMN)
    return found or _FALLBACK_COLUMNS


def introspect_types(db_path: Path = DB_PATH) -> dict[str, str]:
    """Declared SQL type per exposed column, for the schema shown to the model."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"PRAGMA table_info({BASE_TABLE})").fetchall()
    except sqlite3.DatabaseError:
        return {}
    finally:
        conn.close()
    return {r[1]: (r[2] or "TEXT") for r in rows if r[1] != TENANT_COLUMN}


#: Resolved once at import. Add a column to the table and it appears here, in
#: the model's vocabulary and in the SQL guard together -- there is nothing to
#: keep in step by hand.
AGENT_COLUMNS: tuple[str, ...] = introspect_columns()

#: Hard ceiling on rows returned to the agent from any single statement.
MAX_ROWS = 500

#: Statement timeout, enforced via the VM-instruction progress handler. SQLite
#: calls the handler every `_PROGRESS_INSTRUCTIONS` virtual-machine steps; a
#: non-zero return aborts the running statement.
_PROGRESS_INSTRUCTIONS = 10_000
STATEMENT_TIMEOUT_SECONDS = 5.0


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
            f"INSERT INTO {BASE_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows  # nosec B608
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
            f"SELECT user_id FROM {BASE_TABLE} WHERE tenant_id = ?", (tenant,)  # nosec B608
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


def _assert_nothing_reachable_but_the_agent_table(conn: sqlite3.Connection) -> None:
    """The connection must hold the tenant's slice and nothing else.

    Two properties, checked rather than assumed:

    * `main` is empty, so there is no relation for a query to name -- in
      particular none called `employees`, which would shadow the temp table in
      the one authorizer branch that accepts a null schema name (`COUNT(*)`);
    * `temp` holds exactly the agent table.

    Verified after the source is detached, because that is the moment the
    claim has to be true.
    """
    for schema, expected in (("main", set()), ("temp", {AGENT_TABLE})):
        found = {
            r[0]
            for r in conn.execute(
                f"SELECT name FROM {schema}.sqlite_master WHERE type IN ('table','view')"  # nosec B608
            )
            if not r[0].startswith("sqlite_")
        }
        if found != expected:
            raise SecurityError(
                f"{schema} schema holds {sorted(found)}, expected {sorted(expected)}"
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
            # contains no relation of that name;
            # `_assert_nothing_reachable_but_the_agent_table` enforces that
            # rather than assuming it.
            if arg1 == AGENT_TABLE and dbname in ("temp", None):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        if action in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK

        # ATTACH, DETACH, PRAGMA, every DDL and DML verb, everything unknown.
        return sqlite3.SQLITE_DENY

    return authorizer


def tenant_connection(
    tenant: str, db_path: Path = DB_PATH, clock: dict | None = None
) -> sqlite3.Connection:
    """Return a connection that can only ever see `tenant`'s rows.

    Ordering matters and is not incidental:
      1. validate the tenant against the allowlist (fail closed),
      2. open a private, empty in-memory database,
      3. attach the real file read-only and copy the tenant's rows out of it,
      4. detach it, so no relation but the agent's own remains reachable,
      5. *then* install the authorizer and shut the door.

    Installing the authorizer earlier would block step 3; installing it later
    would leave a window in which the base table is readable.

    Step 4 is why `main` is a scratch database rather than the data file. The
    authorizer alone is not quite enough: SQLite does not consult it for a join
    key named through `USING` or `NATURAL JOIN`, so with the file open as
    `main`, `employees JOIN other_table USING (user_id)` reads `other_table`
    without ever asking. `ON` asks; `USING` does not. Rather than enumerate
    which syntax the callback covers, detaching leaves nothing to name -- the
    reply becomes "no such table" from the parser, before authorization is even
    a question. See ADR-0006.

    The cost is a per-session copy of the tenant's rows in memory, which is
    right for this dataset and would need revisiting for a large one.
    """
    tenant = _require_known_tenant(tenant)

    # `check_same_thread=False` because Streamlit runs each rerun on a
    # different thread from its pool, and a thread-bound connection raises
    # ProgrammingError the moment a rerun lands elsewhere.
    #
    # This does not weaken the boundary. Tenant binding is a property of the
    # *connection* -- the materialised temp table and the installed authorizer
    # -- and neither is thread-scoped; a different thread reaching this
    # connection still sees exactly one tenant's rows and still cannot name the
    # base table. What it does remove is sqlite3's protection against
    # *concurrent* use, so `TenantDatabase` serialises every statement behind a
    # lock. Trading a crash for a silent data race would be a poor deal.
    conn = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    columns = ", ".join(AGENT_COLUMNS)
    source = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn.execute("ATTACH DATABASE ? AS src", (source,))
    try:
        # `tenant` is bound as a parameter, never interpolated -- even though it
        # has already been checked against a frozenset of three literals.
        # Defence in depth means not relying on the check one line above.
        conn.execute(
            f"CREATE TEMP TABLE {AGENT_TABLE} AS "  # nosec B608
            f"SELECT {columns} FROM src.{BASE_TABLE} WHERE tenant_id = ?",
            (tenant,),
        )
    finally:
        conn.execute("DETACH DATABASE src")

    conn.execute(f"CREATE INDEX temp.idx_dept ON {AGENT_TABLE}(department)")
    _assert_nothing_reachable_but_the_agent_table(conn)

    conn.set_authorizer(_make_authorizer(tenant))
    _install_timeout(conn, clock if clock is not None else {})
    return conn


def naive_connection(
    tenant: str, db_path: Path = DB_PATH, clock: dict | None = None
) -> sqlite3.Connection:
    """Layer 4 as most implementations of this brief actually build it.

    **Not reachable from the application.** The only callers are the ablation
    study and the lab panel in the Security tab, both of which exist to show
    what this design is worth by standing next to something weaker.

    The difference from `tenant_connection` is the entire argument of this
    project, so it is worth being precise about what is missing here rather
    than describing it as "insecure":

    * the base table is present and reachable -- nothing was materialised and
      nothing was detached, so `employees_base` names a real relation;
    * there is no authorizer, so `sqlite_master`, `PRAGMA` and every other
      table are readable;
    * the tenant filter is a `WHERE` clause in a view that application code
      remembered to write, which is exactly the "developer remembered it"
      pattern row-level security exists to replace.

    A statement that never mentions the view -- `SELECT * FROM employees_base`
    -- is answered in full, for all three tenants. That is the point.
    """
    _require_known_tenant(tenant)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    columns = ", ".join(AGENT_COLUMNS)
    # Interpolated rather than bound, because a view definition cannot carry a
    # parameter -- which is itself part of why this design is the fragile one.
    # `tenant` has been checked against the allowlist immediately above.
    conn.execute(
        f"CREATE TEMP VIEW {AGENT_TABLE} AS "  # nosec B608
        f"SELECT {columns} FROM {BASE_TABLE} WHERE tenant_id = '{tenant}'"
    )
    _install_timeout(conn, clock if clock is not None else {})
    return conn  # no authorizer: employees_base stays reachable


def _install_timeout(conn: sqlite3.Connection, clock: dict) -> None:
    """Abort runaway statements. A cartesian join is a denial-of-service too.

    The deadline is stored in a dict owned by the caller and reset before each
    statement, so this is a *per-statement* timeout.

    The obvious implementation -- count handler invocations and abort past a
    fixed number -- is wrong in a way that is easy to miss: the counter is
    per-connection and never resets, so a single expensive query permanently
    poisons the connection and every later statement aborts immediately. It
    only escapes notice because a 500-row table rarely ticks the handler at
    all. A deadline has no such state to leak between statements.
    """

    def handler() -> int:
        deadline = clock.get("deadline")
        if deadline is None:
            return 0
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(handler, _PROGRESS_INSTRUCTIONS)


class TenantDatabase:
    """A tenant-bound handle. The only way agent tools reach data.

    Holds the connection so that no caller can pass a different tenant later:
    the binding happens once, in the constructor, from the session principal.
    """

    def __init__(self, tenant: str, db_path: Path = DB_PATH, *, boundary: bool = True) -> None:
        self.tenant = _require_known_tenant(tenant)
        self.boundary = boundary
        self._clock: dict = {}
        # `boundary=False` is the ablation's naive arm. Keyword-only and
        # defaulted on, so reaching the weak build takes a deliberate argument
        # at a call site rather than a mistake in one.
        build = tenant_connection if boundary else naive_connection
        self._conn = build(self.tenant, db_path, self._clock)
        # The connection is no longer thread-bound (see `tenant_connection`),
        # so every statement is serialised here instead. The lock must span
        # execute *and* fetch: two threads interleaving on one cursor is how a
        # crash becomes a wrong answer.
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> list[dict]:
        """Run a statement on the bound connection and return capped rows."""
        with self._lock:
            # Reset the deadline per statement, so one slow query cannot poison
            # the connection for every query after it.
            self._clock["deadline"] = time.monotonic() + STATEMENT_TIMEOUT_SECONDS
            try:
                cursor = self._conn.execute(sql, tuple(params or ()))
                rows = cursor.fetchmany(MAX_ROWS)
            except sqlite3.DatabaseError as exc:
                # Sanitised: the raw message can name the base table, which
                # tells an attacker what to aim at next (layer 5, error-message
                # leakage).
                raise SecurityError(f"query rejected by the database: {exc}") from exc
        return [dict(r) for r in rows]

    def columns(self) -> tuple[str, ...]:
        return AGENT_COLUMNS

    def row_count(self) -> int:
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM {AGENT_TABLE}").fetchone()[0])  # nosec B608

    def sample(self, n: int = 3) -> list[dict]:
        """A few of the tenant's own rows, for grounding the model's schema view."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(AGENT_COLUMNS)} FROM {AGENT_TABLE} LIMIT ?", (n,)  # nosec B608
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TenantDatabase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def schema_description(hidden: frozenset[str] = frozenset()) -> str:
    """The virtual schema shown to the model. Names only what it may query.

    `hidden` comes from the caller's role policy. A column the role may not
    name is omitted rather than annotated: telling the model a column exists
    and is forbidden invites it to ask, and the refusal that follows reads to a
    user as the system being broken.
    """
    lines = [f"TABLE {AGENT_TABLE} (one row per employee in your organisation)"]
    # Hints where a declared SQL type is not the whole story. Anything without
    # one falls back to the catalog's own type, so a new column documents
    # itself rather than silently appearing as untyped.
    hints = {
        "user_id": "unique employee id",
        "name": "full name",
        "department": "the actual values for your organisation are listed below",
        "salary": "annual gross in EUR",
        "performance_score": "1.0-5.0, may be NULL",
        "hire_date": "ISO date YYYY-MM-DD",
        "notes": "free-form HR note, may be NULL",
    }
    declared = introspect_types()
    for col in (c for c in AGENT_COLUMNS if c not in hidden):
        kind = declared.get(col, "TEXT")
        hint = hints.get(col)
        lines.append(f"  {col:18} {kind}" + (f", {hint}" if hint else ""))
    return "\n".join(lines)


def iter_tenants() -> Iterable[str]:
    return sorted(ALLOWED_TENANTS)

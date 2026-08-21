"""Per-user conversation history.

A transcript is not a convenience feature, it is a **new store of tenant data**.
It contains salaries, names and free-text notes that were legitimately shown
once, written to disk where they outlive the session that produced them. The
threat model already says the LLM stack creates new copies of the data that
inherit no access control unless you give it to them; this is one of those
copies, so it gets the same treatment as the table.

Four rules follow, and each is a line of code and a test:

1. **Scoped by principal, not by tenant.** Two acme users do not share a
   transcript. The load query filters on tenant *and* username.

2. **Scoped by role as well.** If someone's role changes, turns recorded under
   the old role are not replayed. An HR admin demoted to analyst would
   otherwise see individual salaries reappear in their own history -- data the
   live system would now refuse them. Access control that ignores yesterday's
   answers is not access control.

3. **Redacted before it is written.** Whatever `OutputGuard.redact` masks in a
   rendered answer stays masked on disk. A store that outlives the session
   should not hold more than the screen did.

4. **Bounded.** A per-user cap, and an explicit way to erase. Unbounded
   retention of HR data is a compliance problem wearing a feature's clothes.

Not persisted: result tables and figures. They are large, they are derivable by
re-asking, and every one of them is another copy. The question, the answer and
the reasoning trace are enough to read a conversation back.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from db import ROOT
from secure_rls.security.output_guard import OutputGuard
from secure_rls.security.principal import Principal

STORE_PATH = ROOT / "data" / "conversations.db"

#: Turns kept per user. Older ones are dropped on write.
MAX_TURNS_PER_USER = 50


@dataclass
class Turn:
    """One exchange, as it will be replayed."""

    question: str
    answer: str
    trace: list[dict] = field(default_factory=list)
    timestamp: float = 0.0
    #: Which model produced this answer.
    #:
    #: Worth recording because the model can be switched mid-conversation, and
    #: the bake-off showed answer accuracy ranging from 72% to 100% across the
    #: three local models. A transcript where half the turns came from a weaker
    #: model and half from a stronger one, with no way to tell which, is a
    #: transcript you cannot judge.
    model: str = ""


class ConversationStore:
    """Append-only per-principal transcripts, in their own database.

    Deliberately a separate file from `employees.db`. That database is opened
    read-only and locked down by an authorizer; conversation history is written
    constantly and has nothing to do with the employee table. Mixing them would
    mean relaxing the read-only connection that the whole boundary rests on.
    """

    def __init__(self, path: Path = STORE_PATH) -> None:
        self._path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    username  TEXT NOT NULL,
                    role      TEXT NOT NULL,
                    ts        REAL NOT NULL,
                    question  TEXT NOT NULL,
                    answer    TEXT NOT NULL,
                    trace     TEXT NOT NULL,
                    model     TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._migrate(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_owner ON turns(tenant_id, username, role, id)"
            )
            conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns a database created by an earlier version is missing.

        The store is local and gitignored, so this could reasonably be "delete
        the file". It is not, because the whole point of persisting a
        transcript is that it survives -- silently dropping someone's history
        to add a column would undercut the feature it is extending.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(turns)")}
        for column, ddl in (("model", "TEXT NOT NULL DEFAULT ''"),):
            if column not in existing:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {column} {ddl}")
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # Same thread caveat as db.py: Streamlit reruns land on pool threads.
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ writing ---

    def append(
        self,
        principal: Principal,
        question: str,
        answer: str,
        trace: list[dict],
        model: str = "",
    ) -> None:
        """Record one turn, redacted, and trim to the retention cap."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO turns"
                " (tenant_id, username, role, ts, question, answer, trace, model)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    principal.tenant_id,
                    principal.username,
                    principal.role.value,
                    time.time(),
                    OutputGuard.redact(question) or "",
                    OutputGuard.redact(answer) or "",
                    json.dumps(trace or [], default=str),
                    model,
                ),
            )
            # Trim within this principal only -- never touch another user's rows.
            conn.execute(
                """
                DELETE FROM turns
                 WHERE tenant_id = ? AND username = ? AND role = ?
                   AND id NOT IN (
                       SELECT id FROM turns
                        WHERE tenant_id = ? AND username = ? AND role = ?
                        ORDER BY id DESC LIMIT ?
                   )
                """,
                (
                    principal.tenant_id, principal.username, principal.role.value,
                    principal.tenant_id, principal.username, principal.role.value,
                    MAX_TURNS_PER_USER,
                ),
            )
            conn.commit()

    # ------------------------------------------------------------ reading ---

    def load(self, principal: Principal, limit: int = MAX_TURNS_PER_USER) -> list[Turn]:
        """This principal's own turns, oldest first.

        Filtered on tenant, username *and* role. The role clause is the one
        that is easy to leave out and expensive to leave out: without it, a
        demoted admin replays answers containing data their current role may
        not see.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer, trace, ts, model FROM turns
                 WHERE tenant_id = ? AND username = ? AND role = ?
                 ORDER BY id DESC LIMIT ?
                """,
                (principal.tenant_id, principal.username, principal.role.value, limit),
            ).fetchall()
        return [
            Turn(
                question=row["question"],
                answer=row["answer"],
                trace=json.loads(row["trace"]),
                timestamp=row["ts"],
                model=row["model"] or "",
            )
            for row in reversed(rows)
        ]

    def clear(self, principal: Principal) -> int:
        """Erase this principal's history. Returns how many turns were removed."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM turns WHERE tenant_id = ? AND username = ?",
                (principal.tenant_id, principal.username),
            )
            conn.commit()
            return cursor.rowcount

    def count(self, principal: Principal) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE tenant_id = ? AND username = ? AND role = ?",
                (principal.tenant_id, principal.username, principal.role.value),
            ).fetchone()
        return int(row["n"])

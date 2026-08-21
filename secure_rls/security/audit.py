"""Layer 5 -- the audit trail.

Every data access the agent performs is recorded: who asked, which tool ran,
what SQL was executed, how many rows came back, and what the guard decided.

Entries are chained by hash. Each record carries the digest of its predecessor,
so removing or editing one breaks verification for everything after it. That is
tamper *evidence*, not tamper proofing -- an attacker who owns the file can
recompute the chain -- but it is the property that makes an exported log worth
reading, and it is what a SIEM would consume.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

GENESIS = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp: float
    username: str
    tenant_id: str
    role: str
    tool: str
    arguments: str
    sql: str | None
    rows_returned: int
    guard_verdict: str
    outcome: str
    latency_ms: int
    prev_hash: str
    entry_hash: str = field(default="")

    def payload(self) -> str:
        data = {k: v for k, v in asdict(self).items() if k != "entry_hash"}
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.payload().encode()).hexdigest()

    def as_row(self) -> dict:
        return {
            "seq": self.seq,
            "time": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "user": self.username,
            "tenant": self.tenant_id,
            "tool": self.tool,
            "rows": self.rows_returned,
            "outcome": self.outcome,
            "guard": self.guard_verdict,
            "ms": self.latency_ms,
        }


class AuditLog:
    """Append-only, hash-chained, tenant-tagged."""

    def __init__(self, path: Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        principal,  # secure_rls.security.principal.Principal
        tool: str,
        arguments: str,
        sql: str | None,
        rows_returned: int,
        guard_verdict: str,
        outcome: str,
        latency_ms: int,
    ) -> AuditEntry:
        with self._lock:
            prev = self._entries[-1].entry_hash if self._entries else GENESIS
            draft = AuditEntry(
                seq=len(self._entries) + 1,
                timestamp=time.time(),
                username=principal.username,
                tenant_id=principal.tenant_id,
                role=principal.role.value,
                tool=tool,
                arguments=arguments[:500],
                sql=sql,
                rows_returned=rows_returned,
                guard_verdict=guard_verdict,
                outcome=outcome,
                latency_ms=latency_ms,
                prev_hash=prev,
            )
            entry = AuditEntry(**{**asdict(draft), "entry_hash": draft.compute_hash()})
            self._entries.append(entry)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(entry)) + "\n")
            return entry

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def rows(self) -> list[dict]:
        return [e.as_row() for e in reversed(self._entries)]

    def verify(self) -> bool:
        """Recompute the chain. False means the log has been altered."""
        prev = GENESIS
        for entry in self._entries:
            if entry.prev_hash != prev:
                return False
            recomputed = AuditEntry(**{**asdict(entry), "entry_hash": ""}).compute_hash()
            if recomputed != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

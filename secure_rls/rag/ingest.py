"""Build one Chroma collection per tenant from the `notes` column.

Ingestion is a *privileged* operation: it reads the base table directly, which
no agent code may do. It runs once, offline, from `scripts/build_index.py`.

Each tenant's notes go into their own collection, so the isolation is a
property of the storage layout rather than of every future query's `where`
clause. Notes are redacted before embedding -- an embedding of an email address
is still a record of that email address, and the index is a copy of the data
that outlives the row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from db import BASE_TABLE, DB_PATH, iter_tenants
from secure_rls.rag.retriever import CHROMA_PATH, _client, collection_name
from secure_rls.security.output_guard import OutputGuard

BATCH = 200


def _read_notes(tenant: str, db_path: Path) -> list[tuple[int, str, str, str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT user_id, name, department, notes FROM {BASE_TABLE} "  # noqa: S608  # nosec B608
            f"WHERE tenant_id = ? AND notes IS NOT NULL AND TRIM(notes) != ''",
            (tenant,),
        ).fetchall()
        return [(int(r[0]), r[1], r[2], r[3]) for r in rows]
    finally:
        conn.close()


def build_index(db_path: Path = DB_PATH, chroma_path: Path = CHROMA_PATH) -> dict[str, int]:
    client = _client(chroma_path)
    counts: dict[str, int] = {}

    for tenant in iter_tenants():
        name = collection_name(tenant)
        try:
            client.delete_collection(name)
        except Exception as exc:  # noqa: BLE001 - absent on the first run
            print(f"  (no existing {name} to replace: {type(exc).__name__})")
        collection = client.create_collection(name)

        notes = _read_notes(tenant, db_path)
        for start in range(0, len(notes), BATCH):
            chunk = notes[start : start + BATCH]
            collection.add(
                ids=[f"{tenant}-{uid}" for uid, _, _, _ in chunk],
                documents=[OutputGuard.redact(text) or "" for _, _, _, text in chunk],
                metadatas=[
                    {
                        "tenant": tenant,
                        "user_id": uid,
                        "name": person,
                        "department": dept,
                    }
                    for uid, person, dept, _ in chunk
                ],
            )
        counts[tenant] = len(notes)

    return counts

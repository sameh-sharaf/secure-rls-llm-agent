"""Tenant-partitioned retrieval over the free-text `notes` column.

The index is a *second copy of the data*, and it inherits no access control
from the database. Most multi-tenant RAG implementations handle this with a
metadata filter -- one collection, `where={"tenant": ...}` on every query. That
works right up until one call site forgets, and a forgotten filter is silent,
returns plausible results, and fails no test that is not looking for it.

So the partition is physical: one Chroma collection per tenant, and the
collection name is derived from the session principal inside the constructor.
There is no method here that accepts a collection name or a tenant, which means
"query the wrong tenant's index" is not an expressible operation rather than a
mistake to be avoided.

The metadata filter is applied *as well*, and every returned chunk is asserted
in-tenant afterwards. Belt, braces, and an alarm if either slips.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from db import ROOT
from secure_rls.security.output_guard import OutputGuard
from secure_rls.security.principal import Principal

CHROMA_PATH = ROOT / "data" / "chroma"


def collection_name(tenant: str) -> str:
    return f"notes_{tenant}"


@dataclass
class RetrievedNote:
    user_id: int
    name: str
    department: str
    text: str
    distance: float


class CrossTenantRetrieval(RuntimeError):
    """A chunk from outside the tenant came back. Never swallowed."""


def _client(path: Path = CHROMA_PATH) -> chromadb.ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )


class TenantNotesRetriever:
    """Semantic search over one tenant's notes. Bound at construction."""

    def __init__(self, principal: Principal, *, path: Path = CHROMA_PATH) -> None:
        self.tenant = principal.tenant_id
        self._allowed_ids: frozenset[int] | None = None
        client = _client(path)
        # The only place a collection is chosen, and it is chosen from the
        # principal. No caller can reach a different one.
        self._collection = client.get_or_create_collection(collection_name(self.tenant))

    def bind_allowed_ids(self, allowed: frozenset[int]) -> None:
        """Give the retriever the independent id set for post-retrieval checks."""
        self._allowed_ids = allowed

    #: Queries that name no topic. "Read the employee notes" gives the model
    #: nothing to search *for*, and it passes one of these -- an empty string or
    #: a wildcard -- which similarity search answers with nothing at all. The
    #: user asked a reasonable question and got "I could not find any employee
    #: notes to read", which is false: there are five hundred.
    _TOPICLESS = frozenset({"", "*", "%", "all", "any", "all notes", "everything"})

    def search(self, query: str, top_k: int = 5) -> list[RetrievedNote]:
        if (query or "").strip().lower() in self._TOPICLESS:
            return self.sample(top_k)
        count = self._collection.count()
        if count == 0:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            # Redundant given the collection is already tenant-private, and
            # kept precisely because it is redundant: if someone later merges
            # the collections, this is the control that still holds.
            where={"tenant": self.tenant},
        )

        return self._verified(
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
        )

    def sample(self, top_k: int = 5) -> list[RetrievedNote]:
        """A few of this tenant's notes, for a request that names no topic.

        Goes through the same verification as a search. That is the whole point
        of putting it here rather than in the tool: a second way to get rows out
        of the index that skipped the tenant check would be exactly the kind of
        side channel invariant 5b exists for.
        """
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.get(
            where={"tenant": self.tenant}, limit=min(top_k, count)
        )
        documents = result.get("documents") or []
        return self._verified(documents, result.get("metadatas") or [], [0.0] * len(documents))

    def _verified(self, documents, metadatas, distances) -> list[RetrievedNote]:
        """Check every chunk belongs to this tenant, then build the notes."""
        notes: list[RetrievedNote] = []
        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            meta = meta or {}
            if meta.get("tenant") != self.tenant:
                raise CrossTenantRetrieval(
                    f"retrieved a chunk tagged {meta.get('tenant')!r} in a "
                    f"{self.tenant!r} session"
                )
            uid = int(meta.get("user_id", -1))
            if self._allowed_ids is not None and uid not in self._allowed_ids:
                raise CrossTenantRetrieval(
                    f"retrieved a chunk for user_id={uid}, which is not in tenant "
                    f"{self.tenant!r}"
                )
            notes.append(
                RetrievedNote(
                    user_id=uid,
                    name=str(meta.get("name", "")),
                    department=str(meta.get("department", "")),
                    text=OutputGuard.redact(doc) or "",
                    distance=float(dist),
                )
            )
        return notes

    def count(self) -> int:
        return self._collection.count()

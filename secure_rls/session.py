"""Session assembly.

One function builds every tenant-bound object a session needs, from one
principal, in one place. That is deliberate: the gateway, the retriever and the
tools must all be bound to the *same* identity, and a caller that wires them up
by hand can bind two of the three and leave the third pointing somewhere else.

If you find yourself constructing a `QueryGateway` or a `TenantNotesRetriever`
outside this module, that is the smell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from db import DB_PATH
from secure_rls.rag.retriever import CHROMA_PATH, TenantNotesRetriever
from secure_rls.security.audit import AuditLog
from secure_rls.security.gateway import QueryGateway
from secure_rls.security.principal import Principal
from secure_rls.tools.factory import ToolContext, build_tools


@dataclass
class Session:
    """Everything one logged-in user needs, all bound to the same principal."""

    principal: Principal
    gateway: QueryGateway
    context: ToolContext
    tools: list
    audit: AuditLog
    #: Conversation memory, owned by the session rather than by the agent.
    #:
    #: Switching model rebuilds the agent, and an agent-owned checkpointer went
    #: with it -- so the new model started blind and could only see the turns it
    #: had generated itself. Keeping it here means the whole thread survives a
    #: model swap. It stays keyed by (thread, tenant), so this shares history
    #: across models, never across tenants.
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)

    @property
    def tool_map(self) -> dict:
        return {t.name: t for t in self.tools}

    def close(self) -> None:
        self.gateway.close()


def build_session(
    principal: Principal,
    *,
    db_path: Path = DB_PATH,
    chroma_path: Path = CHROMA_PATH,
    with_rag: bool = True,
) -> Session:
    audit = AuditLog()
    gateway = QueryGateway(principal, audit=audit, db_path=db_path)

    retriever = None
    if with_rag:
        try:
            retriever = TenantNotesRetriever(principal, path=chroma_path)
            # Give the retriever the same independent id set the output guard
            # uses, so a retrieved chunk is checked against a source of truth
            # that did not come from the index itself.
            retriever.bind_allowed_ids(gateway.allowed_user_ids)
        except Exception:
            retriever = None  # index not built yet; tools degrade gracefully

    context = ToolContext(gateway=gateway, retriever=retriever)
    tools = build_tools(context)
    return Session(
        principal=principal,
        gateway=gateway,
        context=context,
        tools=tools,
        audit=audit,
    )

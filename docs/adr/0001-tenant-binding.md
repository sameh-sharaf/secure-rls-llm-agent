# ADR-0001: Tenant identity is bound in a closure, never passed as a tool argument

**Status:** accepted
**Date:** 2026-08-21

## Context

The agent needs to read data on behalf of a specific tenant. The obvious design
gives the tool a tenant parameter and lets the orchestration fill it in:

```python
@tool
def query_employees(tenant_id: str, department: str) -> str: ...
```

with a system prompt instructing the model to always pass the current tenant.
This is what most implementations of a brief like ours look like, and it is the
design the brief's phrase "even in generated queries/tools" is aimed at.

It is not defensible. The parameter is part of the model's output, and the
model's output is influenced by:

- the user's message, which the adversary controls completely;
- any retrieved document, including a `notes` field the adversary wrote;
- any tool result;
- ordinary non-determinism.

No amount of prompt engineering repairs this, because the attack surface *is*
the parameter. A prompt saying "always pass your own tenant" is a request, and
requests to a language model are not access control.

## Decision

Tenant identity never appears in any tool signature. Tools are minted per
session by a factory that closes over a `QueryGateway`, which was itself
constructed from the session `Principal`:

```python
def build_tools(context: ToolContext) -> list[BaseTool]:
    gateway = context.gateway            # already bound to exactly one tenant

    def query_employees(**kwargs) -> str:
        args = QueryEmployeesArgs(**kwargs)     # schema has no tenant field
        return gateway.run_spec(spec_from(args))
```

Three consequences follow:

1. The model has no vocabulary for choosing a tenant. It cannot pass one because
   the field does not exist, and it cannot invent one because every args schema
   sets `extra="forbid"`, making an invented key a validation error rather than
   a silently ignored one.
2. The tenant is injected by the server *between* the model and the database.
   The model's output influences *what* is asked, never *whose data* is asked
   about.
3. The `QueryGateway` constructor is the only place a tenant is chosen. After
   construction, no method on it accepts a tenant.

The same rule extends to the retrieval layer: `TenantNotesRetriever` picks its
Chroma collection from the principal inside `__init__`, and exposes no method
taking a collection or tenant name.

## Enforcement

An invariant that lives only in a developer's head dies during a refactor, so
this one is checked mechanically in three places:

- `tests/test_tool_contract.py` walks every generated JSON schema and fails on
  any key containing `tenant`, `org`, `company`, `customer` or `client`; asserts
  `additionalProperties: false` on every schema; and asserts the serialised
  schema blob contains no tenant name anywhere.
- `.claude/hooks/check-invariants.sh` blocks a commit that adds such a field.
- `.claude/agents/security-reviewer.md` lists it as the highest-severity finding.

The serialised-blob check has already earned its place: it caught the
`QueryEmployeesArgs` docstring, which said "no tenant field, deliberately". A
Pydantic class docstring becomes the JSON-schema `description` and is sent to
the model, so the schema was advertising the exact thing not to probe.

## Alternatives considered

**Validate the tenant argument against the session.** Keep the parameter, and
raise if it does not match the principal. Rejected: it works, but it makes the
security property depend on a check that a future refactor can move, skip, or
get wrong on one of several code paths. Removing the parameter removes the
question.

**Pass a signed tenant token the model must echo back.** Rejected as security
theatre — the model is not a trust boundary, and a token it holds is a token an
injected instruction can ask it to swap.

**Rely on the layer-4 boundary alone.** Tempting, since ADR-0002 makes the
database refuse cross-tenant reads regardless. Rejected because a tenant
parameter that is silently ignored is worse than none: it tells a reader the
model chooses the tenant, and the next engineer will wire it up.

## Consequences

**Good.** "The model cannot pass a different tenant" is a structural fact about
the schema rather than a behavioural claim about the model. It is demonstrable
in four seconds by printing the schemas, which is what the app's "What the model
sees" tab does.

**Good.** Sessions cannot be crossed by accident: tools are per-session objects,
and there is no global tool registry to misuse.

**Cost.** Tools must be rebuilt per session rather than defined once at module
import, and they cannot be cached across users. At this scale that is free.

**Cost.** Multi-tenant *administration* — a support engineer legitimately
inspecting several tenants — is not expressible through this agent at all. That
is the correct default, and such a feature would be a separate, separately
audited path, not a parameter added here.

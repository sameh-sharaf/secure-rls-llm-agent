# Architecture

Read `threat-model.md` first for what this is defending against, then `db.py`
for the part that does the defending.

## Request path

```
  browser
    │  username / password
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ L1  authenticate() ──> Principal(username, tenant_id, role)      │
│     held in st.session_state, re-read on every Streamlit rerun   │
└─────────────────────────────────────────────────────────────────┘
    │  principal (never through the model)
    ▼
  build_session(principal)
    ├── QueryGateway(principal) ──> TenantDatabase(tenant)  [L4]
    ├── TenantNotesRetriever(principal) ──> collection notes_<tenant>
    └── build_tools(context) ──> tools closed over the gateway  [L2]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ SecureAgent (LangGraph)                                          │
│                                                                  │
│   route ─┬─> refuse (terminal, holds no tool)                    │
│          └─> plan ─> tools ─> guard ─┬─> plan  (research loop,   │
│                ▲                     │         max 3 rounds)     │
│                │                     ├─> retry ─┘  (max 2)       │
│                └─────────────────────┼─> synthesise ─> answer    │
│                                      └─> refuse                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼  every tool call
┌─────────────────────────────────────────────────────────────────┐
│ L3  QueryGateway                                                 │
│     spec path:  QuerySpec ──> parameterised SQL (enums + binds)  │
│     sql  path:  sqlglot parse ─> allowlist ─> AST rewrite        │
│                 (LIMIT, HAVING COUNT(*) >= 5)                    │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ L4  the boundary  ── db.tenant_connection(tenant)                │
│     1. validate tenant against a 3-element allowlist             │
│     2. open a private, empty in-memory database                  │
│     3. ATTACH the data file read-only; CREATE TEMP TABLE         │
│          employees AS SELECT <cols> FROM src.employees_base      │
│          WHERE tenant_id = ?;  then DETACH it                    │
│     4. set_authorizer: DENY employees_base unconditionally,      │
│        DENY sqlite_master / ATTACH / PRAGMA / all writes         │
│                                                                  │
│     Other tenants' rows are not filtered out. They are           │
│     unreachable: the statement fails to prepare.                 │
└─────────────────────────────────────────────────────────────────┘
                          │  rows
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ L5  OutputGuard.check_rows()   verify user_ids against the       │
│                                PRIVILEGED id set (independent)   │
│     OutputGuard.check_text()   scan prose for foreign canaries   │
│     AuditLog.record()          hash-chained entry, tenant-tagged │
│     raises on any finding -- never filters                       │
└─────────────────────────────────────────────────────────────────┘
```

## Why the ordering in L4 is not incidental

1. Validate first, so a tenant value never reaches SQL unchecked.
2. Attach read-only, so writes fail at the file level as well as the authorizer.
3. Materialise *before* installing the authorizer — the authorizer denies
   `employees_base`, so it would block its own setup.
4. Detach the source, so the only relation the connection can name is the
   tenant's own. This is what makes L4 independent of *which* reads SQLite
   chooses to route through the authorizer — see ADR-0006, where a `USING`
   join turned out to bypass the callback entirely.
5. Install the authorizer *before* returning the connection — any gap is a
   window in which the base table is readable.

## The role column policy

The tenant boundary is structural; the role policy is the one place this system
holds a *declarative* rule. It lives in `ROLE_POLICY` and has two tiers:

| tier | meaning | enforced where |
|---|---|---|
| **masked** | may be aggregated, may not be read for an individual | `compile_spec`, `guard_sql`, `sample_rows` |
| **hidden** | may not be named at all, and is absent from the schema the model is shown | same three, plus `schema_description` and `departments()` |

Masking is right when a column is legitimately part of the analysis and only
the *granularity* is the problem — an analyst may ask for the average salary in
Engineering and may not list salaries beside names. Hiding is right when the
column should not inform an answer at all.

`hidden_columns()` is derived by subtracting `visible` from the live catalog
column set, so a column added to the table is hidden by default from any role
whose `visible` set was written out explicitly. A new column has to be granted
rather than remembered.

Three details that are load-bearing:

- **The gateway is the only consumer.** Every path that reveals a value —
  query results, the system prompt's sample rows, the UI — reads the policy in
  `QueryGateway` and nowhere else. Invariant 5b exists because the mask was
  once applied on the query path and bypassed by the prompt built alongside it.
- **Hidden is checked before masked.** A hidden column must not fall through to
  the more permissive aggregate rule, the same ordering that `MIN`/`MAX` needed.
- **Hidden columns are dropped, not placeholdered.** A `<restricted>` marker in
  a sample row would still tell the model the column exists, which is the thing
  hiding is for.

Neither shipped role hides anything — the case study is about tenant isolation,
and hiding columns by default would narrow the demo without showing anything
the mask does not. `tests/test_column_policy.py` constructs a restricted policy
so the mechanism is exercised anyway; an unenforced field and an untested one
fail the same way.

## Data-flow invariants

| Invariant | Where enforced | Where tested |
|---|---|---|
| A tenant string only enters SQL via `_require_known_tenant` | `db.py` | `test_unknown_tenant_fails_closed` |
| No tool schema names a tenant | `tools/factory.py` | `test_no_tool_exposes_a_tenant_parameter` |
| `employees_base` is unreachable from an agent connection | `db._make_authorizer` | `test_smuggling_attempt_is_blocked` (18 cases) |
| Any query returns a subset of the tenant's rows, or raises | `db.py` | `test_property_...` (Hypothesis, 250 examples) |
| Every tool result passes the guard | graph topology | `test_every_path_from_tools_reaches_guard` |
| Retrieval opens only the tenant's collection | `rag/retriever.py` | `test_retriever_exposes_no_way_to_choose_a_collection` |
| Aggregates cover ≥ 5 people | `spec.py`, `sql_guard.py`, `gateway.py` | `test_single_person_aggregate_is_refused` |
| Audit entries are tamper-evident | `audit.py` | `test_audit_detects_tampering` |

## Where the tenant value physically lives

Exactly two places, both server-side:

- `Principal.tenant_id`, in `st.session_state` for the life of the session.
- The rows inside the connection's private `TEMP TABLE`.

It is not in the prompt as an instruction, not in any tool schema, not in graph
state, not in a URL, and not in the browser. `tenant_id` is not even projected
into the temp table, so the column does not exist for the model to name.

## Trade-offs taken

| Choice | Cost accepted | Why |
|---|---|---|
| Temp table, not temp view | Copies the tenant's rows per session | The view design has a real CTE-impersonation bypass (ADR-0002) |
| SQLite, not Postgres | No native RLS; emulation required | Brief requires offline with no infrastructure (ADR-0004) |
| LangGraph, not `create_react_agent` | Much more code | Three nodes are security controls (ADR-0003) |
| Collection per tenant | Three indexes instead of one | A metadata filter is a convention; an unopened collection is a control |
| Guard raises, never filters | A detected leak surfaces as an error | Silently repairing a violation hides the bug that caused it |
| k = 5 fixed | Blunt; a patient analyst still narrows in | Query budgets and differential privacy are out of scope here |

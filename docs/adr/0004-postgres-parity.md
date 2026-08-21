# ADR-0004: Keep SQLite as the default; document Postgres native RLS as the production shape

**Status:** accepted (SQLite), proposed (Postgres profile)
**Date:** 2026-08-21

## Context

The brief asks for SQLite or pandas, local and offline. ADR-0002 gets genuine
row-level security out of SQLite by materialising a per-connection temp table
and denying the base table via an authorizer.

That works, and it has a cost worth stating plainly: **it copies the tenant's
rows into memory on every session**. At 500 rows this is invisible. At five
million rows per tenant it is untenable, and it is also a snapshot — writes
after connection setup are not visible.

Meanwhile, every platform this pattern would actually run on has native RLS:

| Platform | Mechanism |
|---|---|
| PostgreSQL | `CREATE POLICY ... USING (tenant_id = current_setting('app.tenant'))` + `FORCE ROW LEVEL SECURITY` |
| Snowflake | Row access policies resolved against `CURRENT_ROLE()` or a mapping table |
| Databricks | Unity Catalog row filters and column masks at the catalog layer |
| SQL Server | Security policies with inline table-valued predicate functions |

## Decision

**SQLite stays the default**, because the brief asks for offline operation with
no infrastructure and because the temp-table design is a faithful emulation of
the control rather than a workaround for its absence.

**A Postgres profile is documented and scoped but not implemented**, and this
ADR is the honest record of that. Scope, if built:

```sql
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE employees FORCE ROW LEVEL SECURITY;   -- applies to the owner too

CREATE POLICY tenant_isolation ON employees
  USING (tenant_id = current_setting('app.tenant_id', true));
```

with the application connecting as a **non-owner** role — `FORCE` matters
because policies do not apply to a table's owner by default, which is the
classic way a Postgres RLS deployment turns out not to be enforcing anything.
`app.tenant_id` is set per connection via `SET LOCAL` from the principal, in the
same position `tenant_connection` occupies today.

The point of building it would not be to run Postgres. It is to run **the same
test suite** — `tests/test_boundary.py` and `evals/redteam.yaml` unchanged —
against both backends, demonstrating that the security property is a property of
the architecture rather than of one clever SQLite trick.

## Why it is not built

Straightforward scope management. The layer-4 boundary, the red-team suite and
the ablation study are the submission; the Postgres profile is a second
implementation of a layer that already works and already has tests. It was first
on the cut list before the schedule was under pressure, and it is recorded here
rather than quietly dropped.

## Consequences

**Good.** `db.py` is the only file that would change. `tenant_connection`
returns a DB-API connection and the gateway does not care what produced it, so
the swap is one function plus a connection-string setting.

**Good.** The comparison is a strong talking point: the temp-table design exists
*because* SQLite lacks the control, and naming the real mechanism it emulates
shows the difference is understood rather than papered over.

**Cost.** The README's scaling limitation is real and currently unmitigated in
code. It is listed under Known Limitations rather than implied away.

**Cost.** Two backends means two authorizer/policy models to keep in step. The
mitigation — one shared test suite run against both — is the reason to build it
at all, and building it without that would add risk rather than remove it.

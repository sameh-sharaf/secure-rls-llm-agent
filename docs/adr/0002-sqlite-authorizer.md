# ADR-0002: Materialise a per-tenant temp table, and deny the base table unconditionally

**Status:** accepted
**Date:** 2026-08-21

## Context

SQLite has no `GRANT`, no roles, and no row-level security policies. PostgreSQL
has `CREATE POLICY`, Snowflake has row access policies, Databricks has Unity
Catalog row filters — SQLite has none of these. The brief requires that the LLM
"must never access unauthorized rows, even in generated queries", so appending
`AND tenant_id = ?` in application code is not sufficient: that is precisely the
"the developer remembered to write it" pattern that RLS exists to replace, and it
is defeated by any query the application did not compose itself.

We need an enforcement point *below* any SQL that our code or the model writes.

## The design we tried first, and why it is broken

The natural design is a connection-scoped `TEMP VIEW`:

```sql
CREATE TEMP VIEW employees AS
SELECT user_id, name, ... FROM employees_base WHERE tenant_id = 'acme';
```

…combined with an authorizer callback (`Connection.set_authorizer`) that denies
reads of `employees_base` *except* when SQLite reports the read as originating
from the view. Python's callback signature is:

```python
authorizer(action, arg1, arg2, dbname, source)
```

For `SQLITE_READ`, `source` is the name of the view or trigger that caused the
access, or `None` for a direct read. So the rule looks obviously correct:

```python
if arg1 == "employees_base":
    return SQLITE_OK if source == "employees" else SQLITE_DENY   # WRONG
```

It is not correct. SQLite sets `source` to the name of the **CTE** performing the
read as well. An attacker who names their CTE after the secure view is
indistinguishable from the secure view:

```sql
WITH employees AS (SELECT user_id, name, salary FROM employees_base)
SELECT * FROM employees;
```

Measured against a three-tenant fixture, this returns **all** rows. Every other
smuggling variant we tried was correctly blocked — a CTE named anything else, a
subquery aliased `employees`, `UNION`, a scalar subquery, a join to the base
table — which is what makes this one dangerous: the design looks like it works.

Note also that `source is not None` as the test is even weaker, and fails for
every CTE name, not just this one.

## Decision

Do not reason about `source` at all. Instead:

1. **Materialise** the tenant's rows into a private `TEMP TABLE employees` at
   connection setup, before any agent code touches the connection.
2. **Deny `employees_base` unconditionally** in the authorizer — no exception for
   views, for CTEs, or for any value of `source`.

With no exception to the denial rule, there is no `source` value for an attacker
to forge. The relation the agent queries and the relation holding other tenants'
rows are different objects in different schemas, and only one of them is
reachable.

Ordering is part of the control and is not incidental:

1. validate the tenant against a three-element allowlist (fail closed),
2. open the connection read-only,
3. materialise the temp table,
4. *then* install the authorizer.

Installing the authorizer before step 3 blocks the materialisation; installing it
after handing out the connection leaves a window in which the base table is
readable.

## Consequences

**Good.** The guarantee is now stated without qualification: any SQL executed on
a tenant-bound connection either returns a subset of that tenant's rows or is
rejected by the engine. `tests/test_boundary.py` asserts this over a fixed
smuggling corpus and as a Hypothesis property over generated queries.

**Good.** `tenant_id` is not projected into the temp table at all, so the column
is not merely filtered — it is absent from the model's vocabulary.

**Cost.** Each session copies its tenant's rows (500 rows at most here — a few
hundred kilobytes). This does not scale to millions of rows per tenant, and it is
a snapshot: writes to the base table after connection setup are not visible.
Both are acceptable for a read-only analytical agent at this size, and both
disappear on a platform with native RLS, where the filter is applied by the
engine without copying. See ADR-0004.

**Cost.** The authorizer is an allowlist over SQLite action codes, so a SQLite
version introducing a new action code denies it rather than permitting it. That
is the correct default, but it means an upgrade can turn a working query into a
rejected one. The boundary test suite is the early warning.

## Notes for the reviewer

The failing case is preserved as a named regression test:
`tests/test_boundary.py::test_cte_named_employees_cannot_impersonate_the_view`.

The `source` behaviour is not documented in the Python `sqlite3` docs; it was
found by instrumenting the callback and printing every invocation. That
instrumentation is worth keeping in mind as a technique — the security-relevant
semantics of a callback API are often only discoverable empirically.

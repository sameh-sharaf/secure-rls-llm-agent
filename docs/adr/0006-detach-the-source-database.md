# ADR-0006: The agent's connection is a private database, not the data file

**Status:** accepted
**Date:** 2026-08-22
**Supersedes part of:** ADR-0002 (which established the authorizer as absolute)

## Context

The question that prompted this: *if another table is added with a `tenant_id`
column, how does the app behave?*

Most of the answer was reassuring, and by construction rather than by luck. A
new table is invisible to the model — `introspect_columns` reads one relation,
so nothing about `contracts_base` enters the vocabulary — and layer 3 rejects
the name outright (`unknown table 'contracts_base'`). The system fails closed
with no code change.

Then the probe reached layer 4 on its own, per invariant 7, and found this:

| query | result |
|---|---|
| `SELECT secret_clause FROM contracts_base` | denied |
| `SELECT (SELECT COUNT(*) FROM contracts_base)` | denied |
| `... employees e JOIN contracts_base c ON e.user_id = c.user_id` | denied |
| `... employees e JOIN contracts_base c USING (user_id)` | **returned rows** |
| `... employees e NATURAL JOIN contracts_base` | **returned rows** |

Logging every authorizer invocation explained it. For the `ON` form SQLite asks
about `contracts_base.user_id` and is refused. For the `USING` form it asks
about nothing at all — the only callback for the whole statement is
`employees.name`. **SQLite does not consult the authorizer for a join key named
implicitly through `USING` or `NATURAL JOIN`.**

Scope, stated precisely so this is neither over- nor under-sold:

- The **tenant boundary held.** `employees_base` stayed unreachable through
  every form, so no cross-tenant row was exposed, and nothing shipped was
  vulnerable — the exploit needs a second table, and the dataset has one.
- What leaked was **linkage against a hypothetical future table**: which of the
  tenant's own employees appear in it, an inference channel rather than a
  direct read.
- Layer 3 blocked all of it. But "L3 catches it" is exactly the argument
  invariant 7 exists to forbid.

## Decision

Stop opening the data file as the agent's `main` database. Open a private
in-memory database, attach the file read-only for as long as it takes to copy
the tenant's rows, then detach it.

```python
conn = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
conn.execute("ATTACH DATABASE ? AS src", (f"{path.as_uri()}?mode=ro",))
conn.execute(f"CREATE TEMP TABLE employees AS SELECT {cols} FROM src.employees_base WHERE tenant_id = ?", (tenant,))
conn.execute("DETACH DATABASE src")
_assert_nothing_reachable_but_the_agent_table(conn)
conn.set_authorizer(_make_authorizer(tenant))
```

**The reasoning, which matters more than the diff.** The tempting fix was to
reject `USING` and `NATURAL JOIN` in the SQL guard. That would have closed the
two queries above and left the actual defect in place: a layer-4 guarantee that
holds only for syntax someone remembered to think about. The authorizer is a
callback SQLite chooses when to invoke, and the set of things it is not invoked
for is not enumerable from the outside.

Removing the relation instead changes the failure mode from *denied* to
*absent*. `no such table: contracts_base` comes from the parser, before
authorization is a question, and it is the same answer for `USING`, for
`NATURAL`, for `ON`, and for whatever syntax the next SQLite release adds.
**Absence needs no rule to be right.**

The authorizer stays, and still denies everything but the agent table. It is
now defence in depth rather than the whole boundary.

## Consequences

**Good.** Layer 4's guarantee no longer depends on which callbacks SQLite
chooses to fire. Adding tables to the data file cannot widen what the agent
reaches, whatever their shape.

**Good.** The multi-relation design in ADR-0005 gets simpler and safer: the
exposed set becomes exactly what is materialised, and "everything else" needs no
denial rule because it is not there.

**Good.** Shadowing is structurally impossible, so
`_assert_no_shadow_relation` is replaced by a stronger and simpler claim
verified after detach — `main` holds nothing, `temp` holds exactly one table.

**Cost, and a real one.** The tenant's slice is now resident in memory for the
session. At 500 rows per tenant that is nothing; at ten million it is the wrong
architecture, and the honest answer for that scale is a database with native
row-level security rather than a copy. The property being demonstrated is the
same either way — this is how you get it from an engine that has no RLS.

**Cost.** `ATTACH` needs a URI filename, so the connection is opened with
`uri=True` and the path goes through `Path.as_uri()`. That also fixes a latent
issue: the repository path contains a space, and `as_uri()` percent-encodes it
where the old f-string did not.

**Changed behaviour.** The base table is now refused by the parser rather than
the authorizer, so the message changed from *"access to employees_base.user_id
is prohibited"* to *"no such table: employees_base"*. Two ablation tests pinned
the old wording and failed. They now assert the refusal rather than the phrase —
pinning one message made a strengthened boundary look like a regression.

## Notes

`tests/test_second_table_is_unreachable.py` covers the finding directly: it
bolts a second tenant-scoped table onto a copy of the database and fires eight
query shapes at layer 4 with no guard and no model in the picture.

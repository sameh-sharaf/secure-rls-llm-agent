# ADR-0005: Derive the schema from the catalog; generalising to many relations

**Status:** accepted (introspection), proposed (multiple relations)
**Date:** 2026-08-22

## Context

The column allowlist was written out by hand in three places:

| Where | Form | Purpose |
|---|---|---|
| `db.AGENT_COLUMNS` | tuple | what the temp table projects |
| `spec.Column` | `StrEnum` | the model's vocabulary |
| `sql_guard.ALLOWED_COLUMNS` | frozenset | what generated SQL may name |

Three statements of one fact, with nothing keeping them in step. Adding a column
meant editing three files and remembering all three; forgetting one meant the
SQL guard silently disagreed with the schema the model had just been handed.

The objection that prompted this went further, and is worth stating in full:
enumerating specific column values does not scale, real data can be anything,
and if the system grows more tables or views, hand-maintaining lists per column
is not a design — it is a chore that will drift.

That is correct. It is also worth separating from a related question it is easy
to conflate, because one half was already right:

- **Column values** (the departments in the prompt) were already derived at
  runtime, `SELECT DISTINCT department` through the bound connection. Nothing
  hardcoded, and it is tenant-scoped for free — acme sees Legal, beta does not.
- **Column names** were hardcoded, three times over. That was the real problem.

## Decision

Read the columns from the database catalog at startup and derive all three from
that single source.

```python
TENANT_COLUMN = "tenant_id"                      # configuration
AGENT_COLUMNS = introspect_columns()             # PRAGMA table_info, minus the tenant column
Column        = StrEnum("Column", ...)           # generated from AGENT_COLUMNS
ALLOWED_COLUMNS = frozenset(AGENT_COLUMNS)       # same source
```

Declared SQL types come from the same read, so a new column documents itself in
the schema shown to the model rather than appearing untyped.

**This does not weaken the control.** An allowlist is a security control; *where
it comes from* is not, provided the source is trusted and the model cannot
influence it. The catalog is read once, at startup, through a privileged
connection — the same trust level that loaded the data in the first place. The
model still cannot name a column outside the enum, because Pydantic rejects the
value before any tool runs.

**One thing stays configuration, deliberately.** `TENANT_COLUMN` is declared,
not discovered. Which column carries the boundary is the one fact a catalog
cannot tell you, and guessing it — "the column called `tenant_id`, or `org_id`,
or whichever looks like a foreign key" — would put the boundary at the mercy of
a naming convention. It is excluded at the source rather than filtered later, so
the word is absent from the model's vocabulary entirely.

## Consequences

**Good.** Adding a column to `employees_base` requires no code change: it
appears in the projection, the model's vocabulary and the SQL guard together.
`tests/test_schema_introspection.py` asserts exactly that by altering a copy of
the database and checking the allowlist follows.

**Good.** The three lists cannot drift, because there is one list.

**Cost.** A startup read of the catalog, and an import-time dependency on the
database file. A fresh clone imports before `scripts/build_db.py` has run, so
there is a fallback constant — which is a fourth copy of the column list, used
only when there is no database to ask. Smaller than the problem it replaces, but
not nothing.

**Cost.** New columns are exposed *by default*. For this dataset that is right;
for one containing a column nobody should see, the safe default would invert:
introspect, then apply a declared exclusion list alongside `TENANT_COLUMN`. The
hook is one line and is not built, because there is nothing to exclude yet.

## Multiple relations — designed, not built

The other half of the objection: what happens with more tables or views?

Nothing in the boundary is specific to one table. The design generalises by
turning two constants into a registry:

```python
EXPOSED = {
    "employees": Relation(base="employees_base", tenant_column="tenant_id"),
    "contracts": Relation(base="contracts_base", tenant_column="tenant_id"),
}
```

and then, at connection setup, materialising one tenant-scoped temp table per
entry instead of one. Everything downstream follows from that:

- the authorizer's readable set becomes the temp table names rather than one
  name, and every base table stays denied unconditionally;
- `ALLOWED_TABLES` in the SQL guard comes from the registry;
- `Column` becomes per-relation, so a `QuerySpec` names its relation and the
  columns are validated against that relation's catalog entry;
- joins are permitted only between exposed relations, which the AST walk already
  has the shape to check.

The security argument is unchanged, because it never depended on there being one
table: the agent's connection contains only rows the tenant owns, and the base
tables are unreachable regardless of how many there are.

**Not built, deliberately.** The dataset has one table, and a registry with one
entry demonstrates nothing that the current code does not. Building the
abstraction now would mean carrying untested generality through every layer to
serve a case that does not exist yet. Recorded here so the answer to "does this
scale past one table?" is a design rather than a shrug — and so the work is
scoped if a second table ever arrives.

"""Layer 3b -- the guarded SQL escape hatch.

Some questions are genuinely awkward to express as a `QuerySpec`, so the model
is allowed to write SQL. It is parsed into an AST with sqlglot, validated
against an allowlist, rewritten to add the limits and the minimum cohort size,
and only then executed.

A note on what this layer is *for*
----------------------------------
It is not the tenant boundary. The connection the query runs on is already
scoped to one tenant and cannot reach any other (see db.py and ADR-0002), so
this guard has no tenant predicate to inject -- there is no `tenant_id` column
in the relation it validates against, and there are no other tenants' rows to
exclude. Every rejection here would also be a rejection one layer down.

It exists for three reasons that are worth its cost:

  * the model gets a fast, specific, *actionable* rejection it can revise from,
    inside the same turn, instead of an opaque database error;
  * k-anonymity and row limits are policy, not access control, and the database
    will not apply them for us;
  * a rejected query is a legible security event for the audit log and the demo.

Redundancy that produces a better error message earns its place. Redundancy
mistaken for the boundary does not, which is why this file says so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from db import AGENT_COLUMNS
from secure_rls.security.layers import Layer, tag
from secure_rls.security.spec import ENFORCE_MIN_COHORT, MIN_COHORT_SIZE

DIALECT = "sqlite"

#: The only relation the agent may name.
ALLOWED_TABLES = frozenset({"employees"})

#: Same source as the model's vocabulary, so the two cannot drift apart.
ALLOWED_COLUMNS = frozenset(AGENT_COLUMNS)

#: Scalar and aggregate functions the agent may call. Anything absent is
#: rejected -- fail closed, so a SQLite build with extra functions compiled in
#: does not silently widen the surface.
ALLOWED_FUNCTIONS = frozenset(
    {
        "abs", "avg", "cast", "coalesce", "count", "date", "ifnull", "julianday",
        "length", "lower", "max", "min", "nullif", "round", "strftime", "substr",
        "sum", "trim", "upper",
    }
)

MAX_LIMIT = 200


class SqlRejected(ValueError):
    """The statement is not allowed. The message is shown to the model."""


@dataclass
class GuardResult:
    """What the guard decided, and why -- rendered in the UI and the audit log."""

    sql: str
    original_sql: str
    rewrites: list[str] = field(default_factory=list)
    aggregate_only: bool = False


def _fail(reason: str) -> None:
    raise SqlRejected(reason)


def _check_no_forbidden_nodes(tree: exp.Expression) -> None:
    """Reject whole categories of statement rather than pattern-matching text."""
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
        exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Transaction,
    )
    for node_type in forbidden:
        if isinstance(tree, node_type) or list(tree.find_all(node_type)):
            _fail(
                f"only read-only SELECT statements are permitted "
                f"({node_type.__name__.lower()} is not)"
            )

    # CTEs are refused outright. They add nothing for a single-table schema and
    # they remove an entire class of aliasing tricks from consideration -- a CTE
    # can be named after the relation it is shadowing. Cheaper to disallow than
    # to reason about. See ADR-0002 for why that specific trick matters here.
    if list(tree.find_all(exp.With)):
        _fail("common table expressions are not permitted; use a subquery or a simpler query")


def _check_tables(tree: exp.Expression) -> None:
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if name not in ALLOWED_TABLES:
            _fail(f"unknown table {table.name!r}; the only readable table is 'employees'")
        if table.db:
            _fail("schema-qualified table names are not permitted")


def _check_columns(tree: exp.Expression) -> None:
    for star in tree.find_all(exp.Star):
        # COUNT(*) is a star too, and a legitimate one. Only a star in a
        # projection position is a problem: `SELECT *` would surface whatever
        # columns the relation happens to carry, putting the projection outside
        # policy control. `employees.*` parses as Column(this=Star) and is
        # caught here as well, because its parent is a Column, not a function.
        if isinstance(star.parent, exp.Func):
            continue
        _fail("SELECT * is not permitted; name the columns you need")
    for column in tree.find_all(exp.Column):
        name = (column.name or "").lower()
        if name and name not in ALLOWED_COLUMNS:
            _fail(
                f"unknown column {column.name!r}; available columns are: "
                f"{', '.join(sorted(ALLOWED_COLUMNS))}"
            )


def _check_functions(tree: exp.Expression) -> None:
    for func in tree.find_all(exp.Func):
        name = (func.sql_name() or "").lower()
        if name and name not in ALLOWED_FUNCTIONS:
            _fail(f"function {name}() is not permitted")


def _check_masked_columns(tree: exp.Expression, masked: frozenset[str]) -> bool:
    """Enforce the role's column policy on model-written SQL.

    An analyst may ask for the average salary; they may not list salaries next
    to names. The obvious rule -- "allowed inside any aggregate" -- is not quite
    right, and the model bake-off found the gap: `SELECT MAX(salary)` is an
    aggregate by syntax and a single individual's pay by content. MIN and MAX
    select one row's value rather than combining many, so they are treated as
    row-level reads of a masked column. See EXTREMAL_AGGREGATES in spec.py.
    """
    if not masked:
        return False

    # A masked column inside MIN()/MAX() discloses one person, so reject it
    # before the more permissive aggregate rule below can wave it through.
    for node in tree.find_all(exp.Min, exp.Max):
        for column in node.find_all(exp.Column):
            name = (column.name or "").lower()
            if name in masked:
                verb = node.__class__.__name__.upper()
                raise tag(
                    SqlRejected(
                        f"your role may not read {name} for individual employees, and "
                        f"{verb}({name}) reports one specific person's {name}. "
                        f"Use AVG({name}) or a median instead, and say plainly which "
                        f"statistic you computed -- do not present it as the {verb.lower()}"
                    ),
                    Layer.L1,
                )

    aggregate_nodes = tuple(tree.find_all(exp.AggFunc))
    for column in tree.find_all(exp.Column):
        name = (column.name or "").lower()
        if name not in masked:
            continue
        inside_aggregate = any(column in agg.find_all(exp.Column) for agg in aggregate_nodes)
        if not inside_aggregate:
            raise tag(
                SqlRejected(
                    f"your role may not read {name} for individual employees; "
                    f"aggregate it instead, for example AVG({name})"
                ),
                Layer.L1,
            )
    return True


def _apply_limit(select: exp.Select, rewrites: list[str]) -> None:
    existing = select.args.get("limit")
    if existing is None:
        select.limit(MAX_LIMIT, copy=False)
        rewrites.append(f"appended LIMIT {MAX_LIMIT}")
        return
    try:
        value = int(existing.expression.name)
    except (AttributeError, ValueError):
        select.limit(MAX_LIMIT, copy=False)
        rewrites.append(f"replaced a non-literal LIMIT with {MAX_LIMIT}")
        return
    if value > MAX_LIMIT:
        select.limit(MAX_LIMIT, copy=False)
        rewrites.append(f"lowered LIMIT {value} to {MAX_LIMIT}")


def _apply_k_anonymity(select: exp.Select, rewrites: list[str]) -> None:
    """Attach a minimum cohort size to any grouped aggregate.

    Rewritten on the AST and re-generated from the tree -- never spliced into
    the SQL text. String surgery on SQL is the bug class this whole layer exists
    to avoid, and a trailing comment defeats it.
    """
    if not ENFORCE_MIN_COHORT:
        return
    if not select.args.get("group"):
        return
    if not list(select.find_all(exp.AggFunc)):
        return

    condition = exp.GTE(
        this=exp.Count(this=exp.Star()),
        expression=exp.Literal.number(MIN_COHORT_SIZE),
    )
    having = select.args.get("having")
    if having is None:
        select.set("having", exp.Having(this=condition))
    else:
        select.set("having", exp.Having(this=exp.And(this=having.this, expression=condition)))
    rewrites.append(f"required COUNT(*) >= {MIN_COHORT_SIZE} per group (k-anonymity)")


def guard_sql(sql: str, *, masked_columns: frozenset[str] = frozenset()) -> GuardResult:
    """Validate and rewrite a model-written statement, or raise `SqlRejected`."""
    if not sql or not sql.strip():
        _fail("empty statement")

    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except Exception as exc:  # sqlglot raises several types
        _fail(f"could not parse the statement: {exc}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        _fail(f"exactly one statement is permitted, found {len(statements)}")

    tree = statements[0]
    if not isinstance(tree, exp.Select):
        _fail("only SELECT statements are permitted")

    _check_no_forbidden_nodes(tree)
    _check_tables(tree)
    _check_columns(tree)
    _check_functions(tree)
    _check_masked_columns(tree, masked_columns)

    rewrites: list[str] = []
    _apply_k_anonymity(tree, rewrites)
    _apply_limit(tree, rewrites)

    aggregate_only = bool(list(tree.find_all(exp.AggFunc))) and not tree.args.get("group")

    return GuardResult(
        sql=tree.sql(dialect=DIALECT),
        original_sql=sql.strip(),
        rewrites=rewrites,
        aggregate_only=aggregate_only,
    )

"""Layer 3a -- the structured query path.

The default way the agent reads data. The model does not write SQL here: it
fills in a typed specification, and the server compiles that into a fully
parameterised statement. No string the model produces is ever concatenated into
SQL -- columns come from enums, operators come from enums, and values are bound
parameters.

This path handles the large majority of real questions. `sql_guard` is the
escape hatch for the rest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from db import AGENT_COLUMNS
from secure_rls.security.layers import Layer, tag

#: Minimum number of underlying rows behind any reported aggregate.
#:
#: RLS answers "which rows may you read". It says nothing about "how precisely
#: may you summarise them", and an average over one person is that person's
#: salary. That is a k-anonymity problem rather than an access-control one, and
#: it is the difference between protecting *rows* and protecting *inference*.
MIN_COHORT_SIZE = 5

#: Whether to enforce it. **Off by default, deliberately.**
#:
#: A cohort floor silently drops small groups from every grouped result, so a
#: department of four vanishes from a chart and the numbers stop adding up.
#: That cost is paid on every ordinary question, while the attack it prevents
#: -- narrowing an aggregate onto one person -- is a statistical-inference
#: problem that a cohort floor only partially addresses anyway. Scoped out of
#: this project and recorded as future work, where it belongs alongside query
#: budgets and differential privacy.
#:
#: What this does NOT relax: the tenant boundary, which is enforced below the
#: query layer and is unaffected, and `EXTREMAL_AGGREGATES` below, which is a
#: role control rather than an inference one.
#:
#: The honest consequence, stated rather than buried: with this off, a role
#: barred from reading an individual salary can still reach one by narrowing an
#: average onto a single person. Set to True to restore the floor.
ENFORCE_MIN_COHORT = False


#: Rows a *listing* query returns, and the ceiling the model may raise it to.
#:
#: Not a security control. The tenant boundary is enforced below the query
#: layer and is unaffected by these; what they bound is how much of a result
#: gets pushed into the model's context, which costs synthesis latency and, past
#: a point, coherence.
#:
#: The default was 25 and was too small to be honest about. Every department in
#: every tenant is larger than that -- the biggest is 86 -- so any request of
#: the form "report on the Sales team" silently returned a third of the team.
#: The truncation notice fired, correctly, and the model then wrote its report
#: from the truncated rows anyway. A default that is wrong for every realistic
#: question is not a default, it is a trap. 100 covers the largest department in
#: every tenant, which is the natural unit of a question here.
#:
#: Aggregates are unaffected: `LIMIT` applies to result rows, so `AVG(salary)`
#: still reads the whole tenant and returns one row, and the gateway's
#: percentile path explicitly overrides the cap to read all of it.
DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 200


#: The columns the model may name, derived from the database catalog.
#:
#: This was a hand-written enum. It is generated now for the reason the author
#: of the third copy always discovers too late: `db.AGENT_COLUMNS`, this enum
#: and `sql_guard.ALLOWED_COLUMNS` were three statements of one fact, and
#: nothing made them agree. Adding a column meant editing three files and
#: hoping; forgetting one meant the guard silently disagreed with the schema
#: the model was given.
#:
#: A generated enum is exactly as strong a control as a written one. The model
#: still cannot name a column outside it -- Pydantic rejects the value -- and
#: the contents still come from a trusted, privileged read rather than from
#: anything the model said.
Column = StrEnum("Column", {name.upper(): name for name in AGENT_COLUMNS})


class Aggregate(StrEnum):
    COUNT = "count"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    #: SQLite has no MEDIAN function, so this one is computed by the gateway in
    #: pandas over the tenant-bound rows rather than compiled into SQL. It is
    #: here because "what is the median salary" is an obvious HR question, and
    #: the correctness suite caught the model flailing without it -- asked for a
    #: median, it selected raw salaries and gave up.
    MEDIAN = "median"
    #: Percentiles, same mechanism as the median.
    #:
    #: These exist so that "what is the highest salary?" has an answer for a
    #: role that may not read one. MAX reports exactly what the top earner is
    #: paid; p90 describes the top of the range without being any individual's
    #: figure. Refusing a reasonable question with no alternative is how a
    #: policy stops being a boundary and starts being an obstacle -- and an
    #: obstacle is what people route around.
    P75 = "p75"
    P90 = "p90"


#: Aggregates that return an actual individual's value rather than a summary of
#: many, and therefore disclose that individual.
#:
#: This distinction is not the same as k-anonymity and is not covered by it.
#: MAX(salary) over a cohort of five hundred passes every cohort-size check and
#: still reports exactly what the top earner is paid -- and "the top earner" is
#: a salient, targetable identity. AVG, SUM and COUNT combine values; MIN and
#: MAX select one.
#:
#: Found by the model bake-off: an analyst barred from reading individual
#: salaries asked for the highest paid person, and qwen2.5 correctly answered
#: "999,999 EUR" by way of MAX(). The tenant boundary held throughout -- this is
#: a *role* boundary failure, and the leak-rate metric was blind to it because
#: it only ever measured cross-tenant disclosure.
#:
#: MEDIAN is deliberately not in this set. On an odd-sized cohort it can equal
#: some individual's value, but "the median earner" is not an identity anyone
#: can target, and blocking it would remove a genuinely useful statistic. That
#: is a judgement call, and it is recorded here rather than left implicit.
EXTREMAL_AGGREGATES = frozenset({"min", "max"})

#: Aggregates the gateway computes in pandas rather than compiling to SQL.
GATEWAY_COMPUTED = {"median": 0.5, "p75": 0.75, "p90": 0.90}


class Operator(StrEnum):
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    LIKE = "like"
    IN = "in"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class Predicate(BaseModel):
    """One filter condition. The value is always bound, never interpolated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: Column
    op: Operator
    value: str | float | int | list[str | float | int] | None = None

    @field_validator("value")
    @classmethod
    def _list_must_be_small(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) > 50:
            raise ValueError("IN list may contain at most 50 values")
        return v


class Metric(BaseModel):
    """An aggregate to compute, e.g. avg(salary)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agg: Aggregate
    #: The column being aggregated. `None` only for COUNT, which compiles to
    #: COUNT(*) and names no column at all.
    #:
    #: This used to default to `user_id`, which meant a metric built without a
    #: column silently became an aggregate over an identifier rather than an
    #: error. A default here is the same mistake `metric_column` made one layer
    #: up: it converts "the caller did not say" into "the caller said this",
    #: and the result is a number that answers a question nobody asked.
    column: Column | None = None
    alias: str | None = None

    @model_validator(mode="after")
    def _an_aggregate_must_say_what_it_aggregates(self) -> Metric:
        """Fail closed: no column, no aggregate -- except COUNT, which needs none.

        The tool contract already resolves or refuses `metric_column` before a
        spec is built (`_resolve_metric_column` in tools/factory.py), and that
        is where the useful error message and the model's retry live. This is
        the backstop for every other caller: the gateway's percentile path, the
        evals, and anything added later that builds a spec by hand.
        """
        if self.agg is not Aggregate.COUNT and self.column is None:
            raise ValueError(f"{self.agg.value} needs a column; only count may omit one")
        return self

    def sql(self) -> str:
        if self.agg.value in GATEWAY_COMPUTED:
            raise SpecError(f"{self.agg.value} is computed by the gateway, not compiled to SQL")
        func = self.agg.value.upper()
        inner = "*" if self.agg is Aggregate.COUNT else self.column.value
        return f"{func}({inner}) AS {self.output_name()}"

    def output_name(self) -> str:
        if self.alias and self.alias.replace("_", "").isalnum():
            return self.alias
        return f"{self.agg.value}_{'rows' if self.agg is Aggregate.COUNT else self.column.value}"


class QuerySpec(BaseModel):
    """A read request, expressed structurally.

    Note what is absent: there is no `tenant_id` field, and no `table` field.
    The tenant comes from the session principal via the bound connection, and
    there is exactly one table the agent can reach.
    """

    model_config = ConfigDict(extra="forbid")

    select: list[Column] = Field(default_factory=list)
    #: Return unique combinations of the selected columns.
    #:
    #: Without this, "what departments are there?" has no representation: a
    #: spec needs a select or a metric, and `group_by` alone was rejected -- so
    #: the model bolted a COUNT onto the question to make the request valid and
    #: answered something nobody asked. A missing verb in the query language
    #: shows up as the model doing something odd, not as an error.
    distinct: bool = False
    metrics: list[Metric] = Field(default_factory=list)
    filters: list[Predicate] = Field(default_factory=list)
    group_by: list[Column] = Field(default_factory=list)
    order_by: Column | None = None
    descending: bool = True
    limit: int = Field(default=DEFAULT_ROW_LIMIT, ge=1, le=MAX_ROW_LIMIT)

    @model_validator(mode="after")
    def _distinct_and_metrics_are_two_different_questions(self) -> QuerySpec:
        """`distinct` with `metrics` is ambiguous, so it is refused rather than dropped.

        The compiler used to silently ignore `distinct` whenever a metric was
        present -- `SELECT DISTINCT` became plain `SELECT` -- and the result was
        a wrong answer rather than a refusal. Asked for the number of distinct
        departments, a model sending `select=[department], distinct=True,
        metrics=[count]` got back `department = Engineering; count_rows = 500`:
        one arbitrary department beside the count of *every* row.

        The request has two plausible readings and the compiler cannot pick:
        `COUNT(DISTINCT department)`, or a count per department. Both are
        expressible -- the second as `group_by`, the first through `run_sql` --
        so naming them beats guessing between them.
        """
        if self.distinct and self.metrics:
            raise ValueError(
                "`distinct` cannot be combined with `metrics`: for a count per "
                "group use group_by, and for a count of distinct values use "
                "run_sql with COUNT(DISTINCT column)"
            )
        return self

    @model_validator(mode="after")
    def _a_projected_column_beside_an_aggregate_is_refused(self) -> QuerySpec:
        """A bare column next to an ungrouped aggregate has no defined value.

        `SELECT department, COUNT(*) FROM employees` is not valid SQL under
        `ONLY_FULL_GROUP_BY` and most engines reject it. SQLite accepts it and
        fills the bare column from a row of its own choosing, so the query
        returned `department = Engineering; count_rows = 500` -- one arbitrary
        department beside the count of every row in the tenant. The number is
        right, the department is noise, and nothing said so.

        SQLite does define the bare column for a lone MIN/MAX -- it comes from
        the extremal row -- which is why `select=[salary], metrics=[max]`
        looked correct while `metrics=[avg]` did not. Relying on that is
        relying on one engine's documented quirk to make a query mean what it
        appears to mean, and this project's whole argument is against depending
        on engine behaviour nobody enumerated. Refusing every case is the same
        answer for every aggregate, on every engine.

        Grouping is unaffected: a column in `group_by` has exactly one value
        per output row, which is the shape that makes the projection sound.
        """
        if not self.metrics:
            return self
        bare = [c.value for c in self.select if c not in self.group_by]
        if bare:
            raise ValueError(
                f"{', '.join(bare)} cannot be selected alongside an aggregate: the "
                f"value would come from an arbitrary row. Add it to group_by to get "
                f"one aggregate per value, or drop it from select to get one "
                f"aggregate over everything"
            )
        return self

    @field_validator("filters")
    @classmethod
    def _cap_filters(cls, v: list[Predicate]) -> list[Predicate]:
        if len(v) > 8:
            raise ValueError("at most 8 filters")
        return v


class CompiledQuery(BaseModel):
    """The result of compiling a spec: SQL plus its bound parameters."""

    model_config = ConfigDict(frozen=True)

    sql: str
    params: list[Any]
    k_anonymity_applied: bool
    aggregate_only: bool


class SpecError(ValueError):
    """The spec is not expressible under the current policy."""


def _compile_predicate(pred: Predicate) -> tuple[str, list[Any]]:
    col = pred.column.value  # from an enum -- cannot be arbitrary text
    if pred.op is Operator.IS_NULL:
        return f"{col} IS NULL", []
    if pred.op is Operator.NOT_NULL:
        return f"{col} IS NOT NULL", []
    if pred.op is Operator.IN:
        values = pred.value if isinstance(pred.value, list) else [pred.value]
        if not values:
            raise SpecError("IN requires at least one value")
        placeholders = ", ".join("?" for _ in values)
        return f"{col} IN ({placeholders})", list(values)
    if pred.op is Operator.LIKE:
        return f"{col} LIKE ?", [str(pred.value)]
    if pred.value is None:
        raise SpecError(f"operator {pred.op.value} requires a value")
    return f"{col} {pred.op.value} ?", [pred.value]


def _referenced_columns(spec: QuerySpec) -> set[str]:
    """Every column the spec touches, not only the ones it projects.

    Filters and ordering count. A predicate on a column the role may not see is
    an inference channel -- `WHERE hidden > 100000` narrows the population
    without ever returning the value.
    """
    names = {c.value for c in spec.select}
    names |= {c.value for c in spec.group_by}
    names |= {m.column.value for m in spec.metrics if m.column is not None}
    names |= {f.column.value for f in spec.filters}
    if spec.order_by is not None:
        names.add(spec.order_by.value)
    return names


def _non_aggregate_positions(spec: QuerySpec) -> list[tuple[str, str]]:
    """(column, position) for every reference *outside* an aggregate.

    Masking permits exactly one thing: combining many values into one. Every
    other way of naming the column -- projecting it, grouping by it, filtering
    on it, ordering by it -- reaches an individual, so the aggregate exemption
    must not extend to any of them.

    `metrics` is deliberately absent: that is the exempt position, policed
    separately by `EXTREMAL_AGGREGATES`. A column appearing in both a metric
    and a filter is still refused, because the filter is a disclosure channel
    regardless of what else the spec asks for.
    """
    found = [(c.value, "select") for c in spec.select]
    found += [(c.value, "group_by") for c in spec.group_by]
    found += [(f.column.value, "filter") for f in spec.filters]
    if spec.order_by is not None:
        found.append((spec.order_by.value, "order_by"))
    return found


#: How to explain a masked column refused in each position. Each says what the
#: caller *can* do instead: a refusal with no alternative is an obstacle, and an
#: obstacle is what people route around.
_MASK_REFUSAL = {
    "select": (
        "your role may not read {col} for individual employees. You can still "
        "describe the range: p90 for the top of it, median for typical, or "
        "average {col} by department"
    ),
    "group_by": (
        "your role may not group by {col}: each group is one distinct value of a "
        "column you may not read for an individual. Group by department or "
        "hire_date instead and aggregate {col} within the group"
    ),
    "filter": (
        "your role may not filter on {col}: a predicate on a column you cannot "
        "read still discloses it, one comparison at a time. Filter on department, "
        "hire_date or performance_score instead, and aggregate {col} within that "
        "group"
    ),
    "order_by": (
        "your role may not order by {col}: ranking people by a column you cannot "
        "read discloses who sits at the top of it. Order by another column, or "
        "ask for an aggregate such as average or p90"
    ),
}


def check_masked_columns(spec: QuerySpec, masked_columns: frozenset[str]) -> None:
    """Enforce the role's column mask across every position of a spec.

    Split out of `compile_spec` because it has a second caller: the gateway's
    percentile path builds its own internal spec and so never reaches the
    compiler with the *caller's* spec in hand. Enforcement that lives only in
    the compiler is enforcement one dispatch branch can step around, which is
    invariant 5b applied to the query path rather than the prompt.
    """
    if not masked_columns:
        return

    for column, position in _non_aggregate_positions(spec):
        if column in masked_columns:
            raise tag(
                SpecError(_MASK_REFUSAL[position].format(col=column)),
                Layer.L1,  # the role decides; layer 3 only enforces the decision
            )

    for metric in spec.metrics:
        if metric.column is None:  # COUNT(*) names no column to mask
            continue
        if metric.column.value in masked_columns and metric.agg.value in EXTREMAL_AGGREGATES:
            raise tag(
                SpecError(
                    f"your role may not read {metric.column.value} for individual "
                    f"employees, and {metric.agg.value.upper()}({metric.column.value}) "
                    f"reports one specific person's {metric.column.value}. For the top of "
                    f"the range use p90, or use an average or median -- and say plainly "
                    f"which statistic you computed, never presenting it as the "
                    f"{metric.agg.value}"
                ),
                Layer.L1,
            )


def compile_spec(
    spec: QuerySpec,
    *,
    masked_columns: frozenset[str] = frozenset(),
    hidden_columns: frozenset[str] = frozenset(),
    limit_override: int | None = None,
) -> CompiledQuery:
    """Turn a validated spec into parameterised SQL.

    `masked_columns` comes from the caller's role policy (layer 1). A masked
    column may still be aggregated -- an analyst can ask for the average salary
    in Engineering -- but may not be selected for a named individual.

    `hidden_columns` is the stronger form from the same policy: the role may not
    name the column at all, in any position. Checked first, because a hidden
    column must not fall through to the more permissive aggregate rule below.

    `limit_override` raises the row cap above what a `QuerySpec` can express.
    It exists for one internal caller: the gateway's median path, which must
    read the whole tenant to compute a correct statistic. The model cannot
    reach it -- `QuerySpec.limit` is still capped at 200 by validation, and
    this argument is keyword-only and never populated from tool input.
    """
    if not spec.select and not spec.metrics and not spec.group_by:
        raise SpecError("a query must select at least one column, metric or grouping")

    aggregate_only = bool(spec.metrics) and not spec.select

    forbidden = _referenced_columns(spec) & hidden_columns
    if forbidden:
        raise tag(
            SpecError(
                f"your role may not access {', '.join(sorted(forbidden))}. "
                f"Ask about the columns in the schema you were given"
            ),
            Layer.L1,
        )

    check_masked_columns(spec, masked_columns)

    projections: list[str] = [c.value for c in spec.group_by]
    projections += [c.value for c in spec.select if c not in spec.group_by]
    projections += [m.sql() for m in spec.metrics]

    # `group_by` with no metric is a request for the distinct values of those
    # columns, which is what "what departments are there?" means. `spec.distinct`
    # with metrics cannot reach here -- QuerySpec refuses that combination rather
    # than dropping the keyword, which is what this line used to do.
    distinct = spec.distinct or (bool(spec.group_by) and not spec.metrics)
    keyword = "SELECT DISTINCT" if distinct else "SELECT"

    # ruff flags f-string SQL construction, correctly in general. It is safe
    # here for a reason worth stating rather than silencing globally: every
    # element of `projections` originates from the `Column` / `Aggregate` enums
    # or from `Metric.output_name()`, which is alphanumeric-checked. No value
    # the model produced reaches this string -- values are bound below.
    sql = f"{keyword} {', '.join(projections)} FROM employees"  # noqa: S608
    params: list[Any] = []

    if spec.filters:
        clauses = []
        for pred in spec.filters:
            clause, bound = _compile_predicate(pred)
            clauses.append(clause)
            params.extend(bound)
        sql += " WHERE " + " AND ".join(clauses)

    k_applied = False
    # GROUP BY only earns its place when something is being aggregated; with no
    # metric, SELECT DISTINCT above already expresses the question.
    if spec.group_by and spec.metrics:
        sql += " GROUP BY " + ", ".join(c.value for c in spec.group_by)
        if spec.metrics and ENFORCE_MIN_COHORT:
            # Grouped aggregates get a minimum cohort size, always. A group of
            # one is an individual disclosure wearing an aggregate's clothes.
            sql += f" HAVING COUNT(*) >= {MIN_COHORT_SIZE}"
            k_applied = True

    if spec.order_by is not None:
        direction = "DESC" if spec.descending else "ASC"
        sql += f" ORDER BY {spec.order_by.value} {direction}"
    elif spec.metrics and spec.group_by:
        sql += f" ORDER BY {spec.metrics[0].output_name()} DESC"

    sql += " LIMIT ?"
    params.append(limit_override if limit_override is not None else spec.limit)

    return CompiledQuery(
        sql=sql,
        params=params,
        k_anonymity_applied=k_applied,
        aggregate_only=aggregate_only,
    )


def cohort_size_query(spec: QuerySpec) -> tuple[str, list[Any]]:
    """Count the rows an ungrouped aggregate would be computed over.

    Grouped aggregates are protected by the injected HAVING clause. An ungrouped
    one -- "average salary of employees hired on 2020-01-15 whose name starts
    with J" -- needs the same protection applied after the fact, because there
    is no group to attach a HAVING to.
    """
    sql = "SELECT COUNT(*) AS n FROM employees"
    params: list[Any] = []
    if spec.filters:
        clauses = []
        for pred in spec.filters:
            clause, bound = _compile_predicate(pred)
            clauses.append(clause)
            params.extend(bound)
        sql += " WHERE " + " AND ".join(clauses)
    return sql, params


SpecLiteral = Literal["query_spec"]

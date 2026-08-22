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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    column: Column = Column.USER_ID
    alias: str | None = None

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
    limit: int = Field(default=50, ge=1, le=200)

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


def compile_spec(
    spec: QuerySpec,
    *,
    masked_columns: frozenset[str] = frozenset(),
    limit_override: int | None = None,
) -> CompiledQuery:
    """Turn a validated spec into parameterised SQL.

    `masked_columns` comes from the caller's role policy (layer 1). A masked
    column may still be aggregated -- an analyst can ask for the average salary
    in Engineering -- but may not be selected for a named individual.

    `limit_override` raises the row cap above what a `QuerySpec` can express.
    It exists for one internal caller: the gateway's median path, which must
    read the whole tenant to compute a correct statistic. The model cannot
    reach it -- `QuerySpec.limit` is still capped at 200 by validation, and
    this argument is keyword-only and never populated from tool input.
    """
    if not spec.select and not spec.metrics and not spec.group_by:
        raise SpecError("a query must select at least one column, metric or grouping")

    aggregate_only = bool(spec.metrics) and not spec.select

    for column in spec.select:
        if column.value in masked_columns:
            raise tag(
                SpecError(
                    f"your role may not read {column.value} for individual employees. "
                    f"You can still describe the range: p90 for the top of it, median "
                    f"for typical, or average {column.value} by department"
                ),
                Layer.L1,  # the role decides; layer 3 only enforces the decision
            )

    for metric in spec.metrics:
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

    projections: list[str] = [c.value for c in spec.group_by]
    projections += [c.value for c in spec.select if c not in spec.group_by]
    projections += [m.sql() for m in spec.metrics]

    # `group_by` with no metric is a request for the distinct values of those
    # columns, which is what "what departments are there?" means.
    distinct = spec.distinct or (bool(spec.group_by) and not spec.metrics)
    keyword = "SELECT DISTINCT" if distinct and not spec.metrics else "SELECT"

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

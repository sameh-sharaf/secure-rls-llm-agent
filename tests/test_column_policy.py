"""The role column policy: masked columns, hidden columns, and one dead field.

`ColumnPolicy.visible` was declared, populated from a hand-written column list
and never read by anything. Harmless while nothing consumed it, and a trap the
moment someone assumed it was enforced -- so it is wired up here rather than
deleted, because the two-tier distinction is worth having:

* a **masked** column may be aggregated but not read for an individual;
* a **hidden** column may not be named at all, and is absent from the schema
  the model is shown.

Neither shipped role hides anything. These tests construct a restricted policy
so the mechanism is exercised, which is the only honest way to have it: an
unenforced field and an untested one fail the same way.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import AGENT_COLUMNS, schema_description  # noqa: E402
from secure_rls.security.gateway import QueryGateway  # noqa: E402
from secure_rls.security.principal import (  # noqa: E402
    ROLE_POLICY,
    ColumnPolicy,
    Role,
    authenticate,
)
from secure_rls.security.spec import (  # noqa: E402
    Aggregate,
    Column,
    Metric,
    Operator,
    Predicate,
    QuerySpec,
    SpecError,
    check_masked_columns,
    compile_spec,
)
from secure_rls.security.sql_guard import SqlRejected, guard_sql  # noqa: E402

HIDDEN = frozenset({"notes"})


@pytest.fixture
def restricted() -> Iterator[QueryGateway]:
    """An analyst whose role cannot see `notes` at all."""
    original = ROLE_POLICY[Role.ANALYST]
    ROLE_POLICY[Role.ANALYST] = ColumnPolicy(
        visible=frozenset(AGENT_COLUMNS) - HIDDEN, row_level_salary=False
    )
    gw = QueryGateway(authenticate("acme_analyst", "acme123"))
    try:
        yield gw
    finally:
        gw.close()
        ROLE_POLICY[Role.ANALYST] = original


# --------------------------------------------------------------------------
# the shipped policy
# --------------------------------------------------------------------------

def test_no_shipped_role_hides_a_column() -> None:
    """The case study is about tenant isolation, not column restriction.

    If this ever fails, someone narrowed the demo. That may be deliberate --
    but it should be a decision, not a side effect.
    """
    for role, policy in ROLE_POLICY.items():
        assert policy.hidden_columns() == frozenset(), f"{role} hides {policy.hidden_columns()}"


def test_visible_is_derived_from_the_catalog() -> None:
    """It was a hand-written list -- the drift ADR-0005 removed everywhere else."""
    for policy in ROLE_POLICY.values():
        assert policy.visible == frozenset(AGENT_COLUMNS)


def test_a_new_column_is_hidden_by_default_from_a_restricted_role() -> None:
    """Fail closed: a column added to the table must be granted, not inherited."""
    policy = ColumnPolicy(visible=frozenset({"user_id", "name"}), row_level_salary=False)
    assert "salary" in policy.hidden_columns()
    assert "user_id" not in policy.hidden_columns()


def test_masking_and_hiding_are_different_things() -> None:
    analyst = ROLE_POLICY[Role.ANALYST]
    assert analyst.masked_columns() == frozenset({"salary"})
    assert analyst.hidden_columns() == frozenset()


# --------------------------------------------------------------------------
# enforcement -- the structured path
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        QuerySpec(select=[Column.NOTES]),
        QuerySpec(group_by=[Column.NOTES]),
        QuerySpec(metrics=[Metric(agg=Aggregate.COUNT, column=Column.NOTES)]),
        # A predicate never returns the value and still narrows the population.
        QuerySpec(
            select=[Column.NAME],
            filters=[Predicate(column=Column.NOTES, op=Operator.LIKE, value="%promoted%")],
        ),
        QuerySpec(select=[Column.NAME], order_by=Column.NOTES),
    ],
    ids=["select", "group_by", "metric", "filter", "order_by"],
)
def test_a_hidden_column_is_refused_in_every_position(spec: QuerySpec) -> None:
    with pytest.raises(SpecError, match="may not access notes"):
        compile_spec(spec, hidden_columns=HIDDEN)


MASKED = frozenset({"salary"})


@pytest.mark.parametrize(
    "spec",
    [
        QuerySpec(select=[Column.SALARY]),
        # Each group is one distinct salary, printed beside its count.
        QuerySpec(metrics=[Metric(agg=Aggregate.COUNT)], group_by=[Column.SALARY]),
        # Binary search: twenty of these recover an exact salary, and not one
        # of them ever asks for the column.
        QuerySpec(
            select=[Column.NAME],
            filters=[Predicate(column=Column.SALARY, op=Operator.GTE, value=150000)],
        ),
        # Ranking people by a column you may not read names the top earner.
        QuerySpec(select=[Column.NAME], order_by=Column.SALARY),
    ],
    ids=["select", "group_by", "filter", "order_by"],
)
def test_a_masked_column_is_refused_outside_an_aggregate(spec: QuerySpec) -> None:
    """The aggregate exemption applies to `metrics` and to nothing else.

    The hidden-column rule was written across every position from the start;
    the mask was checked only in `select`, so `filters` and `order_by` reached
    the database on the structured path while `sql_guard` refused the identical
    query written as SQL. Two implementations of one policy, disagreeing.
    """
    with pytest.raises(SpecError, match="salary"):
        compile_spec(spec, masked_columns=MASKED)


@pytest.mark.parametrize(
    "spec",
    [
        QuerySpec(metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)]),
        QuerySpec(
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            group_by=[Column.DEPARTMENT],
        ),
        QuerySpec(
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            filters=[Predicate(column=Column.DEPARTMENT, op=Operator.EQ, value="Sales")],
        ),
        QuerySpec(select=[Column.NAME], order_by=Column.PERFORMANCE_SCORE),
    ],
    ids=["avg", "avg_by_dept", "avg_filtered_by_dept", "order_by_other"],
)
def test_the_mask_does_not_block_legitimate_aggregates(spec: QuerySpec) -> None:
    """Closing the positions above must not close the ones the role is for.

    An analyst exists to compute salary statistics. A rule that refuses those
    too would score just as well on any leak metric and be worth nothing.
    """
    compile_spec(spec, masked_columns=MASKED)  # must not raise


@pytest.mark.parametrize("agg", [Aggregate.MEDIAN, Aggregate.P75, Aggregate.P90])
def test_the_mask_permits_the_gateway_computed_statistics(agg: Aggregate) -> None:
    """Median and percentiles never reach `compile_spec` -- they are computed in
    pandas by the gateway -- so the policy is asserted against the check itself.

    These are the statistics an analyst is told to reach for in place of a
    maximum. If the mask refused them the refusal message would be advice the
    system does not honour.
    """
    check_masked_columns(
        QuerySpec(metrics=[Metric(agg=agg, column=Column.SALARY)]), MASKED
    )  # must not raise


def test_the_percentile_path_enforces_the_mask_on_the_caller_s_spec() -> None:
    """`run_spec` dispatches to the percentile path before compiling anything.

    That branch builds its own internal spec and compiles *that*, deliberately
    without a mask, so a filter on a masked column in the caller's spec reached
    the database while the ordinary path refused it. Enforcement that lives
    only in the compiler is enforcement a dispatch branch can step around.
    """
    gw = QueryGateway(authenticate("acme_analyst", "acme123"))
    try:
        spec = QuerySpec(
            metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)],
            filters=[Predicate(column=Column.SALARY, op=Operator.GTE, value=150000)],
        )
        with pytest.raises(SpecError, match="may not filter on salary"):
            gw.run_spec(spec)
        # The same statistic without the smuggled predicate still works.
        assert gw.run_spec(
            QuerySpec(metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)])
        ).rows
    finally:
        gw.close()


def test_the_structured_and_sql_paths_agree_on_the_mask() -> None:
    """One policy, two implementations -- pinned against drifting apart again."""
    for spec, sql in [
        (QuerySpec(select=[Column.NAME], order_by=Column.SALARY),
         "SELECT name FROM employees ORDER BY salary DESC"),
        (QuerySpec(select=[Column.NAME],
                   filters=[Predicate(column=Column.SALARY, op=Operator.GT, value=150000)]),
         "SELECT name FROM employees WHERE salary > 150000"),
        (QuerySpec(metrics=[Metric(agg=Aggregate.COUNT)], group_by=[Column.SALARY]),
         "SELECT salary, COUNT(*) FROM employees GROUP BY salary"),
    ]:
        with pytest.raises(SpecError):
            compile_spec(spec, masked_columns=MASKED)
        with pytest.raises(SqlRejected):
            guard_sql(sql, masked_columns=MASKED)


def test_hiding_beats_the_aggregate_exemption() -> None:
    """A masked column survives inside AVG(); a hidden one does not."""
    spec = QuerySpec(metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)])
    compile_spec(spec, masked_columns=frozenset({"salary"}))          # allowed
    with pytest.raises(SpecError):
        compile_spec(spec, hidden_columns=frozenset({"salary"}))      # not allowed


# --------------------------------------------------------------------------
# enforcement -- model-written SQL
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT notes FROM employees",
        "SELECT COUNT(notes) FROM employees",
        "SELECT name FROM employees WHERE notes LIKE '%promoted%'",
        "SELECT name FROM employees ORDER BY notes",
        "SELECT department, COUNT(*) FROM employees GROUP BY department HAVING MAX(notes) > 'a'",
    ],
)
def test_layer_3_refuses_a_hidden_column_in_sql(sql: str) -> None:
    with pytest.raises(SqlRejected, match="may not access notes"):
        guard_sql(sql, hidden_columns=HIDDEN)


def test_sql_without_the_hidden_column_still_works() -> None:
    """A policy that blocks everything is easy and useless."""
    assert guard_sql("SELECT department, AVG(salary) FROM employees GROUP BY department").sql


# --------------------------------------------------------------------------
# the side channels -- invariant 5b
# --------------------------------------------------------------------------

def test_the_prompt_schema_omits_a_hidden_column() -> None:
    text = schema_description(HIDDEN)
    assert "notes" not in text
    assert "salary" in text, "hiding one column must not blank the schema"


def test_sample_rows_drop_the_key_rather_than_masking_it(restricted: QueryGateway) -> None:
    """A placeholder would still tell the model the column is there."""
    rows = restricted.sample_rows(3)
    assert rows
    for row in rows:
        assert "notes" not in row
        assert row["salary"] == QueryGateway.MASKED_PLACEHOLDER  # masked, not hidden
        assert "name" in row


def test_the_gateway_refuses_a_hidden_column_end_to_end(restricted: QueryGateway) -> None:
    with pytest.raises(SpecError, match="may not access notes"):
        restricted.run_spec(QuerySpec(select=[Column.NOTES]))
    with pytest.raises(SqlRejected, match="may not access notes"):
        restricted.run_sql("SELECT notes FROM employees")


def test_the_tenant_s_own_queries_still_work(restricted: QueryGateway) -> None:
    result = restricted.run_spec(
        QuerySpec(
            select=[Column.DEPARTMENT],
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            group_by=[Column.DEPARTMENT],
        )
    )
    assert result.rows


# ------------------------------------------------- no silent substitutions ---
# Two defaults used to turn "the caller did not say" into "the caller said
# this": `Metric.column` fell back to `user_id`, and the compiler dropped
# `distinct` whenever a metric was present. Both produced a confident answer to
# a question nobody asked, which is worse than a refusal.


@pytest.mark.parametrize("agg", ["avg", "sum", "min", "max", "median", "p75", "p90"])
def test_an_aggregate_without_a_column_is_refused_not_defaulted(agg: str) -> None:
    with pytest.raises(ValidationError) as exc:
        Metric(agg=Aggregate(agg))
    assert "needs a column" in str(exc.value)


def test_count_is_the_one_aggregate_that_needs_no_column() -> None:
    """COUNT compiles to COUNT(*), so there is no column to demand or invent."""
    metric = Metric(agg=Aggregate.COUNT)
    assert metric.column is None
    assert metric.sql() == "COUNT(*) AS count_rows"


def test_a_countless_spec_still_compiles_with_no_column_named() -> None:
    compiled = compile_spec(QuerySpec(metrics=[Metric(agg=Aggregate.COUNT)]))
    assert "COUNT(*)" in compiled.sql
    assert "user_id" not in compiled.sql


def test_distinct_with_metrics_is_refused_rather_than_silently_dropped() -> None:
    """The bug: `select=[department], distinct, count` answered with one
    arbitrary department beside the count of every row."""
    with pytest.raises(ValidationError) as exc:
        QuerySpec(
            select=[Column.DEPARTMENT],
            distinct=True,
            metrics=[Metric(agg=Aggregate.COUNT)],
        )
    message = str(exc.value)
    assert "distinct" in message
    # The refusal has to name the two ways to ask what was meant.
    assert "group_by" in message and "COUNT(DISTINCT" in message


def test_distinct_without_metrics_still_compiles_to_select_distinct() -> None:
    compiled = compile_spec(QuerySpec(select=[Column.DEPARTMENT], distinct=True))
    assert compiled.sql.startswith("SELECT DISTINCT department")

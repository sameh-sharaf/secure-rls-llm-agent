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

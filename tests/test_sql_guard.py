"""Layer 3 tests: the query gateway."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secure_rls.security.spec import (  # noqa: E402
    MIN_COHORT_SIZE,
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

REJECTED = [
    ("base table", "SELECT * FROM employees_base"),
    ("named base table", "SELECT user_id FROM employees_base"),
    ("cte", "WITH x AS (SELECT user_id FROM employees) SELECT user_id FROM x"),
    ("cte impersonation", "WITH employees AS (SELECT 1) SELECT * FROM employees"),
    ("two statements", "SELECT user_id FROM employees; DROP TABLE employees"),
    ("delete", "DELETE FROM employees"),
    ("insert", "INSERT INTO employees (user_id) VALUES (1)"),
    ("update", "UPDATE employees SET salary = 1"),
    ("drop", "DROP TABLE employees"),
    ("create", "CREATE TABLE x (a INT)"),
    ("pragma", "PRAGMA table_info(employees)"),
    ("attach", "ATTACH DATABASE 'x.db' AS x"),
    ("select star", "SELECT * FROM employees"),
    ("unknown column", "SELECT tenant_id FROM employees"),
    ("unknown column 2", "SELECT secret FROM employees"),
    ("schema qualified", "SELECT user_id FROM main.employees"),
    ("banned function", "SELECT load_extension('evil') FROM employees"),
    ("union to base", "SELECT user_id FROM employees UNION SELECT user_id FROM employees_base"),
    ("subquery to base", "SELECT user_id FROM employees WHERE user_id IN (SELECT user_id FROM employees_base)"),
    ("garbage", "not sql at all !!!"),
    ("empty", "   "),
]


@pytest.mark.parametrize("label,sql", REJECTED, ids=[r[0] for r in REJECTED])
def test_rejected(label: str, sql: str) -> None:
    with pytest.raises(SqlRejected):
        guard_sql(sql)


def test_tenant_id_is_not_a_column_the_guard_knows() -> None:
    """There is no tenant predicate to inject because there is no such column."""
    with pytest.raises(SqlRejected, match="unknown column"):
        guard_sql("SELECT tenant_id, COUNT(*) FROM employees GROUP BY tenant_id")


def test_count_star_is_allowed() -> None:
    """COUNT(*) contains a Star node but is not `SELECT *`."""
    result = guard_sql("SELECT COUNT(*) FROM employees")
    assert "COUNT(*)" in result.sql.upper().replace(" ", "")


def test_qualified_star_is_rejected() -> None:
    with pytest.raises(SqlRejected, match="SELECT \\*"):
        guard_sql("SELECT employees.* FROM employees")


def test_simple_select_passes_and_gets_a_limit() -> None:
    result = guard_sql("SELECT name, department FROM employees")
    assert "LIMIT 200" in result.sql
    assert any("LIMIT" in r for r in result.rewrites)


def test_high_limit_is_lowered() -> None:
    result = guard_sql("SELECT name FROM employees LIMIT 100000")
    assert "LIMIT 200" in result.sql
    assert any("lowered" in r for r in result.rewrites)


def test_grouped_aggregate_is_not_floored_by_default(cohort_floor) -> None:
    """The floor is off by default -- small departments stay in the result.

    It silently dropped groups from every grouped answer, so a department of
    four vanished from a chart and the numbers stopped adding up. That cost is
    paid on every ordinary question; the inference attack it partly addresses
    is scoped to future work. See ENFORCE_MIN_COHORT in spec.py.
    """
    result = guard_sql("SELECT department, AVG(salary) FROM employees GROUP BY department")
    assert "HAVING" not in result.sql.upper()
    assert not any("k-anonymity" in r for r in result.rewrites)


def test_grouped_aggregate_gets_k_anonymity_when_the_floor_is_on(cohort_floor) -> None:
    """The capability is intact and one flag away."""
    with cohort_floor(True):
        result = guard_sql("SELECT department, AVG(salary) FROM employees GROUP BY department")
    assert f"COUNT(*) >= {MIN_COHORT_SIZE}" in result.sql.replace("  ", " ")
    assert any("k-anonymity" in r for r in result.rewrites)


def test_k_anonymity_survives_a_trailing_comment(cohort_floor) -> None:
    """The rewrite is on the AST, so commenting out the tail cannot defeat it.

    This is the regression for the string-concatenation approach: appending
    ` HAVING COUNT(*) >= 5` as text to a statement ending in `--` is a no-op.
    """
    sql = "SELECT department, AVG(salary) FROM employees GROUP BY department --"
    with cohort_floor(True):
        result = guard_sql(sql)
    assert "HAVING" in result.sql.upper()
    assert f"COUNT(*) >= {MIN_COHORT_SIZE}" in result.sql.replace("  ", " ")


def test_existing_having_is_preserved_and_extended(cohort_floor) -> None:
    sql = "SELECT department, AVG(salary) FROM employees GROUP BY department HAVING AVG(salary) > 1"
    with cohort_floor(True):
        result = guard_sql(sql)
    upper = result.sql.upper()
    assert "AVG(SALARY) > 1" in upper.replace("  ", " ")
    assert f"COUNT(*) >= {MIN_COHORT_SIZE}" in result.sql.replace("  ", " ")


# ------------------------------------------------------------ role policy ---

MASKED = frozenset({"salary"})


def test_analyst_cannot_select_individual_salary() -> None:
    with pytest.raises(SqlRejected, match="individual employees"):
        guard_sql("SELECT name, salary FROM employees", masked_columns=MASKED)


def test_analyst_can_aggregate_salary() -> None:
    result = guard_sql(
        "SELECT department, AVG(salary) FROM employees GROUP BY department",
        masked_columns=MASKED,
    )
    assert "AVG" in result.sql.upper()


def test_hr_admin_can_select_individual_salary() -> None:
    result = guard_sql("SELECT name, salary FROM employees", masked_columns=frozenset())
    assert "salary" in result.sql.lower()


# ------------------------------------------------------------------ specs ---

def test_spec_compiles_to_parameterised_sql() -> None:
    spec = QuerySpec(
        select=[Column.NAME, Column.DEPARTMENT],
        filters=[Predicate(column=Column.DEPARTMENT, op=Operator.EQ, value="Engineering")],
        limit=10,
    )
    compiled = compile_spec(spec)
    assert compiled.sql == (
        "SELECT name, department FROM employees WHERE department = ? LIMIT ?"
    )
    assert compiled.params == ["Engineering", 10]


def test_spec_grouped_metric_gets_k_anonymity(cohort_floor) -> None:
    spec = QuerySpec(
        metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
        group_by=[Column.DEPARTMENT],
    )
    with cohort_floor(True):
        compiled = compile_spec(spec)
    assert f"HAVING COUNT(*) >= {MIN_COHORT_SIZE}" in compiled.sql
    assert compiled.k_anonymity_applied


def test_spec_grouped_metric_has_no_floor_by_default() -> None:
    compiled = compile_spec(
        QuerySpec(
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            group_by=[Column.DEPARTMENT],
        )
    )
    assert "HAVING" not in compiled.sql.upper()
    assert not compiled.k_anonymity_applied


# -- distinct ---------------------------------------------------------------
# "What departments are there?" had no representation: a spec needed a select
# or a metric, and group_by alone was rejected -- so the model bolted a COUNT
# onto the question to make the request valid and answered something nobody
# asked. A missing verb in the query language shows up as the model behaving
# oddly, not as an error.


def test_distinct_selects_unique_values() -> None:
    compiled = compile_spec(QuerySpec(select=[Column.DEPARTMENT], distinct=True))
    assert compiled.sql.startswith("SELECT DISTINCT department FROM employees")
    assert "COUNT" not in compiled.sql.upper()


def test_group_by_without_a_metric_means_distinct() -> None:
    """The shape a model naturally reaches for when listing values."""
    compiled = compile_spec(QuerySpec(group_by=[Column.DEPARTMENT]))
    assert compiled.sql.startswith("SELECT DISTINCT department FROM employees")
    assert "GROUP BY" not in compiled.sql.upper()


def test_group_by_with_a_metric_still_aggregates() -> None:
    compiled = compile_spec(
        QuerySpec(
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            group_by=[Column.DEPARTMENT],
        )
    )
    assert "GROUP BY department" in compiled.sql
    assert "DISTINCT" not in compiled.sql.upper()


def test_distinct_is_not_applied_to_a_plain_select() -> None:
    compiled = compile_spec(QuerySpec(select=[Column.NAME]))
    assert compiled.sql.startswith("SELECT name FROM employees")


def test_distinct_returns_real_unique_departments() -> None:
    """End to end: the answer should be the department list, not a count."""
    from secure_rls.security.gateway import QueryGateway
    from secure_rls.security.principal import authenticate

    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    try:
        result = gw.run_spec(QuerySpec(select=[Column.DEPARTMENT], distinct=True))
        names = {r["department"] for r in result.rows}
        assert "Legal" in names  # acme-only department
        assert len(result.rows) == len(names), "duplicates leaked through DISTINCT"
        assert 5 <= len(names) <= 10
    finally:
        gw.close()


def test_spec_rejects_unknown_field() -> None:
    with pytest.raises(Exception):
        QuerySpec(select=[Column.NAME], tenant_id="beta")  # type: ignore[call-arg]


def test_spec_has_no_tenant_field() -> None:
    """The security invariant, asserted against the schema itself."""
    fields = set(QuerySpec.model_fields)
    assert not any("tenant" in f.lower() for f in fields), fields


def test_spec_masked_column_blocked_for_row_level_select() -> None:
    spec = QuerySpec(select=[Column.NAME, Column.SALARY])
    with pytest.raises(SpecError, match="individual employees"):
        compile_spec(spec, masked_columns=MASKED)


def test_spec_in_list_is_bound_not_interpolated() -> None:
    spec = QuerySpec(
        select=[Column.NAME],
        filters=[
            Predicate(
                column=Column.DEPARTMENT,
                op=Operator.IN,
                value=["Engineering", "Sales'; DROP TABLE employees --"],
            )
        ],
    )
    compiled = compile_spec(spec)
    assert "DROP" not in compiled.sql
    assert "IN (?, ?)" in compiled.sql
    assert "Sales'; DROP TABLE employees --" in compiled.params

"""One definition of the row cap, and a default that is not a trap.

The caps existed four times over -- the tool schema said 25, `QuerySpec` said
50, the SQL guard said 200, the database said 500 -- with two different
defaults for the same idea and nothing keeping any of them in step.

The 25 was the one that hurt. Every department in every tenant is larger than
that, so "give me a report on the Sales team" returned a third of the team, the
truncation notice fired correctly, and the model wrote its report from the
truncated rows anyway. Found in a real transcript, not by a test, because every
eval case asks a question whose answer fits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import ALLOWED_TENANTS, MAX_ROWS, TenantDatabase  # noqa: E402
from secure_rls.security.spec import (  # noqa: E402
    DEFAULT_ROW_LIMIT,
    MAX_ROW_LIMIT,
    Aggregate,
    Column,
    Metric,
    QuerySpec,
    compile_spec,
)
from secure_rls.security.sql_guard import MAX_LIMIT, guard_sql  # noqa: E402
from secure_rls.tools.factory import QueryEmployeesArgs  # noqa: E402


def test_every_cap_comes_from_one_definition() -> None:
    field = QueryEmployeesArgs.model_fields["limit"]
    assert field.default == DEFAULT_ROW_LIMIT
    assert QuerySpec.model_fields["limit"].default == DEFAULT_ROW_LIMIT
    assert MAX_LIMIT == MAX_ROW_LIMIT


def test_the_caps_are_ordered_sensibly() -> None:
    """A default above the ceiling, or a ceiling above the fetch cap, is a bug."""
    assert DEFAULT_ROW_LIMIT <= MAX_ROW_LIMIT <= MAX_ROWS


@pytest.mark.parametrize("tenant", sorted(ALLOWED_TENANTS))
def test_the_default_covers_the_largest_department(tenant: str) -> None:
    """The property the old default failed, stated as data rather than a number.

    If a future dataset has a department bigger than the default, this fails --
    which is exactly when someone should be told, because that is the moment
    "report on this team" starts silently truncating again.
    """
    db = TenantDatabase(tenant)
    try:
        rows = db.execute("SELECT department, COUNT(*) AS n FROM employees GROUP BY department")
    finally:
        db.close()
    largest = max(r["n"] for r in rows)
    assert largest <= DEFAULT_ROW_LIMIT, (
        f"{tenant}'s largest department has {largest} people; a listing query at "
        f"the default limit of {DEFAULT_ROW_LIMIT} would be silently truncated"
    )


def test_the_model_may_still_raise_it_to_the_ceiling() -> None:
    compiled = compile_spec(QuerySpec(select=[Column.NAME], limit=MAX_ROW_LIMIT))
    # Bound, not interpolated -- the limit is model-supplied input like any other
    # (invariant 2), so it appears in `params` rather than in the SQL text.
    assert compiled.sql.endswith("LIMIT ?")
    assert MAX_ROW_LIMIT in compiled.params


def test_the_ceiling_still_binds() -> None:
    with pytest.raises(Exception):
        QuerySpec(select=[Column.NAME], limit=MAX_ROW_LIMIT + 1)
    guarded = guard_sql(f"SELECT name FROM employees LIMIT {MAX_ROW_LIMIT + 500}")
    assert f"LIMIT {MAX_ROW_LIMIT}" in guarded.sql


def test_aggregates_ignore_the_row_cap() -> None:
    """A LIMIT applies to result rows, so an average still reads the whole tenant.

    Worth pinning: if this were ever false, every statistic in the app would be
    computed over an arbitrary slice and would still look perfectly plausible.
    """
    db = TenantDatabase("acme")
    try:
        spec = QuerySpec(metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)], limit=1)
        compiled = compile_spec(spec)
        limited = db.execute(compiled.sql, compiled.params)[0]
        whole = db.execute("SELECT AVG(salary) AS a FROM employees")[0]["a"]
    finally:
        db.close()
    assert abs(next(iter(limited.values())) - whole) < 1e-9

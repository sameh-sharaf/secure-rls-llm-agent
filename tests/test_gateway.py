"""Gateway tests: layers 1, 3 and 5 acting together."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secure_rls.security.gateway import CohortTooSmall, QueryGateway  # noqa: E402
from secure_rls.security.output_guard import LeakDetected, OutputGuard  # noqa: E402
from secure_rls.security.principal import (  # noqa: E402
    AuthenticationError,
    Role,
    authenticate,
    demo_accounts,
)
from secure_rls.security.spec import (  # noqa: E402
    Aggregate,
    Column,
    Metric,
    Operator,
    Predicate,
    QuerySpec,
)
from secure_rls.security.sql_guard import SqlRejected  # noqa: E402


@pytest.fixture
def acme_admin() -> QueryGateway:
    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    yield gw
    gw.close()


@pytest.fixture
def acme_analyst() -> QueryGateway:
    gw = QueryGateway(authenticate("acme_analyst", "acme123"))
    yield gw
    gw.close()


# ------------------------------------------------------------------ auth ---

def test_valid_login() -> None:
    p = authenticate("beta_admin", "beta123")
    assert p.tenant_id == "beta"
    assert p.role is Role.HR_ADMIN


@pytest.mark.parametrize(
    "user,password",
    [("acme_admin", "wrong"), ("nobody", "acme123"), ("", ""), ("acme_admin", "")],
)
def test_invalid_login(user: str, password: str) -> None:
    with pytest.raises(AuthenticationError):
        authenticate(user, password)


def test_every_demo_account_authenticates() -> None:
    for username, tenant, _ in demo_accounts():
        p = authenticate(username, f"{tenant}123")
        assert p.tenant_id == tenant


def test_cache_keys_are_tenant_scoped() -> None:
    a = authenticate("acme_analyst", "acme123")
    b = authenticate("beta_analyst", "beta123")
    assert a.cache_key("avg salary") != b.cache_key("avg salary")


# ---------------------------------------------------------------- reads ----

def test_spec_query_returns_only_tenant_rows(acme_admin: QueryGateway) -> None:
    result = acme_admin.run_spec(
        QuerySpec(select=[Column.NAME, Column.DEPARTMENT, Column.SALARY], limit=200)
    )
    assert result.row_count == 200
    assert result.verdict.ok


def test_total_rows_matches_tenant_size() -> None:
    sizes = {"acme": 500, "beta": 300, "gamma": 200}
    for tenant, expected in sizes.items():
        gw = QueryGateway(authenticate(f"{tenant}_admin", f"{tenant}123"))
        try:
            assert gw.total_rows() == expected
        finally:
            gw.close()


def test_grouped_aggregate_applies_k_anonymity(acme_admin: QueryGateway) -> None:
    result = acme_admin.run_spec(
        QuerySpec(
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            group_by=[Column.DEPARTMENT],
        )
    )
    assert result.rewrites
    assert all(r["avg_salary"] is not None for r in result.rows)


def test_single_person_aggregate_is_refused(acme_admin: QueryGateway) -> None:
    """The differencing attack: an average over one person is that person's pay."""
    with pytest.raises(CohortTooSmall):
        acme_admin.run_spec(
            QuerySpec(
                metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
                filters=[
                    Predicate(column=Column.NAME, op=Operator.EQ, value="ZZ_CANARY_ACME")
                ],
            )
        )


def test_sql_path_rejects_base_table(acme_admin: QueryGateway) -> None:
    with pytest.raises(SqlRejected):
        acme_admin.run_sql("SELECT user_id, name FROM employees_base")


def test_sql_path_executes_valid_query(acme_admin: QueryGateway) -> None:
    result = acme_admin.run_sql(
        "SELECT department, COUNT(*) AS n FROM employees GROUP BY department"
    )
    assert result.row_count > 0
    assert "COUNT(*) >= 5" in result.sql.replace("  ", " ")


# ----------------------------------------------------------- role policy ---

def test_analyst_cannot_read_individual_salary(acme_analyst: QueryGateway) -> None:
    with pytest.raises(Exception, match="individual employees"):
        acme_analyst.run_spec(QuerySpec(select=[Column.NAME, Column.SALARY]))


def test_analyst_can_aggregate_salary(acme_analyst: QueryGateway) -> None:
    result = acme_analyst.run_spec(
        QuerySpec(
            metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
            group_by=[Column.DEPARTMENT],
        )
    )
    assert result.row_count > 0


# -- extremal aggregates ----------------------------------------------------
# Found by the model bake-off, not by design review: an analyst barred from
# reading individual salaries asked "who is the highest paid person", and
# qwen2.5 answered "999,999 EUR" via MAX(). The tenant boundary held; the role
# boundary did not, and the leak-rate metric was blind to it because it only
# measures cross-tenant disclosure.


@pytest.mark.parametrize("agg", ["max", "min"])
def test_analyst_cannot_reach_an_individual_salary_through_min_or_max(
    acme_analyst: QueryGateway, agg: str
) -> None:
    from secure_rls.security.spec import Aggregate, Metric

    with pytest.raises(Exception, match="one specific person"):
        acme_analyst.run_spec(
            QuerySpec(metrics=[Metric(agg=Aggregate(agg), column=Column.SALARY)])
        )


@pytest.mark.parametrize("agg", ["MAX", "MIN"])
def test_min_max_blocked_on_the_sql_path_too(acme_analyst: QueryGateway, agg: str) -> None:
    with pytest.raises(SqlRejected, match="one specific person"):
        acme_analyst.run_sql(f"SELECT {agg}(salary) AS m FROM employees")


def test_analyst_can_still_use_combining_aggregates(acme_analyst: QueryGateway) -> None:
    """AVG, SUM, COUNT and MEDIAN combine many values; MIN and MAX select one."""
    from secure_rls.security.spec import Aggregate, Metric

    for agg in (Aggregate.AVG, Aggregate.SUM, Aggregate.COUNT, Aggregate.MEDIAN):
        result = acme_analyst.run_spec(
            QuerySpec(metrics=[Metric(agg=agg, column=Column.SALARY)])
        )
        assert result.rows


def test_admin_may_still_use_min_and_max(acme_admin: QueryGateway) -> None:
    """The rule is a role policy, not a blanket ban on extremal aggregates."""
    from secure_rls.security.spec import Aggregate, Metric

    result = acme_admin.run_spec(
        QuerySpec(metrics=[Metric(agg=Aggregate.MAX, column=Column.SALARY)])
    )
    assert result.rows[0]["max_salary"] == 999999


def test_unmasked_columns_are_unaffected(acme_analyst: QueryGateway) -> None:
    """Only masked columns get the extremal restriction."""
    from secure_rls.security.spec import Aggregate, Metric

    result = acme_analyst.run_spec(
        QuerySpec(metrics=[Metric(agg=Aggregate.MAX, column=Column.PERFORMANCE_SCORE)])
    )
    assert result.rows


def test_admin_can_read_individual_salary(acme_admin: QueryGateway) -> None:
    result = acme_admin.run_spec(QuerySpec(select=[Column.NAME, Column.SALARY], limit=5))
    assert all("salary" in r for r in result.rows)


# ---------------------------------------------------------------- median ---
# SQLite has no MEDIAN, so the gateway computes it in pandas. Added after the
# correctness suite caught the model flailing: asked for a median it had no way
# to express, it selected raw salaries and gave up.


@pytest.mark.parametrize("tenant", ["acme", "beta", "gamma"])
def test_median_matches_pandas_ground_truth(tenant: str) -> None:
    import pandas as pd

    from db import CSV_PATH
    from secure_rls.security.spec import Aggregate, Metric

    frame = pd.read_csv(CSV_PATH)
    truth = float(frame[frame.tenant_id == tenant].salary.median())

    gw = QueryGateway(authenticate(f"{tenant}_admin", f"{tenant}123"))
    try:
        result = gw.run_spec(
            QuerySpec(metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)])
        )
        assert result.rows[0]["median_salary"] == pytest.approx(truth)
    finally:
        gw.close()


def test_median_reads_the_whole_tenant_not_the_spec_row_cap() -> None:
    """acme has 500 rows; a median over the first 200 is a different number.

    Regression for the first implementation, which reused QuerySpec's 200-row
    validation cap for its internal fetch and silently produced a wrong answer.
    """
    import pandas as pd

    from db import CSV_PATH
    from secure_rls.security.spec import Aggregate, Metric

    frame = pd.read_csv(CSV_PATH)
    acme = frame[frame.tenant_id == "acme"].salary
    assert len(acme) > 200, "fixture no longer exercises the cap"
    truncated = float(acme.head(200).median())
    full = float(acme.median())

    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    try:
        got = gw.run_spec(
            QuerySpec(metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)])
        ).rows[0]["median_salary"]
    finally:
        gw.close()
    assert got == pytest.approx(full)
    if truncated != full:
        assert got != pytest.approx(truncated)


def test_grouped_median_drops_small_cohorts() -> None:
    from secure_rls.security.spec import Aggregate, Metric

    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    try:
        result = gw.run_spec(
            QuerySpec(
                metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)],
                group_by=[Column.DEPARTMENT],
            )
        )
        assert result.rows
        assert any("k-anonymity" in r for r in result.rewrites)
    finally:
        gw.close()


def test_median_over_one_person_is_refused() -> None:
    from secure_rls.security.spec import Aggregate, Metric

    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    try:
        with pytest.raises(CohortTooSmall):
            gw.run_spec(
                QuerySpec(
                    metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)],
                    filters=[
                        Predicate(column=Column.NAME, op=Operator.EQ, value="ZZ_CANARY_ACME")
                    ],
                )
            )
    finally:
        gw.close()


def test_median_is_tenant_scoped() -> None:
    """The median path fetches rows itself -- it must not escape the boundary."""
    import pandas as pd

    from db import CSV_PATH
    from secure_rls.security.spec import Aggregate, Metric

    frame = pd.read_csv(CSV_PATH)
    global_median = float(frame.salary.median())

    gw = QueryGateway(authenticate("gamma_admin", "gamma123"))
    try:
        got = gw.run_spec(
            QuerySpec(metrics=[Metric(agg=Aggregate.MEDIAN, column=Column.SALARY)])
        ).rows[0]["median_salary"]
    finally:
        gw.close()
    gamma_median = float(frame[frame.tenant_id == "gamma"].salary.median())
    assert got == pytest.approx(gamma_median)
    assert got != pytest.approx(global_median)


# ---------------------------------------------------------- output guard ---

def test_output_guard_raises_on_foreign_user_id() -> None:
    guard = OutputGuard(tenant="acme", allowed_user_ids=frozenset({1, 2, 3}))
    with pytest.raises(LeakDetected, match="does not belong"):
        guard.check_rows([{"user_id": 1}, {"user_id": 999}])


def test_output_guard_raises_on_foreign_canary() -> None:
    guard = OutputGuard(tenant="acme", allowed_user_ids=frozenset({1}))
    with pytest.raises(LeakDetected, match="canary"):
        guard.check_rows([{"user_id": 1, "name": "ZZ_CANARY_BETA"}])


def test_output_guard_allows_own_canary() -> None:
    guard = OutputGuard(tenant="acme", allowed_user_ids=frozenset({1}))
    verdict = guard.check_rows([{"user_id": 1, "name": "ZZ_CANARY_ACME"}])
    assert verdict.ok


def test_output_guard_scans_generated_text() -> None:
    guard = OutputGuard(tenant="acme", allowed_user_ids=frozenset({1}))
    with pytest.raises(LeakDetected):
        guard.check_text("The top earner is ZZ_CANARY_GAMMA on 999999.")


def test_redaction() -> None:
    text = "contact bob@example.com or +420 777 123 456 re DE89370400440532013000"
    out = OutputGuard.redact(text)
    assert "bob@example.com" not in out
    assert "DE89370400440532013000" not in out


# ----------------------------------------------------------------- audit ---

def test_audit_chain_is_verifiable(acme_admin: QueryGateway) -> None:
    acme_admin.run_spec(QuerySpec(select=[Column.NAME], limit=3))
    acme_admin.run_spec(QuerySpec(select=[Column.DEPARTMENT], limit=3))
    assert len(acme_admin.audit.entries()) == 2
    assert acme_admin.audit.verify()


def test_audit_records_rejections(acme_admin: QueryGateway) -> None:
    with pytest.raises(SqlRejected):
        acme_admin.run_sql("SELECT user_id FROM employees_base")
    entries = acme_admin.audit.entries()
    assert entries[-1].outcome.startswith("rejected")
    assert acme_admin.audit.verify()


def test_audit_detects_tampering(acme_admin: QueryGateway) -> None:
    from dataclasses import asdict

    from secure_rls.security.audit import AuditEntry

    acme_admin.run_spec(QuerySpec(select=[Column.NAME], limit=3))
    entries = acme_admin.audit._entries
    tampered = AuditEntry(**{**asdict(entries[0]), "rows_returned": 9999})
    entries[0] = tampered
    assert not acme_admin.audit.verify()

"""Layer attribution: which of L1-L5 refused.

Showing the layer turns a generic refusal into a statement about where the
boundary actually sits. It is also a cheap correctness check on the design: if
an attack that should die at the database is being turned away at the query
gateway, the gateway is doing work the boundary was meant to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import SecurityError  # noqa: E402
from secure_rls.security.gateway import CohortTooSmall, QueryGateway  # noqa: E402
from secure_rls.security.layers import Layer, layer_of, tag  # noqa: E402
from secure_rls.security.output_guard import LeakDetected  # noqa: E402
from secure_rls.security.principal import authenticate  # noqa: E402
from secure_rls.security.spec import (  # noqa: E402
    Aggregate,
    Column,
    Metric,
    Operator,
    Predicate,
    QuerySpec,
    SpecError,
)
from secure_rls.security.sql_guard import SqlRejected  # noqa: E402
from secure_rls.tools.factory import refusal_layer  # noqa: E402


@pytest.mark.parametrize(
    "exc,expected",
    [
        (SecurityError("engine refused"), Layer.L4),
        (SqlRejected("unknown table"), Layer.L3),
        (SpecError("bad spec"), Layer.L3),
        (CohortTooSmall("too few"), Layer.L3),
        (LeakDetected("canary"), Layer.L5),
        (ValueError("not a policy refusal"), None),
    ],
)
def test_layer_by_exception_type(exc: Exception, expected) -> None:
    assert layer_of(exc) is expected


def test_explicit_tag_wins_over_the_type_default() -> None:
    """Role policy is decided by L1 even though the check runs inside L3."""
    exc = tag(SpecError("your role may not read salary"), Layer.L1)
    assert layer_of(exc) is Layer.L1


def test_layer_labels_are_human_readable() -> None:
    assert Layer.L4.label == "L4 database boundary"
    assert Layer.L1.label == "L1 identity & role policy"


# ------------------------------------------------- end-to-end attribution ---


def _gateway(user: str) -> QueryGateway:
    tenant = user.split("_")[0]
    return QueryGateway(authenticate(user, f"{tenant}123"))


def test_role_refusal_is_attributed_to_l1() -> None:
    gw = _gateway("acme_analyst")
    try:
        with pytest.raises(SpecError) as caught:
            gw.run_spec(QuerySpec(select=[Column.NAME, Column.SALARY]))
        assert layer_of(caught.value) is Layer.L1
    finally:
        gw.close()


def test_unknown_table_is_attributed_to_l3() -> None:
    gw = _gateway("acme_admin")
    try:
        with pytest.raises(SqlRejected) as caught:
            gw.run_sql("SELECT user_id FROM employees_base")
        assert layer_of(caught.value) is Layer.L3
    finally:
        gw.close()


def test_cohort_refusal_is_attributed_to_l3(cohort_floor) -> None:
    """Only reachable with the k-anonymity floor on; it is off by default."""
    gw = _gateway("acme_admin")
    try:
        with cohort_floor(True), pytest.raises(CohortTooSmall) as caught:
            gw.run_spec(
                QuerySpec(
                    metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
                    filters=[
                        Predicate(column=Column.NAME, op=Operator.EQ, value="ZZ_CANARY_ACME")
                    ],
                )
            )
        assert layer_of(caught.value) is Layer.L3
    finally:
        gw.close()


def test_database_refusal_is_attributed_to_l4() -> None:
    """With the query gateway bypassed, the engine refuses -- and says so."""
    from evals.ablation import _Patches

    patches = _Patches()
    patches.disable_l3()
    try:
        gw = _gateway("acme_admin")
        try:
            with pytest.raises(SecurityError) as caught:
                gw.run_sql("SELECT user_id FROM employees_base LIMIT 5")
            assert layer_of(caught.value) is Layer.L4
        finally:
            gw.close()
    finally:
        patches.restore()


def test_tool_refusal_message_names_the_layer() -> None:
    assert refusal_layer(SecurityError("x")) == "L4 database boundary"
    assert refusal_layer(SqlRejected("x")) == "L3 query gateway"
    assert refusal_layer(tag(SpecError("x"), Layer.L1)) == "L1 identity & role policy"

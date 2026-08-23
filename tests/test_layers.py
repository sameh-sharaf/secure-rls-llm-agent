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
    from secure_rls.security.gateway import QueryGateway
    from secure_rls.security.layers import LayerConfig
    from secure_rls.security.principal import authenticate

    gw = QueryGateway(
        authenticate("acme_admin", "acme123"),
        layers=LayerConfig(l3_query_gateway=False),
    )
    try:
        with pytest.raises(SecurityError) as caught:
            gw.run_sql("SELECT user_id FROM employees_base LIMIT 5")
        assert layer_of(caught.value) is Layer.L4
    finally:
        gw.close()


def test_tool_refusal_message_names_the_layer() -> None:
    assert refusal_layer(SecurityError("x")) == "L4 database boundary"
    assert refusal_layer(SqlRejected("x")) == "L3 query gateway"
    assert refusal_layer(tag(SpecError("x"), Layer.L1)) == "L1 identity & role policy"


# ------------------------------------------------------- LayerConfig itself ---


def test_layers_1_and_2_are_not_switches() -> None:
    """The absence of those fields is a design statement, not an oversight.

    L1 constructs the session and L2 is the shape of the tool schema. Adding a
    switch for either would mean shipping a code path that builds a session
    with no principal, or a tool that takes a tenant argument -- which
    invariant 1 and the pre-commit hook exist to prevent.
    """
    from secure_rls.security.layers import LayerConfig

    fields = set(LayerConfig.__dataclass_fields__)
    assert fields == {"l3_query_gateway", "l4_database_boundary", "l5_output_guard"}
    for forbidden in ("l1", "l2", "identity", "tool_contract", "principal", "tenant"):
        assert not any(forbidden in f for f in fields), forbidden


def test_every_layer_is_on_by_default() -> None:
    """Fail closed: the default constructor is the shipping configuration."""
    from secure_rls.security.layers import ALL_LAYERS, LayerConfig

    assert LayerConfig().all_on
    assert ALL_LAYERS.all_on
    assert LayerConfig().disabled() == []
    assert LayerConfig().describe() == "all layers active"


def test_a_weakened_gateway_is_recorded_in_its_own_audit_log() -> None:
    """Switching a control off is itself a security event, so it gets an entry."""
    from secure_rls.security.gateway import QueryGateway
    from secure_rls.security.layers import Layer, LayerConfig
    from secure_rls.security.principal import authenticate

    layers = LayerConfig(l3_query_gateway=False, l5_output_guard=False)
    assert layers.disabled() == [Layer.L3, Layer.L5]

    gw = QueryGateway(authenticate("acme_admin", "acme123"), layers=layers)
    try:
        entries = [e for e in gw.audit.entries() if e.tool == "__gateway__"]
        assert len(entries) == 1
        assert "L3" in entries[0].arguments and "L5" in entries[0].arguments
    finally:
        gw.close()


def test_the_default_gateway_records_no_such_event() -> None:
    from secure_rls.security.gateway import QueryGateway
    from secure_rls.security.principal import authenticate

    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    try:
        assert not [e for e in gw.audit.entries() if e.tool == "__gateway__"]
    finally:
        gw.close()

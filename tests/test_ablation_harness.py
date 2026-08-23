"""Tests for the ablation harness itself.

An ablation study that silently measures nothing is worse than no ablation: it
produces a confident green table. That is not hypothetical here. The first run
patched `sql_guard.guard_sql`, but the gateway does
`from ...sql_guard import guard_sql`, which binds the function object into the
gateway's namespace at import time -- so layer 3 was never disabled in any arm,
and every arm reported 0.00% including the one designed to leak.

These tests drive the gateway directly, with no model involved, and assert
which layer stops the attack in each configuration. They run in under a second,
so there is no excuse for not knowing whether the harness works before spending
half an hour of GPU time on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import SecurityError  # noqa: E402
from secure_rls.security.layers import LayerConfig  # noqa: E402
from secure_rls.security.output_guard import LeakDetected  # noqa: E402
from secure_rls.security.principal import authenticate  # noqa: E402
from secure_rls.security.sql_guard import SqlRejected  # noqa: E402

# ORDER BY DESC so the rows are gamma's (user_id 801-1000). Without it the
# first 20 rows of the base table are acme's own, and a genuine cross-tenant
# read looks identical to a legitimate one.
ATTACK = "SELECT user_id, name, salary FROM employees_base ORDER BY user_id DESC LIMIT 20"


_LAYER_FIELD = {
    "l3": "l3_query_gateway",
    "l4": "l4_database_boundary",
    "l5": "l5_output_guard",
}


def _config(*disable: str) -> LayerConfig:
    return LayerConfig(**{_LAYER_FIELD[d]: False for d in disable})


def _run_attack(*disable: str):
    """Run ATTACK with the named layers disabled. Returns rows or raises.

    The weakened stack is built by the constructor. It used to be built by
    patching module globals, which is why this file exists at all -- the
    harness patched a name the gateway never looked up and reported a
    confident 0.00% for every arm, including the one designed to leak.
    """
    from secure_rls.security.gateway import QueryGateway

    gateway = QueryGateway(authenticate("acme_admin", "acme123"), layers=_config(*disable))
    try:
        return gateway.run_sql(ATTACK).rows
    finally:
        gateway.close()


def test_full_stack_blocks_at_layer_3() -> None:
    with pytest.raises(SqlRejected):
        _run_attack()


# The base table is denied twice over, and which denial fires first changed
# when the source database stopped being `main` (ADR-0006): the parser now
# reports "no such table" before the authorizer is ever consulted. Both are L4
# refusing, so the test asserts the refusal rather than the wording -- pinning
# one message would have made a strengthened boundary look like a regression.
_L4_DENIED = "no such table|prohibited|not authorized"


def test_layer_4_alone_holds() -> None:
    """The claim that matters: with the query gateway gone, the engine refuses."""
    with pytest.raises(SecurityError, match=_L4_DENIED):
        _run_attack("l3")


def test_layer_3_alone_holds() -> None:
    """L3 and L4 are independently sufficient against generated SQL."""
    with pytest.raises(SqlRejected):
        _run_attack("l4")


def test_layer_5_backstops_when_both_enforcement_layers_are_gone() -> None:
    with pytest.raises(LeakDetected, match="does not belong"):
        _run_attack("l3", "l4")


def test_the_naive_build_leaks() -> None:
    """Without L3, L4 or L5 -- an app-code WHERE and a model writing SQL.

    If this test ever passes without leaking, the ablation harness has stopped
    disabling anything and its results are meaningless.
    """
    rows = _run_attack("l3", "l4", "l5")
    ids = {int(r["user_id"]) for r in rows if r.get("user_id") is not None}
    foreign = {i for i in ids if i > 500}  # acme owns 1-500
    assert foreign, "the naive arm did not leak; the harness is not disabling anything"


def test_a_weakened_gateway_does_not_weaken_a_live_one() -> None:
    """The property that made configuration worth the refactor.

    Streamlit serves every signed-in browser from one process. While the
    layers were disabled by patching `OutputGuard.check_rows` on the *class*
    and `db.tenant_connection` in the module, running one experiment removed
    those controls for every other session in the process, and an exception
    between patch and restore left them off with nothing saying so.

    Built by constructor argument, a weakened gateway is just another object.
    This test holds one open and asserts a normally-constructed gateway is
    still fully armed -- at the same time, in the same process.
    """
    from secure_rls.security.gateway import QueryGateway

    weak = QueryGateway(authenticate("acme_admin", "acme123"), layers=_config("l3", "l4", "l5"))
    strong = QueryGateway(authenticate("acme_admin", "acme123"))
    try:
        # The weak one leaks, as its arm is designed to.
        leaked = {int(r["user_id"]) for r in weak.run_sql(ATTACK).rows}
        assert {i for i in leaked if i > 500}

        # The strong one, alive at the same moment, refuses at layer 3...
        with pytest.raises(SqlRejected):
            strong.run_sql(ATTACK)
        # ...and its output guard is still the real one, not a stub.
        with pytest.raises(LeakDetected):
            strong.verify_rows([{"user_id": 999, "name": "ZZ_CANARY_GAMMA"}])
    finally:
        weak.close()
        strong.close()


def test_every_arm_starts_from_a_clean_stack() -> None:
    """Arms must not contaminate each other, whatever the mechanism."""
    with pytest.raises(LeakDetected):
        _run_attack("l3", "l4")
    with pytest.raises(SqlRejected):
        _run_attack()          # L3 still there
    with pytest.raises(SecurityError, match=_L4_DENIED):
        _run_attack("l3")      # L4 still there
    with pytest.raises(LeakDetected, match="does not belong"):
        _run_attack("l3", "l4")  # L5 still there

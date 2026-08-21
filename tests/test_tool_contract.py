"""Layer 2 tests: the tool contract.

The invariant these protect is the one the whole design rests on -- the model
has no vocabulary for choosing a tenant. Refactors are where invariants die, so
this is asserted mechanically against the generated schemas rather than trusted
to review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secure_rls.security.gateway import QueryGateway  # noqa: E402
from secure_rls.security.principal import authenticate  # noqa: E402
from secure_rls.tools.factory import ToolContext, build_tools, tool_schemas  # noqa: E402

#: Anything that smells like a way to name an organisation.
FORBIDDEN_KEYS = ("tenant", "org", "organisation", "organization", "company", "customer", "client")


@pytest.fixture
def context() -> ToolContext:
    gw = QueryGateway(authenticate("acme_admin", "acme123"))
    ctx = ToolContext(gateway=gw)
    yield ctx
    gw.close()


def _walk_keys(node, found: list[str], path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_keys(value, found, f"{path}.{key}")
            if isinstance(key, str):
                found.append(f"{path}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk_keys(item, found, f"{path}[{i}]")


def test_no_tool_exposes_a_tenant_parameter(context: ToolContext) -> None:
    """THE invariant. If this fails, the architecture is gone."""
    tools = build_tools(context)
    assert tools, "no tools were built"

    for tool in tools:
        schema = tool.args_schema.model_json_schema()
        properties = schema.get("properties", {})
        for name in properties:
            lowered = name.lower()
            for forbidden in FORBIDDEN_KEYS:
                assert forbidden not in lowered, (
                    f"tool {tool.name!r} exposes parameter {name!r}. "
                    f"Tenant identity must be bound at construction time, never passed "
                    f"by the model. See secure_rls/tools/factory.py."
                )


def test_no_tenant_word_anywhere_in_the_serialised_schemas(context: ToolContext) -> None:
    """Belt and braces: not in a property name, an enum, a default or a title."""
    blob = tool_schemas(build_tools(context)).lower()
    for forbidden in ("tenant", "acme", "beta", "gamma"):
        assert forbidden not in blob, f"{forbidden!r} leaked into the tool schemas"


def test_schemas_forbid_extra_fields(context: ToolContext) -> None:
    """An invented field must be a validation error, not a silently ignored key."""
    for tool in build_tools(context):
        schema = tool.args_schema.model_json_schema()
        assert schema.get("additionalProperties") is False, (
            f"tool {tool.name!r} accepts extra fields; set model_config extra='forbid'"
        )


def test_invented_tenant_argument_is_rejected(context: ToolContext) -> None:
    """The concrete attack: the model emits {'tenant_id': 'beta'}."""
    tools = {t.name: t for t in build_tools(context)}
    with pytest.raises(Exception):
        tools["query_employees"].args_schema(select=["name"], tenant_id="beta")


def test_tools_read_only_their_own_tenant(context: ToolContext) -> None:
    tools = {t.name: t for t in build_tools(context)}
    out = tools["query_employees"].invoke({"select": ["name"], "limit": 200})
    assert "ZZ_CANARY_BETA" not in out
    assert "ZZ_CANARY_GAMMA" not in out


def test_two_sessions_get_independent_tool_sets() -> None:
    """Tools are minted per session; they are not shared or cached across tenants."""
    a_gw = QueryGateway(authenticate("acme_admin", "acme123"))
    b_gw = QueryGateway(authenticate("beta_admin", "beta123"))
    try:
        a_tools = {t.name: t for t in build_tools(ToolContext(gateway=a_gw))}
        b_tools = {t.name: t for t in build_tools(ToolContext(gateway=b_gw))}
        a_out = a_tools["query_employees"].invoke(
            {"select": ["name"], "filters": [{"column": "name", "op": "like", "value": "ZZ%"}]}
        )
        b_out = b_tools["query_employees"].invoke(
            {"select": ["name"], "filters": [{"column": "name", "op": "like", "value": "ZZ%"}]}
        )
        assert "ZZ_CANARY_ACME" in a_out and "ZZ_CANARY_BETA" not in a_out
        assert "ZZ_CANARY_BETA" in b_out and "ZZ_CANARY_ACME" not in b_out
    finally:
        a_gw.close()
        b_gw.close()


def test_sql_tool_returns_actionable_refusal(context: ToolContext) -> None:
    """A refusal the model can act on beats an opaque error."""
    tools = {t.name: t for t in build_tools(context)}
    out = tools["run_sql"].invoke({"sql": "SELECT user_id FROM employees_base"})
    assert "REFUSED" in out
    assert "employees" in out


def test_schemas_are_serialisable_for_the_demo(context: ToolContext) -> None:
    blob = tool_schemas(build_tools(context))
    parsed = json.loads(blob)
    assert set(parsed) == {
        "query_employees",
        "run_sql",
        "plot_chart",
        "detect_anomalies",
        "search_notes",
    }

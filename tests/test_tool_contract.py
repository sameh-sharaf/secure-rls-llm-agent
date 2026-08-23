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

from pydantic import ValidationError  # noqa: E402

from secure_rls.security.gateway import QueryGateway  # noqa: E402
from secure_rls.security.principal import authenticate  # noqa: E402
from secure_rls.tools.factory import (  # noqa: E402
    PlotArgs,
    QueryEmployeesArgs,
    ToolContext,
    build_tools,
    tool_schemas,
)

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


# ---------------------------------------------------------- custom charts ---
# The five presets could not express "compare average salary to headcount by
# department" -- two measures on scales three orders of magnitude apart, which
# needs a secondary axis. Rather than add a sixth preset, the model now
# describes the chart declaratively and the server compiles it. Generated
# plotting code stays out of the loop: that would be a sandbox problem and a
# route around QueryGateway, since such code could read whatever it liked.


def test_custom_chart_combines_two_series_on_two_axes(context: ToolContext) -> None:
    tools = {t.name: t for t in build_tools(context)}
    out = tools["plot_chart"].invoke(
        {
            "x": "department",
            "series": [
                {"metric": "avg", "column": "salary", "mark": "bar", "axis": "left"},
                {"metric": "count", "mark": "line", "axis": "right"},
            ],
        }
    )
    assert "REFUSED" not in out
    figure = context.artifacts[-1].payload
    assert [t.type for t in figure.data] == ["bar", "scatter"]
    assert "yaxis2" in figure.layout, "second series was not put on a secondary axis"


def test_custom_chart_is_one_query(context: ToolContext) -> None:
    """Series share a query; a chart is not N round trips."""
    tools = {t.name: t for t in build_tools(context)}
    tools["plot_chart"].invoke(
        {"x": "department", "series": [{"metric": "avg"}, {"metric": "count"}]}
    )
    sql = context.artifacts[-1].sql
    assert sql.count("SELECT") == 1, "series should share a single grouped query"
    assert "GROUP BY department" in sql


def test_custom_chart_obeys_the_role_policy() -> None:
    """A chart is not a way around the column policy."""
    gw = QueryGateway(authenticate("acme_analyst", "acme123"))
    ctx = ToolContext(gateway=gw)
    try:
        tools = {t.name: t for t in build_tools(ctx)}
        blocked = tools["plot_chart"].invoke(
            {"x": "department", "series": [{"metric": "max", "column": "salary"}]}
        )
        assert "REFUSED" in blocked and "one specific person" in blocked
        allowed = tools["plot_chart"].invoke(
            {"x": "department", "series": [{"metric": "avg", "column": "salary"}]}
        )
        assert "REFUSED" not in allowed
    finally:
        gw.close()


def test_custom_chart_needs_a_dimension(context: ToolContext) -> None:
    tools = {t.name: t for t in build_tools(context)}
    out = tools["plot_chart"].invoke({"series": [{"metric": "count"}]})
    assert "REFUSED" in out and "`x`" in out


def test_custom_chart_caps_series_count(context: ToolContext) -> None:
    tools = {t.name: t for t in build_tools(context)}
    out = tools["plot_chart"].invoke(
        {"x": "department", "series": [{"metric": "count"}] * 4}
    )
    assert "at most 3 series" in out


def test_presets_still_work(context: ToolContext) -> None:
    tools = {t.name: t for t in build_tools(context)}
    for preset in ("salary_by_department", "headcount_by_department", "salary_distribution"):
        out = tools["plot_chart"].invoke({"chart": preset})
        assert "REFUSED" not in out, preset


def test_chart_schema_still_has_no_tenant_parameter(context: ToolContext) -> None:
    """The new fields must not have widened the contract."""
    schema = {t.name: t for t in build_tools(context)}["plot_chart"].args_schema
    blob = json.dumps(schema.model_json_schema()).lower()
    for forbidden in ("tenant", "acme", "beta", "gamma", "employees_base"):
        assert forbidden not in blob


# ----------------------------------------------- a bare value where a list goes ---
# Reported from the running app: "and how many employees in operations?" put a
# Pydantic error in front of the user beside the correct answer, because
# qwen2.5 sent `filters` as one object rather than a list of one.

@pytest.mark.parametrize(
    "kwargs, field, expected_len",
    [
        ({"filters": {"column": "department", "op": "=", "value": "Operations"}}, "filters", 1),
        ({"select": "department"}, "select", 1),
        ({"metrics": "count"}, "metrics", 1),
        ({"group_by": "department"}, "group_by", 1),
        ({"select": ["name", "salary"]}, "select", 2),          # lists still work
        ({"filters": []}, "filters", 0),                        # empty stays empty
    ],
    ids=["filters_object", "select_str", "metrics_str", "group_by_str", "list", "empty"],
)
def test_a_single_value_is_accepted_where_a_list_is_expected(
    kwargs: dict, field: str, expected_len: int
) -> None:
    assert len(getattr(QueryEmployeesArgs(**kwargs), field)) == expected_len


def test_coercing_the_shape_does_not_widen_the_contract() -> None:
    """The shape is forgiving; the vocabulary is not.

    This is the test that makes the coercion safe to keep. Wrapping a bare
    value in a list must not become a way to smuggle a field, a column or a
    tenant past the schema -- every element still goes through `FilterArg` and
    the `Column` enum with `extra="forbid"`.
    """
    rejected = [
        {"tenant_id": "beta"},
        {"filters": {"column": "tenant_id", "op": "=", "value": "beta"}},
        {"filters": {"column": "department", "op": "=", "value": "X", "tenant": "beta"}},
        {"select": "tenant_id"},
        {"select": "employees_base"},
        {"metrics": "exfiltrate"},
    ]
    for kwargs in rejected:
        with pytest.raises(ValidationError):
            QueryEmployeesArgs(**kwargs)


def test_a_bare_series_object_is_accepted_by_the_chart_tool() -> None:
    args = PlotArgs(x="department", series={"metric": "avg", "column": "salary"})
    assert len(args.series) == 1
    assert args.series[0].metric == "avg"


# ------------------------------------------------------- the layer trace ---
# Observability for the Security tab: what each layer received and produced.
# Nothing here is consulted by a control, so these tests pin that it stays
# accurate rather than that it stays present.


def test_the_layer_trace_records_a_completed_call(context: ToolContext) -> None:
    tools = {t.name: t for t in build_tools(context)}
    context.reset()
    tools["query_employees"].invoke({"select": ["name"], "limit": 2})

    assert len(context.layer_traces) == 1
    t = context.layer_traces[0]
    assert t["refused_by"] is None
    assert "FROM employees" in t["l3_sql"]
    assert t["l4_rows"] == 2
    assert t["l5_verdict"]


def test_the_layer_trace_stops_at_the_refusing_layer(context: ToolContext) -> None:
    """A refused call must not carry SQL or rows it never produced."""
    tools = {t.name: t for t in build_tools(context)}
    context.reset()
    tools["run_sql"].invoke({"sql": "SELECT * FROM employees_base"})

    t = context.layer_traces[0]
    assert t["refused_by"] == "L3 query gateway"
    assert "l3_sql" not in t and "l4_rows" not in t


def test_the_trace_never_carries_a_tenant(context: ToolContext) -> None:
    """It is rendered in the UI, so it is an output like any other."""
    tools = {t.name: t for t in build_tools(context)}
    context.reset()
    tools["query_employees"].invoke({"select": ["name"], "limit": 2})
    blob = json.dumps(context.layer_traces, default=str).lower()
    for forbidden in ("tenant", "employees_base", "beta", "gamma"):
        assert forbidden not in blob, forbidden


def test_reset_clears_the_trace(context: ToolContext) -> None:
    """Traces are per turn; one question must not render another's."""
    tools = {t.name: t for t in build_tools(context)}
    tools["query_employees"].invoke({"select": ["name"], "limit": 1})
    assert context.layer_traces
    context.reset()
    assert context.layer_traces == []
    assert context.raw_args == {}


# --------------------------------------------------------- metric_column ---
# `metric_column` used to default to `salary` unconditionally, so a model that
# named a metric and forgot the column got a correctly-computed answer to a
# question nobody asked. These pin the resolve-or-refuse rule that replaced it.


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # Explicit always wins.
        ({"metrics": ["avg"], "metric_column": "performance_score"}, "performance_score"),
        # Exactly one numeric column in `select`: nothing to guess about.
        ({"metrics": ["max"], "select": ["salary"]}, "salary"),
        ({"metrics": ["avg"], "select": ["performance_score"]}, "performance_score"),
        # Non-numeric columns in `select` are not candidates, so one numeric
        # column beside them is still unambiguous.
        ({"metrics": ["avg"], "select": ["name", "salary"]}, "salary"),
    ],
    ids=["explicit", "select_salary", "select_score", "select_mixed"],
)
def test_metric_column_is_resolved_when_it_is_unambiguous(kwargs: dict, expected: str) -> None:
    assert QueryEmployeesArgs(**kwargs).metric_column.value == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        # The case that motivated this: a measure named, no column, and a
        # grouping column that is not the thing being measured.
        {"metrics": ["avg"], "group_by": ["department"]},
        # Nothing at all to infer from.
        {"metrics": ["sum"]},
        # `select` names a dimension, not a measure.
        {"metrics": ["avg"], "select": ["department"]},
        # Two numeric candidates is ambiguous, not a coin toss.
        {"metrics": ["avg"], "select": ["salary", "performance_score"]},
        # A count alongside a real aggregate does not excuse the aggregate.
        {"metrics": ["count", "avg"], "group_by": ["department"]},
    ],
    ids=["group_by_only", "bare_metric", "dimension_only", "ambiguous", "count_plus_avg"],
)
def test_an_aggregate_without_a_column_is_refused_rather_than_guessed(kwargs: dict) -> None:
    with pytest.raises(ValidationError) as exc:
        QueryEmployeesArgs(**kwargs)
    assert "metric_column" in str(exc.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metrics": ["count"]},
        {"metrics": ["count"], "group_by": ["department"]},
        {"select": ["name"]},
        {},
    ],
    ids=["count", "count_grouped", "no_metrics", "empty"],
)
def test_a_count_needs_no_metric_column(kwargs: dict) -> None:
    """COUNT compiles to COUNT(*), so there is no column to demand."""
    assert QueryEmployeesArgs(**kwargs).metric_column is None


def test_a_refused_aggregate_names_the_columns_it_would_accept() -> None:
    """The message is read by the model, which gets one retry to act on it."""
    with pytest.raises(ValidationError) as exc:
        QueryEmployeesArgs(metrics=["avg"], group_by=["department"])
    # The validator's own message, not Pydantic's rendering of it: the latter
    # appends a dump of the input, which would make any assertion here pass for
    # the wrong reason.
    message = exc.value.errors()[0]["msg"]
    assert "salary" in message and "performance_score" in message
    # The grouping column is not a candidate, and neither is the primary key.
    assert "department" not in message
    assert "user_id" not in message

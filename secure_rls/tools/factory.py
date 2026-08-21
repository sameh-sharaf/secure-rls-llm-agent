"""Layer 2 -- the tool contract.

Every tool is built as a closure over one `QueryGateway`, which was itself
constructed from the session principal. The consequence is the single most
important property in this codebase:

    THERE IS NO `tenant_id` PARAMETER ON ANY TOOL.

The model cannot pass one, cannot forge one, and has no word for one. Tenant
identity is injected by the server between the model and the database. Anything
the model can name, the model can be persuaded to change -- so it is not given
the name.

Two mechanical protections keep this true as the code changes:

  * `tests/test_tool_contract.py` walks every tool's JSON schema and fails if a
    tenant-like key appears anywhere in it;
  * `.claude/hooks/check-invariants.sh` blocks a commit that reintroduces one.

An invariant that lives only in a developer's head is an invariant that dies
during a refactor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from secure_rls.security.gateway import CohortTooSmall, QueryGateway, QueryResult
from secure_rls.security.layers import layer_of
from secure_rls.security.output_guard import LeakDetected
from secure_rls.security.spec import (
    Aggregate,
    Column,
    Metric,
    Operator,
    Predicate,
    QuerySpec,
    SpecError,
)
from secure_rls.security.sql_guard import SqlRejected

MAX_PREVIEW_ROWS = 12


def _go():
    """plotly.graph_objects, imported lazily to keep tool import cheap."""
    import plotly.graph_objects as go

    return go


@dataclass
class Artifact:
    """Something to render in the UI alongside the answer."""

    kind: Literal["table", "chart", "notes"]
    title: str
    payload: Any
    sql: str | None = None
    rewrites: list[str] = field(default_factory=list)


@dataclass
class ToolContext:
    """Side channel between tools and the UI.

    Tools return short text to the model -- a model does not need 200 rows to
    answer "which department pays most" -- while the full result is kept here
    for rendering. Keeping these separate also keeps token cost proportional to
    the question rather than to the dataset.
    """

    gateway: QueryGateway
    retriever: Any | None = None  # TenantNotesRetriever, bound to the same principal
    artifacts: list[Artifact] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.artifacts.clear()
        self.rejections.clear()


# ---------------------------------------------------------------- schemas ---
# Every schema sets extra="forbid": a field the model invents is a validation
# error, not a silently ignored key.


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterArg(_Base):
    column: Column = Field(description="Column to filter on")
    op: Operator = Field(description="Comparison operator")
    value: str | float | int | list[str | float | int] | None = Field(
        default=None, description="Value to compare against; omit for is_null/not_null"
    )


# NOTE (developers): there is deliberately no organisation-selecting field on
# this schema, and there must never be one -- see the module docstring. This is
# a comment rather than a docstring on purpose: a class docstring becomes the
# schema `description` and is sent to the model, and pointing the model at the
# thing it cannot do is an invitation to probe it.
class QueryEmployeesArgs(_Base):
    """Read employee records for your organisation."""

    select: list[Column] = Field(
        default_factory=list, description="Columns to return for individual employees"
    )
    distinct: bool = Field(
        default=False,
        description=(
            "Return only unique values of the selected columns. Use this for "
            "'what departments are there', 'which job titles exist' and similar -- "
            "do not add a count unless a count was asked for."
        ),
    )
    metrics: list[Literal["count", "avg", "sum", "min", "max", "median", "p75", "p90"]] = Field(
        default_factory=list,
        description=(
            "Aggregates to compute. Use p90 for 'the highest salary' when your role "
            "may not read an individual one -- it describes the top of the range "
            "without being any one person's figure."
        ),
    )
    metric_column: Column = Field(
        default=Column.SALARY, description="Column the aggregates apply to"
    )
    filters: list[FilterArg] = Field(default_factory=list, description="Conditions to apply")
    group_by: list[Column] = Field(default_factory=list, description="Columns to group by")
    order_by: Column | None = Field(default=None, description="Column to sort by")
    descending: bool = Field(default=True, description="Sort descending")
    limit: int = Field(default=25, ge=1, le=200, description="Maximum rows to return")


class RunSqlArgs(_Base):
    sql: str = Field(
        description=(
            "A single read-only SELECT over the table `employees`. "
            "No CTEs, no SELECT *, no other tables."
        )
    )


class Series(_Base):
    """One measure drawn on a chart.

    A typed description of *what to draw*, not code that draws it. The model
    fills this in; the server compiles it to a query and a figure. That keeps
    generated plotting code out of the loop entirely -- a chart tool that
    executed model-written code would be both a sandbox problem and a way
    around QueryGateway, since such code could read whatever it liked.
    """

    metric: Literal["count", "avg", "sum", "min", "max", "median", "p75", "p90"] = Field(
        description="How to aggregate"
    )
    column: Column = Field(
        default=Column.SALARY, description="Column to aggregate; ignored for count"
    )
    mark: Literal["bar", "line"] = Field(default="bar", description="Draw as bars or a line")
    axis: Literal["left", "right"] = Field(
        default="left",
        description=(
            "Which y-axis. Put measures with very different ranges on opposite axes -- "
            "for example average salary on the left and headcount on the right."
        ),
    )
    label: str | None = Field(default=None, description="Optional legend label")


class PlotArgs(_Base):
    """Build a chart from `x` + `series`, or name one of the ready-made charts."""

    # `x` and `series` come first deliberately. With the preset listed first, a
    # model asked for a combined bar-and-line chart reached for that slot and
    # invented `chart="bar_line"` -- a literal that does not exist -- rather
    # than composing the chart it was asked for. Field order is part of the
    # description the model reads, so the general path leads.
    x: Column | None = Field(
        default=None,
        description=(
            "Dimension for the horizontal axis, e.g. department. Use this with "
            "`series` to build any chart."
        ),
    )
    series: list[Series] = Field(
        default_factory=list,
        description=(
            "Measures to draw against `x`. Use two or more to compare quantities on "
            "different scales -- e.g. average salary as a bar on the left axis and "
            "headcount as a line on the right. This is how you draw a combined "
            "bar-and-line chart."
        ),
    )
    chart: (
        Literal[
            "salary_by_department",
            "salary_distribution",
            "headcount_by_department",
            "performance_vs_salary",
            "hires_per_year",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Optional shortcut for one of these five exact charts. Only these values "
            "are valid -- do not invent another. For anything else, use `x` + `series`."
        ),
    )
    title: str | None = Field(default=None, description="Optional chart title")


class AnomalyArgs(_Base):
    column: Literal["salary", "performance_score"] = Field(
        default="salary", description="Column to search for outliers"
    )
    sensitivity: float = Field(
        default=1.5, ge=1.0, le=3.0, description="IQR multiplier; higher finds fewer outliers"
    )


class SearchNotesArgs(_Base):
    query: str = Field(description="What to look for in the free-text HR notes")
    top_k: int = Field(default=5, ge=1, le=10, description="How many notes to return")


# ------------------------------------------------------------------ utils ---


def _frame(result: QueryResult) -> pd.DataFrame:
    return pd.DataFrame(result.rows)


def _limit_of(result: QueryResult) -> int | None:
    """The LIMIT the query ran with, if it is visible in the bound params."""
    for param in reversed(result.params or []):
        if isinstance(param, int):
            return param
    return None


def _summarise(result: QueryResult, note: str = "") -> str:
    """Compact, model-facing rendering of a result set."""
    if not result.rows:
        return (
            "No rows matched. Note this means no rows in *your organisation* matched -- "
            "you can only ever see your own organisation's employees."
        )
    frame = _frame(result)

    # A single-row result is almost always one aggregate, and leading with the
    # row count buries the answer behind a number that looks like one. Asked
    # for the Marketing headcount, llama3.1 read "1 row(s) returned" and
    # answered "there is 1 employee in Marketing" -- the real figure, 62, was
    # on the next line. The format of a tool result is part of the interface to
    # the model, and a misleading preamble produces wrong answers from right
    # data. State the value first.
    if result.row_count == 1:
        row = frame.iloc[0].to_dict()
        lines = ["; ".join(f"{key} = {value}" for key, value in row.items())]
        if note:
            lines.append(note)
        if result.rewrites:
            lines.append("Policy applied: " + "; ".join(result.rewrites))
        return "\n".join(lines)

    preview = frame.head(MAX_PREVIEW_ROWS)

    # When the result fills the limit exactly it is almost certainly truncated,
    # and the row count is not a total. qwen2.5 fetched 25 rows -- the default
    # limit -- and answered "there are 25 employees in Marketing"; the real
    # figure is 62. Say so in the result rather than hoping the model infers it.
    truncated = result.row_count >= limit if (limit := _limit_of(result)) else False
    header = (
        f"{result.row_count} row(s) returned -- this is the row limit, so the result is "
        f"cut off and this count is NOT a total. Use metrics=['count'] for a total."
        if truncated
        else f"{result.row_count} row(s) returned."
    )
    lines = [header]
    if note:
        lines.append(note)
    if result.rewrites:
        lines.append("Policy applied: " + "; ".join(result.rewrites))
    lines.append(preview.to_string(index=False))
    if result.row_count > MAX_PREVIEW_ROWS:
        lines.append(f"... {result.row_count - MAX_PREVIEW_ROWS} further rows not shown.")
    return "\n".join(lines)


def refusal_layer(exc: Exception) -> str:
    """Which layer refused, as a short label for the trace and the UI."""
    layer = layer_of(exc)
    return layer.label if layer else "policy"


def _explain_refusal(exc: Exception) -> str:
    """Turn a policy rejection into something the model can act on this turn.

    The layer is named in the message. It costs nothing, and it turns a generic
    "I can't do that" into a statement about where the boundary actually is --
    which is the point of the whole design.
    """
    where = refusal_layer(exc)
    if isinstance(exc, CohortTooSmall):
        return f"REFUSED [{where}] (minimum cohort size): {exc}"
    if isinstance(exc, SqlRejected):
        return f"REFUSED [{where}] (query policy): {exc}. Rewrite within the allowed schema."
    if isinstance(exc, SpecError):
        return f"REFUSED [{where}] (request policy): {exc}"
    if isinstance(exc, LeakDetected):
        return f"BLOCKED [{where}]: {exc}"
    return f"REFUSED [{where}]: {exc}"


# ------------------------------------------------------------------ tools ---


def build_tools(context: ToolContext) -> list[BaseTool]:
    """Mint the tool set for one session.

    `context.gateway` is captured in every closure below. That capture is the
    security boundary's handshake: after this function returns, there is no
    argument any caller -- model or otherwise -- can supply to reach a different
    tenant's data.
    """
    gateway = context.gateway

    # -- structured read ---------------------------------------------------
    def query_employees(**kwargs: Any) -> str:
        args = QueryEmployeesArgs(**kwargs)
        spec = QuerySpec(
            select=args.select,
            distinct=args.distinct,
            metrics=[
                Metric(agg=Aggregate(m), column=args.metric_column) for m in args.metrics
            ],
            filters=[
                Predicate(column=f.column, op=f.op, value=f.value) for f in args.filters
            ],
            group_by=args.group_by,
            order_by=args.order_by,
            descending=args.descending,
            limit=args.limit,
        )
        try:
            result = gateway.run_spec(spec)
        except Exception as exc:
            context.rejections.append(str(exc))
            return _explain_refusal(exc)

        context.artifacts.append(
            Artifact(
                kind="table",
                title="Query result",
                payload=_frame(result),
                sql=result.display_sql(),
                rewrites=result.rewrites,
            )
        )
        return _summarise(result)

    # -- SQL escape hatch --------------------------------------------------
    def run_sql(sql: str) -> str:
        try:
            result = gateway.run_sql(sql)
        except Exception as exc:
            context.rejections.append(str(exc))
            return _explain_refusal(exc)

        context.artifacts.append(
            Artifact(
                kind="table",
                title="SQL result",
                payload=_frame(result),
                sql=result.sql,
                rewrites=result.rewrites,
            )
        )
        return _summarise(result)

    # -- charts ------------------------------------------------------------
    def _plot_custom(x: Column, series: list[Series], title: str | None) -> str:
        """Compile a declarative chart spec into one query and one figure.

        The model says *what* to draw; this decides *how*. Two measures on very
        different scales -- average salary in the hundred-thousands, headcount
        in the tens -- need a secondary axis or the smaller one is a flat line
        on the floor, so `axis` is part of the spec rather than something the
        model has to solve with code.

        All series are fetched in a single grouped query, so k-anonymity, the
        role column policy and the tenant boundary apply exactly as they do to
        a text answer. `min`/`max` on a masked column are refused here for the
        same reason they are refused anywhere else.
        """
        from plotly.subplots import make_subplots

        spec = QuerySpec(
            metrics=[
                Metric(agg=Aggregate(s.metric), column=s.column, alias=f"s{i}")
                for i, s in enumerate(series)
            ],
            group_by=[x],
            limit=50,
        )
        try:
            result = gateway.run_spec(spec, tool="plot")
        except Exception as exc:
            context.rejections.append(str(exc))
            return _explain_refusal(exc)

        frame = _frame(result)
        if frame.empty:
            return "No data to plot for your organisation."

        heading = title or " and ".join(
            s.label or f"{s.metric} {s.column.value}" for s in series
        ) + f" by {x.value}"

        needs_right = any(s.axis == "right" for s in series)
        figure = make_subplots(specs=[[{"secondary_y": needs_right}]])
        for i, s in enumerate(series):
            column = f"s{i}"
            if column not in frame:
                continue
            name = s.label or (
                "headcount" if s.metric == "count" else f"{s.metric} {s.column.value}"
            )
            trace = (
                _go().Bar(x=frame[x.value], y=frame[column], name=name)
                if s.mark == "bar"
                else _go().Scatter(
                    x=frame[x.value], y=frame[column], name=name, mode="lines+markers"
                )
            )
            figure.add_trace(trace, secondary_y=(s.axis == "right"))

        figure.update_layout(title=heading, xaxis_title=x.value, legend_title=None)
        context.artifacts.append(
            Artifact(kind="chart", title=heading, payload=figure, sql=result.display_sql())
        )
        drawn = ", ".join(
            f"{s.label or s.metric} as a {s.mark} on the {s.axis}" for s in series
        )
        return (
            f"Chart '{heading}' rendered from {result.row_count} groups of your "
            f"organisation's data ({drawn})."
        )

    def plot_chart(
        chart: str | None = None,
        x: Column | None = None,
        series: list | None = None,
        title: str | None = None,
    ) -> str:
        import plotly.express as px

        parsed = [s if isinstance(s, Series) else Series(**s) for s in (series or [])]
        if parsed:
            if x is None:
                return (
                    "REFUSED [L2 tool contract] (request policy): a custom chart needs `x`, "
                    "the dimension to plot against -- for example department."
                )
            if len(parsed) > 3:
                return (
                    "REFUSED [L2 tool contract] (request policy): at most 3 series per chart."
                )
            return _plot_custom(Column(x), parsed, title)

        if chart is None:
            return (
                "REFUSED [L2 tool contract] (request policy): choose a preset `chart`, or "
                "give `x` and `series` for a custom chart."
            )

        specs: dict[str, QuerySpec] = {
            "salary_by_department": QuerySpec(
                metrics=[Metric(agg=Aggregate.AVG, column=Column.SALARY)],
                group_by=[Column.DEPARTMENT],
                limit=50,
            ),
            "headcount_by_department": QuerySpec(
                metrics=[Metric(agg=Aggregate.COUNT)],
                group_by=[Column.DEPARTMENT],
                limit=50,
            ),
            "salary_distribution": QuerySpec(
                select=[Column.DEPARTMENT, Column.SALARY], limit=200
            ),
            "performance_vs_salary": QuerySpec(
                select=[Column.PERFORMANCE_SCORE, Column.SALARY, Column.DEPARTMENT], limit=200
            ),
            "hires_per_year": QuerySpec(select=[Column.HIRE_DATE], limit=200),
        }
        try:
            result = gateway.run_spec(specs[chart], tool="plot")
        except Exception as exc:
            context.rejections.append(str(exc))
            return _explain_refusal(exc)

        frame = _frame(result)
        if frame.empty:
            return "No data to plot for your organisation."

        heading = title or chart.replace("_", " ").title()
        if chart == "salary_by_department":
            fig = px.bar(frame, x="department", y="avg_salary", title=heading)
        elif chart == "headcount_by_department":
            fig = px.bar(frame, x="department", y="count_rows", title=heading)
        elif chart == "salary_distribution":
            fig = px.box(frame, x="department", y="salary", title=heading)
        elif chart == "performance_vs_salary":
            fig = px.scatter(
                frame, x="performance_score", y="salary", color="department", title=heading
            )
        else:
            frame["year"] = pd.to_datetime(frame["hire_date"]).dt.year
            counts = frame.groupby("year").size().reset_index(name="hires")
            fig = px.bar(counts, x="year", y="hires", title=heading)

        context.artifacts.append(
            Artifact(kind="chart", title=heading, payload=fig, sql=result.display_sql())
        )
        return f"Chart '{heading}' rendered from {result.row_count} rows of your organisation's data."

    # -- anomalies ---------------------------------------------------------
    def detect_anomalies(column: str = "salary", sensitivity: float = 1.5) -> str:
        col = Column(column)
        spec = QuerySpec(select=[Column.USER_ID, Column.NAME, Column.DEPARTMENT, col], limit=200)
        try:
            result = gateway.run_spec(spec, tool="detect_anomalies")
        except Exception as exc:
            context.rejections.append(str(exc))
            return _explain_refusal(exc)

        frame = _frame(result)
        if frame.empty or column not in frame:
            return "Not enough data to look for outliers."

        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) < 8:
            return "Too few data points to identify outliers reliably."

        # The quartiles are computed on this tenant's rows only. Fitting across
        # all tenants would encode the other tenants' distributions into the
        # scores -- derived data inherits the classification of its inputs.
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - sensitivity * iqr, q3 + sensitivity * iqr
        numeric = pd.to_numeric(frame[column], errors="coerce")
        outliers = frame[(numeric < low) | (numeric > high)].copy()

        if outliers.empty:
            return f"No {column} outliers found (IQR fence {low:,.0f} to {high:,.0f})."

        context.artifacts.append(
            Artifact(
                kind="table",
                title=f"{column} outliers",
                payload=outliers,
                sql=result.display_sql(),
                rewrites=[f"IQR fence fitted on this organisation only: {low:,.0f}-{high:,.0f}"],
            )
        )
        return (
            f"{len(outliers)} outlier(s) in {column}, using an IQR fence of "
            f"{low:,.0f} to {high:,.0f} fitted on your organisation's data:\n"
            f"{outliers.head(MAX_PREVIEW_ROWS).to_string(index=False)}"
        )

    # -- notes retrieval ---------------------------------------------------
    def search_notes(query: str, top_k: int = 5) -> str:
        if context.retriever is None:
            return "Note search is unavailable; the index has not been built."
        try:
            notes = context.retriever.search(query, top_k=top_k)
        except Exception as exc:
            context.rejections.append(str(exc))
            return _explain_refusal(exc)

        if not notes:
            return "No notes in your organisation matched that description."

        context.artifacts.append(
            Artifact(
                kind="notes",
                title=f"Notes matching: {query}",
                payload=notes,
                rewrites=[f"searched the {gateway.principal.tenant_id} note index only"],
            )
        )
        # Retrieved text is untrusted input: it was written by employees, and
        # an employee can write an instruction into a notes field. Delimiting
        # it is a mitigation, not a control -- what makes it safe is that a
        # fully compromised model still holds no tool that can cross a tenant.
        rendered = [f"{n.name} ({n.department}): {n.text}" for n in notes]
        return gateway.wrap_untrusted(rendered)

    tools: list[BaseTool] = [
        StructuredTool.from_function(
            func=query_employees,
            name="query_employees",
            description=(
                "Read employee records for your organisation. Use `select` for individual "
                "rows, `select` with `distinct` for the unique values of a column, or "
                "`metrics` with `group_by` for aggregates. Answer the question asked: "
                "listing what exists is not the same as counting it. Prefer this over raw SQL."
            ),
            args_schema=QueryEmployeesArgs,
        ),
        StructuredTool.from_function(
            func=run_sql,
            name="run_sql",
            description=(
                "Run one read-only SELECT over the `employees` table when the structured "
                "query tool cannot express the question. Returns an explanatory refusal if "
                "the statement is outside policy."
            ),
            args_schema=RunSqlArgs,
        ),
        StructuredTool.from_function(
            func=plot_chart,
            name="plot_chart",
            description=(
                "Draw a chart of your organisation's workforce data. Use a preset "
                "`chart` when one fits, or build a custom chart with `x` (the "
                "dimension) and `series` (one or more measures). Combine series to "
                "compare quantities on different scales -- put average salary as bars "
                "on the left axis and headcount as a line on the right."
            ),
            args_schema=PlotArgs,
        ),
        StructuredTool.from_function(
            func=detect_anomalies,
            name="detect_anomalies",
            description=(
                "Flag statistical outliers in salary or performance, using an IQR fence "
                "fitted on your organisation's data alone."
            ),
            args_schema=AnomalyArgs,
        ),
        StructuredTool.from_function(
            func=search_notes,
            name="search_notes",
            description=(
                "Semantic search over free-text HR notes for your organisation. Use for "
                "questions SQL cannot answer, such as retention risk or promotion signals."
            ),
            args_schema=SearchNotesArgs,
        ),
    ]
    return tools


def tool_schemas(tools: list[BaseTool]) -> str:
    """The JSON schemas the model actually receives.

    Rendered in the UI during the demo: the absence of `tenant_id` is more
    convincing than any paragraph of documentation, and it takes four seconds.
    """
    return json.dumps(
        {t.name: t.args_schema.model_json_schema() for t in tools},
        indent=2,
        default=str,
    )

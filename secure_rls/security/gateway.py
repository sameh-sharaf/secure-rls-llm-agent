"""The query gateway: the single place data is read on behalf of an agent.

One gateway is constructed per session, from the session principal. It owns the
tenant-bound connection, applies the role's column policy, enforces the minimum
cohort size, runs every result past the output guard, and writes the audit
entry. Tools hold a gateway; tools never hold a connection, a tenant string, or
a raw SQL executor.

The constructor is the only place a tenant is chosen. After that there is no
method on this object that accepts a tenant, which is what makes "the model
cannot pick a tenant" a structural fact rather than a policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from db import DB_PATH, MAX_ROWS, SecurityError, TenantDatabase, tenant_user_ids
from secure_rls.security.audit import AuditLog
from secure_rls.security.output_guard import GuardVerdict, OutputGuard
from secure_rls.security.principal import Principal
from secure_rls.security.spec import (
    ENFORCE_MIN_COHORT,
    GATEWAY_COMPUTED,
    MIN_COHORT_SIZE,
    QuerySpec,
    SpecError,
    cohort_size_query,
    compile_spec,
)
from secure_rls.security.sql_guard import GuardResult, SqlRejected, guard_sql


@dataclass
class QueryResult:
    """What a tool gets back. Carries its own provenance for the UI and audit."""

    rows: list[dict]
    sql: str
    params: list = field(default_factory=list)
    rewrites: list[str] = field(default_factory=list)
    verdict: GuardVerdict | None = None
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def display_sql(self) -> str:
        """SQL with parameters substituted, for display only -- never executed."""
        out = self.sql
        for param in self.params:
            literal = f"'{param}'" if isinstance(param, str) else str(param)
            out = out.replace("?", literal, 1)
        return out


class CohortTooSmall(ValueError):
    """An aggregate would have been computed over too few people."""


class QueryGateway:
    """Tenant-bound data access. Constructed once per session."""

    def __init__(
        self,
        principal: Principal,
        *,
        audit: AuditLog | None = None,
        db_path: Path = DB_PATH,
    ) -> None:
        self.principal = principal
        self.audit = audit or AuditLog()
        # The tenant is read from the principal exactly here, and never again.
        self._db = TenantDatabase(principal.tenant_id, db_path)
        self._guard = OutputGuard(
            tenant=principal.tenant_id,
            allowed_user_ids=tenant_user_ids(principal.tenant_id, db_path),
        )

    # -------------------------------------------------------------- reads ---

    def run_spec(self, spec: QuerySpec, *, tool: str = "query_db") -> QueryResult:
        """Structured path: compile a typed spec and execute it."""
        if any(m.agg.value in GATEWAY_COMPUTED for m in spec.metrics):
            return self._run_percentile(spec, tool=tool)
        started = time.perf_counter()
        masked = self.principal.policy.masked_columns()

        try:
            compiled = compile_spec(spec, masked_columns=masked)
            if compiled.aggregate_only and not spec.group_by:
                self._require_cohort(spec)
            rows = self._db.execute(compiled.sql, compiled.params)
        except (SpecError, SecurityError, CohortTooSmall) as exc:
            self._audit(tool, spec.model_dump_json(), None, 0, "n/a", f"rejected: {exc}", started)
            raise

        verdict = self._guard.check_rows(rows)
        self._audit(
            tool, spec.model_dump_json(), compiled.sql, len(rows), verdict.summary, "ok", started
        )
        rewrites = (
            [f"required COUNT(*) >= {MIN_COHORT_SIZE} per group (k-anonymity)"]
            if compiled.k_anonymity_applied
            else []
        )
        return QueryResult(
            rows=rows,
            sql=compiled.sql,
            params=compiled.params,
            rewrites=rewrites,
            verdict=verdict,
        )

    def run_sql(self, sql: str, *, tool: str = "query_sql") -> QueryResult:
        """Escape hatch: validate and rewrite model-written SQL, then execute."""
        started = time.perf_counter()
        masked = self.principal.policy.masked_columns()

        try:
            guarded: GuardResult = guard_sql(sql, masked_columns=masked)
            rows = self._db.execute(guarded.sql, [])
            if guarded.aggregate_only:
                self._require_cohort_for_sql(guarded.sql)
        except (SqlRejected, SecurityError, CohortTooSmall) as exc:
            self._audit(tool, sql, None, 0, "n/a", f"rejected: {exc}", started)
            raise

        verdict = self._guard.check_rows(rows)
        self._audit(tool, sql, guarded.sql, len(rows), verdict.summary, "ok", started)
        return QueryResult(
            rows=rows, sql=guarded.sql, rewrites=guarded.rewrites, verdict=verdict
        )

    def _run_percentile(self, spec: QuerySpec, *, tool: str) -> QueryResult:
        """Median, computed in pandas over rows the boundary already filtered.

        SQLite has no MEDIAN function. Rather than teach the model a SQL trick
        involving ORDER BY and OFFSET -- which would push complexity onto the
        least reliable component -- the gateway fetches the column through the
        ordinary bound path and computes the statistic itself.

        The rows still come from the tenant-bound connection, still pass the
        output guard, and still get audited. The k-anonymity rule is applied to
        each group afterwards, exactly as the HAVING clause would have.
        """
        import pandas as pd

        started = time.perf_counter()
        metric = next(m for m in spec.metrics if m.agg.value in GATEWAY_COMPUTED)
        quantile = GATEWAY_COMPUTED[metric.agg.value]

        # No mask is applied to this fetch: a median is an aggregate, so a
        # masked column is legitimate here for exactly the reason an analyst may
        # ask for an average. The individual values never leave this method.
        fetch = QuerySpec(
            select=[*spec.group_by, metric.column],
            filters=spec.filters,
        )
        try:
            # MAX_ROWS, not the spec's 200-row cap: a median over the first 200
            # of 500 rows is simply a wrong number. Exact up to MAX_ROWS; beyond
            # that a warehouse percentile function is the right answer, not a
            # bigger fetch.
            compiled = compile_spec(fetch, limit_override=MAX_ROWS)
            rows = self._db.execute(compiled.sql, compiled.params)
        except (SpecError, SecurityError) as exc:
            self._audit(tool, spec.model_dump_json(), None, 0, "n/a", f"rejected: {exc}", started)
            raise

        frame = pd.DataFrame(rows)
        column = metric.column.value
        alias = metric.output_name()

        if frame.empty:
            out: list[dict] = []
        elif spec.group_by:
            keys = [c.value for c in spec.group_by]
            grouped = frame.groupby(keys)[column]
            summary = grouped.agg(
                **{alias: lambda g: g.quantile(quantile), "count": "size"}
            ).reset_index()
            if ENFORCE_MIN_COHORT:
                summary = summary[summary["count"] >= MIN_COHORT_SIZE]
            summary = summary.drop(columns=["count"])
            out = summary.to_dict("records")
        else:
            if ENFORCE_MIN_COHORT and len(frame) < MIN_COHORT_SIZE:
                raise CohortTooSmall(
                    f"that statistic covers only {len(frame)} employee(s); at least "
                    f"{MIN_COHORT_SIZE} are required before a statistic can be reported"
                )
            out = [{alias: float(frame[column].quantile(quantile))}]

        verdict = self._guard.check_rows(out)
        self._audit(
            tool, spec.model_dump_json(), compiled.sql, len(out), verdict.summary, "ok", started
        )
        return QueryResult(
            rows=out,
            sql=compiled.sql,
            params=compiled.params,
            rewrites=[
                f"{metric.agg.value} computed by the gateway over {len(frame)} tenant "
                f"rows (SQLite has no percentile functions)",
                *([f"groups smaller than {MIN_COHORT_SIZE} dropped (k-anonymity)"]
                  if ENFORCE_MIN_COHORT else []),
            ],
            verdict=verdict,
        )

    # ------------------------------------------------------------ helpers ---

    def _require_cohort(self, spec: QuerySpec) -> None:
        """Block an ungrouped aggregate computed over fewer than k people."""
        if not ENFORCE_MIN_COHORT:
            return
        sql, params = cohort_size_query(spec)
        rows = self._db.execute(sql, params)
        n = int(rows[0]["n"]) if rows else 0
        if n < MIN_COHORT_SIZE:
            raise CohortTooSmall(
                f"that aggregate covers only {n} employee(s); at least {MIN_COHORT_SIZE} are "
                f"required before a statistic can be reported, because a statistic over a "
                f"handful of people discloses those people"
            )

    def _require_cohort_for_sql(self, guarded_sql: str) -> None:
        """Same rule for the SQL path, applied to the statement's own filter."""
        if not ENFORCE_MIN_COHORT:
            return
        lowered = guarded_sql.lower()
        where = ""
        if " where " in lowered:
            start = lowered.index(" where ") + len(" where ")
            for terminator in (" group by ", " order by ", " limit ", " having "):
                if terminator in lowered[start:]:
                    end = start + lowered[start:].index(terminator)
                    where = guarded_sql[start:end]
                    break
            else:
                where = guarded_sql[start:]
        count_sql = "SELECT COUNT(*) AS n FROM employees"
        if where.strip():
            count_sql += f" WHERE {where.strip()}"
        rows = self._db.execute(count_sql, [])
        n = int(rows[0]["n"]) if rows else 0
        if n < MIN_COHORT_SIZE:
            raise CohortTooSmall(
                f"that aggregate covers only {n} employee(s); at least {MIN_COHORT_SIZE} are "
                f"required before a statistic can be reported"
            )

    def _audit(
        self,
        tool: str,
        arguments: str,
        sql: str | None,
        rows: int,
        verdict: str,
        outcome: str,
        started: float,
    ) -> None:
        self.audit.record(
            principal=self.principal,
            tool=tool,
            arguments=arguments,
            sql=sql,
            rows_returned=rows,
            guard_verdict=verdict,
            outcome=outcome,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # -------------------------------------------------------------- misc ----

    def verify_rows(self, rows: list[dict]) -> GuardVerdict:
        """Re-check an arbitrary result set. Used by the graph's guard node."""
        return self._guard.check_rows(rows)

    def check_answer(self, text: str) -> GuardVerdict:
        """Scan the model's final prose for foreign canaries before display."""
        return self._guard.check_text(text)

    def redact(self, text: str | None) -> str | None:
        return self._guard.redact(text)

    def wrap_untrusted(self, chunks: list[str]) -> str:
        return self._guard.wrap_untrusted(chunks)

    @property
    def allowed_user_ids(self) -> frozenset[int]:
        """The independent id set, for binding into the retriever's post-check."""
        return self._guard.allowed_user_ids

    #: Shown in place of a value the caller's role may not read.
    MASKED_PLACEHOLDER = "<restricted for your role>"

    def sample_rows(self, n: int = 3) -> list[dict]:
        """A few of the tenant's own rows, with the role's column policy applied.

        These rows are injected into the system prompt to ground the model's
        idea of the schema, and they are also rendered in the UI. Both are
        *outputs*, and both were bypassing the column mask.

        The bug this prevents was found by the model bake-off and is worth
        stating plainly: an analyst -- a role explicitly barred from reading an
        individual's salary -- had three real employees' salaries handed to them
        in the system prompt before asking anything. Asked for the highest
        salary, llama3.1 answered a real one. It was not hallucinating; it was
        reciting its own prompt.

        The boundary was enforced carefully on the query path and then leaked
        around through a side channel built alongside it. Masking belongs here,
        in the one method every caller goes through, rather than in each caller.
        """
        masked = self.principal.policy.masked_columns()
        rows = self._db.sample(n)
        if not masked:
            return rows
        return [
            {
                key: (self.MASKED_PLACEHOLDER if key in masked else value)
                for key, value in row.items()
            }
            for row in rows
        ]

    def total_rows(self) -> int:
        return self._db.row_count()

    def close(self) -> None:
        self._db.close()

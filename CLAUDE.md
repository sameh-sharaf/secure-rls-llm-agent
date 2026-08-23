# secure-rls — working notes for coding agents

A conversational data analyst over a multi-tenant HR dataset, where the LLM is
**structurally incapable** of reading another tenant's rows.

Read `docs/architecture.md` for the full picture and `docs/threat-model.md`
before changing anything under `secure_rls/security/`.

## The invariants

These are not style preferences. Breaking one removes the property the whole
project exists to demonstrate.

1. **No tool may take a tenant-selecting parameter.** Not `tenant_id`, not
   `org`, not `company`, not a `table` argument. Tenant identity is captured in
   a closure at tool-construction time, from the session principal. Anything
   the model can name, the model can be persuaded to change.
   *Enforced by:* `tests/test_tool_contract.py`, `.claude/hooks/check-invariants.sh`

2. **Never build SQL by string concatenation.** Structured queries compile from
   enums with bound parameters (`secure_rls/security/spec.py`). Model-written
   SQL is rewritten on the sqlglot AST and re-generated from the tree
   (`sql_guard.py`). A trailing `--` defeats string splicing; it cannot defeat
   an AST rewrite.

3. **All data access goes through `QueryGateway`.** No module outside
   `db.py` and `secure_rls/security/` may open a database connection. If you
   need data in a new tool, take a gateway.

4. **The agent's connection holds the tenant's slice and nothing else.**
   `main` is a private in-memory database; the data file is attached read-only
   only long enough to copy the rows out, then detached. The base table is
   therefore *absent*, not merely denied. The authorizer stays and still denies
   `employees_base` unconditionally -- no `source` value, view name or CTE name
   grants an exception (ADR-0002) -- but it is now defence in depth rather than
   the whole boundary, because it cannot be relied on alone (ADR-0006).

5. **The output guard raises; it never filters.** Silently removing an
   offending row would hide the bug that produced it.

5b. **Every path that shows a value is an output, including the prompt.**
   Sample rows injected into the system prompt, rows rendered in the UI, and
   rows returned by a tool are all subject to the role's column policy. Apply
   the mask in `QueryGateway`, where every caller goes through it, never in the
   caller. A boundary enforced on the query path and bypassed by a side channel
   built alongside it is not a boundary.

5c. **`MIN` and `MAX` are row-level reads wearing an aggregate's clothes.**
   They select one row's value rather than combining many, so on a masked
   column they disclose an individual and no cohort-size rule will catch it.
   See `EXTREMAL_AGGREGATES` in `spec.py`.

5d. **The aggregate exemption applies to `metrics` and to nothing else.**
   A masked column may be combined into a statistic. It may not be projected,
   grouped by, filtered on or ordered by -- a predicate discloses the value one
   comparison at a time, and an ordering discloses who sits at the top of it.
   Enforced for every position by `check_masked_columns` in `spec.py`, which is
   the *only* place the rule is written: it had two implementations that
   disagreed for a while, and `sql_guard` was the one that happened to be right.
   Call it from any new path that accepts a caller's `QuerySpec` -- the
   gateway's percentile branch compiles its own spec and had to be given the
   call explicitly.

5e. **Never substitute a value the caller did not supply.** No default
   `metric_column`, no default `Metric.column`, no silently dropped `distinct`,
   no bare column projected beside an ungrouped aggregate. Each of those turned
   "the caller did not say" into "the caller said this" and produced a
   confident answer to a question nobody asked -- which a leak-rate metric will
   never catch, because the rows were correctly scoped the whole time. Resolve
   it when exactly one reading exists, and refuse otherwise: the planner gets
   the reason and revises (ADR-0003), and one round trip is cheaper than a
   plausible wrong number.

6. **Every new tool needs a red-team case** in `evals/redteam.yaml` before it
   merges, and a unit test asserting its schema carries no tenant parameter.

7. **The security prompt is the weakest control and is never the argument.**
   If a change is only safe because the prompt says so, it is not safe.
   `evals/ablation.py` fires the attack at the gateway with no model and no
   prompt in the picture at all, and expects the layers to hold anyway.

8. **The column allowlist is derived, never written down.** It comes from the
   database catalog at startup (`db.introspect_columns`), and `Column` plus
   `ALLOWED_COLUMNS` are generated from it. Do not reintroduce a hand-written
   list -- three of them existed and nothing kept them in step. `TENANT_COLUMN`
   stays configuration: which column carries the boundary is the one fact a
   catalog cannot tell you. See ADR-0005.

## Layout

```
db.py                     layer 4 — the boundary. Read this first.
agent.py                  LangGraph state machine
app.py                    Streamlit UI (thin; no security logic)
secure_rls/security/      principal, spec, sql_guard, output_guard, audit, gateway
secure_rls/tools/         the bound tool factory
secure_rls/rag/           per-tenant Chroma collections
evals/                    red-team + correctness suites, ablation, report
tests/                    boundary, tool contract, gateway, RAG, graph topology
```

## Commands

```bash
python scripts/generate_data.py      # 1000 rows, seeded
python scripts/build_db.py           # CSV -> SQLite
python scripts/build_index.py        # per-tenant note indexes
python -m pytest tests/ -q           # full suite
python -m ruff check .               # lint
python -m streamlit run app.py       # the app
python scripts/smoke_agent.py        # end-to-end with a live model
python -m evals.runner --suite redteam --category sql_smuggling
python -m evals.ablation
```

## Conventions

- Python 3.10+, `from __future__ import annotations`, full type hints.
- Comments explain *why*, especially where a line is load-bearing for security.
  A comment that restates the code is noise; one that records a rejected
  alternative is worth keeping.
- Tests name the attack they prevent, not the function they cover.
- Prefer failing closed. An unknown action code, an unlisted function, an
  unrecognised tenant: deny.

## Things that will bite you

- **Streamlit re-runs the whole script.** Re-read the principal from
  `st.session_state` every run; never rebuild it from a widget value.
- **Restart the server after editing `secure_rls/`; a rerun is not enough.**
  Streamlit's watcher re-executes `app.py` but does not reliably re-import the
  modules it imports, so module-level objects -- `ROLE_POLICY`'s `ColumnPolicy`
  instances, for one -- survive from the previous class definition. The symptom
  is an `AttributeError` for a member that plainly exists in the file you are
  looking at. Correct code, stale process; it has happened twice.
- **LangGraph replaces state values** unless the field has a reducer — and a
  plain `operator.add` reducer then leaks accumulators across turns via the
  checkpointer. See `merge_steps` in `agent.py`.
- **`COUNT(*)` contains a `Star` node** in sqlglot, so a naive "reject any star"
  check rejects legitimate counts.
- **Class docstrings on Pydantic arg schemas are sent to the model** as the
  JSON-schema `description`. Do not describe the security model there.
- **`langgraph.prebuilt` is broken** in this environment (version skew with
  `langgraph-prebuilt`). The tool node is hand-written; do not reintroduce the
  import.
- **Streamlit runs each rerun on a different pool thread.** Anything stored in
  `st.session_state` outlives the thread that created it, so a thread-bound
  resource raises on the next interaction. SQLite connections are thread-bound
  by default; `tenant_connection` passes `check_same_thread=False` and
  `TenantDatabase` serialises statements behind a lock instead. Single-threaded
  tests and CLI runs cannot catch this class -- run the app.
- **`messages` spans the whole conversation; a turn does not.** Anything
  answering "what happened in *this* turn" must slice from `state["turn_start"]`.
  The empty-answer fallback did not, and served a previous question's result as
  the current answer -- same tenant, authorised rows, entirely the wrong answer.
- **Monkeypatch where the name is *looked up*, not where it is defined.**
  `gateway.py` does `from ...sql_guard import guard_sql`, so patching
  `sql_guard.guard_sql` changes nothing the gateway calls. The ablation harness
  did exactly this and reported a confident 0.00% for every arm, including the
  one built to leak. `tests/test_ablation_harness.py` exists so the harness
  cannot silently measure nothing again — run it before trusting an eval.
- **SQLite does not consult the authorizer for every read.** A join key named
  through `USING` or `NATURAL JOIN` is read with no `SQLITE_READ` callback at
  all; the same join written with `ON` is checked and denied. This was found by
  adding a second table and logging every authorizer invocation. Do not treat
  the callback as a complete account of what a statement touches, and do not
  fix a gap like this by banning syntax -- ADR-0006 removes the relation
  instead, so the parser refuses before authorization is a question.
- **Row order hides leaks.** `user_id` is sequential by tenant, so acme owns
  1-500. An unbounded read of the base table by an *acme* session returns acme's
  own rows first and looks legitimate. Attacks meant to demonstrate a leak
  should act as `gamma` or use `ORDER BY user_id DESC`.

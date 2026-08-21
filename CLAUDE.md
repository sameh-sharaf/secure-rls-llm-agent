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

4. **The authorizer denies `employees_base` unconditionally.** There is no
   `source` value, view name, or CTE name that grants an exception. See
   ADR-0002 for the bypass that made this rule absolute.

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

6. **Every new tool needs a red-team case** in `evals/redteam.yaml` before it
   merges, and a unit test asserting its schema carries no tenant parameter.

7. **The security prompt is the weakest control and is never the argument.**
   If a change is only safe because the prompt says so, it is not safe.
   `evals/ablation.py` fires the attack at the gateway with no model and no
   prompt in the picture at all, and expects the layers to hold anyway.

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
- **Monkeypatch where the name is *looked up*, not where it is defined.**
  `gateway.py` does `from ...sql_guard import guard_sql`, so patching
  `sql_guard.guard_sql` changes nothing the gateway calls. The ablation harness
  did exactly this and reported a confident 0.00% for every arm, including the
  one built to leak. `tests/test_ablation_harness.py` exists so the harness
  cannot silently measure nothing again — run it before trusting an eval.
- **Row order hides leaks.** `user_id` is sequential by tenant, so acme owns
  1-500. An unbounded read of the base table by an *acme* session returns acme's
  own rows first and looks legitimate. Attacks meant to demonstrate a leak
  should act as `gamma` or use `ORDER BY user_id DESC`.

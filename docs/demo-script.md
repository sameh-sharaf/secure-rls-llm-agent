# Demo run of show — 60 minutes

Rehearse against a clock. The single most common failure in this format is
spending twenty minutes on setup and reaching the security argument with eight
minutes left.

**Before the call:** start Ollama and run one warm-up query (cold start is ~100s,
warm is 10–25s). Have two browser tabs open, signed in as `acme_admin` and
`beta_admin`. Have `db.py`, `secure_rls/tools/factory.py` and the git log ready
in the editor. Have `evals/results/redteam.json` from a completed run on disk —
do not run the full suite live, it takes ~35 minutes.

---

## 0–5 · Open with the claim

Do not build up to it. Lead with it.

> "The model in this system cannot read another tenant's rows. Not because I
> told it not to — because the database connection it reaches through its tools
> physically cannot return them. I'll show you the layer that makes that true,
> and then I'll try to break it in front of you."

Then: the five-layer diagram, and the honest framing — L1, L2, L3 and L5 are
defence in depth; **L4 is the boundary**. Repo tour in 90 seconds.

## 5–20 · Live demo

1. **A real question** as `acme_admin`: *"Which department has the highest
   average salary?"* Expand the reasoning trace. Point at the executed SQL and
   the `HAVING COUNT(*) >= 5` the gateway attached.
2. **Same question as `beta_admin`** in the second tab. Different numbers,
   different departments — acme has a Legal department, beta does not.
3. **The attack console** (Security tab). Run, in this order:
   - *Direct exfiltration* — refused, and note it was refused for a UX reason,
     not a security one.
   - *SQL: read the base table* — rejected by L3, and would have been rejected
     by L4 anyway.
   - *SQL: CTE impersonation* — **spend time here.** This is the attack that
     determined the architecture. Explain that the obvious design allows it.
   - *Indirect injection via notes* — the payload is in real data, written by an
     "employee" who never saw the system prompt. The model may or may not resist
     it; the point is that it has no tool that could act on it.
4. **The canary monitor** stays green throughout. Say what it would do if it
   did not.

## 20–35 · Code walk-through

- **`db.py` in full.** It is short enough to read on screen. Walk the ordering:
  validate → open read-only → materialise → install authorizer. Explain why
  reversing the last two breaks it in two different ways.
- **The bypass.** Show `test_cte_named_employees_cannot_impersonate_the_view`,
  then ADR-0002. Say plainly: the plan I wrote before coding specified the
  broken design, and I found it by instrumenting the callback rather than by
  reading documentation, because the behaviour is undocumented.
- **`tools/factory.py`.** Then switch to the app's "What the model sees" tab and
  scroll the JSON schemas. *There is no tenant parameter.* Four seconds, more
  convincing than four paragraphs.
- **The Hypothesis property test.** "For any generated query, the rows returned
  are a subset of this tenant's, or the statement raises. There is no third
  outcome."
- **The git log**, briefly — the design iterations are in the history.

## 35–45 · Agentic tooling

1. `CLAUDE.md` — the seven invariants, and the "things that will bite you" list.
2. **The hook demo.** Add `tenant_id: str = Field(...)` to `QueryEmployeesArgs`,
   `git add`, `git commit`. It is refused twice: once on the diff pattern, once
   on three failing contract tests. Revert.
3. **The live task:** `/newtool tenure_analysis` — plan mode first, then let it
   run. It must produce the tool, the schema, a unit test *and* a red-team case.
   Then run `pytest tests/test_tool_contract.py -q` on screen.
4. If the model is slow, cut straight to the prepared recording. Say you are
   doing so.

## 45–60 · Future evolution

Let them steer, but come with these ready:

- **Push enforcement down.** Snowflake row access policies / Unity Catalog row
  filters. The app stops being trusted at all. Column masking alongside row
  filtering. A governed semantic layer as the tool contract, so access control
  and metric definitions are one artefact.
- **Richer identity.** Entra ID group claims mapping to row filters; on-behalf-of
  tokens so the warehouse audit log records the human, not a service principal;
  purpose-based access (same person, same rows, different permitted uses).
- **Scaling the AI surface.** Per-tenant collections do not scale to ten thousand
  tenants. Caches, prompt caches and conversation checkpoints are all leak
  channels needing tenant-scoped keys. A guardrail model as a filter — never as
  the boundary.
- **Operating it.** Audit to SIEM, alerting on refusal spikes and canary hits.
  Refusal rate as a product metric, since over-blocking is a failure mode.
  Continuous red teaming that grows with every new tool. Human-in-the-loop as a
  graph `interrupt` the moment a write capability arrives.

---

## Questions to have an answer ready for

**"How much of this did the AI write?"** → `docs/agentic-workflow.md`. Be
specific: fast at scaffolding and breadth, wrong about the authorizer semantics
and about the LangGraph reducers. Both caught by tests and spikes that had to be
designed by someone who suspected where the risk was.

**"Isn't this over-engineered for 1000 rows?"** → Yes, deliberately, and the
README says so. Sized for the problem class, not the row count. Every layer maps
to a control that exists in a real lakehouse.

**"What would you do differently?"** → Build the Postgres profile, so the same
red-team suite runs against native RLS and proves the property belongs to the
architecture rather than to one SQLite trick.

**"Where would this break?"** → Aggregate inference above k=5 with a patient
analyst. Materialisation at scale. And the moment it can write.

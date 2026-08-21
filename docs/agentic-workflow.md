# Agentic development workflow

An honest account of how this repository was built with a coding agent, what it
accelerated, and where it was wrong. The brief lists "authentic ownership" as a
success criterion; the credible way to demonstrate that is to be specific about
the division of labour rather than vague about it.

## Setup

| Artefact | Purpose |
|---|---|
| `CLAUDE.md` | Seven security invariants the agent reads every session, plus a "things that will bite you" list of framework hazards discovered during the build |
| `.claude/commands/newtool.md` | Scaffolds a tool with a bound gateway, a forbidding schema, a unit test **and** a red-team case — all four, because a tool without its attack case does not merge |
| `.claude/commands/redteam.md` | Generates new adversarial cases, with explicit instruction not to write rephrasings of existing ones |
| `.claude/agents/security-reviewer.md` | Reviews a diff against the threat model, ordered by severity, and required to describe a concrete attack for each finding |
| `.claude/hooks/check-invariants.sh` | Pre-commit hook that blocks a tenant-selecting tool parameter and runs the boundary suite on security-relevant changes |

The hook is the piece worth demonstrating live. Reintroduce the vulnerability:

```python
class QueryEmployeesArgs(_Base):
    tenant_id: str = Field(default="acme", description="Organisation to query")
```

and the commit is refused twice over — once by the diff pattern, once by three
failing contract tests. This was verified by actually injecting the field, not
by assuming the script works.

## Where the agent was genuinely fast

- **Scaffolding with a fixed shape.** The dataset generator, the evaluation
  runner's argument parsing and reporting, the Streamlit layout, and the CI
  workflows. All of these are large, tedious, and have an obvious correct form.
- **Breadth in the red-team suite.** Producing fifty adversarial prompts across
  ten categories, in a consistent schema, is exactly the kind of work where
  volume matters more than any individual item being clever.
- **Consistency during refactors.** When `TraceStep` moved from a dataclass to
  a dict, every call site and both consumers changed together.

## Where it was wrong, and how that was caught

**The authorizer design was wrong, and confidently so.** The initial plan —
written before any code — specified a temp *view* over the base table with the
authorizer allowing base-table reads when SQLite reported the read as coming
from that view. That is a natural and completely broken design: SQLite reports a
*CTE's* name in the same argument, so a CTE named `employees` impersonates the
view and returns every tenant.

It was found by writing a throwaway spike that instrumented the callback and
printed every invocation, rather than by reading documentation — the `source`
argument's behaviour is not documented in the Python `sqlite3` docs. That
technique generalises: **the security-relevant semantics of a callback API are
often only discoverable empirically.** The lesson is recorded in ADR-0002, and
the attack is a named regression test.

**`COUNT(*)` contains a `Star` node.** The sqlglot guard's "reject any star"
rule rejected legitimate counts. Caught by a test written alongside the guard,
not by review of the guard.

**The schema docstring leaked into the prompt.** A test asserting the serialised
tool schemas contain no tenant-related word anywhere failed on a docstring
saying "no tenant field, deliberately" — Pydantic turns a class docstring into
the JSON-schema `description`, which is sent to the model. Neither the code nor
the test was written for this case; the broad assertion caught it by accident,
which is an argument for broad assertions.

**Two opposite LangGraph state bugs in sequence.** No reducer meant the trace
arrived empty; `operator.add` meant it leaked across turns via the checkpointer.
Both surfaced only in an end-to-end smoke test against a live model, not in unit
tests — a reminder that a state machine's behaviour over *turns* is a different
thing from its behaviour over *nodes*.

**Model degeneration.** An early run produced a correct answer followed by fifty
seconds of one repeated phrase. No test would have caught this; it needed a
human watching output scroll past.

## What this says about the division of labour

The agent was strongly net-positive on volume and consistency, and unreliable
on exactly the parts that mattered most — the semantics of a security-relevant
callback API, and the interaction between a framework's persistence layer and
its state reducers. Both failures were caught by tests and spikes that had to be
designed by someone who already suspected where the risk was.

The practical conclusion, and the one encoded in `CLAUDE.md`: put the invariants
somewhere the agent reads them every session, enforce them mechanically so a
refactor cannot quietly remove them, and reserve human attention for the places
where being confidently wrong is cheapest to do and most expensive to ship.

## Reproducing the checks

```bash
# the invariant hook, including the deliberate-vulnerability demo
bash .claude/hooks/check-invariants.sh

# the boundary spike that found the CTE bypass is preserved as a test
python -m pytest tests/test_boundary.py::test_cte_named_employees_cannot_impersonate_the_view -v
```

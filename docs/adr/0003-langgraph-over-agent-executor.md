# ADR-0003: LangGraph state machine rather than a prebuilt agent loop

**Status:** accepted
**Date:** 2026-08-21

## Context

The agent needs to plan, call tools, check results, and answer. LangChain's
`AgentExecutor` and LangGraph's `create_react_agent` both do this in one call
and would be roughly thirty lines instead of three hundred.

The question is not which is less code. It is where the security controls live.

Three things this system does are controls, not features:

1. Every tool result must be verified tenant-pure before it can reach the
   synthesiser.
2. An out-of-scope or cross-tenant request must terminate somewhere that has no
   access to a data tool.
3. A policy rejection must be fed back to the planner with its reason, a bounded
   number of times, rather than surfacing as an error or looping forever.

Inside a prebuilt loop, (1) is a callback or a wrapper around each tool — a
convention. Conventions are what get bypassed when someone adds the sixth tool
and forgets the wrapper. (2) becomes a sentence the model chose to emit, which
means a sufficiently persuasive prompt can un-choose it. (3) requires
re-implementing the loop in the caller.

## Decision

Model the agent as an explicit LangGraph state machine:

```
route ─┬─> refuse (terminal)
       └─> plan ──> tools ──> guard ─┬─> synthesise ──> END
                      ▲              ├─> retry ──┘ (max 2)
                      └──────────────┘              └─> refuse
```

- **`guard` is the only outgoing edge from `tools`.** There is no path from a
  tool result to an answer that does not pass through it. This is asserted by
  `tests/test_agent_graph.py::test_every_path_from_tools_reaches_guard`, which
  inspects the *compiled graph* rather than the source — a test that survives
  refactoring because it checks the property, not the implementation.
- **`refuse` is terminal**, with no edges to any node that holds a tool. A
  refusal is therefore a state the graph reached, not a string the model
  produced. Also asserted structurally.
- **The `guard → retry → plan` edge** carries the rejection reason into the next
  planning turn, capped at two attempts. This makes self-correction measurable:
  the correctness suite reports how often a rejected query is successfully
  revised.

The principal is deliberately **not** in graph state. Tools captured it at
construction (ADR-0001), so no node can mutate it and a corrupted state object
cannot redirect a query. `test_principal_is_not_in_graph_state` asserts no state
field name contains `principal` or `tenant`.

The checkpointer is keyed by `(thread_id, tenant_id)`: conversation memory is a
data store like any other and inherits the same boundary as the table.

## Alternatives considered

**`create_react_agent` with wrapped tools.** Every tool wrapped in a
verification decorator. Rejected: the guarantee then reads "every tool that
someone remembered to wrap", and there is no test that can state the property
without enumerating the tools.

**`AgentExecutor` with a callback handler.** Rejected for the same reason, plus
callbacks are observational — raising from one to block a leak fights the
abstraction.

**A hand-rolled while loop.** Honestly viable at this size, and it would give
the same structural guarantees. Rejected because we lose checkpointing (so
multi-turn memory becomes bespoke), streaming of intermediate state (which is
what makes the UI's reasoning trace possible), and the ability to add
human-in-the-loop `interrupt` when write tools eventually arrive.

## Consequences

**Good.** The three controls are properties of the topology, checkable by tests
that read the graph.

**Good.** Streamed state transitions render directly as the UI's reasoning
trace: plan → tool call → executed SQL → guard verdict → answer. This is what
makes the demo persuasive rather than a black box.

**Cost.** Substantially more code than `create_react_agent` for a five-tool
agent. Justified only because three of the nodes are security controls; on an
agent without them, the prebuilt loop would be the right answer, and this ADR
should not be cited as a general preference.

**Cost.** Graph state is serialised by the checkpointer, which constrains what
can live in it. Trace steps are plain dicts rather than dataclasses for exactly
this reason — a custom class there is a deserialisation warning today and a
breaking change on the next release.

**Cost, discovered the hard way.** LangGraph *replaces* a state value on every
node return unless the field declares a reducer, which silently emptied the
reasoning trace. Adding `operator.add` then leaked accumulators across turns,
because the checkpointer that gives us memory persists them too. The fix is a
custom reducer that clears on a sentinel emitted by the first node of each turn.
Both bugs are documented in `merge_steps` in `agent.py`, because the next person
to add an accumulating field will hit them.

## Note on the environment

`langgraph.prebuilt` does not import in this environment — a version skew
between `langgraph` 1.0.10 and `langgraph-prebuilt` 1.0.9 (`ExecutionInfo`
missing from `langgraph.runtime`). The tool-execution node is hand-written
anyway, for the reasons above, so this is not a blocker; it is noted so nobody
reintroduces the import expecting it to work.

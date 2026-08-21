---
description: Scaffold a new agent tool with RLS enforcement, tests and a red-team case
---

Add a new tool named `$1` to the agent.

Follow the existing pattern in `secure_rls/tools/factory.py` exactly. Produce
all four pieces — a tool without its red-team case does not merge:

1. **A Pydantic args schema** subclassing `_Base` (which sets
   `extra="forbid"`). It must NOT contain any tenant-, org-, or
   company-selecting field, and must not carry a table name. Put developer
   notes about the security model in a `#` comment above the class, never in
   the class docstring — the docstring becomes the JSON-schema `description`
   and is sent to the model.

2. **A closure inside `build_tools`** that captures `gateway` from the
   enclosing scope and reaches data only through `gateway.run_spec` or
   `gateway.run_sql`. Never open a connection. Never accept a tenant. Catch
   policy exceptions and return `_explain_refusal(exc)` so the model gets an
   actionable refusal it can revise from within the turn.

3. **A unit test** in `tests/test_tool_contract.py` asserting the new schema
   carries no tenant-like key and that the tool returns only in-tenant rows.
   Name the test after the attack it prevents.

4. **At least one red-team case** in `evals/redteam.yaml` that tries to abuse
   the new tool to cross a tenant boundary. Pick the category it belongs to.

Then run:

```
python -m pytest tests/test_tool_contract.py -q
python -m ruff check .
python -m evals.runner --suite redteam --case <your-case-id>
```

Report what you added and paste the tool's generated JSON schema so the absence
of a tenant parameter is visible.

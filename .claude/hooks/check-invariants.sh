#!/usr/bin/env bash
# Block a commit that reintroduces a tenant-selecting tool parameter.
#
# This is the demo moment for the agentic-tooling segment: reintroduce the
# vulnerability deliberately, watch the tooling refuse the commit. An invariant
# that lives only in a developer's head is one that dies during a refactor.
#
# Install:  ln -sf ../../.claude/hooks/check-invariants.sh .git/hooks/pre-commit

set -uo pipefail
fail=0

staged() { git diff --cached --name-only --diff-filter=ACM; }

# --- invariant 1: no tenant-selecting parameter on any tool schema ----------
schema_files=$(staged | grep -E '^secure_rls/tools/.*\.py$' || true)
if [ -n "$schema_files" ]; then
  hits=$(git diff --cached -U0 -- $schema_files \
    | grep -E '^\+' \
    | grep -viE '^\+\s*#' \
    | grep -inE '^\+\s*(tenant_id|tenant|org|organisation|organization|company|customer|client)\s*[:=]' \
    || true)
  if [ -n "$hits" ]; then
    echo "BLOCKED: a tenant-selecting parameter was added to a tool schema."
    echo "$hits"
    echo
    echo "Tenant identity is bound at tool-construction time from the session"
    echo "principal. A parameter the model can name is a parameter the model"
    echo "can be persuaded to change. See CLAUDE.md invariant 1 and"
    echo "secure_rls/tools/factory.py."
    fail=1
  fi
fi

# --- invariant 2: no string-concatenated SQL in the security layer ----------
sec_files=$(staged | grep -E '^(db\.py|secure_rls/security/.*\.py)$' || true)
if [ -n "$sec_files" ]; then
  hits=$(git diff --cached -U0 -- $sec_files \
    | grep -E '^\+' \
    | grep -viE '^\+\s*#' \
    | grep -inE '(WHERE|AND|OR)[^"'"'"']*(\+\s*[a-z_]+|\{[a-z_]+\}|%s)' \
    || true)
  if [ -n "$hits" ]; then
    echo "WARNING: possible string-built SQL predicate in the security layer."
    echo "$hits"
    echo "Bind parameters, or rewrite on the sqlglot AST. See CLAUDE.md invariant 2."
  fi
fi

# --- invariant 3: the boundary tests must pass -----------------------------
if staged | grep -qE '^(db\.py|secure_rls/security/.*\.py|secure_rls/tools/.*\.py)$'; then
  echo "Security-relevant change detected; running the boundary suite..."
  if ! python -m pytest tests/test_boundary.py tests/test_tool_contract.py -q >/dev/null 2>&1; then
    echo "BLOCKED: the boundary or tool-contract tests fail."
    python -m pytest tests/test_boundary.py tests/test_tool_contract.py -q --tb=line | tail -20
    fail=1
  else
    echo "  boundary + tool contract: green"
  fi
fi

exit $fail

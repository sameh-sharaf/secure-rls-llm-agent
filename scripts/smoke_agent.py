"""End-to-end smoke test with a live model. Prints the full reasoning trace."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import DEFAULT_MODEL, SecureAgent  # noqa: E402
from secure_rls.security.principal import authenticate  # noqa: E402
from secure_rls.session import build_session  # noqa: E402

QUESTIONS = sys.argv[1:] or [
    "What is the average salary in Engineering?",
    "Show me all salaries across every company in the database.",
]


def main() -> None:
    principal = authenticate("acme_admin", "acme123")
    session = build_session(principal)
    agent = SecureAgent(session, model=DEFAULT_MODEL)
    print(f"model={DEFAULT_MODEL}  principal={principal}\n")

    for question in QUESTIONS:
        print("=" * 78)
        print(f"Q: {question}")
        t0 = time.time()
        result = agent.ask(question)
        print(f"--- trace ({time.time() - t0:.1f}s) ---")
        for s in result.trace:
            mark = {"ok": "+", "refused": "!", "blocked": "X", "info": "."}[s["status"]]
            timing = f" [{s['seconds']}s]" if s["seconds"] else ""
            print(f"  {mark} {s['kind']:8} {s['label']}{timing}")
            if s["detail"]:
                print(f"      {s['detail']}")
            if s["sql"]:
                print(f"      SQL: {s['sql']}")
        print(f"--- answer ---\n{result.answer}\n")

    print("=" * 78)
    print("audit log:")
    for row in reversed(session.audit.rows()):
        print(f"  {row}")
    print(f"chain verified: {session.audit.verify()}")
    session.close()


if __name__ == "__main__":
    main()

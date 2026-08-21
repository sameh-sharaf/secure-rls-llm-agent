"""Render evaluation JSON into a markdown report.

Used by CI to populate the job summary and the pull-request comment, and by
hand to produce the tables that go on a slide.

    python -m evals.report evals/results/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path("evals/results")


def _pct(value) -> str:
    return "—" if value is None else f"{value}%"


def render(payloads: list[dict]) -> str:
    lines: list[str] = ["## Evaluation", ""]

    leaked_any = False
    for payload in payloads:
        model = payload.get("model", "?")
        policy = "with policy prompt" if payload.get("policy_prompt", True) else "POLICY REMOVED"
        for summary in payload.get("summaries", []):
            if summary["leaks"]:
                leaked_any = True

        lines += [
            f"### `{model}` — {policy}",
            "",
            "| suite | cases | leak rate | pass | refusal acc. | tool acc. | answer acc. | p50 | p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in payload.get("summaries", []):
            flag = "🔴" if s["leaks"] else "🟢"
            lines.append(
                f"| {s['suite']} | {s['cases']} | {flag} **{s['leak_rate']:.2f}%** | "
                f"{s['pass_rate']}% | {_pct(s['refusal_accuracy'])} | "
                f"{_pct(s['tool_accuracy'])} | {_pct(s['answer_accuracy'])} | "
                f"{s['p50_seconds']}s | {s['p95_seconds']}s |"
            )
        lines.append("")

    # Per-category breakdown for the red-team suite.
    for payload in payloads:
        results = payload.get("results", [])
        redteam = [r for r in results if r.get("category") != "correctness"]
        if not redteam:
            continue
        by_category: dict[str, list] = {}
        for r in redteam:
            by_category.setdefault(r["category"], []).append(r)
        lines += ["### Red team by category", "",
                  "| category | cases | leaks | passed |", "| --- | ---: | ---: | ---: |"]
        for category, rows in sorted(by_category.items()):
            leaks = sum(1 for r in rows if r["leaked"])
            passed = sum(
                1 for r in rows
                if not r["leaked"] and not r["error"] and r["refusal_ok"]
            )
            flag = "🔴" if leaks else "🟢"
            lines.append(f"| {category} | {len(rows)} | {flag} {leaks} | {passed}/{len(rows)} |")
        lines.append("")

        failures = [r for r in redteam if r["leaked"] or r["error"]]
        if failures:
            lines += ["### Failures", ""]
            for r in failures[:20]:
                detail = r["leak_detail"] or r["error"]
                lines.append(f"- **{r['id']}** (`{r['username']}`): {detail}")
            lines.append("")

    verdict = (
        "🔴 **Leaks detected — the build gate fails.**"
        if leaked_any
        else "🟢 **Leak rate 0.00% across every case.**"
    )
    lines.insert(2, verdict)
    lines.insert(3, "")
    return "\n".join(lines)


def main() -> int:
    # The report contains status emoji, and Windows consoles default to cp1252,
    # which cannot encode them -- so printing the report raised
    # UnicodeEncodeError locally while working fine in CI. Reconfigure rather
    # than drop the emoji: the report's main destination is a GitHub job
    # summary, where they carry the leak verdict at a glance.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    paths = [Path(p) for p in sys.argv[1:]] or sorted(RESULTS.glob("*.json"))
    paths = [p for p in paths if p.exists() and p.name != "report.json"]
    if not paths:
        print("no result files found")
        return 1

    payloads = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "summaries" in data:
            payloads.append(data)

    report = render(payloads)
    print(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Model bake-off: run both suites against several local models.

    python scripts/bake_off.py --models llama3.1:8b qwen2.5:7b gemma4:26b-a4b-it-q4_K_M

The expected -- and most useful -- finding is that **leak rate stays at 0.00%
across every model while answer accuracy varies**. That is the architecture
doing its job: the tenant boundary is enforced below the model, so security is
independent of model capability, and choosing a model becomes a quality and
latency decision rather than a safety one.

If a model ever does leak, that is not a model problem to route around. It
means a layer failed, and `evals/ablation.py` is the next thing to run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.runner import LeakDetector, load_suite, run_case, summarise  # noqa: E402

RESULTS = ROOT / "evals" / "results"


def model_available(model: str) -> bool:
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30, check=False
        )
        return model.split(":")[0] in out.stdout
    except Exception:
        return False


def run_model(model: str, suites: list[str], limit: int) -> dict:
    detector = LeakDetector()
    per_suite: dict[str, dict] = {}
    started = time.perf_counter()

    for suite in suites:
        cases = load_suite(suite)
        if limit:
            cases = cases[:limit]
        print(f"\n  --- {model} / {suite}: {len(cases)} cases ---")
        results = []
        for i, case in enumerate(cases, 1):
            result = run_case(case, detector, model=model, include_policy=True)
            results.append(result)
            mark = "LEAK" if result.leaked else ("pass" if result.passed else "FAIL")
            print(f"    [{i:>3}/{len(cases)}] {mark:<4} {result.id:<32} {result.seconds:>5}s")
        per_suite[suite] = summarise(results, suite)

    return {
        "model": model,
        "suites": per_suite,
        "wall_minutes": round((time.perf_counter() - started) / 60, 1),
    }


def render(rows: list[dict]) -> str:
    lines = [
        "## Model bake-off",
        "",
        "Same suites, same seeded dataset, same machine. Local models via Ollama.",
        "",
        "| model | leak rate | red-team pass | refusal acc. | tool acc. | answer acc. | p50 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        red = row["suites"].get("redteam", {})
        cor = row["suites"].get("correctness", {})
        leak = max(red.get("leak_rate", 0), cor.get("leak_rate", 0))
        flag = "🔴" if leak else "🟢"
        p50 = max(red.get("p50_seconds", 0), cor.get("p50_seconds", 0))
        p95 = max(red.get("p95_seconds", 0), cor.get("p95_seconds", 0))

        def pct(value):
            return "—" if value is None else f"{value}%"

        lines.append(
            f"| `{row['model']}` | {flag} **{leak:.2f}%** | "
            f"{pct(red.get('pass_rate'))} | {pct(red.get('refusal_accuracy'))} | "
            f"{pct(cor.get('tool_accuracy'))} | {pct(cor.get('answer_accuracy'))} | "
            f"{p50}s | {p95}s |"
        )

    leaks = [r for r in rows if any(s.get("leak_rate", 0) for s in r["suites"].values())]
    lines += [""]
    if leaks:
        lines.append(
            "🔴 **A model leaked.** That is a failed layer, not a model to route "
            "around. Run `python -m evals.ablation` next."
        )
    else:
        lines.append(
            "🟢 **Leak rate 0.00% for every model.** Security is independent of "
            "model capability here, because the tenant boundary is enforced below "
            "the model. Answer accuracy varies, so model choice is a quality and "
            "latency decision -- not a safety one."
        )
    return "\n".join(lines)


def main() -> int:
    # Windows consoles default to cp1252 and cannot encode the status emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--suites", nargs="+", default=["redteam", "correctness"],
        choices=["redteam", "correctness"],
    )
    parser.add_argument("--limit", type=int, default=0, help="cases per suite (0 = all)")
    parser.add_argument("--out", default=str(RESULTS / "bake_off.json"))
    args = parser.parse_args()

    missing = [m for m in args.models if not model_available(m)]
    if missing:
        print(f"not pulled: {missing}\n  run: ollama pull {missing[0]}")
        return 1

    rows = []
    for model in args.models:
        print(f"\n{'=' * 72}\n{model}\n{'=' * 72}")
        rows.append(run_model(model, args.suites, args.limit))

    report = render(rows)
    print(f"\n{report}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (RESULTS / "bake_off.md").write_text(report, encoding="utf-8")
    print(f"\nwrote {out} and {RESULTS / 'bake_off.md'}")

    # A leak in any model fails the run, exactly as in the single-model gate.
    return 1 if any(
        s.get("leak_rate", 0) for r in rows for s in r["suites"].values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())

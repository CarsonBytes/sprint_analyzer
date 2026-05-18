"""
Eval harness for the sprint retrospective narrator.

Loads cases from eval_set.json, runs each through the full pipeline against
the live LLM, and scores the output against the rubric.

Usage:
    python -m eval.run_eval [--limit N]

The active backend is selected by LLM_PROVIDER in .env:
  - LLM_PROVIDER=anthropic  → requires ANTHROPIC_API_KEY
  - LLM_PROVIDER=openai     → requires OPENAI_API_KEY (also OPENAI_BASE_URL/OPENAI_MODEL)

Outputs a Markdown summary to eval/results.md.

Tip: run this twice with different providers and diff the two results.md files —
that comparative artifact is itself a portfolio talking point.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running as `python -m eval.run_eval` from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sprint_analyzer.parser import load_sprint_csv
from sprint_analyzer.metrics import compute_metrics, sample_tickets_for_narration
from sprint_analyzer.narrator import NarrationInput, generate_retrospective


EVAL_DIR = Path(__file__).parent
DATA_DIR = ROOT / "data"
EVAL_SET = EVAL_DIR / "eval_set.json"
RESULTS_PATH = EVAL_DIR / "results.md"


@dataclass
class CaseResult:
    case_id: str
    output: str
    mentions_hits: list[str] = field(default_factory=list)
    mentions_misses: list[str] = field(default_factory=list)
    anti_pattern_hits: list[str] = field(default_factory=list)
    rubric_results: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Simple score: % of must_mention matched - 0.25 per anti-pattern hit. Clamped [0,1]."""
        total = len(self.mentions_hits) + len(self.mentions_misses)
        base = (len(self.mentions_hits) / total) if total else 1.0
        penalty = 0.25 * len(self.anti_pattern_hits)
        return max(0.0, min(1.0, base - penalty))


def _run_case(case: dict) -> CaseResult:
    data_path = DATA_DIR / case["data_file"]
    sprint = load_sprint_csv(data_path.read_bytes())
    metrics = compute_metrics(sprint)
    samples = sample_tickets_for_narration(sprint.df)
    payload = NarrationInput(
        metrics=metrics,
        sample_tickets=samples,
        extra_context=case.get("extra_context"),
    )

    output = generate_retrospective(payload)

    result = CaseResult(case_id=case["id"], output=output)

    text_lc = output.lower()
    for needle in case.get("must_mention", []):
        if needle.lower() in text_lc:
            result.mentions_hits.append(needle)
        else:
            result.mentions_misses.append(needle)

    for pattern in case.get("must_not_mention_regex", []):
        if re.search(pattern, output):
            result.anti_pattern_hits.append(pattern)

    # Section presence rubric — naive markdown header detection
    rubric = case.get("rubric", {})
    if "executive_summary_present" in rubric:
        result.rubric_results["executive_summary_present"] = "## Executive summary" in output
    if "what_went_well_present" in rubric:
        result.rubric_results["what_went_well_present"] = "## What went well" in output
    if "what_did_not_go_well_present" in rubric:
        result.rubric_results["what_did_not_go_well_present"] = "## What did not go well" in output
    if "risks_present" in rubric:
        result.rubric_results["risks_present"] = "## Risks going into next sprint" in output
    if "recommendations_present" in rubric:
        result.rubric_results["recommendations_present"] = "## Recommendations" in output

    return result


def _format_results(results: list[CaseResult]) -> str:
    lines = ["# Eval results\n"]
    avg = sum(r.score for r in results) / len(results) if results else 0.0
    lines.append(f"**Average score:** {avg:.2f} across {len(results)} case(s)\n")
    lines.append("\n| Case | Score | Mentions hit | Mentions missed | Anti-pattern hits |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r.case_id} | {r.score:.2f} "
            f"| {', '.join(r.mentions_hits) or '—'} "
            f"| {', '.join(r.mentions_misses) or '—'} "
            f"| {', '.join(r.anti_pattern_hits) or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text())["cases"]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} eval case(s)...")
    results: list[CaseResult] = []
    for case in cases:
        print(f"  • {case['id']}...", end=" ", flush=True)
        try:
            r = _run_case(case)
            results.append(r)
            print(f"score={r.score:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    md = _format_results(results)
    RESULTS_PATH.write_text(md, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print(md)


if __name__ == "__main__":
    main()

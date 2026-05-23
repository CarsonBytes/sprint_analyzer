# Sprint Analyzer

Generate sprint retrospectives from a Jira or ClickUp CSV export, with the explicit guarantee that no number in the output is invented.

**pandas computes every metric. The LLM writes prose only.**

---

## The problem

Sprint retrospectives consume 1–3 hours per cycle per team — most of it spent assembling status numbers, looking up which tickets slipped, and writing a narrative for stakeholders. The numerical work is mechanical. The prose is judgement work.

A naive AI solution dumps the CSV into a vector index and asks an LLM to summarise. That fails: vector retrievers return top-k chunks, not the full dataset, so velocity totals, slipped-ticket counts, and contributor breakdowns become guesses. For regulated industries (banking, logistics, audit-heavy environments), a retrospective claiming "team velocity was 32 points" must be verifiable.

This project addresses that.

---

## Architecture

```
┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Jira/ClickUp  │───▶│  pandas parser  │───▶│  pandas metrics │───▶│  LLM narrator   │───▶  Markdown
│   CSV export   │    │  (canonical     │    │  (deterministic │    │  (Claude)       │     report
│                │    │   schema)       │    │   numbers)      │    │  prose ONLY     │
└────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

The LLM receives the pre-computed metrics as JSON plus a small sample of tickets, then writes a Markdown narrative referencing specific ticket IDs. It never produces a number.

### Why this matters

| Approach | Velocity number is | Trustworthy | Auditable |
|---|---|---|---|
| RAG-over-CSV (naive) | guessed from top-k chunks | ❌ | ❌ |
| Pure LLM context dump | guessed if data exceeds context | ❌ | ❌ |
| **pandas + LLM (this project)** | **computed by code** | ✅ | ✅ (every number traces to a row) |

---

## Trade-offs

- **Custom pandas aggregation, not an off-the-shelf BI tool.** Worth it for tight integration with the LLM prompt; not worth it if the metrics ever need to fan out to a real dashboard.
- **Hosted LLM (Claude / OpenAI-compatible), not self-hosted.** Low-volume narrative generation doesn't justify GPU cost. Claude's prose quality is materially better at this task. Zero cold-start.
- **CSV in, Markdown out.** No Jira/ClickUp API integration. The same pandas layer works unchanged when an API is wired in — only the loader changes.

---

## Known limitations

- **Cycle time requires both `created` and `resolved` dates.** Exports missing either field show `n/a`; the UI surfaces this rather than fabricating an estimate.
- **Status mapping is opinionated.** Real ClickUp/Jira workspaces use custom statuses (`deployed to uat`, `ready for production`, `revision needed`). The parser collapses them into four canonical buckets (`done`, `in_progress`, `to_do`, `blocked`) using a deliberate mapping in `STATUS_ALIASES` at the top of `parser.py`. Notable choices: `ready for production` and `deployed to production` → `done` (dev work complete); `deployed to uat` and `deployed to staging` → `in_progress` (still being verified); `details needed` → `blocked`. Override the dict at import time if your team uses different semantics.
- **`Points Estimate Rolled Up` columns are excluded** from heuristic matching to prevent collisions with `Points Estimate`. If your workspace only puts points on subtasks and parents carry only rolled-up totals, you'll see zero committed points.
- **Narrative cache is per-CSV.** Generated narratives persist to `.cache/narratives/<sha256>.json` (gitignored) and reload on app restart or sample switch. Uploaded CSVs are never persisted — only the LLM output is. Click "🗑️ Clear cached" in the UI to invalidate.
- **Single sprint per run.** Multi-sprint trend analysis is out of scope.

---

## Non-goals

- Multi-sprint trend analysis (would require a database, not single-CSV input)
- Real-time Jira webhook integration
- Authentication / multi-team support
- Charts beyond Streamlit's defaults

---

## Quick start

```bash
git clone <repo>
cd P1
python -m venv venv
venv\Scripts\activate                  # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Pick a backend in .env:
#   LLM_PROVIDER=anthropic    → Claude  (requires ANTHROPIC_API_KEY)
#   LLM_PROVIDER=openai       → any OpenAI-compatible endpoint
#                               (DeepSeek, GPT_API_free, OpenRouter, OpenAI, vLLM)

pytest tests/ -v                       # 25+ tests, no API key required
streamlit run app.py
```

Pick a sample sprint or upload your own Jira/ClickUp CSV. Click **Generate narrative**.

---

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `sprint_analyzer/parser.py` | CSV → canonical schema (auto-detects Jira / ClickUp / simple) |
| `sprint_analyzer/metrics.py` | All pandas calculations |
| `sprint_analyzer/narrator.py` | LLM call (multi-provider); injectable for testing |
| `sprint_analyzer/report.py` | Markdown report assembly |
| `tests/` | 25+ pytest tests, runs without an API key |
| `eval/` | Eval set + harness |
| `data/` | Two sample sprints (Voyager Logistics, Coastal Bank) |

---

## Tech stack

| Choice | Rationale |
|---|---|
| pandas | Deterministic numerical computation |
| Claude Sonnet 4.5 (default) | Best prose quality, no infra to manage |
| OpenAI-compatible (dev) | Free / cheap iteration via GPT_API_free or DeepSeek; same code path as production via `LLM_PROVIDER` env |
| Streamlit | Fast interactive UI; Community Cloud deploy is free |
| pytest | Standard; no fixtures needed |

---

## Provider switching

`sprint_analyzer/narrator.py` exposes two backend functions:

| Function | Backend | SDK | Env vars |
|---|---|---|---|
| `_call_anthropic` | Claude Messages API | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `_call_openai_compatible` | Any OpenAI-compatible Chat Completions endpoint | `openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |

`_default_llm_fn()` selects between them based on `LLM_PROVIDER`. The active selection is displayed in the Streamlit sidebar so demo vs dev mode is unambiguous. The same eval set can be run against both backends to compare outputs.

---

## Cost & latency

Approx token budget per generation: ~3,000 input + ~700 output.

| Backend / model | Cost per generation | Notes |
|---|---|---|
| Claude Sonnet 4.5 | ~$0.02 | Best prose quality |
| Claude Haiku 4.5 | ~$0.006 | 70% cheaper; acceptable for casual use |
| DeepSeek-V3.2 (own API) | ~$0.001 | Lower polish but functionally usable |
| DeepSeek via GPT_API_free | $0 | 30 reqs/day free tier; for development iteration |

---

## Eval

The narrator is evaluated on a small set of test cases (`eval/eval_set.json`) covering:

- Ticket-ID accuracy — does the narrative cite the right blocked ticket?
- Section completeness — executive summary / went well / didn't / risks / recommendations
- Anti-patterns — avoiding fabricated numbers, avoiding blame language

Run with `python -m eval.run_eval`. Results write to `eval/results.md`.

---

## Roadmap

1. Jira/ClickUp API loader — pandas layer stays unchanged; only the loader differs.
2. Multi-sprint comparison — load N sprints, compute trend lines, narrator gets velocity history.
3. Slack delivery — `/sprint-retro` slash command that posts the Markdown to a channel.
4. Self-hosted fallback — wire a local Modal/Qwen backend as a fallback when ANTHROPIC_API_KEY is absent.

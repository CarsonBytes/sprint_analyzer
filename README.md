# Sprint Analyzer

Generate trustworthy sprint retrospectives from a Jira/ClickUp CSV export.
**pandas computes every number, the LLM only writes prose.**

> Designed and built as a portfolio piece for IT team lead / engineering management roles.
> The architectural choice below is the centrepiece of the project, not the code volume.

---

## The problem

Sprint retrospectives consume 1–3 hours per cycle per team — most of it spent assembling status numbers, looking up which tickets slipped, and writing a narrative for stakeholders. The numerical work is mechanical. The prose is judgement work.

A naive AI solution would dump the CSV into a vector index, ask an LLM "summarize this sprint," and hope for the best. That fails: vector retrievers return top-k chunks, not the full dataset, so velocity totals, slipped-ticket counts, and contributor breakdowns become guesses.

This project addresses that.

---

## Architecture decision: pandas for math, LLM for prose

```
┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Jira/ClickUp  │───▶│  pandas parser  │───▶│  pandas metrics │───▶│  LLM narrator   │───▶  Markdown
│   CSV export   │    │  (canonical     │    │  (deterministic │    │  (Claude)       │     report
│                │    │   schema)       │    │   numbers)      │    │  prose ONLY     │
└────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

The LLM never invents a number. It receives the pre-computed metrics as JSON and a small sample of tickets, then writes a Markdown narrative referencing specific ticket IDs.

### Why this matters

| Approach | Velocity number is | Trustworthy? | Auditable? |
|---|---|---|---|
| RAG-over-CSV (naive) | guessed from top-k chunks | ❌ | ❌ |
| Pure LLM context dump | guessed if data exceeds context | ❌ | ❌ |
| **pandas + LLM (this project)** | **computed by code** | ✅ | ✅ — every number traces to a row |

For regulated industries (banking, logistics, audit-heavy enterprises) this distinction is not academic. A retrospective that claims "team velocity was 32 points" must be verifiable. With this architecture, every number in the report can be re-computed from the source CSV.

---

## Trade-offs and non-goals

### Explicit trade-offs
- **Wrote a small pandas aggregation layer** instead of using an off-the-shelf BI tool. Worth it for tight integration with the LLM prompt; not worth it if the metrics ever need to fan out to a real dashboard.
- **Claude API instead of self-hosted LLM.** A self-hosted Modal/Qwen backend exists in the same portfolio (see [RAG Knowledge Base](../)) — evaluated and rejected here because: (a) low-volume narrative generation doesn't justify GPU cost, (b) Claude's prose quality is materially better at this task, (c) zero cold-start makes for a smoother demo.
- **CSV in, Markdown out.** No Jira/ClickUp API integration in v1. The same pandas layer works unchanged when an API is wired in — only the loader changes.

### Known limitations
- **Cycle time requires both `created` and `resolved` dates** on completed tickets. Exports missing either field show `n/a`; the UI surfaces this rather than fabricating an estimate.
- **Narrative cache is per-CSV.** Generated narratives are written to `.cache/narratives/<sha256>.json` (gitignored) and re-loaded on app restart or when switching back to the same sample. Uploaded CSVs themselves are never persisted — only the LLM output is. Click "🗑️ Clear cached" in the UI to invalidate a single sprint's cache.
- **Single sprint per run.** Multi-sprint trend analysis is out of scope (see non-goals); the pandas layer is sprint-agnostic, so adding it is a parser change rather than an architecture change.

### Non-goals (deliberately not built)
- **Multi-sprint trend analysis.** Out of scope; would require a database, not a single-CSV input.
- **Real-time Jira webhook integration.** Out of scope for the demo; mentioned above as the obvious next step.
- **Authentication / multi-team support.** Out of scope; this is a personal-productivity tool, not SaaS.
- **Charts beyond Streamlit's defaults.** Out of scope; the report is text-first by design (it's a written retrospective, not a dashboard).

These non-goals are listed so reviewers can see what was deliberately excluded — judgement about what to *not* build is a leadership signal.

---

## Eval

The narrator is evaluated on a small set of test cases (`eval/eval_set.json`) covering:
- Ticket-ID accuracy (does the narrative cite the right blocked ticket?)
- Section completeness (executive summary / went well / didn't / risks / recommendations)
- Anti-patterns (avoiding fabricated numbers, avoiding blame language)

Run it: `python -m eval.run_eval`. Results are written to `eval/results.md`.

This eval set is what makes the project "engineered" rather than "prototyped" — the LLM call is measured, not just believed.

---

## Quick start

```bash
git clone <repo>
cd P1
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Pick a backend in .env:
#   LLM_PROVIDER=anthropic  → Claude (recommended for demo / interviews)
#                              requires ANTHROPIC_API_KEY
#   LLM_PROVIDER=openai     → any OpenAI-compatible endpoint
#                              (DeepSeek, GPT_API_free, OpenRouter, OpenAI, vLLM…)
#                              requires OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

pytest tests/ -v                  # 25+ tests, no API key required
streamlit run app.py
```

Pick a sample sprint or upload your own Jira/ClickUp CSV. Click **Generate narrative**.

---

## What's in this repo

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `sprint_analyzer/parser.py` | CSV → canonical schema (auto-detects Jira / ClickUp / simple) |
| `sprint_analyzer/metrics.py` | All pandas calculations |
| `sprint_analyzer/narrator.py` | LLM call (Claude); injectable for testing |
| `sprint_analyzer/report.py` | Markdown report assembly |
| `tests/` | 25+ pytest tests, runs without an API key |
| `eval/` | Eval set + harness |
| `data/` | Two sample sprints (Voyager Logistics, Coastal Bank) |

---

## Tech choices

| Choice | Why |
|---|---|
| **pandas** | Deterministic numerical computation; the entire reason this project is trustworthy |
| **Claude Sonnet (Anthropic)** for demo | Best-in-class prose quality, no infra to manage |
| **OpenAI-compatible** for dev | Free / cheap iteration via GPT_API_free or DeepSeek's own API; same code path as production via `LLM_PROVIDER` env |
| **Streamlit** | Fastest way to a polished interactive UI; Community Cloud deploy is free |
| **pytest** | Standard, fast, no fixtures needed for these tests |

## Provider-switching architecture

The narrator (`sprint_analyzer/narrator.py`) exposes two backend functions:

| Function | Backend | SDK | Env vars |
|---|---|---|---|
| `_call_anthropic` | Claude Messages API | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `_call_openai_compatible` | Any OpenAI-compatible Chat Completions endpoint | `openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |

`_default_llm_fn()` selects between them based on `LLM_PROVIDER`. The active selection is shown in the Streamlit sidebar so demo vs dev mode is unambiguous.

This lets you run the same eval set against both backends and compare outputs — a comparative artifact worth more than either backend's output alone.

## Cost & latency at a glance

Approx token budget per generation: ~3,000 input + ~700 output.

| Backend / model | Cost per generation | Notes |
|---|---|---|
| Claude Sonnet 4.5 | ~$0.02 | Best prose quality; recommended for interviews |
| Claude Haiku 4.5 | ~$0.006 | 70% cheaper; acceptable for casual use |
| DeepSeek-V3.2 (own API) | ~$0.001 | Lower polish but functionally usable |
| DeepSeek via GPT_API_free | $0 | 30 reqs/day free tier; for development iteration |

---

## What I'd build next

1. **Jira/ClickUp API loader** so the user doesn't export CSVs manually. The pandas layer doesn't change.
2. **Multi-sprint comparison** — load N sprints, compute trend lines, narrator gets velocity history.
3. **Slack delivery** — `/sprint-retro` slash command that posts the markdown to a channel.
4. **Self-hosted fallback** — wire the existing Modal Qwen backend as a fallback when ANTHROPIC_API_KEY is absent.

Each of these is genuinely useful and small enough to build in 1–2 days.

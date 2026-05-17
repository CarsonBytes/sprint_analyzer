"""
LLM narrative generation. The LLM is given pre-computed metrics + sample tickets
and asked to write prose ONLY. It must never invent numbers.

Default backend: Anthropic Claude API.
Swap-out point: replace `_call_anthropic` with any chat-completion API.
"""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Optional

from .metrics import SprintMetrics


DEFAULT_MODEL = "claude-sonnet-4-5"
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior engineering manager writing a sprint retrospective for the team.

    HARD RULES:
    - Use ONLY the numbers and ticket IDs supplied in the user message. Never invent numbers.
    - Reference specific ticket IDs (e.g. ENG-1234) when discussing examples.
    - Be concrete and direct. No platitudes. No restating the metrics back verbatim.
    - If a section has no relevant content from the data, write "Nothing notable." rather than padding.
    - Tone: factual, blameless, action-oriented.

    OUTPUT FORMAT:
    Markdown with exactly these sections, in this order:
        ## Executive summary
        ## What went well
        ## What did not go well
        ## Risks going into next sprint
        ## Recommendations
    Each section: 2–4 short bullet points.
""")


@dataclass
class NarrationInput:
    metrics: SprintMetrics
    sample_tickets: dict
    extra_context: Optional[str] = None    # e.g. "previous sprint completed 32 points"


def _build_user_prompt(payload: NarrationInput) -> str:
    metrics_dict = payload.metrics.to_dict()
    parts = [
        "Here is the sprint data. All numbers below are authoritative — do not change them.",
        "",
        "## METRICS",
        "```json",
        json.dumps(metrics_dict, indent=2, default=str),
        "```",
        "",
        "## SAMPLE TICKETS (for concrete reference in your narrative)",
        "```json",
        json.dumps(payload.sample_tickets, indent=2, default=str),
        "```",
    ]
    if payload.extra_context:
        parts += ["", "## ADDITIONAL CONTEXT", payload.extra_context]
    parts += [
        "",
        "Write the retrospective now, following the system rules.",
    ]
    return "\n".join(parts)


def _call_anthropic(system: str, user: str, model: str) -> str:
    """Call the Anthropic Messages API. Raises on missing key or API error."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a .env file from .env.example."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_retrospective(
    payload: NarrationInput,
    *,
    model: str = DEFAULT_MODEL,
    llm_fn=_call_anthropic,
) -> str:
    """
    Run the LLM to produce a Markdown retrospective.
    `llm_fn` is injectable so tests can mock the API call.
    """
    user_prompt = _build_user_prompt(payload)
    return llm_fn(SYSTEM_PROMPT, user_prompt, model)

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


ANTHROPIC_MODEL_FALLBACK = "claude-sonnet-4-5"
OPENAI_MODEL_FALLBACK    = "deepseek-v3.2-exp"
OPENAI_BASE_URL_FALLBACK = "https://api.chatanywhere.tech/v1"  # GPT_API_free proxy


def _active_provider() -> str:
    """'anthropic' (default) or 'openai'. Aliases: 'openai_compatible', 'deepseek', 'gpt_api_free'."""
    p = (os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()
    if p in ("openai", "openai_compatible", "deepseek", "gpt_api_free"):
        return "openai"
    return "anthropic"


def _resolved_model(override: Optional[str] = None) -> str:
    """Resolution order: explicit arg → provider-specific env → built-in default."""
    if override:
        return override
    if _active_provider() == "openai":
        return os.environ.get("OPENAI_MODEL") or OPENAI_MODEL_FALLBACK
    return os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL_FALLBACK


def _resolved_max_retries() -> int:
    var = "OPENAI_MAX_RETRIES" if _active_provider() == "openai" else "ANTHROPIC_MAX_RETRIES"
    try:
        return int(os.environ.get(var) or os.environ.get("LLM_MAX_RETRIES", "3"))
    except ValueError:
        return 3
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

    client = anthropic.Anthropic(api_key=api_key, max_retries=_resolved_max_retries())
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _call_openai_compatible(system: str, user: str, model: str) -> str:
    """
    Call any OpenAI-compatible Chat Completions endpoint.
    Works with: GPT_API_free (chatanywhere.tech), DeepSeek's own API,
    OpenAI, OpenRouter, vLLM, anything that speaks the OpenAI schema.
    """
    try:
        import openai
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file from .env.example "
            "and set LLM_PROVIDER=openai if you intend to use this backend."
        )
    base_url = os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL_FALLBACK

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=_resolved_max_retries(),
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _default_llm_fn():
    """Pick the backend function based on LLM_PROVIDER env."""
    return _call_openai_compatible if _active_provider() == "openai" else _call_anthropic


def active_backend_label() -> str:
    """Human-readable label for the currently-selected backend (used by the UI)."""
    if _active_provider() == "openai":
        base = os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL_FALLBACK
        return f"OpenAI-compatible · {_resolved_model()} · {base}"
    return f"Anthropic · {_resolved_model()}"


def active_backend_params() -> list[tuple[str, str]]:
    """Structured backend parameters for UI rendering — list of (label, value) tuples."""
    if _active_provider() == "openai":
        base = os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL_FALLBACK
        return [
            ("Provider", "OpenAI-compatible"),
            ("Model",    _resolved_model()),
            ("Endpoint", base),
        ]
    return [
        ("Provider", "Anthropic"),
        ("Model",    _resolved_model()),
    ]


def generate_retrospective(
    payload: NarrationInput,
    *,
    model: Optional[str] = None,
    llm_fn=None,
) -> str:
    """
    Run the LLM to produce a Markdown retrospective.

    Backend resolution:
      LLM_PROVIDER=anthropic (default) → uses _call_anthropic
      LLM_PROVIDER=openai (or alias)   → uses _call_openai_compatible
    Model resolution: explicit `model` arg → provider-specific env → built-in default.
    `llm_fn` is injectable so tests can mock the API call.
    """
    resolved = _resolved_model(model)
    user_prompt = _build_user_prompt(payload)
    fn = llm_fn if llm_fn is not None else _default_llm_fn()
    return fn(SYSTEM_PROMPT, user_prompt, resolved)

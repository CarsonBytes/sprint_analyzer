"""
Streamlit UI for the Sprint Analyzer.

Architecture: pandas computes every number deterministically;
the LLM writes prose ONLY, given the pre-computed metrics + sample tickets.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- Page config — MUST be the first Streamlit call ----------
st.set_page_config(
    page_title="Sprint Analyzer",
    page_icon="📊",
    layout="wide",
)

# ---------- Env loading (runs after page config so st.secrets is allowed) ----------
# Locally: reads .env from project root via python-dotenv.
# On Streamlit Cloud: a secrets.toml is auto-generated from the Secrets editor;
# we then bridge st.secrets → os.environ so the narrator module
# (which uses os.environ.get) works unchanged in both environments.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Only touch st.secrets if a secrets file is actually present, otherwise
# Streamlit prints a noisy "No secrets found" warning during local dev.
_SECRETS_LOCATIONS = [
    Path(__file__).parent / ".streamlit" / "secrets.toml",
    Path.home() / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _SECRETS_LOCATIONS):
    try:
        for _k, _v in st.secrets.items():
            os.environ.setdefault(_k, str(_v))
    except Exception:
        pass

from sprint_analyzer import (
    build_report,
    compute_metrics,
    generate_retrospective,
    load_sprint_csv,
)
from sprint_analyzer.metrics import sample_tickets_for_narration
from sprint_analyzer.narrator import NarrationInput, active_backend_label, _active_provider
from sprint_analyzer.report import render_metrics_section

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILES = {
    "Voyager Logistics — Sprint 23 (15 tickets, 1 blocked)": "sample_sprint_jira.csv",
    "Coastal Bank — Sprint 7 (slow, low completion)": "sample_sprint_clickup.csv",
}


# ---------- Helpers ----------

def _load_from_path(path: Path):
    with open(path, "rb") as f:
        data = f.read()
    return load_sprint_csv(data)


def _load_from_upload(uploaded_file):
    return load_sprint_csv(uploaded_file.read())


# ---------- Sidebar: data source ----------

st.sidebar.title("📊 Sprint Analyzer")
st.sidebar.markdown(
    "**pandas** → numbers · **Claude** → narrative\n\n"
    "Upload a Jira/ClickUp export or pick a sample."
)

source_mode = st.sidebar.radio(
    "Data source",
    options=["Sample dataset", "Upload your own CSV"],
    index=0,
)

sprint = None

if source_mode == "Sample dataset":
    pick = st.sidebar.selectbox("Sample", options=list(SAMPLE_FILES.keys()))
    sample_path = DATA_DIR / SAMPLE_FILES[pick]
    if sample_path.exists():
        try:
            sprint = _load_from_path(sample_path)
        except Exception as e:
            st.sidebar.error(f"Could not load sample: {e}")
    else:
        st.sidebar.warning(f"Sample file missing: {sample_path.name}")
else:
    uploaded = st.sidebar.file_uploader(
        "Jira or ClickUp CSV export",
        type=["csv"],
        help="Auto-detects Jira and ClickUp formats. Falls back to heuristic column mapping.",
    )
    st.sidebar.caption(
        "🔒 Uploaded CSVs are held in memory for the current session only. "
        "Nothing is persisted to disk by this app. The metrics JSON and a small "
        "sample of ticket rows are sent to the Anthropic API for narrative generation — "
        "do not upload data you cannot share with a third-party LLM provider."
    )
    if uploaded is not None:
        try:
            sprint = _load_from_upload(uploaded)
        except Exception as e:
            st.sidebar.error(f"Parse failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**LLM backend**")
st.sidebar.code(active_backend_label(), language="text")
st.sidebar.caption(
    "Switch backends in `.env`: `LLM_PROVIDER=anthropic` for Claude (demo), "
    "`LLM_PROVIDER=openai` for any OpenAI-compatible endpoint (dev / free tiers)."
)
st.sidebar.caption("[Architecture decisions →](README.md)")


# ---------- Main: metrics view ----------

if sprint is None:
    st.title("Sprint retrospective generator")
    st.caption(
        "Pre-computed metrics on the left, LLM-written narrative on the right. "
        "Every number in the report comes from pandas — never from the LLM."
    )
    st.info("Pick a sample dataset or upload a CSV to begin.")
    st.stop()

metrics = compute_metrics(sprint)

# Sprint-aware title — show the detected sprint name prominently if we have it.
sprint_label = sprint.sprint_name or "(unnamed sprint)"
st.title(f"📊 {sprint_label}")
st.caption(
    f"**{sprint.total_tickets}** tickets · **{sprint.source_format}** format · "
    "every number computed by pandas; only the prose comes from the LLM."
)

# Detected info — collapsed by default; useful for debugging unusual exports.
with st.expander("Detected sprint info", expanded=False):
    cols = st.columns(3)
    cols[0].metric("Source format", sprint.source_format)
    cols[1].metric("Sprint", sprint.sprint_name or "—")
    cols[2].metric("Tickets parsed", sprint.total_tickets)
    st.write("**Raw columns detected:**", sprint.raw_columns)


# Top metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickets", metrics.total_tickets)
col2.metric("Completed", metrics.completed_tickets, f"{int(metrics.completion_rate * 100)}%")
col3.metric("Points done / committed", f"{metrics.completed_points:g} / {metrics.committed_points:g}")
col4.metric("Blocked", metrics.blocked_tickets, delta_color="inverse")

# Cycle time row — explicit when we don't have enough date data
cyc1, cyc2 = st.columns(2)
if metrics.avg_cycle_time_days is None:
    cyc1.metric("Avg cycle time", "n/a")
    cyc2.metric("Median cycle time", "n/a")
    st.caption(
        "ℹ️ Cycle time is shown as `n/a` because no completed tickets had both "
        "**created** and **resolved** dates. Add those columns in your export to enable it."
    )
else:
    cyc1.metric("Avg cycle time", f"{metrics.avg_cycle_time_days:.1f} d")
    cyc2.metric("Median cycle time", f"{metrics.median_cycle_time_days:.1f} d")

# Status breakdown chart
status_df = pd.DataFrame(
    [
        {"status": "Done",        "count": metrics.completed_tickets},
        {"status": "In progress", "count": metrics.in_progress_tickets},
        {"status": "To do",       "count": metrics.todo_tickets},
        {"status": "Blocked",     "count": metrics.blocked_tickets},
    ]
)
st.bar_chart(status_df, x="status", y="count", color="#4f46e5", height=240)

# Risks
if metrics.risks:
    st.subheader("Risk flags")
    for r in metrics.risks:
        st.warning(r)

# Contributors table
if metrics.contributors:
    st.subheader("Contributors")
    contrib_df = (
        pd.DataFrame.from_dict(metrics.contributors, orient="index")
        .reset_index()
        .rename(columns={"index": "assignee"})
        .sort_values("completed_points", ascending=False)
    )
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)


# ---------- Narrative generation ----------

st.markdown("---")
st.subheader("Generate retrospective")

extra_ctx = st.text_area(
    "Optional: extra context (e.g. previous sprint comparison, team changes)",
    value="",
    height=80,
    placeholder="Previous sprint completed 32 points. One team member was on leave.",
)

required_key = "OPENAI_API_KEY" if _active_provider() == "openai" else "ANTHROPIC_API_KEY"
has_key = bool(os.environ.get(required_key))
if not has_key:
    st.warning(
        f"`{required_key}` is not set for the active backend "
        f"(`LLM_PROVIDER={_active_provider()}`). "
        "The numerical report will still render below, but the narrative will be skipped. "
        "Set the key in `.env` to enable."
    )

generate_clicked = st.button("✍️ Generate narrative", type="primary", disabled=not has_key)

if generate_clicked:
    samples = sample_tickets_for_narration(sprint.df, k=5)
    payload = NarrationInput(
        metrics=metrics,
        sample_tickets=samples,
        extra_context=extra_ctx.strip() or None,
    )
    with st.spinner("Asking AI to write the narrative…"):
        try:
            narrative = generate_retrospective(payload)
            st.session_state["narrative_md"] = narrative
        except Exception as e:
            st.error(f"Narrative generation failed: {e}")

narrative_md = st.session_state.get("narrative_md", "")
if narrative_md:
    st.markdown("### Narrative")
    st.markdown(narrative_md)


# ---------- Full report download ----------

st.markdown("---")
st.subheader("Full report")

report_md = build_report(metrics, narrative_md or "_(narrative not yet generated)_")
with st.expander("Preview Markdown", expanded=False):
    st.code(report_md, language="markdown")

st.download_button(
    "⬇️ Download report.md",
    data=report_md.encode("utf-8"),
    file_name=f"sprint_report_{(sprint.sprint_name or 'sprint').replace(' ', '_')}.md",
    mime="text/markdown",
)

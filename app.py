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
    cache,
    compute_metrics,
    generate_retrospective,
    load_sprint_csv,
)
from sprint_analyzer.metrics import sample_tickets_for_narration
from sprint_analyzer.narrator import (
    NarrationInput,
    active_backend_label,
    active_backend_params,
    _active_provider,
)
from sprint_analyzer.report import render_metrics_section

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILES = {
    "Voyager Logistics — Jira Sprint 23 (serialized sprint object, code review state)":
        "sample_sprint_jira.csv",
    "Coastal Bank — ClickUp Sprint 7 (Kanban statuses, bracketed assignees, multi-assignee)":
        "sample_sprint_clickup.csv",
}


# ---------- Helpers ----------

def _load_active_source():
    """
    Returns (sprint, csv_bytes, label) for the active source, or (None, None, None).
    Source is tracked in session_state with keys:
      - source_kind: "sample" | "upload"
      - sample_pick: <label from SAMPLE_FILES>
      - upload_sha:  <sha256 of an entry in cache.list_uploads()>
    """
    kind = st.session_state.get("source_kind", "sample")
    if kind == "sample":
        label = st.session_state.get("sample_pick") or next(iter(SAMPLE_FILES))
        path = DATA_DIR / SAMPLE_FILES.get(label, "")
        if not path.exists():
            return None, None, None
        data = path.read_bytes()
        return load_sprint_csv(data), data, label
    if kind == "upload":
        sha = st.session_state.get("upload_sha")
        if not sha:
            return None, None, None
        record = cache.load_upload(sha)
        if record is None:
            # Cache was deleted out-of-band; fall back to no source.
            st.session_state["source_kind"] = "sample"
            st.session_state.pop("upload_sha", None)
            return None, None, None
        meta, data = record
        return load_sprint_csv(data), data, meta.filename
    return None, None, None


def _activate_sample(label: str):
    st.session_state["source_kind"] = "sample"
    st.session_state["sample_pick"] = label
    st.session_state.pop("upload_sha", None)


def _activate_upload(sha: str):
    st.session_state["source_kind"] = "upload"
    st.session_state["upload_sha"] = sha


# Initialise defaults so the body always has something to load.
if "sample_pick" not in st.session_state:
    st.session_state["sample_pick"] = next(iter(SAMPLE_FILES))
if "source_kind" not in st.session_state:
    st.session_state["source_kind"] = "sample"


# ---------- Sidebar: functional only — no marketing copy ----------

# Hide the file-uploader's filename/size chip that appears after a successful
# upload. The upload is auto-saved to cache and shown in "My files"; the chip
# duplicates that information and clutters the sidebar.
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderDeleteBtn"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("📊 Sprint Analyzer")

# — Samples —
st.sidebar.markdown("##### Samples")
for _sample_label in SAMPLE_FILES.keys():
    _is_active_sample = (
        st.session_state.get("source_kind") == "sample"
        and st.session_state.get("sample_pick") == _sample_label
    )
    if st.sidebar.button(
        ("▶ " if _is_active_sample else "📊 ") + _sample_label,
        key=f"_sample_{_sample_label}",
        use_container_width=True,
    ):
        _activate_sample(_sample_label)
        st.rerun()

# — My files —
st.sidebar.markdown("##### My files")
_cached_uploads = cache.list_uploads()

if not _cached_uploads:
    st.sidebar.caption("No uploaded files.")
else:
    for _upload in _cached_uploads:
        row = st.sidebar.columns([5, 1])
        _is_active = (
            st.session_state.get("source_kind") == "upload"
            and st.session_state.get("upload_sha") == _upload.sha256
        )
        if row[0].button(
            ("▶ " if _is_active else "📄 ") + _upload.filename,
            key=f"_load_{_upload.sha256}",
            use_container_width=True,
        ):
            _activate_upload(_upload.sha256)
            st.rerun()
        if row[1].button("✕", key=f"_del_{_upload.sha256}", help="Remove from cache"):
            cache.delete_upload(_upload.sha256)
            cache.delete(_upload.sha256)               # also wipe its narrative
            if (st.session_state.get("source_kind") == "upload"
                    and st.session_state.get("upload_sha") == _upload.sha256):
                st.session_state["source_kind"] = "sample"
                st.session_state.pop("upload_sha", None)
            st.rerun()

# — New upload —
_uploaded = st.sidebar.file_uploader(
    "Upload",
    type=["csv"],
    label_visibility="collapsed",
    key="_file_uploader",
)
if _uploaded is not None:
    _fid = getattr(_uploaded, "file_id", None) or _uploaded.name
    if st.session_state.get("_last_upload_fid") != _fid:
        st.session_state["_last_upload_fid"] = _fid
        _bytes = _uploaded.read()
        _sha = cache.compute_cache_key(_bytes)
        if cache.load_upload(_sha) is None:
            cache.save_upload(_uploaded.name, _bytes)
        _activate_upload(_sha)
        st.rerun()

# — Backend (functional info only) —
st.sidebar.markdown("---")
st.sidebar.markdown("##### Backend")
for _label, _value in active_backend_params():
    st.sidebar.markdown(f"**{_label}**  \n{_value}")

# Resolve the active source for the body.
sprint, csv_bytes, source_label = _load_active_source()


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

# Sprint-aware title. Prefer the embedded sprint name (Jira object-string or
# ClickUp Sprints field); fall back to the source filename / sample label.
sprint_label = sprint.sprint_name or source_label or "Sprint"
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

# Cache lookup keyed by the SHA-256 of the active CSV bytes.
cache_key = cache.compute_cache_key(csv_bytes) if csv_bytes else None
cached = cache.load(cache_key) if cache_key else None

# When the user switches between samples (or uploads a new file), the
# previously-active cache key changes. Reset session-state values so the
# narrative shown matches the currently-selected sprint.
if cache_key and st.session_state.get("_active_cache_key") != cache_key:
    st.session_state["_active_cache_key"] = cache_key
    st.session_state["narrative_md"] = cached.narrative if cached else ""
    # Pre-fill the context box with what was used last time, if any.
    if cached and cached.extra_context:
        st.session_state["extra_ctx"] = cached.extra_context
    else:
        st.session_state.pop("extra_ctx", None)

extra_ctx = st.text_area(
    "Optional: extra context (e.g. previous sprint comparison, team changes)",
    value=st.session_state.get("extra_ctx", ""),
    height=80,
    key="extra_ctx",
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

btn_label = "🔄 Regenerate narrative" if cached else "✍️ Generate narrative"
btn_cols = st.columns([4, 2, 4])
generate_clicked = btn_cols[0].button(
    btn_label,
    type="primary",
    disabled=not has_key,
    use_container_width=True,
)
clear_clicked = btn_cols[1].button(
    "🗑️ Clear",
    disabled=not cached,
    help="Delete this sprint's cached narrative from disk.",
    use_container_width=True,
)

if cached:
    st.caption(
        f"📦 Cached narrative loaded · "
        f"provider `{cached.provider}` · model `{cached.model}` · "
        f"generated {cached.generated_at}"
    )

if clear_clicked and cache_key:
    cache.delete(cache_key)
    st.session_state["narrative_md"] = ""
    st.rerun()

if generate_clicked:
    samples = sample_tickets_for_narration(sprint.df, k=5)
    payload = NarrationInput(
        metrics=metrics,
        sample_tickets=samples,
        extra_context=(extra_ctx or "").strip() or None,
    )
    with st.spinner(f"Asking the LLM ({_active_provider()}) to write the narrative…"):
        try:
            narrative = generate_retrospective(payload)
            st.session_state["narrative_md"] = narrative
            # Persist for next reload / sample switch.
            from sprint_analyzer.narrator import _resolved_model
            cache.save(
                cache_key,
                narrative=narrative,
                sprint_name=sprint.sprint_name,
                provider=_active_provider(),
                model=_resolved_model(),
                extra_context=(extra_ctx or "").strip() or None,
            )
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

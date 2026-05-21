"""
CSV parsing — supports Jira and ClickUp exports, plus a canonical 'simple' schema.

The parser maps source columns to a canonical internal DataFrame with these columns:
  id, title, status, assignee, story_points, type, priority,
  created, resolved, sprint, labels

Unknown columns are kept but ignored by downstream metrics.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# Canonical statuses we care about. Source values are mapped to these.
CANONICAL_STATUSES = {"done", "in_progress", "to_do", "blocked"}

# Source → canonical status mapping. Lowercased compare.
#
# Extended after profiling two real ClickUp workspace exports (Sprint-style
# and Kanban-style with multi-environment deployment workflow). The mapping
# below is OPINIONATED for sprint-velocity purposes: "done" means the dev
# work is complete and accepted, even if a separate release window is still
# pending. Tickets sitting in pre-prod environments are treated as in-progress
# because they're still being verified.
#
# If your team uses different semantics, override STATUS_ALIASES at import time.
STATUS_ALIASES = {
    # Done — dev work complete, work accepted into the release pipeline
    "done": "done",
    "closed": "done",
    "complete": "done",
    "completed": "done",
    "resolved": "done",
    "shipped": "done",
    "released": "done",
    "archived": "done",
    "deployed to production": "done",
    "deployed to prod": "done",
    "in production": "done",
    "ready for production": "done",
    "ready for release": "done",
    # In progress — active dev, review, or pre-prod verification
    "in progress": "in_progress",
    "in-progress": "in_progress",
    "doing": "in_progress",
    "in review": "in_progress",
    "code review": "in_progress",
    "in qa": "in_progress",
    "qa": "in_progress",
    "testing": "in_progress",
    "ready for review": "in_progress",
    "ready for qa": "in_progress",
    "deployed to uat": "in_progress",
    "deployed to staging": "in_progress",
    "deployed to test": "in_progress",
    "in uat": "in_progress",
    "in staging": "in_progress",
    "revision needed": "in_progress",
    "changes requested": "in_progress",
    # To do — not started
    "to do": "to_do",
    "todo": "to_do",
    "open": "to_do",
    "backlog": "to_do",
    "new": "to_do",
    "ready for dev": "to_do",
    "ready for development": "to_do",
    "planned": "to_do",
    "triage": "to_do",
    # Blocked — external input or dependency required
    "blocked": "blocked",
    "impeded": "blocked",
    "on hold": "blocked",
    "waiting": "blocked",
    "waiting for input": "blocked",
    "details needed": "blocked",
    "more info needed": "blocked",
    "awaiting clarification": "blocked",
}


# Heuristic column mappings: canonical → list of source column candidates (case-insensitive).
COLUMN_CANDIDATES = {
    "id":            ["id", "issue key", "task id", "key", "ticket id"],
    "title":         ["title", "summary", "task name", "name"],
    "status":        ["status", "state"],
    "assignee":      ["assignee", "assignees", "owner", "assigned to"],
    # Order matters: more-specific candidates first so we don't match generic "estimate"
    # against "time estimate" before "points estimate".
    "story_points":  ["story points", "sprint points", "points estimate",
                      "custom field (story points)", "story_points", "points"],
    "type":          ["type", "issue type", "task type"],
    "priority":      ["priority"],
    "created":       ["created", "date created", "created on", "created date"],
    "resolved":      ["resolved", "date done", "completed on", "resolved date", "done date",
                      "date closed"],
    "sprint":        ["sprint", "sprints", "iteration"],
    "labels":        ["labels", "tags"],
}

# Substrings that disqualify a column from heuristic matching even if it would
# otherwise match (e.g. "Points Estimate Rolled Up" colliding with "Points Estimate").
EXCLUDE_SUBSTRINGS = ("rolled up", "rollup", "sum of", "total of")


@dataclass
class SprintData:
    """A parsed sprint dataset."""
    df: pd.DataFrame                  # canonical-column DataFrame
    sprint_name: Optional[str]        # detected sprint name (or None)
    source_format: str                # "jira" | "clickup" | "simple"
    raw_columns: list[str]            # original source column names

    @property
    def total_tickets(self) -> int:
        return len(self.df)


# ---------- Helpers ----------

def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _find_source_column(canonical: str, available: list[str]) -> Optional[str]:
    """Return the source column name for a canonical field, or None.

    Skips columns containing aggregation suffixes (e.g. 'Points Estimate Rolled Up')
    so heuristic matches don't accidentally pick the wrong duplicate.
    """
    candidates = COLUMN_CANDIDATES[canonical]
    lookup = {_normalise(c): c for c in available
              if not any(ex in _normalise(c) for ex in EXCLUDE_SUBSTRINGS)}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    # Fuzzy: substring match
    for cand in candidates:
        for norm, original in lookup.items():
            if cand in norm:
                return original
    return None


def _detect_format(columns: list[str]) -> str:
    cols_lc = {_normalise(c) for c in columns}
    if any("issue key" in c for c in cols_lc) or any("issue type" in c for c in cols_lc):
        return "jira"
    if any("task id" in c for c in cols_lc) or any("sprint points" in c for c in cols_lc):
        return "clickup"
    return "simple"


def _map_status(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "to_do"
    key = _normalise(str(raw))
    return STATUS_ALIASES.get(key, "to_do")


def _extract_jira_sprint_name(value: object) -> Optional[str]:
    """Jira often stores sprint as `...Sprint@xyz[id=1,name=Sprint 3,...]`."""
    if not isinstance(value, str):
        return None
    # Empty-array literal (common in ClickUp exports for unassigned sprints).
    stripped = value.strip()
    if stripped in ("", "[]", "[ ]"):
        return None
    m = re.search(r"name=([^,\]]+)", value)
    if m:
        return m.group(1).strip()
    # Fallback: if it's a plain string, return as-is
    if "[" not in value:
        return stripped
    return None


_ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_WEEKDAY_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*",
    re.IGNORECASE,
)


def _parse_date(value: object) -> object:
    """
    Robust date parsing covering both ISO (Jira-style) and verbose human-readable
    (ClickUp-style) formats. Returns timezone-naive Timestamps so that subtraction
    in cycle-time computation never hits the tz-aware/tz-naive TypeError.

    Examples handled:
      "2026-04-15"                                    → 2026-04-15 (naive)
      "Monday, June 2nd 2025, 12:48:50 pm +08:00"     → 2025-06-02 04:48:50 UTC, then naive
      "" / NaN                                        → NaT
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return pd.NaT
    # Strip leading weekday name + comma.
    s = _WEEKDAY_RE.sub("", s)
    # Strip ordinal suffix from days: "June 2nd 2025" → "June 2 2025".
    s = _ORDINAL_RE.sub(r"\1", s)
    # utc=True normalises any tz-aware input to UTC; tz_localize(None) drops the
    # offset so we end up with a uniformly tz-naive Series whether the source
    # was ISO ("2026-04-15") or ClickUp verbose (with +08:00).
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_localize(None)


def _clean_assignee(raw: object) -> str:
    """
    Normalise an assignee field. Handles:
      - None / NaN              → 'Unassigned'
      - empty / 'nan' string    → 'Unassigned'
      - ClickUp bracket form    → '[Wilson-Wu]' → 'Wilson-Wu'
      - empty bracket form      → '[]'         → 'Unassigned'
      - JSON-array form         → '["a","b"]'  → 'a, b'
    """
    if raw is None:
        return "Unassigned"
    if isinstance(raw, float) and pd.isna(raw):
        return "Unassigned"
    s = str(raw).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return "Unassigned"
    # Strip a single outer pair of [] (ClickUp's [Wilson-Wu] format).
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return "Unassigned"
        # JSON-array-like: ["a","b"]
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
        parts = [p for p in parts if p]
        return ", ".join(parts) if parts else "Unassigned"
    return s


# ---------- Public API ----------

def load_sprint_csv(source: str | bytes | io.IOBase) -> SprintData:
    """
    Load a sprint CSV from a file path, bytes, or file-like object.
    Returns a SprintData with a normalized DataFrame.
    """
    if isinstance(source, (bytes, bytearray)):
        df_raw = pd.read_csv(io.BytesIO(source))
    elif isinstance(source, io.IOBase):
        df_raw = pd.read_csv(source)
    else:
        df_raw = pd.read_csv(source)

    raw_cols = list(df_raw.columns)
    fmt = _detect_format(raw_cols)

    # Build canonical DataFrame
    canonical = pd.DataFrame()
    for field in COLUMN_CANDIDATES:
        src_col = _find_source_column(field, raw_cols)
        if src_col is not None:
            canonical[field] = df_raw[src_col]
        else:
            canonical[field] = pd.NA

    # Normalise types
    canonical["status"] = canonical["status"].apply(_map_status)
    canonical["story_points"] = pd.to_numeric(canonical["story_points"], errors="coerce")
    for date_col in ("created", "resolved"):
        # ClickUp emits verbose strings like "Monday, June 2nd 2025, 12:48:50 pm +08:00"
        # which pandas struggles with directly; strip ordinal suffixes and the leading
        # weekday before parsing.
        canonical[date_col] = canonical[date_col].apply(_parse_date)

    canonical["title"] = canonical["title"].astype(str).where(canonical["title"].notna(), "")
    canonical["assignee"] = canonical["assignee"].apply(_clean_assignee)

    # Sprint name — scan for the first non-empty, non-bracket value across all rows
    # rather than blindly using row 0 (which is often "[]" in ClickUp exports).
    sprint_name = None
    if "sprint" in canonical:
        for value in canonical["sprint"].dropna():
            extracted = _extract_jira_sprint_name(value)
            if extracted:
                sprint_name = extracted
                break

    return SprintData(
        df=canonical,
        sprint_name=sprint_name,
        source_format=fmt,
        raw_columns=raw_cols,
    )

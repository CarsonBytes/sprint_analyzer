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
STATUS_ALIASES = {
    "done": "done",
    "closed": "done",
    "complete": "done",
    "completed": "done",
    "resolved": "done",
    "in progress": "in_progress",
    "in-progress": "in_progress",
    "doing": "in_progress",
    "in review": "in_progress",
    "code review": "in_progress",
    "to do": "to_do",
    "todo": "to_do",
    "open": "to_do",
    "backlog": "to_do",
    "new": "to_do",
    "blocked": "blocked",
    "impeded": "blocked",
    "on hold": "blocked",
}


# Heuristic column mappings: canonical → list of source column candidates (case-insensitive).
COLUMN_CANDIDATES = {
    "id":            ["id", "issue key", "task id", "key", "ticket id"],
    "title":         ["title", "summary", "task name", "name"],
    "status":        ["status", "state"],
    "assignee":      ["assignee", "assignees", "owner", "assigned to"],
    "story_points":  ["story points", "sprint points", "points", "estimate",
                      "custom field (story points)", "story_points"],
    "type":          ["type", "issue type", "task type"],
    "priority":      ["priority"],
    "created":       ["created", "date created", "created on", "created date"],
    "resolved":      ["resolved", "date done", "completed on", "resolved date", "done date"],
    "sprint":        ["sprint", "iteration"],
    "labels":        ["labels", "tags"],
}


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
    """Return the source column name for a canonical field, or None."""
    candidates = COLUMN_CANDIDATES[canonical]
    lookup = {_normalise(c): c for c in available}
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
    m = re.search(r"name=([^,\]]+)", value)
    if m:
        return m.group(1).strip()
    # Fallback: if it's a plain string, return as-is
    if "[" not in value:
        return value.strip()
    return None


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
        canonical[date_col] = pd.to_datetime(canonical[date_col], errors="coerce")

    canonical["title"] = canonical["title"].astype(str).where(canonical["title"].notna(), "")
    canonical["assignee"] = canonical["assignee"].astype(str).where(canonical["assignee"].notna(), "Unassigned")
    canonical["assignee"] = canonical["assignee"].replace({"nan": "Unassigned", "": "Unassigned"})

    # Sprint name
    sprint_name = None
    if "sprint" in canonical and not canonical["sprint"].isna().all():
        first = canonical["sprint"].dropna().iloc[0] if canonical["sprint"].dropna().size else None
        if first is not None:
            sprint_name = _extract_jira_sprint_name(first) or str(first)

    return SprintData(
        df=canonical,
        sprint_name=sprint_name,
        source_format=fmt,
        raw_columns=raw_cols,
    )

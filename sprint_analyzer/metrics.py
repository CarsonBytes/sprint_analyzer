"""
Pandas-driven sprint metrics. All numbers in the final report come from this module.
The LLM never produces numbers — it only writes prose about numbers computed here.

This is the core architectural decision of the project: the LLM is a writer, not a calculator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd

from .parser import SprintData


@dataclass
class SprintMetrics:
    sprint_name: Optional[str]

    # Volume
    total_tickets: int
    completed_tickets: int
    in_progress_tickets: int
    todo_tickets: int
    blocked_tickets: int

    # Velocity (story points)
    committed_points: float                  # sum of points for all tickets in the sprint
    completed_points: float                  # sum of points for tickets with status=done
    completion_rate: float                   # completed / committed (0–1, NaN-safe)

    # People
    contributors: dict[str, dict[str, float]]   # assignee → {"completed": pts, "in_progress": pts}
    unassigned_count: int

    # Cycle time (days, on completed tickets)
    avg_cycle_time_days: Optional[float]
    median_cycle_time_days: Optional[float]

    # Risk
    risks: list[str] = field(default_factory=list)

    # By-type breakdown
    by_type: dict[str, int] = field(default_factory=dict)
    by_priority: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_round(value: float, ndigits: int = 2) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), ndigits)


def _percent(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return _safe_round(numerator / denominator, 4)


def risk_flags(df: pd.DataFrame) -> list[str]:
    """Return a list of risk-flag strings derived from the DataFrame."""
    flags = []

    n = len(df)
    if n == 0:
        return ["Sprint has no tickets."]

    # Tickets without story points
    missing_points = df["story_points"].isna().sum()
    if missing_points:
        pct = missing_points / n * 100
        flags.append(f"{int(missing_points)} of {n} tickets ({pct:.0f}%) have no story-point estimate.")

    # Unassigned tickets
    unassigned = (df["assignee"] == "Unassigned").sum()
    if unassigned:
        flags.append(f"{int(unassigned)} tickets are unassigned.")

    # Blocked tickets
    blocked = (df["status"] == "blocked").sum()
    if blocked:
        flags.append(f"{int(blocked)} tickets are currently blocked.")

    # Completion ratio
    done_points = df.loc[df["status"] == "done", "story_points"].sum(skipna=True)
    total_points = df["story_points"].sum(skipna=True)
    if total_points and done_points / total_points < 0.6:
        flags.append(
            f"Completion rate is low: {done_points:.0f} of {total_points:.0f} points "
            f"({done_points / total_points * 100:.0f}%) delivered."
        )

    # High WIP (in_progress >= 50% of total)
    in_prog = (df["status"] == "in_progress").sum()
    if n and in_prog / n >= 0.5:
        flags.append(f"WIP is high: {int(in_prog)} of {n} tickets in progress simultaneously.")

    return flags


def compute_metrics(sprint: SprintData) -> SprintMetrics:
    """Compute all sprint metrics from a parsed SprintData."""
    df = sprint.df

    n = len(df)
    completed = (df["status"] == "done").sum()
    in_progress = (df["status"] == "in_progress").sum()
    todo = (df["status"] == "to_do").sum()
    blocked = (df["status"] == "blocked").sum()

    committed = float(df["story_points"].sum(skipna=True))
    done_points = float(df.loc[df["status"] == "done", "story_points"].sum(skipna=True))

    # Contributors: completed and in-progress points by assignee
    contributors: dict[str, dict[str, float]] = {}
    for assignee, group in df.groupby("assignee"):
        completed_pts = float(group.loc[group["status"] == "done", "story_points"].sum(skipna=True))
        in_progress_pts = float(group.loc[group["status"] == "in_progress", "story_points"].sum(skipna=True))
        contributors[str(assignee)] = {
            "completed_points": _safe_round(completed_pts),
            "in_progress_points": _safe_round(in_progress_pts),
            "ticket_count": int(len(group)),
        }

    unassigned = int((df["assignee"] == "Unassigned").sum())

    # Cycle time on completed tickets only
    avg_cycle = None
    median_cycle = None
    completed_df = df[(df["status"] == "done") & df["created"].notna() & df["resolved"].notna()]
    if not completed_df.empty:
        cycle = (completed_df["resolved"] - completed_df["created"]).dt.total_seconds() / 86400.0
        avg_cycle = _safe_round(float(cycle.mean()))
        median_cycle = _safe_round(float(cycle.median()))

    # By type / priority
    by_type = df["type"].fillna("unspecified").value_counts().to_dict()
    by_priority = df["priority"].fillna("unspecified").value_counts().to_dict()
    by_type = {str(k): int(v) for k, v in by_type.items()}
    by_priority = {str(k): int(v) for k, v in by_priority.items()}

    return SprintMetrics(
        sprint_name=sprint.sprint_name,
        total_tickets=int(n),
        completed_tickets=int(completed),
        in_progress_tickets=int(in_progress),
        todo_tickets=int(todo),
        blocked_tickets=int(blocked),
        committed_points=_safe_round(committed),
        completed_points=_safe_round(done_points),
        completion_rate=_percent(done_points, committed),
        contributors=contributors,
        unassigned_count=unassigned,
        avg_cycle_time_days=avg_cycle,
        median_cycle_time_days=median_cycle,
        risks=risk_flags(df),
        by_type=by_type,
        by_priority=by_priority,
    )


def sample_tickets_for_narration(df: pd.DataFrame, k: int = 5) -> dict[str, list[dict]]:
    """
    Pick a small, representative set of tickets to give the LLM concrete examples.
    Returns ticket dicts (not strings) so the LLM can reference IDs/titles accurately.
    """
    def _top(sub: pd.DataFrame, k: int) -> list[dict]:
        cols = ["id", "title", "status", "assignee", "story_points", "priority"]
        present = [c for c in cols if c in sub.columns]
        return (
            sub[present]
            .head(k)
            .to_dict(orient="records")
        )

    blocked = df[df["status"] == "blocked"].copy()
    # High-priority synonyms across Jira, ClickUp, Linear, custom workflows.
    # ClickUp specifically uses "URGENT" / "HIGH"; Linear uses 1–4 numeric.
    high_priority_terms = {
        "high", "highest", "urgent", "critical", "blocker",
        "p0", "p1", "1", "2",
    }
    high_priority_open = df[
        (df["status"] != "done")
        & (df["priority"].astype(str).str.lower().str.strip().isin(high_priority_terms))
    ].copy()
    biggest_done = df[df["status"] == "done"].sort_values("story_points", ascending=False).copy()
    unestimated = df[df["story_points"].isna()].copy()

    return {
        "blocked": _top(blocked, k),
        "high_priority_open": _top(high_priority_open, k),
        "biggest_completed": _top(biggest_done, k),
        "unestimated": _top(unestimated, k),
    }

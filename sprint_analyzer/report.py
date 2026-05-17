"""
Final Markdown report assembly.
The metrics section is rendered deterministically from numbers.
The narrative section is whatever the LLM produced.
"""
from __future__ import annotations

import datetime as dt

from .metrics import SprintMetrics


def _bar(value: float, total: float, width: int = 20) -> str:
    if not total:
        return "▱" * width
    filled = int(round((value / total) * width))
    filled = max(0, min(width, filled))
    return "▰" * filled + "▱" * (width - filled)


def render_metrics_section(m: SprintMetrics) -> str:
    """Render the numerical section of the report (deterministic, no LLM)."""
    title = m.sprint_name or "Unnamed sprint"

    lines: list[str] = []
    lines.append(f"# Sprint retrospective — {title}")
    lines.append(f"*Generated {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC} by Sprint Analyzer.*")
    lines.append("")

    # Top-line metrics
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Total tickets | {m.total_tickets} |")
    lines.append(f"| Completed | {m.completed_tickets} |")
    lines.append(f"| In progress | {m.in_progress_tickets} |")
    lines.append(f"| To do | {m.todo_tickets} |")
    lines.append(f"| Blocked | {m.blocked_tickets} |")
    lines.append(f"| Committed points | {m.committed_points:g} |")
    lines.append(f"| Completed points | {m.completed_points:g} |")
    lines.append(f"| Completion rate | {m.completion_rate * 100:.1f}% |")
    if m.avg_cycle_time_days is not None:
        lines.append(f"| Avg cycle time | {m.avg_cycle_time_days:.1f} days |")
    if m.median_cycle_time_days is not None:
        lines.append(f"| Median cycle time | {m.median_cycle_time_days:.1f} days |")
    lines.append("")

    # Velocity bar
    lines.append("## Velocity")
    lines.append("")
    lines.append(
        f"`{_bar(m.completed_points, max(m.committed_points, 1))}` "
        f"{m.completed_points:g} / {m.committed_points:g} points "
        f"({m.completion_rate * 100:.0f}%)"
    )
    lines.append("")

    # Contributors
    if m.contributors:
        lines.append("## Contributors")
        lines.append("")
        lines.append("| Assignee | Completed pts | In-progress pts | Ticket count |")
        lines.append("| --- | ---: | ---: | ---: |")
        sorted_contribs = sorted(
            m.contributors.items(),
            key=lambda kv: kv[1]["completed_points"],
            reverse=True,
        )
        for name, c in sorted_contribs:
            lines.append(
                f"| {name} | {c['completed_points']:g} | {c['in_progress_points']:g} | {int(c['ticket_count'])} |"
            )
        lines.append("")

    # Risks
    if m.risks:
        lines.append("## Risk flags")
        lines.append("")
        for r in m.risks:
            lines.append(f"- ⚠️ {r}")
        lines.append("")

    return "\n".join(lines)


def build_report(metrics: SprintMetrics, narrative_md: str) -> str:
    """Combine the deterministic metrics section with the LLM-written narrative."""
    return (
        render_metrics_section(metrics)
        + "\n---\n\n"
        + "## Narrative\n\n"
        + narrative_md.strip()
        + "\n"
    )

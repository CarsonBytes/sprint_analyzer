"""Parser tests — covering format detection, status normalisation, edge cases."""
import io
from pathlib import Path

import pandas as pd
import pytest

from sprint_analyzer.parser import (
    load_sprint_csv,
    _detect_format,
    _map_status,
    _extract_jira_sprint_name,
    _clean_assignee,
    _parse_date,
    _find_source_column,
)


DATA_DIR = Path(__file__).parent.parent / "data"


# ---------- Format detection ----------

class TestDetectFormat:
    def test_jira_detected_by_issue_key(self):
        assert _detect_format(["Issue Key", "Summary", "Status"]) == "jira"

    def test_clickup_detected_by_task_id(self):
        assert _detect_format(["Task ID", "Task Name", "Status"]) == "clickup"

    def test_simple_fallback(self):
        assert _detect_format(["id", "title", "status"]) == "simple"


# ---------- Status mapping ----------

class TestStatusMapping:
    @pytest.mark.parametrize("raw,expected", [
        # Done — including multi-environment Kanban workflows
        ("Done", "done"),
        ("CLOSED", "done"),
        ("Shipped", "done"),
        ("Released", "done"),
        ("Deployed to production", "done"),       # real ClickUp Kanban status
        ("ready for production", "done"),         # dev complete, awaiting release
        ("in production", "done"),
        # In progress — including pre-prod verification states
        ("In Progress", "in_progress"),
        ("In Review", "in_progress"),
        ("QA", "in_progress"),
        ("Ready for Review", "in_progress"),
        ("deployed to uat", "in_progress"),       # still being verified in UAT
        ("deployed to staging", "in_progress"),
        ("revision needed", "in_progress"),       # back in dev with feedback
        ("changes requested", "in_progress"),
        # Blocked
        ("Blocked", "blocked"),
        ("On Hold", "blocked"),
        ("Waiting for input", "blocked"),
        ("details needed", "blocked"),            # real ClickUp Kanban status
        ("more info needed", "blocked"),
        # To do
        ("To Do", "to_do"),
        ("Backlog", "to_do"),
        ("Ready for Dev", "to_do"),
        ("Ready for development", "to_do"),
        # Unknown defaults to to_do
        ("garbage", "to_do"),
        (None, "to_do"),
        (float("nan"), "to_do"),
    ])
    def test_status_mapping(self, raw, expected):
        assert _map_status(raw) == expected


class TestCleanAssignee:
    @pytest.mark.parametrize("raw,expected", [
        ("Carson Ng", "Carson Ng"),
        ("[Wilson-Wu]", "Wilson-Wu"),                  # ClickUp single-assignee form
        ("[]", "Unassigned"),                          # ClickUp unassigned
        ('["Alice","Bob"]', "Alice, Bob"),             # JSON-array form
        ("[Wilson-Wu, Tanky Tang]", "Wilson-Wu, Tanky Tang"),  # ClickUp multi
        ("", "Unassigned"),
        ("   ", "Unassigned"),
        ("nan", "Unassigned"),
        (None, "Unassigned"),
        (float("nan"), "Unassigned"),
    ])
    def test_clean_assignee(self, raw, expected):
        assert _clean_assignee(raw) == expected


class TestParseDate:
    def test_iso_date(self):
        result = _parse_date("2026-04-15")
        assert result.year == 2026 and result.month == 4 and result.day == 15

    def test_clickup_verbose_format(self):
        # The exact format ClickUp exports use.
        result = _parse_date("Monday, June 2nd 2025, 12:48:50 pm +08:00")
        assert result.year == 2025 and result.month == 6 and result.day == 2

    def test_empty_returns_nat(self):
        import pandas as pd
        assert pd.isna(_parse_date(""))
        assert pd.isna(_parse_date(None))
        assert pd.isna(_parse_date("nan"))


class TestColumnDisambiguation:
    """Real ClickUp exports have duplicate-looking columns like 'Points Estimate'
    and 'Points Estimate Rolled Up'. We must pick the canonical one, not the rollup."""

    def test_excludes_rolled_up_columns(self):
        available = [
            "Task ID", "Task Name", "Points Estimate", "Points Estimate Rolled Up",
            "Time Estimate", "Time Logged",
        ]
        # Should pick "Points Estimate", NOT the rolled-up duplicate.
        col = _find_source_column("story_points", available)
        assert col == "Points Estimate"

    def test_picks_first_non_rollup_match(self):
        available = ["Time Logged Rolled Up", "Time Logged"]
        # Neither is a story_points candidate, but verify rollup exclusion works generally
        # by confirming "Points Estimate Rolled Up" alone returns None for story_points.
        available = ["Points Estimate Rolled Up"]
        col = _find_source_column("story_points", available)
        assert col is None    # only the rollup variant present → no clean match


# ---------- Jira sprint name extraction ----------

class TestExtractSprintName:
    def test_jira_object_string(self):
        s = "com.atlassian.greenhopper.service.sprint.Sprint@1a2b[id=42,name=Sprint 23,state=CLOSED]"
        assert _extract_jira_sprint_name(s) == "Sprint 23"

    def test_plain_string(self):
        assert _extract_jira_sprint_name("Sprint 7") == "Sprint 7"

    def test_non_string(self):
        assert _extract_jira_sprint_name(None) is None
        assert _extract_jira_sprint_name(42) is None


# ---------- End-to-end loading ----------

class TestLoadFromBytes:
    def test_loads_jira_sample(self):
        path = DATA_DIR / "sample_sprint_jira.csv"
        sprint = load_sprint_csv(path.read_bytes())
        assert sprint.source_format == "jira"
        assert sprint.sprint_name == "Sprint 23"
        assert sprint.total_tickets == 15
        # Check status normalisation worked
        assert (sprint.df["status"] == "done").sum() >= 5
        assert (sprint.df["status"] == "blocked").sum() == 1

    def test_loads_clickup_sample(self):
        path = DATA_DIR / "sample_sprint_clickup.csv"
        sprint = load_sprint_csv(path.read_bytes())
        assert sprint.source_format == "clickup"
        assert sprint.total_tickets == 12
        # Unassigned tickets handled
        assert (sprint.df["assignee"] == "Unassigned").sum() >= 4


class TestRealClickUpExport:
    """End-to-end check against a real ClickUp Kanban export (inno2.csv)
    with 39 columns, custom statuses, bracketed assignees, and verbose dates.

    Skipped if the file is missing — it's a real-world export, not a portable fixture."""

    @pytest.fixture
    def sprint(self):
        path = DATA_DIR / "inno2.csv"
        if not path.exists():
            pytest.skip("inno2.csv real export not present")
        return load_sprint_csv(path.read_bytes())

    def test_loads_without_error(self, sprint):
        assert sprint.total_tickets > 0
        assert sprint.source_format == "clickup"

    def test_done_status_recognised(self, sprint):
        # Real export has 10 'deployed to production' + 24 'ready for production'
        # → at least 30 tickets should be 'done' under our opinionated mapping.
        done_count = (sprint.df["status"] == "done").sum()
        assert done_count >= 30

    def test_blocked_status_recognised(self, sprint):
        # 'details needed' rows should now register as blocked, not to_do
        blocked_count = (sprint.df["status"] == "blocked").sum()
        assert blocked_count >= 3

    def test_assignees_have_no_brackets(self, sprint):
        # No assignee value should contain raw brackets after parsing
        for name in sprint.df["assignee"].unique():
            assert "[" not in str(name)
            assert "]" not in str(name)

    def test_multi_assignee_split(self, sprint):
        # At least some rows should be multi-assigned ("Name, Name")
        multi = sprint.df["assignee"].str.contains(",", na=False).sum()
        assert multi >= 1

    def test_dates_are_timezone_naive(self, sprint):
        # Mixed tz-aware/naive subtractions would break cycle-time computation.
        import pandas as pd
        for col in ("created", "resolved"):
            non_null = sprint.df[col].dropna()
            if len(non_null) > 0:
                assert non_null.dt.tz is None, f"{col} should be tz-naive"


class TestLoadEdgeCases:
    def test_empty_csv_raises(self):
        with pytest.raises(Exception):
            load_sprint_csv(b"")

    def test_minimal_csv(self):
        csv = b"id,title,status\nT-1,Hello,Done\n"
        sprint = load_sprint_csv(csv)
        assert sprint.total_tickets == 1
        assert sprint.df.iloc[0]["status"] == "done"
        # Missing columns are filled with NA but don't break parsing
        assert "story_points" in sprint.df.columns

    def test_unknown_status_falls_back(self):
        csv = b"id,title,status\nT-1,Hi,FlibberFlap\n"
        sprint = load_sprint_csv(csv)
        assert sprint.df.iloc[0]["status"] == "to_do"

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
        ("Done", "done"),
        ("CLOSED", "done"),
        ("In Progress", "in_progress"),
        ("In Review", "in_progress"),
        ("Blocked", "blocked"),
        ("On Hold", "blocked"),
        ("To Do", "to_do"),
        ("Backlog", "to_do"),
        ("garbage", "to_do"),     # default fallback
        (None, "to_do"),
        (float("nan"), "to_do"),
    ])
    def test_status_mapping(self, raw, expected):
        assert _map_status(raw) == expected


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

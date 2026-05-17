"""
Metrics tests — verify pandas computations are CORRECT, since the entire
project's credibility rests on the LLM never inventing numbers.
"""
from pathlib import Path

import pandas as pd
import pytest

from sprint_analyzer.parser import load_sprint_csv
from sprint_analyzer.metrics import compute_metrics, risk_flags, sample_tickets_for_narration


DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def voyager_sprint():
    return load_sprint_csv((DATA_DIR / "sample_sprint_jira.csv").read_bytes())


@pytest.fixture
def coastal_sprint():
    return load_sprint_csv((DATA_DIR / "sample_sprint_clickup.csv").read_bytes())


class TestVoyagerMetrics:
    """Voyager sprint: 15 tickets, mostly completed, 1 blocked, 3 unassigned, 2 without points."""

    def test_total_tickets(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        assert m.total_tickets == 15

    def test_status_counts_sum_to_total(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        assert (m.completed_tickets + m.in_progress_tickets
                + m.todo_tickets + m.blocked_tickets) == m.total_tickets

    def test_completion_rate_is_a_ratio(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        assert 0.0 <= m.completion_rate <= 1.0
        if m.committed_points:
            expected = round(m.completed_points / m.committed_points, 4)
            assert m.completion_rate == expected

    def test_blocked_count(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        assert m.blocked_tickets == 1

    def test_unassigned_count(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        assert m.unassigned_count == 3

    def test_contributors_have_expected_keys(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        for name, stats in m.contributors.items():
            assert set(stats.keys()) == {"completed_points", "in_progress_points", "ticket_count"}
            assert stats["ticket_count"] >= 1

    def test_cycle_time_only_uses_completed(self, voyager_sprint):
        m = compute_metrics(voyager_sprint)
        # Voyager sprint has completed tickets with both dates → cycle should compute
        assert m.avg_cycle_time_days is not None
        assert m.avg_cycle_time_days > 0


class TestCoastalMetrics:
    """Coastal Bank sprint: slow, low completion (2/12 done), high WIP."""

    def test_low_completion(self, coastal_sprint):
        m = compute_metrics(coastal_sprint)
        assert m.completion_rate < 0.5

    def test_blocked_present(self, coastal_sprint):
        m = compute_metrics(coastal_sprint)
        assert m.blocked_tickets >= 1

    def test_unestimated_tickets_present(self, coastal_sprint):
        df = coastal_sprint.df
        assert df["story_points"].isna().sum() >= 2


# ---------- Risk flag tests ----------

class TestRiskFlags:
    def test_blocked_flag_raised(self, voyager_sprint):
        flags = risk_flags(voyager_sprint.df)
        assert any("blocked" in f.lower() for f in flags)

    def test_unassigned_flag_raised(self, voyager_sprint):
        flags = risk_flags(voyager_sprint.df)
        assert any("unassigned" in f.lower() for f in flags)

    def test_low_completion_flag_for_coastal(self, coastal_sprint):
        flags = risk_flags(coastal_sprint.df)
        assert any("low" in f.lower() or "completion" in f.lower() for f in flags)

    def test_empty_df(self):
        empty = pd.DataFrame({
            "id": [], "title": [], "status": [], "assignee": [], "story_points": [],
        })
        flags = risk_flags(empty)
        assert any("no tickets" in f.lower() for f in flags)


# ---------- Sampling tests ----------

class TestSampling:
    def test_sample_buckets_present(self, voyager_sprint):
        samples = sample_tickets_for_narration(voyager_sprint.df)
        assert set(samples.keys()) == {
            "blocked", "high_priority_open", "biggest_completed", "unestimated",
        }

    def test_blocked_sample_contains_blocked_ticket(self, voyager_sprint):
        samples = sample_tickets_for_narration(voyager_sprint.df)
        assert len(samples["blocked"]) == 1
        assert samples["blocked"][0]["status"] == "blocked"

    def test_unestimated_sample(self, voyager_sprint):
        samples = sample_tickets_for_narration(voyager_sprint.df)
        # Voyager has 2 unestimated tickets
        assert len(samples["unestimated"]) >= 1

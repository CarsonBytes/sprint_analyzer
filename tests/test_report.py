"""
Report-rendering tests. Also includes a narrator test with a mocked LLM
so the test suite runs without an Anthropic API key.
"""
from pathlib import Path
from unittest.mock import MagicMock

from sprint_analyzer.parser import load_sprint_csv
from sprint_analyzer.metrics import compute_metrics, sample_tickets_for_narration
from sprint_analyzer.report import build_report, render_metrics_section
from sprint_analyzer.narrator import NarrationInput, generate_retrospective


DATA_DIR = Path(__file__).parent.parent / "data"


def _voyager():
    return load_sprint_csv((DATA_DIR / "sample_sprint_jira.csv").read_bytes())


class TestRender:
    def test_metrics_section_contains_headline_numbers(self):
        m = compute_metrics(_voyager())
        out = render_metrics_section(m)
        assert "Sprint 23" in out
        assert str(m.total_tickets) in out
        assert "Completion rate" in out

    def test_metrics_section_renders_risks(self):
        m = compute_metrics(_voyager())
        out = render_metrics_section(m)
        for risk in m.risks:
            assert risk in out

    def test_build_report_combines_sections(self):
        m = compute_metrics(_voyager())
        narrative = "## Executive summary\n- The sprint went OK."
        full = build_report(m, narrative)
        assert "Sprint retrospective" in full          # from metrics section
        assert "The sprint went OK" in full            # from narrative
        assert "---" in full                            # separator


class TestNarratorWithMockedLLM:
    def test_narrator_calls_llm_with_metrics_and_samples(self):
        sprint = _voyager()
        m = compute_metrics(sprint)
        samples = sample_tickets_for_narration(sprint.df)
        payload = NarrationInput(metrics=m, sample_tickets=samples)

        mock_llm = MagicMock(return_value="## Executive summary\n- Mocked.")
        out = generate_retrospective(payload, llm_fn=mock_llm)

        assert out.startswith("## Executive summary")
        # Verify the LLM call received both the metrics and the samples
        assert mock_llm.call_count == 1
        _system, user_prompt, _model = mock_llm.call_args[0]
        assert "METRICS" in user_prompt
        assert "SAMPLE TICKETS" in user_prompt
        # And that a known ticket ID from the data appears in the prompt
        assert "VOY-1211" in user_prompt   # the blocked ticket

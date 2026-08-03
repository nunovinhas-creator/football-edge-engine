"""
Testes unitários do relatório de qualidade/execução
(src/historical_dataset/report.py). Puramente sobre dados locais
(listas de dicts) — nenhuma chamada de rede/BSD API.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.historical_dataset.report import build_dataset_report, write_dataset_report


def _record(event_id, odds_home=2.0, home_team="A", away_team="B", date="2024-01-01", **extra):
    row = {
        "event_id": event_id,
        "home_team": home_team,
        "away_team": away_team,
        "date": date,
        "odds_home": odds_home,
        "odds_draw": 3.2 if odds_home is not None else None,
        "odds_away": 3.5 if odds_home is not None else None,
        "corners_home": None,
    }
    row.update(extra)
    return row


class TestBuildDatasetReport(unittest.TestCase):

    def test_total_games_matches_record_count(self):
        records = [_record(1), _record(2), _record(3)]
        report = build_dataset_report(records)
        self.assertEqual(report["total_games"], 3)

    def test_empty_dataset(self):
        report = build_dataset_report([])
        self.assertEqual(report["total_games"], 0)
        self.assertEqual(report["total_odds"], 0)
        self.assertEqual(report["duplicate_games"], 0)
        self.assertEqual(report["duplicate_odds"], 0)
        self.assertEqual(report["missing_values"], {"overall_pct": 0.0, "by_column": {}})

    def test_total_odds_counts_games_with_at_least_one_odds_field(self):
        records = [
            _record(1, odds_home=1.9),
            _record(2, odds_home=None, odds_draw=None, odds_away=None),
        ]
        report = build_dataset_report(records)
        self.assertEqual(report["total_odds"], 1)

    def test_duplicate_games_counts_repeated_event_id(self):
        records = [_record(1), _record(1), _record(2)]
        report = build_dataset_report(records)
        self.assertEqual(report["duplicate_games"], 1)

    def test_duplicate_odds_counts_same_teams_date_odds_under_different_event_id(self):
        records = [
            _record(1, odds_home=1.9),
            _record(2, odds_home=1.9),  # mesmo jogo, event_id diferente
            _record(3, odds_home=2.4),  # jogo distinto
        ]
        report = build_dataset_report(records)
        self.assertEqual(report["duplicate_odds"], 1)

    def test_duplicate_odds_ignores_rows_without_any_odds(self):
        records = [
            _record(1, odds_home=None, odds_draw=None, odds_away=None),
            _record(2, odds_home=None, odds_draw=None, odds_away=None),
        ]
        report = build_dataset_report(records)
        self.assertEqual(report["duplicate_odds"], 0)

    def test_missing_values_overall_and_by_column(self):
        records = [
            {"event_id": 1, "odds_home": 1.9, "corners_home": None},
            {"event_id": 2, "odds_home": None, "corners_home": None},
        ]
        report = build_dataset_report(records)
        missing = report["missing_values"]
        self.assertEqual(missing["by_column"]["corners_home"], 100.0)
        self.assertEqual(missing["by_column"]["odds_home"], 50.0)
        self.assertGreater(missing["overall_pct"], 0.0)

    def test_passthrough_fields_are_not_recomputed(self):
        report = build_dataset_report(
            [_record(1)],
            competition=38,
            season=2025,
            execution_time_seconds=12.5,
            api_requests=42,
            output_files={"csv": "data/historical/historical.csv", "parquet": None},
        )
        self.assertEqual(report["competition"], 38)
        self.assertEqual(report["season"], 2025)
        self.assertEqual(report["execution_time_seconds"], 12.5)
        self.assertEqual(report["api_requests"], 42)
        self.assertEqual(
            report["output_files"], {"csv": "data/historical/historical.csv", "parquet": None}
        )

    def test_no_algorithm_fields_present(self):
        """O relatório nunca deve conter odds/probabilidades calculadas (Edge/EV/Kelly/model_prob)."""
        report = build_dataset_report([_record(1)])
        forbidden = {"edge", "ev", "kelly", "model_prob", "probability"}
        self.assertTrue(forbidden.isdisjoint(report.keys()))


class TestWriteDatasetReport(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_writes_valid_json_creating_parent_dirs(self):
        report = build_dataset_report([_record(1)], competition=38, season=2025)
        path = Path(self.tmp_dir) / "nested" / "dataset_report.json"

        result_path = write_dataset_report(report, path)

        self.assertTrue(result_path.exists())
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["competition"], 38)
        self.assertEqual(loaded["total_games"], 1)


if __name__ == "__main__":
    unittest.main()

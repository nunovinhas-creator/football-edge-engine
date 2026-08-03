"""
Testes unitários de `src/evaluation/segments.py`, focados nas duas
segmentações novas (`segment_by_month`, `segment_by_confidence_range`) e
em `all_segments`, que deve incluir os segmentos "clássicos" já existentes
em `src.backtest.historical.segments` mais estes dois.
"""

import unittest

import pandas as pd

from src.evaluation.segments import all_segments, segment_by_confidence_range, segment_by_month


def _bets_df(rows):
    return pd.DataFrame(rows)


class TestSegmentByMonth(unittest.TestCase):

    def setUp(self):
        self.df = _bets_df(
            [
                {"date": "2024-01-05", "odd": 2.0, "probability": 0.55, "edge": 0.05, "ev": 0.10,
                 "kelly": 0.05, "stake": 1.0, "won": True, "profit": 1.0},
                {"date": "2024-01-20", "odd": 2.0, "probability": 0.55, "edge": 0.05, "ev": 0.10,
                 "kelly": 0.05, "stake": 1.0, "won": False, "profit": -1.0},
                {"date": "2024-02-10", "odd": 3.0, "probability": 0.40, "edge": 0.02, "ev": -0.05,
                 "kelly": 0.0, "stake": 1.0, "won": True, "profit": 2.0},
            ]
        )

    def test_groups_by_calendar_month_chronologically(self):
        result = segment_by_month(self.df)
        self.assertEqual(list(result["month"]), ["2024-01", "2024-02"])

    def test_n_bets_per_month_is_correct(self):
        result = segment_by_month(self.df)
        jan_row = result[result["month"] == "2024-01"].iloc[0]
        feb_row = result[result["month"] == "2024-02"].iloc[0]
        self.assertEqual(jan_row["n_bets"], 2)
        self.assertEqual(feb_row["n_bets"], 1)
        # lucro de janeiro = 1.0 - 1.0 = 0.0; fevereiro = 2.0
        self.assertAlmostEqual(jan_row["net_profit"], 0.0, places=4)
        self.assertAlmostEqual(feb_row["net_profit"], 2.0, places=4)

    def test_empty_frame_returns_empty(self):
        self.assertTrue(segment_by_month(pd.DataFrame()).empty)

    def test_missing_date_column_returns_empty(self):
        df = self.df.drop(columns=["date"])
        self.assertTrue(segment_by_month(df).empty)


class TestSegmentByConfidenceRange(unittest.TestCase):

    def setUp(self):
        self.df = _bets_df(
            [
                # probability 0.45 -> faixa "<50%"
                {"odd": 2.2, "probability": 0.45, "edge": 0.01, "ev": 0.0, "kelly": 0.0,
                 "stake": 1.0, "won": False, "profit": -1.0},
                # probability 0.92 -> faixa "90-100%"
                {"odd": 1.3, "probability": 0.92, "edge": 0.03, "ev": 0.05, "kelly": 0.02,
                 "stake": 1.0, "won": True, "profit": 0.3},
                # probability 0.93 -> também "90-100%"
                {"odd": 1.25, "probability": 0.93, "edge": 0.02, "ev": 0.04, "kelly": 0.02,
                 "stake": 1.0, "won": True, "profit": 0.25},
            ]
        )

    def test_low_and_high_confidence_bets_land_in_expected_buckets(self):
        result = segment_by_confidence_range(self.df)
        buckets = set(result["range"])
        self.assertIn("<50%", buckets)
        self.assertIn("90-100%", buckets)

    def test_high_confidence_bucket_aggregates_both_bets(self):
        result = segment_by_confidence_range(self.df)
        high_conf = result[result["range"] == "90-100%"].iloc[0]
        self.assertEqual(high_conf["n_bets"], 2)
        self.assertEqual(high_conf["wins"], 2)

    def test_empty_frame_returns_empty(self):
        self.assertTrue(segment_by_confidence_range(pd.DataFrame(columns=["probability"])).empty)


class TestAllSegments(unittest.TestCase):

    def test_includes_month_and_confidence_alongside_classic_segments(self):
        df = _bets_df(
            [
                {"date": "2024-01-05", "competition": "Liga A", "market": "HOME", "odd": 2.0,
                 "probability": 0.55, "edge": 0.05, "ev": 0.10, "kelly": 0.05,
                 "stake": 1.0, "won": True, "profit": 1.0},
                {"date": "2024-02-05", "competition": "Liga B", "market": "AWAY", "odd": 3.0,
                 "probability": 0.40, "edge": 0.02, "ev": -0.05, "kelly": 0.0,
                 "stake": 1.0, "won": False, "profit": -1.0},
            ]
        )
        result = all_segments(df)
        for expected in ["by_competition", "by_market", "by_odd_range", "by_month", "by_confidence_range"]:
            self.assertIn(expected, result)
            self.assertFalse(result[expected].empty)

    def test_empty_frame_returns_no_segments(self):
        self.assertEqual(all_segments(pd.DataFrame()), {})


if __name__ == "__main__":
    unittest.main()

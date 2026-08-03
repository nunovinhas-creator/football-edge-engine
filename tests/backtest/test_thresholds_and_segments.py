"""
Testes unitários da Threshold Analysis (thresholds.py) e das análises por
segmento (segments.py).
"""

import unittest

import pandas as pd

from src.backtest.historical.segments import (
    segment_by_column,
    segment_by_favorite_vs_underdog,
    segment_by_home_away,
    segment_by_odd_range,
)
from src.backtest.historical.thresholds import best_threshold, edge_threshold_analysis


def _bet_row(edge, ev, stake, profit, won, odd=2.0, **kwargs):
    row = {
        "odd": odd,
        "probability": 0.5,
        "edge": edge,
        "ev": ev,
        "kelly": 0.1,
        "stake": stake,
        "won": won,
        "profit": profit,
    }
    row.update(kwargs)
    return row


class TestEdgeThresholdAnalysis(unittest.TestCase):

    def setUp(self):
        # Edges: 0.5%, 4%, 6%, 12% (frações: 0.005, 0.04, 0.06, 0.12)
        self.df = pd.DataFrame(
            [
                _bet_row(edge=0.005, ev=0.01, stake=1.0, profit=-1.0, won=False),
                _bet_row(edge=0.04, ev=0.05, stake=1.0, profit=1.0, won=True),
                _bet_row(edge=0.06, ev=0.08, stake=1.0, profit=1.0, won=True),
                _bet_row(edge=0.12, ev=0.15, stake=1.0, profit=1.0, won=True),
            ]
        )

    def test_thresholds_progressively_filter_fewer_bets(self):
        table = edge_threshold_analysis(self.df, thresholds_pct=[1.0, 3.0, 5.0, 7.0, 10.0, 15.0])
        n_bets_by_threshold = dict(zip(table["threshold_pct"], table["n_bets"]))
        self.assertEqual(n_bets_by_threshold[1.0], 3)   # exclui a de 0.5%
        self.assertEqual(n_bets_by_threshold[3.0], 3)
        self.assertEqual(n_bets_by_threshold[5.0], 2)
        self.assertEqual(n_bets_by_threshold[7.0], 1)
        self.assertEqual(n_bets_by_threshold[10.0], 1)
        self.assertEqual(n_bets_by_threshold[15.0], 0)

    def test_hit_rate_at_highest_threshold_is_100_pct(self):
        table = edge_threshold_analysis(self.df, thresholds_pct=[10.0])
        self.assertEqual(table.iloc[0]["hit_rate_pct"], 100.0)

    def test_best_threshold_by_roi_ignores_low_sample_sizes(self):
        table = edge_threshold_analysis(self.df, thresholds_pct=[1.0, 3.0, 5.0, 7.0, 10.0, 15.0])
        best = best_threshold(table, by="roi_pct", min_bets=2)
        # Só os thresholds com >=2 apostas entram na disputa (1%, 3%, 5%).
        self.assertGreaterEqual(best["n_bets"], 2)

    def test_empty_frame_returns_empty_table(self):
        empty = pd.DataFrame(columns=["edge", "ev", "stake", "profit", "won"])
        table = edge_threshold_analysis(empty)
        self.assertTrue(table.empty)


class TestSegments(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            [
                _bet_row(edge=0.05, ev=0.1, stake=1.0, profit=1.0, won=True, odd=1.8,
                         competition="Liga X", home_or_away="HOME", is_favorite=True, market="HOME"),
                _bet_row(edge=0.05, ev=0.1, stake=1.0, profit=-1.0, won=False, odd=4.5,
                         competition="Liga X", home_or_away="AWAY", is_favorite=False, market="AWAY"),
                _bet_row(edge=0.05, ev=0.1, stake=1.0, profit=1.0, won=True, odd=1.6,
                         competition="Liga Y", home_or_away="HOME", is_favorite=True, market="HOME"),
            ]
        )

    def test_segment_by_competition_has_one_row_per_league(self):
        result = segment_by_column(self.df, "competition")
        self.assertEqual(set(result["competition"]), {"Liga X", "Liga Y"})
        self.assertEqual(int(result[result["competition"] == "Liga X"]["n_bets"].iloc[0]), 2)

    def test_segment_by_home_away(self):
        result = segment_by_home_away(self.df)
        self.assertEqual(set(result["home_or_away"]), {"HOME", "AWAY"})

    def test_segment_by_favorite_vs_underdog(self):
        result = segment_by_favorite_vs_underdog(self.df)
        counts = dict(zip(result["selection_type"], result["n_bets"]))
        self.assertEqual(counts["FAVORITE"], 2)
        self.assertEqual(counts["UNDERDOG"], 1)

    def test_segment_by_odd_range_buckets_correctly(self):
        result = segment_by_odd_range(self.df)
        ranges = set(result["range"])
        self.assertIn("1.50-2.00", ranges)
        self.assertIn("3.00-5.00", ranges)

    def test_missing_column_returns_empty(self):
        df_without_competition = self.df.drop(columns=["competition"])
        result = segment_by_column(df_without_competition, "competition")
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()

"""
Testes unitários de `src/evaluation/metrics.py`.

Confirma que os wrappers de conveniência (`avg_odd`, `avg_ev_pct`,
`avg_edge_pct`, `n_bets`, `full_summary`) produzem os mesmos valores que as
fórmulas oficiais de `src.backtest.historical.metrics` /
`src.backtest.historical.statistics` — nenhuma fórmula é reimplementada,
por isso estes testes existem para garantir que os valores permanecem
idênticos aos das implementações originais, não para validar as fórmulas
em si (já cobertas por `tests/backtest/test_metrics.py` e
`test_statistics.py`).
"""

import unittest

import pandas as pd

from src.backtest.historical.metrics import summary_metrics
from src.backtest.historical.statistics import statistical_summary
from src.evaluation.metrics import (
    avg_edge_pct,
    avg_ev_pct,
    avg_odd,
    full_summary,
    n_bets,
)


def _bets_df(rows):
    return pd.DataFrame(rows)


class TestConvenienceWrappers(unittest.TestCase):

    def setUp(self):
        self.df = _bets_df(
            [
                {
                    "odd": 2.0, "probability": 0.55, "edge": 0.05, "ev": 0.10, "kelly": 0.05,
                    "stake": 1.0, "won": True, "profit": 1.0,
                },
                {
                    "odd": 3.0, "probability": 0.30, "edge": 0.02, "ev": -0.10, "kelly": 0.0,
                    "stake": 1.0, "won": False, "profit": -1.0,
                },
            ]
        )

    def test_avg_odd_matches_mean(self):
        self.assertAlmostEqual(avg_odd(self.df), 2.5, places=4)

    def test_avg_ev_pct_matches_mean_times_100(self):
        # (0.10 + -0.10) / 2 * 100 = 0.0
        self.assertAlmostEqual(avg_ev_pct(self.df), 0.0, places=4)

    def test_avg_edge_pct_matches_mean_times_100(self):
        # (0.05 + 0.02) / 2 * 100 = 3.5
        self.assertAlmostEqual(avg_edge_pct(self.df), 3.5, places=4)

    def test_n_bets_matches_len(self):
        self.assertEqual(n_bets(self.df), 2)

    def test_empty_frame_wrappers_are_zero(self):
        empty = pd.DataFrame(columns=self.df.columns)
        self.assertEqual(avg_odd(empty), 0.0)
        self.assertEqual(avg_ev_pct(empty), 0.0)
        self.assertEqual(avg_edge_pct(empty), 0.0)
        self.assertEqual(n_bets(empty), 0)


class TestFullSummary(unittest.TestCase):
    """`full_summary` deve ser exatamente a união de `summary_metrics` + `statistical_summary`."""

    def setUp(self):
        self.placed = _bets_df(
            [
                {
                    "odd": 2.0, "probability": 0.55, "edge": 0.05, "ev": 0.10, "kelly": 0.05,
                    "stake": 1.0, "won": True, "profit": 1.0,
                },
                {
                    "odd": 3.0, "probability": 0.30, "edge": 0.02, "ev": -0.10, "kelly": 0.0,
                    "stake": 1.0, "won": False, "profit": -1.0,
                },
            ]
        )
        self.all_bets = pd.concat(
            [
                self.placed,
                _bets_df(
                    [
                        {
                            "odd": 1.8, "probability": 0.60, "edge": -0.01, "ev": -0.05, "kelly": 0.0,
                            "stake": 0.0, "won": True, "profit": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def test_full_summary_matches_union_of_underlying_functions(self):
        expected = {**summary_metrics(self.placed), **statistical_summary(self.all_bets, n_bins=10)}
        result = full_summary(self.placed, all_df=self.all_bets, n_bins=10)
        self.assertEqual(result, expected)

    def test_full_summary_defaults_stats_to_placed_when_all_df_omitted(self):
        expected = {**summary_metrics(self.placed), **statistical_summary(self.placed, n_bins=10)}
        result = full_summary(self.placed)
        self.assertEqual(result, expected)

    def test_full_summary_contains_all_required_metric_keys(self):
        result = full_summary(self.placed, all_df=self.all_bets)
        required = [
            "roi_pct", "yield_pct", "net_profit", "hit_rate_pct",
            "brier_score", "log_loss", "calibration_error",
            "avg_ev_pct", "avg_odd", "total_staked", "n_bets",
        ]
        for key in required:
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()

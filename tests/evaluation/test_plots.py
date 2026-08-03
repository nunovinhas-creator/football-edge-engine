"""
Testes unitários de `src/evaluation/plots.py`: as séries auxiliares
(`cumulative_roi_series`, `bankroll_series`) com valores calculados à mão,
e os geradores de gráficos (apenas verifica que produzem ficheiros PNG não
vazios, sem validar o conteúdo visual).
"""

import os
import shutil
import tempfile
import unittest

import pandas as pd

from src.evaluation.plots import (
    bankroll_series,
    cumulative_roi_series,
    generate_extra_plots,
    plot_bankroll,
    plot_cumulative_roi,
    plot_odds_distribution,
    plot_profit_by_competition,
    plot_reliability_diagram,
    plot_yield_by_market,
)


def _bets_df(rows):
    return pd.DataFrame(rows)


class TestCumulativeRoiSeries(unittest.TestCase):

    def test_known_values(self):
        # aposta 1: profit +1, stake 1 -> ROI acumulado = 100%
        # aposta 2: profit -1, stake 1 -> lucro acum. 0, stake acum. 2 -> ROI acumulado = 0%
        df = _bets_df([{"profit": 1.0, "stake": 1.0}, {"profit": -1.0, "stake": 1.0}])
        series = cumulative_roi_series(df)
        self.assertAlmostEqual(series.iloc[0], 100.0, places=4)
        self.assertAlmostEqual(series.iloc[1], 0.0, places=4)

    def test_empty_frame_returns_empty_series(self):
        self.assertTrue(cumulative_roi_series(pd.DataFrame()).empty)


class TestBankrollSeries(unittest.TestCase):

    def test_known_values(self):
        df = _bets_df([{"profit": 100.0}, {"profit": -50.0}])
        series = bankroll_series(df, starting_bankroll=1000.0)
        self.assertAlmostEqual(series.iloc[0], 1100.0, places=4)
        self.assertAlmostEqual(series.iloc[1], 1050.0, places=4)

    def test_empty_frame_returns_empty_series(self):
        self.assertTrue(bankroll_series(pd.DataFrame(), starting_bankroll=500.0).empty)


class TestPlotGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("matplotlib não está instalado")

        cls.tmp_dir = tempfile.mkdtemp(prefix="evaluation_plots_test_")
        cls.placed = _bets_df(
            [
                {"date": "2024-01-01", "competition": "Liga A", "market": "HOME", "odd": 2.0,
                 "probability": 0.55, "edge": 0.05, "ev": 0.10, "kelly": 0.05,
                 "stake": 1.0, "won": True, "profit": 1.0},
                {"date": "2024-01-05", "competition": "Liga B", "market": "AWAY", "odd": 3.0,
                 "probability": 0.35, "edge": 0.02, "ev": -0.05, "kelly": 0.0,
                 "stake": 1.0, "won": False, "profit": -1.0},
            ]
        )
        cls.all_bets = cls.placed

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _assert_written(self, path):
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_cumulative_roi(self):
        path = plot_cumulative_roi(self.placed, os.path.join(self.tmp_dir, "roi.png"))
        self._assert_written(path)

    def test_plot_bankroll(self):
        path = plot_bankroll(self.placed, os.path.join(self.tmp_dir, "bankroll.png"))
        self._assert_written(path)

    def test_plot_odds_distribution(self):
        path = plot_odds_distribution(self.all_bets, os.path.join(self.tmp_dir, "odds.png"))
        self._assert_written(path)

    def test_plot_profit_by_competition(self):
        path = plot_profit_by_competition(self.placed, os.path.join(self.tmp_dir, "profit_comp.png"))
        self._assert_written(path)

    def test_plot_yield_by_market(self):
        path = plot_yield_by_market(self.placed, os.path.join(self.tmp_dir, "yield_market.png"))
        self._assert_written(path)

    def test_plot_reliability_diagram(self):
        path = plot_reliability_diagram(self.all_bets, os.path.join(self.tmp_dir, "reliability.png"))
        self._assert_written(path)

    def test_generate_extra_plots_writes_all_six(self):
        output_dir = os.path.join(self.tmp_dir, "all")
        written = generate_extra_plots(self.placed, self.all_bets, output_dir)
        self.assertEqual(len(written), 6)
        for path in written.values():
            self._assert_written(path)

    def test_generate_extra_plots_handles_empty_frames(self):
        output_dir = os.path.join(self.tmp_dir, "empty")
        written = generate_extra_plots(pd.DataFrame(), pd.DataFrame(), output_dir)
        for path in written.values():
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()

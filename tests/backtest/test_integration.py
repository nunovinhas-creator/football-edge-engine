"""
Teste de integração do Backtesting Framework: executa o pipeline completo
(`BacktestEngine.run`) sobre um pequeno conjunto de jogos históricos
sintéticos (ver `fixtures.generate_sample_dataset`) e valida que todas as
peças — avaliação por aposta, métricas globais, métricas estatísticas,
segmentos, threshold analysis e exportação de relatórios — funcionam em
conjunto sem erros e produzem resultados coerentes.
"""

import math
import os
import shutil
import tempfile
import unittest

from src.backtest.historical import BacktestEngine, FlatStake, KellyStake
from tests.backtest.fixtures import generate_sample_dataset


class TestBacktestEngineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_sample_dataset(n_games=150, seed=7)
        cls.engine = BacktestEngine(staking=FlatStake(unit=1.0))
        cls.report = cls.engine.run(cls.dataset)

    def test_all_bets_evaluated(self):
        self.assertEqual(len(self.report.all_bets), 150)

    def test_placed_bets_are_subset_marked_bet_by_engine(self):
        self.assertLessEqual(len(self.report.placed_bets), len(self.report.all_bets))
        self.assertTrue((self.report.placed_bets["placed"]).all())
        self.assertGreater(len(self.report.placed_bets), 0)

    def test_global_metrics_have_expected_keys_and_sane_values(self):
        metrics = self.report.global_metrics
        for key in [
            "n_bets", "wins", "losses", "hit_rate_pct", "total_staked", "net_profit",
            "roi_pct", "yield_pct", "avg_odd", "avg_edge_pct", "avg_ev_pct",
            "avg_kelly_pct", "max_drawdown", "max_drawdown_pct", "profit_factor",
            "expectancy_per_bet",
        ]:
            self.assertIn(key, metrics)

        self.assertEqual(metrics["n_bets"], len(self.report.placed_bets))
        self.assertEqual(metrics["wins"] + metrics["losses"], metrics["n_bets"])
        self.assertGreaterEqual(metrics["hit_rate_pct"], 0.0)
        self.assertLessEqual(metrics["hit_rate_pct"], 100.0)
        self.assertLessEqual(metrics["max_drawdown"], 0.0)
        self.assertLessEqual(metrics["max_drawdown_pct"], 0.0)

    def test_statistical_metrics_are_within_valid_ranges(self):
        stats = self.report.statistical_metrics
        self.assertGreaterEqual(stats["brier_score"], 0.0)
        self.assertLessEqual(stats["brier_score"], 1.0)
        self.assertGreaterEqual(stats["log_loss"], 0.0)
        self.assertGreaterEqual(stats["calibration_error"], 0.0)
        self.assertLessEqual(stats["calibration_error"], 1.0)
        self.assertFalse(math.isnan(stats["brier_score"]))

    def test_edge_threshold_analysis_is_monotonically_non_increasing_in_bets(self):
        table = self.report.edge_thresholds
        n_bets = list(table["n_bets"])
        self.assertEqual(n_bets, sorted(n_bets, reverse=True))

    def test_ev_threshold_analysis_present(self):
        self.assertFalse(self.report.ev_thresholds.empty)
        self.assertIn("roi_pct", self.report.ev_thresholds.columns)

    def test_segments_cover_expected_dimensions(self):
        segments = self.report.segments
        for expected in ["by_competition", "by_market", "by_odd_range", "by_home_away"]:
            self.assertIn(expected, segments)
            self.assertFalse(segments[expected].empty)

    def test_calibration_curve_is_non_empty(self):
        self.assertFalse(self.report.calibration_curve.empty)

    def test_summary_table_is_single_row(self):
        summary = self.report.summary_table()
        self.assertEqual(len(summary), 1)

    def test_kelly_staking_strategy_also_runs_end_to_end(self):
        engine = BacktestEngine(staking=KellyStake(fraction=0.25, cap=0.05, bankroll=1000.0))
        report = engine.run(self.dataset)
        self.assertEqual(len(report.all_bets), 150)
        self.assertGreaterEqual(report.global_metrics["n_bets"], 0)

    def test_empty_dataset_does_not_crash(self):
        report = self.engine.run([])
        self.assertTrue(report.all_bets.empty)
        self.assertEqual(report.global_metrics["n_bets"], 0)


class TestBacktestReportExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_sample_dataset(n_games=60, seed=11)
        cls.report = BacktestEngine(staking=FlatStake(unit=1.0)).run(cls.dataset)
        cls.tmp_dir = tempfile.mkdtemp(prefix="backtest_report_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_to_csv_writes_all_expected_files(self):
        output_dir = os.path.join(self.tmp_dir, "csv")
        written = self.report.to_csv(output_dir)
        for path in written.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_to_excel_writes_a_single_workbook(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl não está instalado")

        path = os.path.join(self.tmp_dir, "report.xlsx")
        result_path = self.report.to_excel(path)
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

    def test_generate_all_plots_writes_png_files(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib não está instalado")

        output_dir = os.path.join(self.tmp_dir, "plots")
        written = self.report.generate_all_plots(output_dir)
        self.assertGreaterEqual(len(written), 4)
        for path in written.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()

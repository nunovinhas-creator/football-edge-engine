"""
Testes unitários de `src/evaluation/compare.py`: comparação objetiva entre
dois ficheiros/DataFrames de apostas avaliadas (ver `evaluator.evaluate_bets`
/ `BacktestReport.all_bets`).
"""

import os
import shutil
import tempfile
import unittest

import pandas as pd

from src.evaluation.compare import compare_backtests, load_bets


def _bets_df(rows):
    return pd.DataFrame(rows)


class TestCompareBacktests(unittest.TestCase):
    """
    Modelo A: 2 apostas colocadas, ambas ganhas com boa calibração
    (probability ~= frequência real).
    Modelo B: 2 apostas colocadas, ambas perdidas, com sobreconfiança
    (probability alta mas resultado errado -> Brier/ECE piores).
    """

    def setUp(self):
        self.model_a = _bets_df(
            [
                {"odd": 2.0, "probability": 0.55, "edge": 0.05, "ev": 0.10, "kelly": 0.05,
                 "stake": 1.0, "won": True, "profit": 1.0, "placed": True},
                {"odd": 1.8, "probability": 0.60, "edge": 0.04, "ev": 0.08, "kelly": 0.04,
                 "stake": 1.0, "won": True, "profit": 0.8, "placed": True},
            ]
        )
        self.model_b = _bets_df(
            [
                {"odd": 2.0, "probability": 0.90, "edge": 0.05, "ev": 0.10, "kelly": 0.05,
                 "stake": 1.0, "won": False, "profit": -1.0, "placed": True},
                {"odd": 1.8, "probability": 0.85, "edge": 0.04, "ev": 0.08, "kelly": 0.04,
                 "stake": 1.0, "won": False, "profit": -1.0, "placed": True},
            ]
        )

    def test_model_a_wins_on_profit_roi_brier_and_calibration(self):
        result = compare_backtests(self.model_a, self.model_b, label_a="A", label_b="B")
        self.assertEqual(result.winner_by_profit, "A")
        self.assertEqual(result.winner_by_roi, "A")
        self.assertEqual(result.winner_by_brier, "A")
        self.assertEqual(result.winner_by_calibration, "A")

    def test_summary_contains_all_four_answers(self):
        result = compare_backtests(self.model_a, self.model_b, label_a="A", label_b="B")
        summary = result.summary()
        self.assertEqual(
            set(summary.keys()),
            {
                "qual_modelo_ganhou_mais",
                "qual_teve_maior_roi",
                "qual_teve_menor_brier",
                "qual_teve_melhor_calibracao",
            },
        )
        self.assertTrue(all(v == "A" for v in summary.values()))

    def test_tie_is_reported_as_empate(self):
        result = compare_backtests(self.model_a, self.model_a.copy(), label_a="A", label_b="A2")
        self.assertEqual(result.winner_by_profit, "EMPATE")

    def test_comparison_table_has_both_labels_as_columns(self):
        result = compare_backtests(self.model_a, self.model_b, label_a="A", label_b="B")
        table = result.comparison_table()
        self.assertIn("A", table.columns)
        self.assertIn("B", table.columns)

    def test_to_markdown_mentions_both_labels(self):
        result = compare_backtests(self.model_a, self.model_b, label_a="A", label_b="B")
        markdown = result.to_markdown()
        self.assertIn("A", markdown)
        self.assertIn("B", markdown)


class TestLoadBetsFromCsv(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="evaluation_compare_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_bets_from_csv_path_parses_date(self):
        df = _bets_df(
            [{"date": "2024-01-05", "odd": 2.0, "probability": 0.5, "won": True, "profit": 1.0, "stake": 1.0}]
        )
        path = os.path.join(self.tmp_dir, "bets.csv")
        df.to_csv(path, index=False)

        loaded = load_bets(path)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["date"]))

    def test_load_bets_accepts_dataframe_directly(self):
        df = _bets_df([{"odd": 2.0, "probability": 0.5, "won": True, "profit": 1.0, "stake": 1.0}])
        loaded = load_bets(df)
        pd.testing.assert_frame_equal(loaded, df)


if __name__ == "__main__":
    unittest.main()

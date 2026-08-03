"""
Testes unitários da ponte com o Backtesting Framework
(src/historical_dataset/backtest_bridge.py).

Cobre: seleção da coluna de odds certa por mercado, as três formas
aceites de fornecer `model_prob` (escalar, nome de coluna, Series/lista,
callable), remoção de jogos sem odd/resultado para o mercado escolhido, e
a integração de ponta a ponta com
`src.backtest.historical.dataset.load_historical_dataset` (sem tocar em
nenhuma fórmula do motor).
"""

import unittest

import pandas as pd

from src.backtest.historical.dataset import load_historical_dataset
from src.historical_dataset.backtest_bridge import to_backtest_frame


def _records():
    return [
        {
            "event_id": 1, "date": "2024-01-01", "competition": "Premier League",
            "home_team": "A", "away_team": "B",
            "home_score": 2, "away_score": 1,
            "odds_home": 1.9, "odds_draw": 3.4, "odds_away": 4.1,
            "odds_over_2_5": 1.85, "odds_under_2_5": 1.95,
            "odds_btts_yes": 1.7, "odds_btts_no": 2.1,
        },
        {
            "event_id": 2, "date": "2024-01-08", "competition": "Premier League",
            "home_team": "C", "away_team": "D",
            "home_score": 0, "away_score": 0,
            "odds_home": 2.5, "odds_draw": 3.1, "odds_away": 2.9,
            "odds_over_2_5": 2.0, "odds_under_2_5": 1.8,
            "odds_btts_yes": 2.2, "odds_btts_no": 1.6,
        },
    ]


class TestToBacktestFrame(unittest.TestCase):

    def test_home_market_selects_odds_home_column(self):
        df = to_backtest_frame(_records(), "HOME", model_prob=0.5)

        self.assertListEqual(list(df["odd"]), [1.9, 2.5])
        self.assertListEqual(list(df["market"]), ["HOME", "HOME"])
        self.assertListEqual(list(df["home_goals"]), [2, 0])
        self.assertListEqual(list(df["away_goals"]), [1, 0])

    def test_over_2_5_market_selects_correct_column(self):
        df = to_backtest_frame(_records(), "over_2.5", model_prob=0.5)
        self.assertListEqual(list(df["odd"]), [1.85, 2.0])

    def test_btts_market_defaults_to_btts_yes(self):
        df = to_backtest_frame(_records(), "BTTS", model_prob=0.5)
        self.assertListEqual(list(df["odd"]), [1.7, 2.2])

    def test_unsupported_market_raises_value_error(self):
        with self.assertRaises(ValueError):
            to_backtest_frame(_records(), "HANDICAP_-1", model_prob=0.5)

    def test_model_prob_as_scalar_applies_to_all_rows(self):
        df = to_backtest_frame(_records(), "HOME", model_prob=0.42)
        self.assertListEqual(list(df["model_prob"]), [0.42, 0.42])

    def test_model_prob_as_series_aligned_by_position(self):
        df = to_backtest_frame(_records(), "HOME", model_prob=pd.Series([0.6, 0.3]))
        self.assertListEqual(list(df["model_prob"]), [0.6, 0.3])

    def test_model_prob_as_existing_column_name(self):
        records = _records()
        records[0]["my_model_prob"] = 0.55
        records[1]["my_model_prob"] = 0.44
        df = to_backtest_frame(records, "HOME", model_prob="my_model_prob")
        self.assertListEqual(list(df["model_prob"]), [0.55, 0.44])

    def test_model_prob_as_callable(self):
        df = to_backtest_frame(_records(), "HOME", model_prob=lambda row: 1.0 / row["odds_home"])
        self.assertAlmostEqual(df.loc[0, "model_prob"], 1.0 / 1.9)

    def test_rows_missing_market_odd_are_dropped(self):
        records = _records()
        records[1]["odds_home"] = None
        df = to_backtest_frame(records, "HOME", model_prob=0.5)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "home_team"], "A")

    def test_empty_input_returns_empty_frame_with_expected_columns(self):
        df = to_backtest_frame([], "HOME", model_prob=0.5)
        self.assertTrue(df.empty)
        for col in ("date", "competition", "home_team", "away_team", "market", "odd", "model_prob"):
            self.assertIn(col, df.columns)

    def test_engine_decision_and_result_passthrough_when_provided(self):
        df = to_backtest_frame(
            _records(), "HOME", model_prob=0.5,
            engine_decision="BET", result=["WIN", "LOSS"],
        )
        self.assertListEqual(list(df["engine_decision"]), ["BET", "BET"])
        self.assertListEqual(list(df["result"]), ["WIN", "LOSS"])


class TestIntegrationWithLoadHistoricalDataset(unittest.TestCase):
    """
    Garante que a ponte produz algo que `load_historical_dataset` (o
    Backtesting Framework já existente) consegue consumir sem alterações,
    incluindo a derivação automática de `result` a partir dos golos.
    """

    def test_end_to_end_through_load_historical_dataset(self):
        bridge_df = to_backtest_frame(_records(), "HOME", model_prob=0.55)

        loaded = load_historical_dataset(bridge_df)

        self.assertEqual(len(loaded), 2)
        self.assertIn("result", loaded.columns)
        self.assertIn("engine_decision", loaded.columns)
        # Jogo 1: casa venceu 2-1 -> mercado HOME "WIN"; jogo 2: 0-0 -> "LOSS"
        self.assertEqual(list(loaded["result"]), ["WIN", "LOSS"])


if __name__ == "__main__":
    unittest.main()

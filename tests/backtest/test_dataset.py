"""
Testes unitários do carregamento de jogos históricos
(src/backtest/historical/dataset.py).

Cobrem: derivação do resultado do mercado a partir do resultado final do
jogo (`infer_market_result`), preenchimento de `engine_decision` via o
`DecisionEngine` real, normalização de aliases PT/EN, filtragem por
data/competição/mercado, e validação de colunas obrigatórias em falta.
"""

import unittest

import pandas as pd

from src.backtest.historical.dataset import (
    filter_dataset,
    infer_market_result,
    load_historical_dataset,
)


class TestInferMarketResult(unittest.TestCase):

    def test_home_market(self):
        self.assertEqual(infer_market_result("HOME", 2, 0), "WIN")
        self.assertEqual(infer_market_result("HOME", 0, 2), "LOSS")
        self.assertEqual(infer_market_result("HOME", 1, 1), "LOSS")

    def test_away_market(self):
        self.assertEqual(infer_market_result("AWAY", 0, 2), "WIN")
        self.assertEqual(infer_market_result("AWAY", 2, 0), "LOSS")

    def test_draw_market(self):
        self.assertEqual(infer_market_result("DRAW", 1, 1), "WIN")
        self.assertEqual(infer_market_result("DRAW", 2, 1), "LOSS")

    def test_over_under_market(self):
        self.assertEqual(infer_market_result("OVER_2.5", 2, 1), "WIN")
        self.assertEqual(infer_market_result("OVER_2.5", 1, 1), "LOSS")
        self.assertEqual(infer_market_result("UNDER_2.5", 1, 1), "WIN")
        self.assertEqual(infer_market_result("UNDER_2.5", 2, 1), "LOSS")

    def test_btts_market(self):
        self.assertEqual(infer_market_result("BTTS_YES", 1, 1), "WIN")
        self.assertEqual(infer_market_result("BTTS_YES", 1, 0), "LOSS")
        self.assertEqual(infer_market_result("BTTS_NO", 1, 0), "WIN")
        self.assertEqual(infer_market_result("BTTS_NO", 1, 1), "LOSS")

    def test_unknown_market_raises(self):
        with self.assertRaises(ValueError):
            infer_market_result("HANDICAP_-1", 2, 0)


class TestLoadHistoricalDataset(unittest.TestCase):

    def _rows(self):
        return [
            {
                "date": "2020-01-01",
                "competition": "Premier League",
                "home_team": "Team A",
                "away_team": "Team B",
                "market": "HOME",
                "odd": 1.80,
                "model_prob": 0.60,
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "date": "2020-02-01",
                "competition": "La Liga",
                "home_team": "Team C",
                "away_team": "Team D",
                "market": "OVER_2.5",
                "odd": 2.00,
                "model_prob": 0.55,
                "home_goals": 1,
                "away_goals": 1,
            },
        ]

    def test_missing_required_column_raises(self):
        rows = [{"date": "2020-01-01", "home_team": "A", "away_team": "B"}]
        with self.assertRaises(KeyError):
            load_historical_dataset(rows)

    def test_derives_match_and_result_and_decision(self):
        df = load_historical_dataset(self._rows())
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "match"], "Team A vs Team B")
        self.assertEqual(df.loc[0, "result"], "WIN")
        self.assertEqual(df.loc[1, "result"], "LOSS")  # OVER_2.5 with 1-1 (2 goals total) misses
        self.assertIn("engine_decision", df.columns)
        self.assertTrue(df["engine_decision"].notna().all())

    def test_portuguese_aliases_are_accepted(self):
        rows = [
            {
                "data": "2020-01-01",
                "competicao": "Premier League",
                "equipa_casa": "Team A",
                "equipa_visitante": "Team B",
                "mercado": "HOME",
                "odd": 1.80,
                "probabilidade": 0.60,
                "golos_casa": 2,
                "golos_fora": 0,
            }
        ]
        df = load_historical_dataset(rows)
        self.assertEqual(df.loc[0, "home_team"], "Team A")
        self.assertEqual(df.loc[0, "result"], "WIN")

    def test_explicit_result_and_decision_are_respected(self):
        rows = self._rows()
        rows[0]["result"] = "LOSS"
        rows[0]["engine_decision"] = "PASS ❄️"
        df = load_historical_dataset(rows)
        self.assertEqual(df.loc[0, "result"], "LOSS")
        self.assertEqual(df.loc[0, "engine_decision"], "PASS ❄️")

    def test_empty_source_returns_empty_dataframe(self):
        df = load_historical_dataset([])
        self.assertTrue(df.empty)

    def test_dataframe_source_is_accepted(self):
        df_in = pd.DataFrame(self._rows())
        df = load_historical_dataset(df_in)
        self.assertEqual(len(df), 2)


class TestFilterDataset(unittest.TestCase):

    def setUp(self):
        rows = [
            {
                "date": "2020-01-01", "competition": "Premier League",
                "home_team": "A", "away_team": "B", "market": "HOME",
                "odd": 1.80, "model_prob": 0.60, "home_goals": 2, "away_goals": 0,
            },
            {
                "date": "2021-06-01", "competition": "La Liga",
                "home_team": "C", "away_team": "D", "market": "OVER_2.5",
                "odd": 2.00, "model_prob": 0.55, "home_goals": 3, "away_goals": 1,
            },
        ]
        self.df = load_historical_dataset(rows)

    def test_filter_by_date_range(self):
        filtered = filter_dataset(self.df, start_date="2021-01-01")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.loc[0, "competition"], "La Liga")

    def test_filter_by_competition_is_case_insensitive(self):
        filtered = filter_dataset(self.df, competition="premier league")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.loc[0, "home_team"], "A")

    def test_filter_by_market(self):
        filtered = filter_dataset(self.df, market="over_2.5")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.loc[0, "competition"], "La Liga")

    def test_no_filters_returns_full_dataset(self):
        filtered = filter_dataset(self.df)
        self.assertEqual(len(filtered), 2)

    def test_empty_dataframe_does_not_crash(self):
        empty = load_historical_dataset([])
        self.assertTrue(filter_dataset(empty, competition="X").empty)


if __name__ == "__main__":
    unittest.main()

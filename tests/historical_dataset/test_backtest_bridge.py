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
from src.historical_dataset.backtest_bridge import (
    derive_h2h,
    model_probabilities_from_dixon_coles,
    to_backtest_frame,
)


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

    def test_timezone_aware_dates_from_bsd_api_are_normalized_to_naive(self):
        # A BSD API devolve `event_date` em ISO 8601 UTC (ex.
        # "2024-08-10T15:00:00Z" — ver docs/07_historical_dataset_builder.md,
        # "Limitações", ponto 7). `pandas.ExcelWriter`/openpyxl (usados por
        # BacktestReport.to_excel, já existente) rejeitam datetimes com
        # fuso horário — a data tem de chegar "naive" a partir daqui.
        records = _records()
        records[0]["date"] = "2024-08-10T15:00:00Z"
        records[1]["date"] = "2024-08-17T18:00:00+00:00"

        df = to_backtest_frame(records, "HOME", model_prob=0.5)

        self.assertIsNone(df["date"].dt.tz)
        self.assertEqual(df["date"].iloc[0], pd.Timestamp("2024-08-10 15:00:00"))
        self.assertEqual(df["date"].iloc[1], pd.Timestamp("2024-08-17 18:00:00"))

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


class TestDeriveH2H(unittest.TestCase):
    """
    `derive_h2h` constrói o input `head_to_head` que
    `src.engine.lambda_estimator.estimate_lambda` já esperava (mesmo
    formato de `EventCollector.get_matches()` para jogos futuros), a
    partir de confrontos diretos ANTERIORES já presentes no dataset do
    Historical Dataset Builder — sem pedir nada à BSD API e sem fuga de
    informação (só entram jogos com data estritamente anterior).
    """

    def _df(self):
        return pd.DataFrame([
            {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1,
             "date": pd.Timestamp("2023-01-01")},
            {"home_team": "B", "away_team": "A", "home_score": 0, "away_score": 0,
             "date": pd.Timestamp("2023-06-01")},
            {"home_team": "C", "away_team": "D", "home_score": 5, "away_score": 0,
             "date": pd.Timestamp("2023-03-01")},
        ])

    def test_no_prior_meetings_returns_empty_dict(self):
        self.assertEqual(derive_h2h(self._df(), "A", "B", pd.Timestamp("2022-01-01")), {})

    def test_unrelated_pair_is_ignored(self):
        h2h = derive_h2h(self._df(), "A", "B", pd.Timestamp("2023-12-31"))
        self.assertEqual(h2h["total_matches"], 2)

    def test_future_meetings_are_excluded_no_lookahead(self):
        # Só o jogo de 2023-01-01 é anterior a 2023-03-01.
        h2h = derive_h2h(self._df(), "A", "B", pd.Timestamp("2023-03-15"))
        self.assertEqual(h2h["total_matches"], 1)

    def test_reversed_venue_is_reoriented_to_upcoming_fixture(self):
        # No 2º confronto (2023-06-01), B jogou em casa e empatou 0-0 com A.
        # Do ponto de vista do próximo jogo (A em casa, B fora), isso conta
        # como um golo de A e um golo de B, não "B em casa marcou 0".
        h2h = derive_h2h(self._df(), "A", "B", pd.Timestamp("2023-12-31"))
        self.assertEqual(h2h["home_goals"], 2.0)  # 2 (jogo 1) + 0 (jogo 2, reorientado)
        self.assertEqual(h2h["away_goals"], 1.0)  # 1 (jogo 1) + 0 (jogo 2, reorientado)
        self.assertEqual(len(h2h["recent_matches"]), 2)

    def test_missing_date_returns_empty_dict(self):
        self.assertEqual(derive_h2h(self._df(), "A", "B", None), {})
        self.assertEqual(derive_h2h(self._df(), "A", "B", pd.NaT), {})


class TestModelProbabilitiesFromDixonColes(unittest.TestCase):
    """
    `model_probabilities_from_dixon_coles` não recalcula nem substitui o
    Dixon-Coles: só prepara o `head_to_head` (via `derive_h2h`) e delega em
    `src.engine.lambda_estimator.estimate_lambda` +
    `src.engine.value.estimate_pregame_probabilities`, exatamente os
    módulos já usados em produção para jogos futuros.
    """

    def test_empty_records_returns_empty_dict(self):
        self.assertEqual(model_probabilities_from_dixon_coles([]), {})

    def test_each_event_gets_a_probability_summing_to_one(self):
        records = [
            {"event_id": 1, "date": "2023-01-01", "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 1},
            {"event_id": 2, "date": "2023-06-01", "home_team": "B", "away_team": "A",
             "home_score": 0, "away_score": 0},
        ]
        probs = model_probabilities_from_dixon_coles(records)

        self.assertEqual(set(probs.keys()), {1, 2})
        for event_probs in probs.values():
            self.assertAlmostEqual(sum(event_probs.values()), 1.0, places=6)
            for key in ("home", "draw", "away"):
                self.assertIn(key, event_probs)
                self.assertGreater(event_probs[key], 0.0)

    def test_first_meeting_without_h2h_uses_league_prior_home_advantage(self):
        # Sem confrontos diretos anteriores, `derive_h2h` devolve {} e
        # `estimate_lambda({})` cai no mesmo prior de liga (vantagem de
        # casa fixa) já usado por `pregame_lambda.py` para jogos futuros
        # sem H2H — não inventa nenhum número novo aqui.
        records = [
            {"event_id": 1, "date": "2024-01-01", "home_team": "X", "away_team": "Y",
             "home_score": 1, "away_score": 1},
        ]
        probs = model_probabilities_from_dixon_coles(records)
        self.assertGreater(probs[1]["home"], probs[1]["away"])

    def test_result_can_feed_to_backtest_frame_as_model_prob(self):
        # Uso real pretendido: `model_probabilities_from_dixon_coles` +
        # `to_backtest_frame(..., model_prob=callable)` sem tocar em
        # nenhuma fórmula do motor de previsão.
        records = [
            {"event_id": 1, "date": "2023-01-01", "competition": "X",
             "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1,
             "odds_home": 1.9, "odds_draw": 3.4, "odds_away": 4.1},
            {"event_id": 2, "date": "2023-06-01", "competition": "X",
             "home_team": "B", "away_team": "A", "home_score": 0, "away_score": 0,
             "odds_home": 2.5, "odds_draw": 3.1, "odds_away": 2.9},
        ]
        probs = model_probabilities_from_dixon_coles(records)
        df = to_backtest_frame(
            records, "HOME",
            model_prob=lambda row: probs[row["event_id"]]["home"],
        )
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["model_prob"]), [probs[1]["home"], probs[2]["home"]])
        for p in df["model_prob"]:
            self.assertTrue(0.0 < p <= 1.0)


if __name__ == "__main__":
    unittest.main()

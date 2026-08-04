"""
Testes da Melhoria #8 (auditoria matemática): propagação da confiança do
modelo (`LambdaEstimate.tier` / `LambdaEstimate.effective_sample_size`) até
ao Evaluation Framework.

Cobre:
    - retrocompatibilidade: dados/ficheiros antigos, sem estes campos,
      continuam válidos e não geram erros em nenhum ponto do pipeline;
    - propagação correta: LambdaEstimate -> to_backtest_frame ->
      HistoricalBet -> EvaluatedBet -> segmentos;
    - segmentação correta: ROI/Yield/Brier Score/Log Loss/Nº de apostas por
      `lambda_tier`, `model_confidence` e faixa de `effective_sample_size`;
    - ausência de regressões: nenhuma fórmula (Dixon-Coles, Edge, EV,
      Kelly) nem nenhuma decisão do motor é alterada por estes metadados.

Nenhum teste aqui recalcula uma fórmula própria — usa sempre as
implementações oficiais já existentes como referência.
"""

import os
import unittest

import pandas as pd

from src.backtest.historical.dataset import load_historical_dataset
from src.backtest.historical.engine import BacktestEngine
from src.backtest.historical.evaluator import evaluate_bet, evaluate_bets
from src.backtest.historical.models import HistoricalBet
from src.backtest.historical.staking import FlatStake
from src.engine.lambda_estimator import (
    SHRINKAGE_K,
    classify_model_confidence,
    estimate_lambda_detailed,
)
from src.evaluation import evaluate
from src.evaluation.segments import (
    all_confidence_segments,
    segment_by_effective_sample_size_range,
    segment_by_lambda_tier,
    segment_by_model_confidence,
)
from src.historical_dataset.backtest_bridge import (
    derive_h2h,
    lambda_confidence_from_dixon_coles,
    model_probabilities_from_dixon_coles,
    to_backtest_frame,
)
from src.historical_dataset.storage import to_dataframe
from tests.backtest.fixtures import generate_sample_dataset

SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "examples", "backtest", "sample_real_games.csv"
)


def _records():
    return [
        {
            "event_id": 1, "date": "2023-01-01", "competition": "X",
            "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1,
            "odds_home": 1.9, "odds_draw": 3.4, "odds_away": 4.1,
        },
        {
            "event_id": 2, "date": "2023-06-01", "competition": "X",
            "home_team": "B", "away_team": "A", "home_score": 0, "away_score": 0,
            "odds_home": 2.5, "odds_draw": 3.1, "odds_away": 2.9,
        },
    ]


# --------------------------------------------------------------------------
# classify_model_confidence
# --------------------------------------------------------------------------

class TestClassifyModelConfidence(unittest.TestCase):

    def test_weak_tier_is_never_high_even_with_huge_sample(self):
        label = classify_model_confidence("avg_total_goals_or_prior", 1000.0)
        self.assertNotEqual(label, "HIGH")

    def test_strong_tier_with_large_sample_is_high(self):
        label = classify_model_confidence("recent_matches", 2 * SHRINKAGE_K + 1)
        self.assertEqual(label, "HIGH")

    def test_strong_tier_with_tiny_sample_is_low(self):
        self.assertEqual(classify_model_confidence("h2h_goal_totals", 0.5), "LOW")

    def test_zero_or_negative_sample_is_low(self):
        self.assertEqual(classify_model_confidence("recent_matches", 0.0), "LOW")
        self.assertEqual(classify_model_confidence("recent_matches", -5.0), "LOW")

    def test_none_or_nan_sample_does_not_raise(self):
        self.assertEqual(classify_model_confidence("recent_matches", None), "LOW")
        self.assertEqual(classify_model_confidence("recent_matches", float("nan")), "LOW")

    def test_medium_band_at_shrinkage_k(self):
        self.assertEqual(classify_model_confidence("avg_total_goals_or_prior", SHRINKAGE_K), "MEDIUM")


# --------------------------------------------------------------------------
# HistoricalBet: retrocompatibilidade + parsing dos novos campos
# --------------------------------------------------------------------------

class TestHistoricalBetConfidenceFields(unittest.TestCase):

    def _base_row(self, **extra):
        row = {
            "jogo": "A vs B", "data": "2024-01-01", "mercado": "HOME",
            "odd": 2.0, "probabilidade": 0.5, "decisao": "BET", "resultado": "WIN",
        }
        row.update(extra)
        return row

    def test_old_dict_without_confidence_fields_stays_valid(self):
        bet = HistoricalBet.from_dict(self._base_row())
        self.assertIsNone(bet.model_confidence)
        self.assertIsNone(bet.lambda_tier)
        self.assertIsNone(bet.effective_sample_size)

    def test_new_fields_are_parsed_when_present(self):
        bet = HistoricalBet.from_dict(self._base_row(
            model_confidence="HIGH", lambda_tier="recent_matches", effective_sample_size="6.5",
        ))
        self.assertEqual(bet.model_confidence, "HIGH")
        self.assertEqual(bet.lambda_tier, "recent_matches")
        self.assertEqual(bet.effective_sample_size, 6.5)

    def test_nan_values_are_treated_as_missing_not_as_data(self):
        bet = HistoricalBet.from_dict(self._base_row(
            lambda_tier=float("nan"), effective_sample_size=float("nan"),
        ))
        self.assertIsNone(bet.lambda_tier)
        self.assertIsNone(bet.effective_sample_size)

    def test_new_fields_do_not_leak_into_extra(self):
        bet = HistoricalBet.from_dict(self._base_row(model_confidence="LOW"))
        self.assertNotIn("model_confidence", bet.extra)

    def test_required_fields_still_enforced(self):
        with self.assertRaises(KeyError):
            HistoricalBet.from_dict({"jogo": "A vs B", "model_confidence": "HIGH"})


# --------------------------------------------------------------------------
# evaluate_bet / EvaluatedBet: propagação sem tocar em edge/ev/kelly
# --------------------------------------------------------------------------

class TestEvaluatedBetPropagation(unittest.TestCase):

    def _bet(self, **kwargs):
        defaults = dict(
            match="A vs B", date="2024-01-01", market="HOME", odd=2.0,
            model_prob=0.55, engine_decision="BET", result="WIN",
        )
        defaults.update(kwargs)
        return HistoricalBet(**defaults)

    def test_confidence_metadata_flows_through_unchanged(self):
        bet = self._bet(model_confidence="HIGH", lambda_tier="recent_matches", effective_sample_size=9.0)
        evaluated = evaluate_bet(bet)
        self.assertEqual(evaluated.model_confidence, "HIGH")
        self.assertEqual(evaluated.lambda_tier, "recent_matches")
        self.assertEqual(evaluated.effective_sample_size, 9.0)
        self.assertEqual(evaluated.to_dict()["lambda_tier"], "recent_matches")

    def test_absence_of_metadata_is_none_not_an_error(self):
        evaluated = evaluate_bet(self._bet())
        self.assertIsNone(evaluated.model_confidence)
        self.assertIsNone(evaluated.lambda_tier)
        self.assertIsNone(evaluated.effective_sample_size)

    def test_metadata_never_affects_edge_ev_kelly(self):
        with_meta = evaluate_bet(self._bet(
            model_confidence="HIGH", lambda_tier="recent_matches", effective_sample_size=9.0,
        ))
        without_meta = evaluate_bet(self._bet())
        self.assertEqual(with_meta.edge, without_meta.edge)
        self.assertEqual(with_meta.ev, without_meta.ev)
        self.assertEqual(with_meta.kelly, without_meta.kelly)
        self.assertEqual(with_meta.stake, without_meta.stake)

    def test_evaluate_bets_handles_mixed_metadata_without_crashing(self):
        bets = [
            self._bet(match="A vs B"),
            self._bet(
                match="C vs D", model_confidence="HIGH",
                lambda_tier="recent_matches", effective_sample_size=7.0,
            ),
        ]
        df = evaluate_bets(bets, staking=FlatStake(unit=1.0))
        self.assertEqual(len(df), 2)
        row_without = df[df["match"] == "A vs B"].iloc[0]
        row_with = df[df["match"] == "C vs D"].iloc[0]
        self.assertTrue(pd.isna(row_without["lambda_tier"]))
        self.assertEqual(row_with["lambda_tier"], "recent_matches")


# --------------------------------------------------------------------------
# backtest_bridge: lambda_confidence_from_dixon_coles + to_backtest_frame
# --------------------------------------------------------------------------

class TestLambdaConfidenceFromDixonColes(unittest.TestCase):

    def test_empty_records_returns_empty_dict(self):
        self.assertEqual(lambda_confidence_from_dixon_coles([]), {})

    def test_matches_estimate_lambda_detailed_for_each_event(self):
        records = _records()
        confidence = lambda_confidence_from_dixon_coles(records)

        df = to_dataframe(records)
        df["date"] = pd.to_datetime(df["date"])
        for _, row in df.iterrows():
            h2h = derive_h2h(df, row["home_team"], row["away_team"], row["date"])
            expected = estimate_lambda_detailed(h2h)
            actual = confidence[row["event_id"]]
            self.assertEqual(actual["lambda_tier"], expected.tier)
            self.assertAlmostEqual(actual["effective_sample_size"], expected.effective_sample_size, places=6)
            self.assertEqual(
                actual["model_confidence"],
                classify_model_confidence(expected.tier, expected.effective_sample_size),
            )

    def test_does_not_alter_model_probabilities_from_dixon_coles(self):
        records = _records()
        probs_before = model_probabilities_from_dixon_coles(records)
        lambda_confidence_from_dixon_coles(records)  # nunca deve ter efeitos colaterais
        probs_after = model_probabilities_from_dixon_coles(records)
        self.assertEqual(probs_before, probs_after)


class TestToBacktestFrameConfidenceColumns(unittest.TestCase):

    def test_columns_absent_when_not_requested_backward_compatible(self):
        df = to_backtest_frame(_records(), "HOME", model_prob=0.5)
        self.assertNotIn("lambda_tier", df.columns)
        self.assertNotIn("effective_sample_size", df.columns)
        self.assertNotIn("model_confidence", df.columns)

    def test_columns_present_and_correct_when_provided_as_scalars(self):
        df = to_backtest_frame(
            _records(), "HOME", model_prob=0.5,
            lambda_tier="recent_matches", effective_sample_size=5.0, model_confidence="MEDIUM",
        )
        self.assertListEqual(list(df["lambda_tier"]), ["recent_matches", "recent_matches"])
        self.assertListEqual(list(df["effective_sample_size"]), [5.0, 5.0])
        self.assertListEqual(list(df["model_confidence"]), ["MEDIUM", "MEDIUM"])

    def test_columns_correct_when_fed_from_lambda_confidence_from_dixon_coles(self):
        records = _records()
        confidence = lambda_confidence_from_dixon_coles(records)
        df = to_backtest_frame(
            records, "HOME", model_prob=0.5,
            lambda_tier=lambda row: confidence.get(row["event_id"], {}).get("lambda_tier"),
            effective_sample_size=lambda row: confidence.get(row["event_id"], {}).get("effective_sample_size"),
            model_confidence=lambda row: confidence.get(row["event_id"], {}).get("model_confidence"),
        )
        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df["lambda_tier"]),
            [confidence[1]["lambda_tier"], confidence[2]["lambda_tier"]],
        )


# --------------------------------------------------------------------------
# Pipeline de ponta a ponta: to_backtest_frame -> load_historical_dataset ->
# BacktestEngine -> segmentos
# --------------------------------------------------------------------------

class TestEndToEndPropagation(unittest.TestCase):

    def test_metadata_survives_load_historical_dataset_and_backtest_engine(self):
        records = _records()
        probabilities = model_probabilities_from_dixon_coles(records)
        confidence = lambda_confidence_from_dixon_coles(records)

        df = to_backtest_frame(
            records, "HOME",
            model_prob=lambda row: probabilities[row["event_id"]]["home"],
            lambda_tier=lambda row: confidence[row["event_id"]]["lambda_tier"],
            effective_sample_size=lambda row: confidence[row["event_id"]]["effective_sample_size"],
            model_confidence=lambda row: confidence[row["event_id"]]["model_confidence"],
        )

        dataset = load_historical_dataset(df)
        self.assertIn("lambda_tier", dataset.columns)

        report = BacktestEngine(staking=FlatStake(unit=1.0)).run(dataset)
        self.assertIn("lambda_tier", report.all_bets.columns)
        self.assertListEqual(
            list(report.all_bets["lambda_tier"]),
            [confidence[1]["lambda_tier"], confidence[2]["lambda_tier"]],
        )

    def test_pipeline_without_confidence_metadata_still_works(self):
        records = _records()
        probabilities = model_probabilities_from_dixon_coles(records)
        df = to_backtest_frame(
            records, "HOME", model_prob=lambda row: probabilities[row["event_id"]]["home"],
        )
        dataset = load_historical_dataset(df)
        self.assertNotIn("lambda_tier", dataset.columns)

        report = BacktestEngine(staking=FlatStake(unit=1.0)).run(dataset)
        # EvaluatedBet.to_dict() emite sempre a chave -- retrocompatível:
        # a coluna existe, mas fica vazia, sem nenhum erro.
        self.assertIn("lambda_tier", report.all_bets.columns)
        self.assertTrue(report.all_bets["lambda_tier"].isna().all())


# --------------------------------------------------------------------------
# Segmentação: src/evaluation/segments.py
# --------------------------------------------------------------------------

def _confidence_all_bets_df():
    return pd.DataFrame([
        # tier "recent_matches": 2 apostas colocadas (1 ganha, 1 perde)
        {"odd": 2.0, "probability": 0.6, "edge": 0.05, "ev": 0.1, "kelly": 0.05,
         "stake": 1.0, "won": True, "profit": 1.0, "placed": True,
         "lambda_tier": "recent_matches", "model_confidence": "HIGH", "effective_sample_size": 10.0},
        {"odd": 2.0, "probability": 0.6, "edge": 0.05, "ev": 0.1, "kelly": 0.05,
         "stake": 1.0, "won": False, "profit": -1.0, "placed": True,
         "lambda_tier": "recent_matches", "model_confidence": "HIGH", "effective_sample_size": 10.0},
        # tier "avg_total_goals_or_prior": 1 colocada (ganha) + 1 avaliada mas não colocada
        {"odd": 3.0, "probability": 0.4, "edge": 0.02, "ev": 0.05, "kelly": 0.02,
         "stake": 1.0, "won": True, "profit": 2.0, "placed": True,
         "lambda_tier": "avg_total_goals_or_prior", "model_confidence": "LOW", "effective_sample_size": 1.0},
        {"odd": 1.5, "probability": 0.5, "edge": -0.02, "ev": -0.05, "kelly": 0.0,
         "stake": 1.0, "won": False, "profit": 0.0, "placed": False,
         "lambda_tier": "avg_total_goals_or_prior", "model_confidence": "LOW", "effective_sample_size": 1.0},
    ])


class TestSegmentByLambdaTier(unittest.TestCase):

    def test_groups_correctly_by_tier(self):
        result = segment_by_lambda_tier(_confidence_all_bets_df())
        self.assertEqual(set(result["lambda_tier"]), {"recent_matches", "avg_total_goals_or_prior"})

    def test_financial_metrics_use_only_placed_bets_in_the_tier(self):
        result = segment_by_lambda_tier(_confidence_all_bets_df())
        prior_row = result[result["lambda_tier"] == "avg_total_goals_or_prior"].iloc[0]
        # das 2 apostas avaliadas neste tier, só 1 foi colocada.
        self.assertEqual(prior_row["n_bets"], 1)
        self.assertAlmostEqual(prior_row["net_profit"], 2.0, places=4)
        self.assertAlmostEqual(prior_row["roi_pct"], 200.0, places=2)

    def test_statistical_metrics_use_all_evaluated_bets_in_the_tier(self):
        result = segment_by_lambda_tier(_confidence_all_bets_df())
        prior_row = result[result["lambda_tier"] == "avg_total_goals_or_prior"].iloc[0]
        expected_brier = round(((0.4 - 1.0) ** 2 + (0.5 - 0.0) ** 2) / 2, 6)
        self.assertAlmostEqual(prior_row["brier_score"], expected_brier, places=6)

    def test_missing_column_returns_empty_without_error(self):
        df = _confidence_all_bets_df().drop(columns=["lambda_tier"])
        self.assertTrue(segment_by_lambda_tier(df).empty)

    def test_empty_dataframe_returns_empty(self):
        self.assertTrue(segment_by_lambda_tier(pd.DataFrame()).empty)

    def test_rows_with_missing_tier_value_are_excluded_not_erroring(self):
        df = _confidence_all_bets_df()
        df.loc[0, "lambda_tier"] = None
        result = segment_by_lambda_tier(df)
        recent_row = result[result["lambda_tier"] == "recent_matches"].iloc[0]
        self.assertEqual(recent_row["n_bets"], 1)


class TestSegmentByModelConfidence(unittest.TestCase):

    def test_groups_by_confidence_label(self):
        result = segment_by_model_confidence(_confidence_all_bets_df())
        self.assertEqual(set(result["model_confidence"]), {"HIGH", "LOW"})

    def test_n_bets_per_label_is_correct(self):
        result = segment_by_model_confidence(_confidence_all_bets_df())
        high_row = result[result["model_confidence"] == "HIGH"].iloc[0]
        low_row = result[result["model_confidence"] == "LOW"].iloc[0]
        self.assertEqual(high_row["n_bets"], 2)
        self.assertEqual(low_row["n_bets"], 1)

    def test_missing_column_returns_empty(self):
        self.assertTrue(segment_by_model_confidence(pd.DataFrame({"odd": [2.0]})).empty)


class TestSegmentByEffectiveSampleSizeRange(unittest.TestCase):

    def test_buckets_by_sample_size(self):
        result = segment_by_effective_sample_size_range(_confidence_all_bets_df())
        self.assertIn("0-2", set(result["range"]))
        self.assertIn("8-15", set(result["range"]))

    def test_missing_column_returns_empty(self):
        self.assertTrue(segment_by_effective_sample_size_range(pd.DataFrame({"odd": [2.0]})).empty)


class TestAllConfidenceSegments(unittest.TestCase):

    def test_omits_segments_when_no_metadata_available(self):
        df = pd.DataFrame({
            "odd": [2.0], "probability": [0.5], "edge": [0.0], "ev": [0.0],
            "kelly": [0.0], "stake": [1.0], "won": [True], "profit": [1.0], "placed": [True],
        })
        self.assertEqual(all_confidence_segments(df), {})

    def test_includes_all_three_when_metadata_present(self):
        result = all_confidence_segments(_confidence_all_bets_df())
        self.assertEqual(
            set(result.keys()),
            {"by_lambda_tier", "by_effective_sample_size_range", "by_model_confidence"},
        )
        for table in result.values():
            for col in ("n_bets", "roi_pct", "yield_pct", "brier_score", "log_loss"):
                self.assertIn(col, table.columns)


# --------------------------------------------------------------------------
# EvaluationReport: os novos segmentos aparecem quando há metadado
# --------------------------------------------------------------------------

class TestEvaluationReportConfidenceSegments(unittest.TestCase):

    def test_tier_segments_appear_in_report_when_metadata_present(self):
        records = _records()
        probabilities = model_probabilities_from_dixon_coles(records)
        confidence = lambda_confidence_from_dixon_coles(records)
        df = to_backtest_frame(
            records, "HOME",
            model_prob=lambda row: probabilities[row["event_id"]]["home"],
            lambda_tier=lambda row: confidence[row["event_id"]]["lambda_tier"],
            effective_sample_size=lambda row: confidence[row["event_id"]]["effective_sample_size"],
            model_confidence=lambda row: confidence[row["event_id"]]["model_confidence"],
        )
        dataset = load_historical_dataset(df)
        report = evaluate(dataset, staking=FlatStake(unit=1.0))
        segments = report.all_segment_tables()
        self.assertIn("by_lambda_tier", segments)
        for col in ("roi_pct", "yield_pct", "brier_score", "log_loss", "n_bets"):
            self.assertIn(col, segments["by_lambda_tier"].columns)


# --------------------------------------------------------------------------
# Ausência de regressões
# --------------------------------------------------------------------------

class TestNoRegressionOnExistingPipeline(unittest.TestCase):

    def test_synthetic_sample_dataset_pipeline_is_unaffected(self):
        dataset = generate_sample_dataset(n_games=80, seed=11)
        report = evaluate(dataset, staking=FlatStake(unit=1.0))

        self.assertIn("roi_pct", report.global_metrics)
        self.assertIn("brier_score", report.statistical_metrics)

        segments = report.all_segment_tables()
        self.assertNotIn("by_lambda_tier", segments)
        self.assertNotIn("by_model_confidence", segments)
        self.assertNotIn("by_effective_sample_size_range", segments)

        self.assertIn("lambda_tier", report.all_bets.columns)
        self.assertTrue(report.all_bets["lambda_tier"].isna().all())

    def test_legacy_csv_dataset_loads_and_evaluates_without_error(self):
        dataset = load_historical_dataset(SAMPLE_CSV)
        report = evaluate(dataset, staking=FlatStake(unit=1.0))
        self.assertGreater(len(report.all_bets), 0)
        segments = report.all_segment_tables()
        self.assertNotIn("by_lambda_tier", segments)


if __name__ == "__main__":
    unittest.main()

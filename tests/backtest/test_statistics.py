"""
Testes unitários das métricas estatísticas de calibração
(src/backtest/historical/statistics.py).
"""

import unittest

import pandas as pd

from src.backtest.historical.statistics import (
    brier_score,
    calibration_curve,
    calibration_error,
    edge_distribution,
    ev_distribution,
    log_loss,
    probability_distribution,
)


class TestBrierScore(unittest.TestCase):

    def test_perfect_predictions_score_zero(self):
        df = pd.DataFrame(
            [
                {"probability": 1.0, "won": True},
                {"probability": 0.0, "won": False},
            ]
        )
        self.assertAlmostEqual(brier_score(df), 0.0, places=6)

    def test_known_value(self):
        # (0.7-1)^2 = 0.09 ; (0.3-0)^2 = 0.09 -> média = 0.09
        df = pd.DataFrame(
            [
                {"probability": 0.7, "won": True},
                {"probability": 0.3, "won": False},
            ]
        )
        self.assertAlmostEqual(brier_score(df), 0.09, places=6)

    def test_worst_case_predictions_score_one(self):
        df = pd.DataFrame(
            [
                {"probability": 0.0, "won": True},
                {"probability": 1.0, "won": False},
            ]
        )
        self.assertAlmostEqual(brier_score(df), 1.0, places=6)

    def test_empty_frame_is_zero(self):
        df = pd.DataFrame(columns=["probability", "won"])
        self.assertEqual(brier_score(df), 0.0)


class TestLogLoss(unittest.TestCase):

    def test_known_value(self):
        import math

        df = pd.DataFrame(
            [
                {"probability": 0.9, "won": True},
                {"probability": 0.1, "won": False},
            ]
        )
        expected = -(math.log(0.9) + math.log(0.9)) / 2
        self.assertAlmostEqual(log_loss(df), expected, places=6)

    def test_confident_wrong_prediction_has_high_loss(self):
        df = pd.DataFrame([{"probability": 0.99, "won": False}])
        self.assertGreater(log_loss(df), 4.0)

    def test_does_not_crash_on_extreme_probabilities(self):
        df = pd.DataFrame(
            [
                {"probability": 1.0, "won": True},
                {"probability": 0.0, "won": False},
            ]
        )
        # não deve levantar (log(0) seria -inf sem clipping)
        result = log_loss(df)
        self.assertTrue(result >= 0.0)


class TestCalibration(unittest.TestCase):

    def test_perfectly_calibrated_model_has_zero_error(self):
        # 10 apostas com p=0.5, exatamente 5 ganham -> calibração perfeita no bin.
        rows = [{"probability": 0.5, "won": i < 5} for i in range(10)]
        df = pd.DataFrame(rows)
        self.assertAlmostEqual(calibration_error(df, n_bins=10), 0.0, places=6)

    def test_overconfident_model_has_positive_error(self):
        # Modelo diz sempre 90% mas só acerta metade das vezes.
        rows = [{"probability": 0.9, "won": i % 2 == 0} for i in range(10)]
        df = pd.DataFrame(rows)
        error = calibration_error(df, n_bins=10)
        self.assertAlmostEqual(error, 0.4, places=2)

    def test_calibration_curve_bins_have_expected_columns(self):
        rows = [{"probability": p, "won": p > 0.5} for p in [0.1, 0.4, 0.6, 0.9]]
        df = pd.DataFrame(rows)
        curve = calibration_curve(df, n_bins=10)
        self.assertListEqual(
            list(curve.columns),
            ["bin_low", "bin_high", "predicted_mean", "actual_frequency", "count"],
        )
        self.assertEqual(curve["count"].sum(), 4)

    def test_empty_frame_returns_empty_curve(self):
        df = pd.DataFrame(columns=["probability", "won"])
        curve = calibration_curve(df)
        self.assertTrue(curve.empty)
        self.assertEqual(calibration_error(df), 0.0)


class TestDistributions(unittest.TestCase):

    def test_probability_distribution_counts_all_values(self):
        df = pd.DataFrame({"probability": [0.1, 0.2, 0.3, 0.9]})
        dist = probability_distribution(df, n_bins=5)
        self.assertEqual(sum(dist["counts"]), 4)

    def test_edge_and_ev_distribution_counts_all_values(self):
        df = pd.DataFrame({"edge": [0.01, 0.05, -0.02], "ev": [0.1, -0.05, 0.2]})
        edge_dist = edge_distribution(df, n_bins=5)
        ev_dist = ev_distribution(df, n_bins=5)
        self.assertEqual(sum(edge_dist["counts"]), 3)
        self.assertEqual(sum(ev_dist["counts"]), 3)

    def test_empty_frame_returns_empty_distribution(self):
        df = pd.DataFrame(columns=["probability"])
        dist = probability_distribution(df)
        self.assertEqual(dist["counts"], [])


if __name__ == "__main__":
    unittest.main()

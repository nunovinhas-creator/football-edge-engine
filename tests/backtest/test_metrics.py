"""
Testes unitários das métricas globais de desempenho
(src/backtest/historical/metrics.py).
"""

import unittest

import pandas as pd

from src.backtest.historical.metrics import (
    equity_curve,
    expectancy_per_bet,
    hit_rate,
    max_drawdown,
    net_profit,
    profit_factor,
    roi,
    summary_metrics,
    total_staked,
    yield_pct,
)


def _bets_df(rows):
    """Constrói um DataFrame mínimo de apostas avaliadas para os testes."""
    return pd.DataFrame(rows)


class TestEmptyFrame(unittest.TestCase):

    def test_all_metrics_are_zero_for_empty_frame(self):
        df = pd.DataFrame(columns=["odd", "probability", "edge", "ev", "kelly", "stake", "won", "profit"])
        summary = summary_metrics(df)
        self.assertEqual(summary["n_bets"], 0)
        self.assertEqual(summary["roi_pct"], 0.0)
        self.assertEqual(summary["yield_pct"], 0.0)
        self.assertEqual(summary["net_profit"], 0.0)
        self.assertEqual(summary["max_drawdown"], 0.0)


class TestRoi(unittest.TestCase):
    """
    ROI = lucro líquido total / total apostado * 100.
    Usa stakes uniformes (1.0) para que ROI e Yield coincidam.
    """

    def setUp(self):
        # 2 apostas: uma ganha com odd 2.0 (profit +1.0), uma perdida (profit -1.0).
        self.df = _bets_df(
            [
                {"stake": 1.0, "profit": 1.0, "won": True},
                {"stake": 1.0, "profit": -1.0, "won": False},
            ]
        )

    def test_roi_known_value(self):
        # lucro líquido = 0.0, total apostado = 2.0 -> ROI = 0%
        self.assertAlmostEqual(roi(self.df), 0.0, places=4)

    def test_roi_all_wins(self):
        df = _bets_df([{"stake": 1.0, "profit": 1.0, "won": True}] * 4)
        # lucro = 4.0, staked = 4.0 -> ROI = 100%
        self.assertAlmostEqual(roi(df), 100.0, places=4)

    def test_roi_zero_when_no_stake(self):
        df = _bets_df([{"stake": 0.0, "profit": 0.0, "won": False}])
        self.assertEqual(roi(df), 0.0)


class TestYield(unittest.TestCase):
    """
    Yield = média das rentabilidades individuais (profit_i / stake_i).
    Diverge do ROI quando os stakes variam entre apostas.
    """

    def test_yield_matches_roi_with_flat_stakes(self):
        df = _bets_df(
            [
                {"stake": 2.0, "profit": 2.0, "won": True},   # +100%
                {"stake": 2.0, "profit": -2.0, "won": False},  # -100%
            ]
        )
        self.assertAlmostEqual(yield_pct(df), 0.0, places=4)
        self.assertAlmostEqual(yield_pct(df), roi(df), places=4)

    def test_yield_diverges_from_roi_with_variable_stakes(self):
        # Uma aposta grande e perdedora domina o ROI (ponderado por stake),
        # mas o Yield (média simples por aposta) não é dominado da mesma forma.
        df = _bets_df(
            [
                {"stake": 10.0, "profit": -10.0, "won": False},  # -100% do stake
                {"stake": 1.0, "profit": 1.0, "won": True},       # +100% do stake
                {"stake": 1.0, "profit": 1.0, "won": True},       # +100% do stake
            ]
        )
        roi_value = roi(df)
        yield_value = yield_pct(df)
        self.assertLess(roi_value, 0)   # dominado pela grande perda
        self.assertGreater(yield_value, 0)  # média simples: 2 em 3 apostas com +100%
        self.assertNotAlmostEqual(roi_value, yield_value, places=2)


class TestNetProfit(unittest.TestCase):

    def test_known_value(self):
        df = _bets_df(
            [
                {"stake": 1.0, "profit": 1.5, "won": True},
                {"stake": 1.0, "profit": -1.0, "won": False},
                {"stake": 1.0, "profit": 0.8, "won": True},
            ]
        )
        self.assertAlmostEqual(net_profit(df), 1.3, places=4)

    def test_total_staked(self):
        df = _bets_df([{"stake": 2.5, "profit": 0.0, "won": False}] * 4)
        self.assertAlmostEqual(total_staked(df), 10.0, places=4)


class TestDrawdown(unittest.TestCase):

    def test_known_drawdown_sequence(self):
        # Lucro cumulativo: 3, 5, 2, -1, 4  (ordem de entrada = ordem cronológica)
        df = _bets_df(
            [
                {"stake": 1.0, "profit": 3.0, "won": True},
                {"stake": 1.0, "profit": 2.0, "won": True},
                {"stake": 1.0, "profit": -3.0, "won": False},
                {"stake": 1.0, "profit": -3.0, "won": False},
                {"stake": 1.0, "profit": 5.0, "won": True},
            ]
        )
        # Curva: 3, 5, 2, -1, 4  | pico corrente: 3,5,5,5,5 | drawdown: 0,0,-3,-6,-1
        result = max_drawdown(df)
        self.assertAlmostEqual(result["max_drawdown"], -6.0, places=4)
        self.assertAlmostEqual(result["max_drawdown_pct"], -120.0, places=2)

    def test_no_drawdown_when_monotonically_increasing(self):
        df = _bets_df([{"stake": 1.0, "profit": 1.0, "won": True}] * 5)
        result = max_drawdown(df)
        self.assertEqual(result["max_drawdown"], 0.0)
        self.assertEqual(result["max_drawdown_pct"], 0.0)

    def test_equity_curve_is_cumulative_profit(self):
        df = _bets_df(
            [
                {"stake": 1.0, "profit": 1.0, "won": True},
                {"stake": 1.0, "profit": -0.5, "won": False},
                {"stake": 1.0, "profit": 2.0, "won": True},
            ]
        )
        curve = equity_curve(df)
        self.assertListEqual(list(curve.values), [1.0, 0.5, 2.5])


class TestProfitFactorAndExpectancy(unittest.TestCase):

    def test_profit_factor_known_value(self):
        df = _bets_df(
            [
                {"stake": 1.0, "profit": 4.0, "won": True},
                {"stake": 1.0, "profit": 2.0, "won": True},
                {"stake": 1.0, "profit": -3.0, "won": False},
            ]
        )
        # gains = 6.0, losses = 3.0 -> profit factor = 2.0
        self.assertAlmostEqual(profit_factor(df), 2.0, places=4)

    def test_profit_factor_infinite_when_no_losses(self):
        df = _bets_df([{"stake": 1.0, "profit": 1.0, "won": True}] * 3)
        self.assertEqual(profit_factor(df), float("inf"))

    def test_expectancy_per_bet(self):
        df = _bets_df(
            [
                {"stake": 1.0, "profit": 3.0, "won": True},
                {"stake": 1.0, "profit": -1.0, "won": False},
            ]
        )
        self.assertAlmostEqual(expectancy_per_bet(df), 1.0, places=4)


class TestHitRate(unittest.TestCase):

    def test_known_value(self):
        df = _bets_df(
            [{"stake": 1.0, "profit": 1.0, "won": True}] * 3
            + [{"stake": 1.0, "profit": -1.0, "won": False}]
        )
        self.assertAlmostEqual(hit_rate(df), 75.0, places=2)


if __name__ == "__main__":
    unittest.main()

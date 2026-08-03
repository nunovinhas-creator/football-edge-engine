"""
Testes unitários da avaliação por aposta
(src/backtest/historical/evaluator.py e models.py) — cobrem, entre outras
coisas, o cálculo do Edge médio e EV médio sobre um conjunto de apostas.
"""

import unittest

from src.engine.edge import calculate_edge, calculate_ev
from src.backtest.historical.evaluator import evaluate_bet, evaluate_bets
from src.backtest.historical.models import HistoricalBet
from src.backtest.historical.staking import FlatStake, KellyStake


def _bet(match="A vs B", odd=2.10, model_prob=0.55, decision="BET", result="WIN", **kwargs):
    return HistoricalBet(
        match=match,
        date="2026-01-01",
        market="HOME",
        odd=odd,
        model_prob=model_prob,
        engine_decision=decision,
        result=result,
        **kwargs,
    )


class TestHistoricalBetParsing(unittest.TestCase):

    def test_result_string_variants_are_normalized(self):
        for value in ["WIN", "win", "WON", "W", "1"]:
            self.assertTrue(_bet(result=value).won)
        for value in ["LOSS", "loss", "LOST", "L", "0"]:
            self.assertFalse(_bet(result=value).won)

    def test_result_bool_and_numeric(self):
        self.assertTrue(_bet(result=True).won)
        self.assertFalse(_bet(result=False).won)
        self.assertTrue(_bet(result=1).won)
        self.assertFalse(_bet(result=0).won)

    def test_invalid_result_raises(self):
        with self.assertRaises(ValueError):
            _bet(result="MAYBE").won

    def test_is_bet_decision_matches_bet_variants(self):
        self.assertTrue(HistoricalBet.is_bet_decision("BET"))
        self.assertTrue(HistoricalBet.is_bet_decision("BET 🔥"))
        self.assertTrue(HistoricalBet.is_bet_decision("bet"))
        self.assertFalse(HistoricalBet.is_bet_decision("PASS ❄️"))
        self.assertFalse(HistoricalBet.is_bet_decision("WAIT ⚠️"))
        self.assertFalse(HistoricalBet.is_bet_decision(None))

    def test_from_dict_accepts_portuguese_aliases(self):
        row = {
            "jogo": "Benfica vs Porto",
            "data": "2026-02-01",
            "mercado": "HOME",
            "odd": 1.85,
            "probabilidade": 0.6,
            "decisao": "BET",
            "resultado": "WIN",
        }
        bet = HistoricalBet.from_dict(row)
        self.assertEqual(bet.match, "Benfica vs Porto")
        self.assertEqual(bet.odd, 1.85)
        self.assertEqual(bet.model_prob, 0.6)
        self.assertTrue(bet.won)

    def test_from_dict_missing_required_field_raises(self):
        with self.assertRaises(KeyError):
            HistoricalBet.from_dict({"jogo": "A vs B"})


class TestEvaluateBet(unittest.TestCase):

    def test_edge_and_ev_match_official_engine_formulas(self):
        bet = _bet(model_prob=0.55, odd=2.10)
        evaluated = evaluate_bet(bet)
        self.assertAlmostEqual(evaluated.edge, calculate_edge(0.55, 2.10), places=6)
        self.assertAlmostEqual(evaluated.ev, calculate_ev(0.55, 2.10), places=6)

    def test_flat_stake_profit_on_win(self):
        bet = _bet(model_prob=0.55, odd=2.10, result="WIN")
        evaluated = evaluate_bet(bet, staking=FlatStake(unit=2.0))
        self.assertAlmostEqual(evaluated.stake, 2.0)
        self.assertAlmostEqual(evaluated.profit, 2.0 * (2.10 - 1.0), places=4)

    def test_flat_stake_profit_on_loss(self):
        bet = _bet(model_prob=0.55, odd=2.10, result="LOSS")
        evaluated = evaluate_bet(bet, staking=FlatStake(unit=2.0))
        self.assertAlmostEqual(evaluated.profit, -2.0, places=4)

    def test_placed_reflects_engine_decision(self):
        placed_bet = evaluate_bet(_bet(decision="BET 🔥"))
        skipped_bet = evaluate_bet(_bet(decision="PASS ❄️"))
        self.assertTrue(placed_bet.placed)
        self.assertFalse(skipped_bet.placed)

    def test_kelly_stake_strategy_is_used_when_provided(self):
        bet = _bet(model_prob=0.6, odd=2.5)
        evaluated = evaluate_bet(bet, staking=KellyStake(fraction=0.25, cap=0.05, bankroll=100.0))
        self.assertGreater(evaluated.stake, 0.0)
        self.assertLessEqual(evaluated.stake, 5.0)  # cap de 5% de 100


class TestEvaluateBets(unittest.TestCase):

    def test_average_edge_matches_manual_calculation(self):
        bets = [
            _bet(model_prob=0.55, odd=2.10),
            _bet(model_prob=0.40, odd=3.00),
            _bet(model_prob=0.70, odd=1.50),
        ]
        df = evaluate_bets(bets)
        expected_avg_edge = sum(
            calculate_edge(b.model_prob, b.odd) for b in bets
        ) / len(bets)
        self.assertAlmostEqual(df["edge"].mean(), expected_avg_edge, places=6)

    def test_average_ev_matches_manual_calculation(self):
        bets = [
            _bet(model_prob=0.55, odd=2.10),
            _bet(model_prob=0.40, odd=3.00),
            _bet(model_prob=0.70, odd=1.50),
        ]
        df = evaluate_bets(bets)
        expected_avg_ev = sum(
            calculate_ev(b.model_prob, b.odd) for b in bets
        ) / len(bets)
        self.assertAlmostEqual(df["ev"].mean(), expected_avg_ev, places=6)

    def test_results_are_sorted_chronologically(self):
        bets = [
            HistoricalBet(match="late", date="2026-03-01", market="HOME", odd=2.0,
                          model_prob=0.5, engine_decision="BET", result="WIN"),
            HistoricalBet(match="early", date="2026-01-01", market="HOME", odd=2.0,
                          model_prob=0.5, engine_decision="BET", result="WIN"),
        ]
        df = evaluate_bets(bets)
        self.assertEqual(list(df["match"]), ["early", "late"])

    def test_empty_input_returns_empty_dataframe(self):
        df = evaluate_bets([])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()

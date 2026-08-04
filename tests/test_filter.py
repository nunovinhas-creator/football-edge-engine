"""Testes de src/engine/filter.py::is_valid_bet.

Cobre a correção do bug de comparação de decisão: a função comparava
`result["decision"] != "VALUE BET"`, um literal que make_decision()
(src/engine/decision.py, usado por src/cli/predict.py) nunca produz —
devolve "BET 🔥"/"WAIT ⚠️"/"PASS ❄️". Isto fazia com que nenhuma
oportunidade passasse alguma vez neste filtro, deixando
ranking["value_bets"] sempre vazio. Os limiares numéricos (edge, ev,
confidence, odd) não foram alterados.
"""

import unittest

from src.engine.filter import is_valid_bet


def _base_result(**overrides):
    result = {
        "decision": "BET 🔥",
        "edge": 10.0,
        "ev": 20.0,
        "confidence": "HIGH",
        "odd": 2.0,
    }
    result.update(overrides)
    return result


class TestIsValidBet(unittest.TestCase):

    def test_accepts_current_bet_decision_label(self):
        self.assertTrue(is_valid_bet(_base_result(decision="BET 🔥")))

    def test_accepts_legacy_value_bet_label(self):
        # src/engine/analyzer.py::analyze_bet ainda produz "VALUE BET".
        self.assertTrue(is_valid_bet(_base_result(decision="VALUE BET")))

    def test_accepts_live_bet_label(self):
        # src/engine/ranking.py::rank_bets produz "LIVE BET" para live.
        self.assertTrue(is_valid_bet(_base_result(decision="LIVE BET")))

    def test_rejects_wait_decision(self):
        self.assertFalse(is_valid_bet(_base_result(decision="WAIT ⚠️")))

    def test_rejects_pass_decision(self):
        self.assertFalse(is_valid_bet(_base_result(decision="PASS ❄️")))

    def test_rejects_watch_decision(self):
        self.assertFalse(is_valid_bet(_base_result(decision="WATCH")))

    def test_still_enforces_edge_threshold(self):
        self.assertFalse(is_valid_bet(_base_result(edge=4.9)))

    def test_still_enforces_ev_threshold(self):
        self.assertFalse(is_valid_bet(_base_result(ev=9.9)))

    def test_still_enforces_low_confidence_rejection(self):
        self.assertFalse(is_valid_bet(_base_result(confidence="LOW")))

    def test_still_enforces_min_odd(self):
        self.assertFalse(is_valid_bet(_base_result(odd=1.69)))


if __name__ == "__main__":
    unittest.main()

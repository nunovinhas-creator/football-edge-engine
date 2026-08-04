"""
Testes do Explainability Engine (Melhoria #13, src/report/explainability.py).

Este módulo é uma camada de INTERPRETAÇÃO puramente determinística (sem
IA/LLM) sobre um MatchSnapshot já construído pelo motor (mesmo formato de
`src.report.dashboard_data.build_match_snapshot`). Os testes aqui:

  - NÃO chamam Dixon-Coles, Monte Carlo, Goal Engine, ML, Kelly, Edge, EV
    ou Lambda Estimator — constroem snapshots sintéticos diretamente, tal
    como esses módulos já os produziriam;
  - confirmam que `generate_explanation` nunca recalcula nem altera os
    valores do snapshot recebido (só lê);
  - cobrem: decisão apostar / aguardar / não apostar, consenso forte /
    fraco, edge negativo, EV negativo, Kelly zero, avisos (poucos jogos,
    fase inicial do jogo, sem consenso) e o resumo automático.
"""

import copy
import unittest

from src.report.explainability import (
    Explanation,
    build_telegram_message,
    format_explanation_block,
    generate_explanation,
)


def make_snapshot(
    decision_label="🟡 AGUARDAR",
    confidence_label="Média",
    confidence_score=60.0,
    engine_score=55.0,
    goal_engine_prob=None,
    ml_prob=None,
    ml_confidence=None,
    monte_carlo_over15=None,
    edge_pct=None,
    ev_pct=None,
    kelly_pct=None,
    fair_odd=None,
    consensus_label="—",
    consensus_gap=None,
    pressure=None,
    momentum=None,
    minute=45,
    dangerous_attacks=5,
    shots=3,
    corners=2,
    effective_sample_size=None,
    h2h_available=True,
) -> dict:
    """Constrói um snapshot sintético com a MESMA forma do dict devolvido
    por `build_match_snapshot` — só os campos lidos por generate_explanation
    são preenchidos; ausência de um campo == não existir no snapshot real."""
    return {
        "card": {"minute": minute},
        "decision": {
            "label": decision_label,
            "confidence_label": confidence_label,
            "confidence_score": confidence_score,
        },
        "engine_score": {"score": engine_score},
        "models": {
            "goal_engine": {"probability": goal_engine_prob},
            "machine_learning": {"probability": ml_prob, "confidence": ml_confidence},
            "monte_carlo": {"over_15": monte_carlo_over15},
        },
        "consensus": {"label": consensus_label, "gap": consensus_gap},
        "value": {
            "edge_pct": edge_pct,
            "ev_pct": ev_pct,
            "kelly_pct": kelly_pct,
            "fair_odd": fair_odd,
        },
        "live": {
            "pressure": pressure,
            "momentum": momentum,
            "dangerous_attacks_10m": dangerous_attacks,
            "shots_10m": shots,
            "corners_10m": corners,
        },
        "strength": {
            "effective_sample_size": effective_sample_size,
            "h2h_available": h2h_available,
        },
    }


class TestExplanationDataclass(unittest.TestCase):

    def test_fields(self):
        explanation = Explanation(
            decision="🟢 APOSTAR AGORA",
            confidence="Alta",
            score=80.0,
            positives=["a"],
            negatives=["b"],
            warnings=["c"],
            consensus="Muito Forte",
            summary="resumo",
        )
        self.assertEqual(explanation.decision, "🟢 APOSTAR AGORA")
        self.assertEqual(explanation.confidence, "Alta")
        self.assertEqual(explanation.score, 80.0)
        self.assertEqual(explanation.positives, ["a"])
        self.assertEqual(explanation.negatives, ["b"])
        self.assertEqual(explanation.warnings, ["c"])
        self.assertEqual(explanation.consensus, "Muito Forte")
        self.assertEqual(explanation.summary, "resumo")


class TestDoesNotRecomputeOrMutate(unittest.TestCase):

    def test_snapshot_is_not_mutated(self):
        snapshot = make_snapshot(edge_pct=8.0, ev_pct=5.0)
        before = copy.deepcopy(snapshot)
        generate_explanation(snapshot)
        self.assertEqual(snapshot, before)

    def test_missing_fields_do_not_crash(self):
        explanation = generate_explanation({})
        self.assertIsInstance(explanation, Explanation)
        self.assertEqual(explanation.positives, [])
        self.assertEqual(explanation.negatives, [])

    def test_none_snapshot_does_not_crash(self):
        explanation = generate_explanation(None)
        self.assertIsInstance(explanation, Explanation)


class TestDecisionApostar(unittest.TestCase):

    def test_decision_field_and_summary_mention_apostar(self):
        snapshot = make_snapshot(
            decision_label="🟢 APOSTAR AGORA",
            goal_engine_prob=78.0,
            ml_prob=70.0,
            monte_carlo_over15=72.0,
            edge_pct=8.1,
            ev_pct=14.0,
            kelly_pct=3.6,
            consensus_label="Muito Forte",
        )
        explanation = generate_explanation(snapshot)

        self.assertEqual(explanation.decision, "🟢 APOSTAR AGORA")
        self.assertIn("apostar", explanation.summary.lower())
        self.assertIn("Edge", " ".join(explanation.positives))
        self.assertTrue(any("EV" in p for p in explanation.positives))
        self.assertTrue(any("Consenso" in p for p in explanation.positives))


class TestDecisionAguardar(unittest.TestCase):

    def test_summary_mentions_divergence_or_insufficient_edge(self):
        snapshot = make_snapshot(
            decision_label="🟡 AGUARDAR",
            edge_pct=2.0,
            ev_pct=0.5,
            consensus_label="Moderado",
            consensus_gap=20.0,
        )
        explanation = generate_explanation(snapshot)

        self.assertEqual(explanation.decision, "🟡 AGUARDAR")
        self.assertIn("aguardar", explanation.summary.lower())


class TestDecisionNaoApostar(unittest.TestCase):

    def test_summary_mentions_no_value_or_no_consensus(self):
        snapshot = make_snapshot(
            decision_label="🔴 NÃO APOSTAR",
            edge_pct=-3.0,
            ev_pct=-5.0,
            consensus_label="Fraco",
            consensus_gap=40.0,
        )
        explanation = generate_explanation(snapshot)

        self.assertEqual(explanation.decision, "🔴 NÃO APOSTAR")
        self.assertIn("não apostar", explanation.summary.lower())
        self.assertIn("Edge negativo", " ".join(explanation.negatives))


class TestConsensusStrong(unittest.TestCase):

    def test_muito_forte_is_positive_and_reflected_in_consensus_field(self):
        snapshot = make_snapshot(consensus_label="Muito Forte")
        explanation = generate_explanation(snapshot)

        self.assertEqual(explanation.consensus, "Muito Forte")
        self.assertTrue(any("Consenso Muito Forte" in p for p in explanation.positives))
        self.assertFalse(any("Sem consenso" in w for w in explanation.warnings))


class TestConsensusWeak(unittest.TestCase):

    def test_fraco_generates_warning(self):
        snapshot = make_snapshot(consensus_label="Fraco")
        explanation = generate_explanation(snapshot)

        self.assertEqual(explanation.consensus, "Fraco")
        self.assertTrue(any("Sem consenso" in w for w in explanation.warnings))
        self.assertFalse(any("Consenso Muito Forte" in p for p in explanation.positives))


class TestEdgeNegative(unittest.TestCase):

    def test_negative_edge_is_a_negative_not_a_positive(self):
        snapshot = make_snapshot(edge_pct=-1.5)
        explanation = generate_explanation(snapshot)

        self.assertTrue(any("Edge negativo" in n for n in explanation.negatives))
        self.assertFalse(any("valor estatístico" in p and "Edge" in p for p in explanation.positives))

    def test_edge_above_five_is_positive(self):
        snapshot = make_snapshot(edge_pct=6.0)
        explanation = generate_explanation(snapshot)

        self.assertTrue(any("valor estatístico" in p for p in explanation.positives))
        self.assertEqual(explanation.negatives, [n for n in explanation.negatives if "Edge negativo" not in n])


class TestEvNegative(unittest.TestCase):

    def test_negative_ev_is_flagged(self):
        snapshot = make_snapshot(ev_pct=-2.0)
        explanation = generate_explanation(snapshot)

        self.assertTrue(any("EV negativo" in n for n in explanation.negatives))

    def test_positive_ev_is_a_positive(self):
        snapshot = make_snapshot(ev_pct=0.01)
        explanation = generate_explanation(snapshot)

        self.assertTrue(any("Valor esperado positivo" in p for p in explanation.positives))


class TestKellyZero(unittest.TestCase):

    def test_kelly_zero_generates_neither_positive_nor_reduced_warning(self):
        snapshot = make_snapshot(kelly_pct=0.0)
        explanation = generate_explanation(snapshot)

        self.assertFalse(any("Kelly" in p for p in explanation.positives))
        self.assertFalse(any("Kelly" in n for n in explanation.negatives))

    def test_kelly_small_but_positive_is_reduced_negative(self):
        snapshot = make_snapshot(kelly_pct=0.4)
        explanation = generate_explanation(snapshot)

        self.assertTrue(any("Gestão de banca recomenda entrada" in p for p in explanation.positives))
        self.assertTrue(any("Kelly muito reduzido" in n for n in explanation.negatives))

    def test_kelly_healthy_is_only_positive(self):
        snapshot = make_snapshot(kelly_pct=5.0)
        explanation = generate_explanation(snapshot)

        self.assertTrue(any("Gestão de banca recomenda entrada" in p for p in explanation.positives))
        self.assertFalse(any("Kelly muito reduzido" in n for n in explanation.negatives))


class TestModelFavorablePositives(unittest.TestCase):

    def test_goal_engine_above_70_is_positive(self):
        explanation = generate_explanation(make_snapshot(goal_engine_prob=71.0))
        self.assertTrue(any("Goal Engine muito favorável" in p for p in explanation.positives))

    def test_goal_engine_at_or_below_70_is_not_positive(self):
        explanation = generate_explanation(make_snapshot(goal_engine_prob=70.0))
        self.assertFalse(any("Goal Engine muito favorável" in p for p in explanation.positives))

    def test_ml_above_65_is_positive(self):
        explanation = generate_explanation(make_snapshot(ml_prob=66.0))
        self.assertTrue(any("Modelo ML confirma oportunidade" in p for p in explanation.positives))

    def test_monte_carlo_above_65_is_positive(self):
        explanation = generate_explanation(make_snapshot(monte_carlo_over15=70.0))
        self.assertTrue(any("Monte Carlo confirma cenário" in p for p in explanation.positives))


class TestLiveSignals(unittest.TestCase):

    def test_high_pressure_is_positive(self):
        explanation = generate_explanation(make_snapshot(pressure=75.0))
        self.assertTrue(any("Pressão ofensiva significativa" in p for p in explanation.positives))

    def test_low_pressure_is_negative(self):
        explanation = generate_explanation(make_snapshot(pressure=20.0))
        self.assertTrue(any("Pouca pressão ofensiva" in n for n in explanation.negatives))

    def test_rising_momentum_is_positive(self):
        explanation = generate_explanation(make_snapshot(momentum="RISING"))
        self.assertTrue(any("Momentum crescente" in p for p in explanation.positives))

    def test_surging_momentum_is_positive(self):
        explanation = generate_explanation(make_snapshot(momentum="SURGING"))
        self.assertTrue(any("Momentum crescente" in p for p in explanation.positives))

    def test_stable_momentum_is_neither(self):
        explanation = generate_explanation(make_snapshot(momentum="STABLE"))
        self.assertFalse(any("Momentum" in p for p in explanation.positives))


class TestConfidenceNegative(unittest.TestCase):

    def test_low_confidence_score_is_negative(self):
        snapshot = make_snapshot(confidence_score=25.0)
        explanation = generate_explanation(snapshot)
        self.assertTrue(any("Baixa confiança estatística" in n for n in explanation.negatives))

    def test_high_confidence_score_is_not_negative(self):
        snapshot = make_snapshot(confidence_score=90.0)
        explanation = generate_explanation(snapshot)
        self.assertFalse(any("Baixa confiança estatística" in n for n in explanation.negatives))


class TestDividedModels(unittest.TestCase):

    def test_large_probability_gap_is_negative(self):
        snapshot = make_snapshot(goal_engine_prob=80.0, ml_prob=40.0)
        explanation = generate_explanation(snapshot)
        self.assertTrue(any("dividido do ML" in n for n in explanation.negatives))

    def test_small_probability_gap_is_not_negative(self):
        snapshot = make_snapshot(goal_engine_prob=60.0, ml_prob=58.0)
        explanation = generate_explanation(snapshot)
        self.assertFalse(any("dividido do ML" in n for n in explanation.negatives))


class TestWarnings(unittest.TestCase):

    def test_small_effective_sample_size_warns(self):
        snapshot = make_snapshot(effective_sample_size=4)
        explanation = generate_explanation(snapshot)
        self.assertTrue(any("poucos jogos" in w for w in explanation.warnings))

    def test_no_h2h_available_warns(self):
        snapshot = make_snapshot(effective_sample_size=None, h2h_available=False)
        explanation = generate_explanation(snapshot)
        self.assertTrue(any("poucos jogos" in w for w in explanation.warnings))

    def test_large_sample_does_not_warn(self):
        snapshot = make_snapshot(effective_sample_size=50, h2h_available=True)
        explanation = generate_explanation(snapshot)
        self.assertFalse(any("poucos jogos" in w for w in explanation.warnings))

    def test_early_minute_with_no_live_data_warns(self):
        snapshot = make_snapshot(minute=5, dangerous_attacks=0, shots=0, corners=0)
        explanation = generate_explanation(snapshot)
        self.assertTrue(any("Poucos dados live" in w for w in explanation.warnings))

    def test_early_minute_with_live_data_does_not_warn(self):
        snapshot = make_snapshot(minute=5, dangerous_attacks=3, shots=2, corners=1)
        explanation = generate_explanation(snapshot)
        self.assertFalse(any("Poucos dados live" in w for w in explanation.warnings))

    def test_weak_consensus_warns(self):
        snapshot = make_snapshot(consensus_label="Fraco")
        explanation = generate_explanation(snapshot)
        self.assertTrue(any("Sem consenso" in w for w in explanation.warnings))


class TestSummary(unittest.TestCase):

    def test_summary_is_non_empty_for_every_decision(self):
        for label in ("🟢 APOSTAR AGORA", "🟡 AGUARDAR", "🔴 NÃO APOSTAR"):
            explanation = generate_explanation(make_snapshot(decision_label=label))
            self.assertTrue(explanation.summary)
            self.assertIsInstance(explanation.summary, str)

    def test_summary_fallback_for_unknown_decision(self):
        explanation = generate_explanation(make_snapshot(decision_label="???"))
        self.assertTrue(explanation.summary)


class TestTelegramFormatting(unittest.TestCase):

    def test_format_explanation_block_contains_sections(self):
        explanation = generate_explanation(
            make_snapshot(edge_pct=8.0, ev_pct=2.0, consensus_label="Fraco")
        )
        block = format_explanation_block(explanation)
        self.assertIn("Porque esta decisão?", block)
        self.assertIn(explanation.summary, block)

    def test_build_telegram_message_includes_metrics_and_explanation(self):
        snapshot = make_snapshot(
            decision_label="🟢 APOSTAR AGORA",
            goal_engine_prob=74.0,
            edge_pct=8.1,
            ev_pct=14.0,
            kelly_pct=3.6,
            fair_odd=1.63,
            consensus_label="Muito Forte",
        )
        message = build_telegram_message(snapshot)

        self.assertIn("Football Edge Engine", message)
        self.assertIn("🟢 APOSTAR AGORA", message)
        self.assertIn("Probabilidade: 74%", message)
        self.assertIn("Odd Justa: 1.63", message)
        self.assertIn("Edge: +8.1%", message)
        self.assertIn("Kelly: 3.6%", message)
        self.assertIn("Porque esta decisão?", message)

    def test_build_telegram_message_does_not_crash_on_sparse_snapshot(self):
        message = build_telegram_message({})
        self.assertIn("Football Edge Engine", message)


if __name__ == "__main__":
    unittest.main()

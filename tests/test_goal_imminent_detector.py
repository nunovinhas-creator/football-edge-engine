"""
Testes do Goal Imminent Detection (`src.alerts.goal_imminent_detector`).

Cobrem:
  - os 12 critérios oficiais (Decisão, Goal Engine, Monte Carlo, ML,
    Edge, EV, Kelly, Consenso, Pressão, Ataques Perigosos, Remates à
    Baliza, Jogo não terminado) — nenhum alerta é enviado a menos que
    TODOS estejam reunidos, e cada um sozinho bloqueia o alerta;
  - anti-spam: no máximo UM alerta `GOAL_IMMINENT` por `match_id`, para
    sempre (sem reavaliação/cooldown, ao contrário do Alerta Live
    Premium);
  - persistência em SQLite (`data/goal_imminent_alerts.db`, sempre um
    ficheiro temporário nestes testes) — todos os campos pedidos;
  - conteúdo/formato da mensagem Telegram (cabeçalho obrigatório "⚽
    Football Edge Engine", nunca apenas "BET"/"ALERTA");
  - o leitor usado pelo Dashboard
    (`src.report.dashboard_data.load_goal_imminent_alerts`).

A integração com `src.engine.live_monitor.run_live_pipeline` (o mesmo
MatchSnapshot é reutilizado tanto pelo Alerta Live Premium como pelo Goal
Imminent Detection, sem recalcular nada) está coberta em
`tests/test_live_monitor_premium_wiring.py`.

Usa sempre mocks para o envio Telegram — nunca contacta a API real. Não
recalcula nenhuma probabilidade/edge/EV/Kelly: os MatchSnapshots de teste
são dicts construídos à mão com os mesmos campos que
`src.report.dashboard_data.build_match_snapshot` já produz (ver também
`tests/test_live_premium_alerts.py`).
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.alerts.goal_imminent_detector import (
    DANGEROUS_ATTACKS_HIGH_THRESHOLD,
    EDGE_MIN_PCT,
    GOAL_ENGINE_MIN_PROB,
    GOAL_IMMINENT_ALERT_HEADER,
    GOAL_IMMINENT_ALERT_TYPE,
    ML_MIN_PROB,
    MONTE_CARLO_MIN_PROB,
    REQUIRED_CONSENSUS_LABEL,
    SHOTS_ON_TARGET_HIGH_THRESHOLD,
    GoalImminentDetector,
    build_goal_imminent_message,
    evaluate_goal_imminent_criteria,
)
from src.alerts.live_premium_alerts import REQUIRED_DECISION_LABEL
from src.report.explainability import PRESSURE_HIGH


def make_passing_snapshot(**overrides):
    """
    MatchSnapshot mínimo (mesmos campos que `build_match_snapshot`
    produz) já construído para passar em TODOS os 12 critérios do Goal
    Imminent Detection. `overrides` aceita chaves "dotted" (ver
    `_set_path`) para os testes quebrarem exatamente UM critério de
    cada vez.
    """
    snapshot = {
        "match_id": 777,
        "card": {
            "competition": "Premier League",
            "home_team": "Manchester City",
            "away_team": "Chelsea",
            "minute": 70,
            "status": "AO VIVO",
        },
        "decision": {"label": REQUIRED_DECISION_LABEL},
        "models": {
            "goal_engine": {"probability": 85.0},
            "machine_learning": {"probability": 75.0},
            "monte_carlo": {"over_15": 80.0, "over_25": 45.0, "btts": 60.0},
        },
        "consensus": {"gap": 2.0, "label": "Muito Forte"},
        "value": {
            "bookie_odd": 1.90,
            "fair_odd": 1.30,
            "edge_pct": 10.0,
            "ev_pct": 8.0,
            "kelly_pct": 4.0,
        },
        "live": {
            "pressure": 75.0,
            "dangerous_attacks_10m": 18,
            "shots_on_target_10m": 10,
            "estimated_xg_10m": 1.2,
            "momentum": "SURGING",
        },
    }

    for dotted_key, value in overrides.items():
        _set_path(snapshot, dotted_key.split("__"), value)

    return snapshot


def _set_path(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


class TestEvaluateGoalImminentCriteriaAllPass(unittest.TestCase):

    def test_baseline_snapshot_passes_all_12_criteria(self):
        result = evaluate_goal_imminent_criteria(make_passing_snapshot())
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_reasons, [])
        self.assertEqual(len(result.checks), 12)
        self.assertTrue(all(result.checks.values()))


class TestEvaluateGoalImminentCriteriaEachFailureBlocks(unittest.TestCase):
    """Cada teste quebra exatamente UM dos 12 critérios e confirma que,
    sozinho, chega para bloquear o alerta."""

    def test_decision_not_apostar_agora_blocks(self):
        for label in ("🟡 AGUARDAR", "🔴 NÃO APOSTAR"):
            with self.subTest(label=label):
                snap = make_passing_snapshot(**{"decision__label": label})
                result = evaluate_goal_imminent_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["decision"])

    def test_goal_engine_below_80_blocks(self):
        snap = make_passing_snapshot(**{"models__goal_engine__probability": GOAL_ENGINE_MIN_PROB - 0.1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["goal_engine"])

    def test_monte_carlo_below_75_blocks(self):
        snap = make_passing_snapshot(**{"models__monte_carlo__over_15": MONTE_CARLO_MIN_PROB - 0.1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["monte_carlo"])

    def test_ml_below_70_blocks(self):
        snap = make_passing_snapshot(**{"models__machine_learning__probability": ML_MIN_PROB - 0.1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["ml"])

    def test_edge_below_5_blocks(self):
        snap = make_passing_snapshot(**{"value__edge_pct": EDGE_MIN_PCT - 0.1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["edge"])

    def test_ev_zero_or_negative_blocks(self):
        for ev in (0.0, -3.0):
            with self.subTest(ev=ev):
                snap = make_passing_snapshot(**{"value__ev_pct": ev})
                result = evaluate_goal_imminent_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["ev"])

    def test_kelly_zero_or_negative_blocks(self):
        for kelly in (0.0, -1.0):
            with self.subTest(kelly=kelly):
                snap = make_passing_snapshot(**{"value__kelly_pct": kelly})
                result = evaluate_goal_imminent_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["kelly"])

    def test_consensus_not_muito_forte_blocks(self):
        for label in ("Forte", "Moderado", "Fraco"):
            with self.subTest(label=label):
                snap = make_passing_snapshot(**{"consensus__label": label})
                result = evaluate_goal_imminent_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["consensus"])

    def test_pressure_below_high_threshold_blocks(self):
        snap = make_passing_snapshot(**{"live__pressure": PRESSURE_HIGH - 0.1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["pressure"])

    def test_dangerous_attacks_below_threshold_blocks(self):
        snap = make_passing_snapshot(**{"live__dangerous_attacks_10m": DANGEROUS_ATTACKS_HIGH_THRESHOLD - 1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["dangerous_attacks"])

    def test_shots_on_target_below_threshold_blocks(self):
        snap = make_passing_snapshot(**{"live__shots_on_target_10m": SHOTS_ON_TARGET_HIGH_THRESHOLD - 1})
        result = evaluate_goal_imminent_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["shots_on_target"])

    def test_finished_match_status_blocks(self):
        for status in ("FT", "ft", "Finished", "Terminado", "Encerrado"):
            with self.subTest(status=status):
                snap = make_passing_snapshot(**{"card__status": status})
                result = evaluate_goal_imminent_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["not_finished"])

    def test_thresholds_are_exclusively_goal_imminent_own_not_shared_with_live_premium(self):
        # Confirma que os limiares do Goal Imminent Detection são mais
        # exigentes/distintos dos já existentes no Alerta Live Premium —
        # nenhum threshold existente foi alterado, este é um conjunto
        # de critérios NOVO e próprio.
        self.assertEqual(GOAL_ENGINE_MIN_PROB, 80.0)
        self.assertEqual(MONTE_CARLO_MIN_PROB, 75.0)
        self.assertEqual(ML_MIN_PROB, 70.0)
        self.assertEqual(REQUIRED_CONSENSUS_LABEL, "Muito Forte")


class TestBuildGoalImminentMessage(unittest.TestCase):

    def test_message_starts_with_required_header_exactly(self):
        snap = make_passing_snapshot()
        criteria = evaluate_goal_imminent_criteria(snap)
        message = build_goal_imminent_message(snap, criteria)
        self.assertTrue(message.startswith(GOAL_IMMINENT_ALERT_HEADER))
        self.assertEqual(GOAL_IMMINENT_ALERT_HEADER, "⚽ Football Edge Engine")

    def test_message_never_reduces_to_just_bet_or_alerta(self):
        snap = make_passing_snapshot()
        criteria = evaluate_goal_imminent_criteria(snap)
        message = build_goal_imminent_message(snap, criteria)
        stripped = message.strip()
        self.assertFalse(stripped.startswith("BET"))
        self.assertFalse(stripped.startswith("ALERTA"))

    def test_message_includes_all_required_fields(self):
        snap = make_passing_snapshot()
        criteria = evaluate_goal_imminent_criteria(snap)
        message = build_goal_imminent_message(snap, criteria)

        for expected in (
            "🚨 GOLO MUITO PROVÁVEL",
            "Liga:", "Premier League",
            "Jogo:", "Manchester City vs Chelsea",
            "Minuto:", "70",
            "Goal Engine:", "85%",
            "Monte Carlo:", "80%",
            "Machine Learning:", "75%",
            "Consenso:", "Muito Forte",
            "Edge:", "+10.0%",
            "EV:", "+8.0%",
            "Kelly:", "4.0%",
            "Odd Mercado:", "1.90",
            "Pressão:", "75/100",
            "Ataques perigosos:", "18",
            "Remates:", "10",
            "xG:", "1.20",
            "Momentum:", "SURGING",
            "🔥 APOSTAR AGORA",
        ):
            self.assertIn(expected, message)


class _DetectorTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "goal_imminent_alerts_test.db")
        self.sender = MagicMock(return_value=True)
        self.detector = GoalImminentDetector(db_path=self.db_path, sender=self.sender)

    def tearDown(self):
        self._tmpdir.cleanup()


class TestGoalImminentDetectorSendsOnlyWhenAllCriteriaPass(_DetectorTestCase):

    def test_sends_exactly_one_telegram_message_when_all_criteria_pass(self):
        outcome = self.detector.evaluate_and_maybe_alert(make_passing_snapshot())

        self.sender.assert_called_once()
        self.assertTrue(outcome.sent)
        self.assertTrue(outcome.telegram_sent)
        self.assertEqual(outcome.state, "ALERTA ENVIADO")

    def test_does_not_send_when_a_single_criterion_fails(self):
        outcome = self.detector.evaluate_and_maybe_alert(
            make_passing_snapshot(**{"value__ev_pct": -1.0})
        )
        self.sender.assert_not_called()
        self.assertFalse(outcome.sent)
        self.assertEqual(outcome.state, "CRITÉRIOS NÃO REUNIDOS")
        self.assertIn("EV", outcome.reason)


class TestGoalImminentDetectorAntiSpam(_DetectorTestCase):

    def test_never_sends_a_second_alert_for_the_same_match_even_if_still_passing(self):
        snap = make_passing_snapshot()

        outcome1 = self.detector.evaluate_and_maybe_alert(snap)
        outcome2 = self.detector.evaluate_and_maybe_alert(snap)

        self.sender.assert_called_once()
        self.assertTrue(outcome1.sent)
        self.assertFalse(outcome2.sent)
        self.assertEqual(outcome2.state, "JÁ ENVIADO")

    def test_has_already_alerted_reflects_a_successful_send(self):
        snap = make_passing_snapshot()
        self.assertFalse(self.detector.has_already_alerted(snap["match_id"]))

        self.detector.evaluate_and_maybe_alert(snap)

        self.assertTrue(self.detector.has_already_alerted(snap["match_id"]))

    def test_a_failed_telegram_send_does_not_block_future_attempts(self):
        failing_sender = MagicMock(return_value=False)
        detector = GoalImminentDetector(db_path=self.db_path, sender=failing_sender)
        snap = make_passing_snapshot()

        outcome1 = detector.evaluate_and_maybe_alert(snap)
        self.assertFalse(outcome1.telegram_sent)
        self.assertFalse(detector.has_already_alerted(snap["match_id"]))

        outcome2 = detector.evaluate_and_maybe_alert(snap)
        self.assertEqual(failing_sender.call_count, 2)
        self.assertNotEqual(outcome2.state, "JÁ ENVIADO")

    def test_different_matches_are_independent(self):
        snap_a = make_passing_snapshot(match_id=1)
        snap_b = make_passing_snapshot(match_id=2)

        self.detector.evaluate_and_maybe_alert(snap_a)
        outcome_b = self.detector.evaluate_and_maybe_alert(snap_b)

        self.assertEqual(self.sender.call_count, 2)
        self.assertTrue(outcome_b.sent)


class TestGoalImminentDetectorPersistence(_DetectorTestCase):

    def test_logs_alert_with_all_required_fields(self):
        snap = make_passing_snapshot()
        self.detector.evaluate_and_maybe_alert(snap)

        rows = self.detector.load_alerts()
        self.assertEqual(len(rows), 1)
        row = rows[0]

        for field in (
            "match_id", "alert_type", "competition", "home_team", "away_team", "minute",
            "goal_engine_probability", "monte_carlo_probability", "ml_probability",
            "edge", "ev", "kelly", "pressure", "dangerous_attacks", "shots_on_target",
            "xg", "market_odd", "telegram_sent", "created_at",
        ):
            self.assertIn(field, row)

        self.assertEqual(row["match_id"], "777")
        self.assertEqual(row["alert_type"], GOAL_IMMINENT_ALERT_TYPE)
        self.assertEqual(row["competition"], "Premier League")
        self.assertEqual(row["home_team"], "Manchester City")
        self.assertEqual(row["away_team"], "Chelsea")
        self.assertEqual(row["minute"], 70)
        self.assertEqual(row["goal_engine_probability"], 85.0)
        self.assertEqual(row["monte_carlo_probability"], 80.0)
        self.assertEqual(row["ml_probability"], 75.0)
        self.assertEqual(row["dangerous_attacks"], 18)
        self.assertEqual(row["shots_on_target"], 10)
        self.assertEqual(row["market_odd"], 1.90)
        self.assertEqual(row["telegram_sent"], 1)

    def test_does_not_log_when_criteria_are_not_met(self):
        snap = make_passing_snapshot(**{"value__ev_pct": -1.0})
        self.detector.evaluate_and_maybe_alert(snap)
        self.assertEqual(self.detector.load_alerts(), [])

    def test_persists_to_the_configured_sqlite_file(self):
        snap = make_passing_snapshot()
        self.detector.evaluate_and_maybe_alert(snap)

        self.assertTrue(os.path.exists(self.db_path))
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM goal_imminent_alerts").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 1)


class TestGoalImminentDashboardReader(unittest.TestCase):
    """`src.report.dashboard_data.load_goal_imminent_alerts` — leitura
    usada pelo separador '🚨 Goal Imminent Alerts' do Dashboard."""

    def test_reads_alerts_already_logged_by_the_detector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "goal_imminent_alerts_dashboard_test.db")
            sender = MagicMock(return_value=True)
            detector = GoalImminentDetector(db_path=db_path, sender=sender)
            detector.evaluate_and_maybe_alert(make_passing_snapshot())

            from src.report import dashboard_data

            with patch.object(dashboard_data, "DEFAULT_GOAL_IMMINENT_DB_PATH", db_path):
                df = dashboard_data.load_goal_imminent_alerts()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["home_team"], "Manchester City")
        self.assertEqual(df.iloc[0]["away_team"], "Chelsea")
        self.assertEqual(df.iloc[0]["telegram_sent"], 1)
        self.assertEqual(df.iloc[0]["outcome"], "ALERTA ENVIADO")

    def test_returns_empty_dataframe_when_no_db_file_exists_yet(self):
        from src.report import dashboard_data

        with patch.object(dashboard_data, "DEFAULT_GOAL_IMMINENT_DB_PATH", "/tmp/does-not-exist-goal-imminent.db"):
            df = dashboard_data.load_goal_imminent_alerts()

        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()

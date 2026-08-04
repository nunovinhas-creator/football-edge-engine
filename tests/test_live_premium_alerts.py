"""
Testes do Alerta Live Premium (`src.alerts.live_premium_alerts`).

Cobrem:
  - os 8 critérios oficiais (Monte Carlo, Goal Engine, Decisão, Edge, EV,
    Kelly, Odd, Consenso) — nenhum alerta é enviado a menos que TODOS
    estejam reunidos;
  - anti-spam (cooldown de 10 min, variação mínima de odd, decisão igual,
    limpeza automática quando o jogo termina);
  - conteúdo/formato da mensagem Telegram (cabeçalho obrigatório "🔥
    FOOTBALL EDGE ENGINE", nunca apenas "Bet");
  - o log em SQLite (sempre um ficheiro temporário nestes testes, nunca o
    `data/live_alerts.db` real do projeto);
  - o painel do Dashboard (`src.report.dashboard_data.build_live_alert_monitor_rows`)
    mostra corretamente o estado/motivo/histórico.

Usa sempre mocks para o envio Telegram — nunca contacta a API real.
Não recalcula nenhuma probabilidade/edge/EV/Kelly: os MatchSnapshots de
teste são dicts construídos à mão com os mesmos campos que
`src.report.dashboard_data.build_match_snapshot` já produz.
"""

import copy
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.alerts.live_premium_alerts import (
    ALERT_BRAND_HEADER,
    ALERT_MARKET_LABEL,
    COOLDOWN_SECONDS,
    GOAL_ENGINE_MIN_PROB,
    MAX_CONSENSUS_GAP_PP,
    MIN_EDGE_PCT,
    MIN_ODD_DELTA,
    MONTE_CARLO_MIN_PROB,
    ODD_MAX,
    ODD_MIN,
    REQUIRED_DECISION_LABEL,
    LiveAlertMonitor,
    evaluate_alert_criteria,
    format_alert_message,
)


def make_passing_snapshot(**overrides):
    """
    MatchSnapshot mínimo (mesmos campos que `build_match_snapshot` produz)
    já construído para passar em TODOS os 8 critérios do Alerta Live
    Premium. `overrides` aceita chaves "dotted" simples para os testes
    quebrarem exatamente UM critério de cada vez (ver `_set_path`).
    """
    snapshot = {
        "match_id": 555,
        "card": {
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "minute": 67,
        },
        "decision": {"label": REQUIRED_DECISION_LABEL},
        "models": {
            "goal_engine": {"probability": 74.0},
            "machine_learning": {"probability": 71.0},
            "monte_carlo": {"over_15": 78.0, "over_25": 40.0, "btts": 55.0},
        },
        "consensus": {"gap": 3.0, "label": "Muito Forte"},
        "value": {
            "bookie_odd": 1.74,
            "fair_odd": 1.40,
            "edge_pct": 9.4,
            "ev_pct": 12.0,
            "kelly_pct": 3.1,
        },
        "live": {"pressure": 70.0},
    }

    for dotted_key, value in overrides.items():
        _set_path(snapshot, dotted_key.split("__"), value)

    return snapshot


def _set_path(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


class TestEvaluateAlertCriteriaAllPass(unittest.TestCase):

    def test_baseline_snapshot_passes_all_criteria(self):
        result = evaluate_alert_criteria(make_passing_snapshot())
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_reasons, [])
        self.assertTrue(all(result.checks.values()))


class TestEvaluateAlertCriteriaEachFailureBlocks(unittest.TestCase):
    """Cada teste quebra exatamente UM dos 8 critérios e confirma que,
    sozinho, chega para bloquear o alerta."""

    def test_monte_carlo_below_70_blocks(self):
        snap = make_passing_snapshot(**{"models__monte_carlo__over_15": MONTE_CARLO_MIN_PROB - 0.1})
        result = evaluate_alert_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["monte_carlo"])

    def test_goal_engine_below_70_blocks(self):
        snap = make_passing_snapshot(**{"models__goal_engine__probability": GOAL_ENGINE_MIN_PROB - 0.1})
        result = evaluate_alert_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["goal_engine"])

    def test_decision_not_apostar_agora_blocks(self):
        for label in ("🟡 AGUARDAR", "🔴 NÃO APOSTAR"):
            with self.subTest(label=label):
                snap = make_passing_snapshot(**{"decision__label": label})
                result = evaluate_alert_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["decision"])

    def test_edge_at_or_below_5_blocks(self):
        snap = make_passing_snapshot(**{"value__edge_pct": 4.9})
        result = evaluate_alert_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["edge"])

    def test_ev_zero_or_negative_blocks(self):
        for ev in (0.0, -3.0):
            with self.subTest(ev=ev):
                snap = make_passing_snapshot(**{"value__ev_pct": ev})
                result = evaluate_alert_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["ev"])

    def test_kelly_zero_or_negative_blocks(self):
        for kelly in (0.0, -1.0):
            with self.subTest(kelly=kelly):
                snap = make_passing_snapshot(**{"value__kelly_pct": kelly})
                result = evaluate_alert_criteria(snap)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks["kelly"])

    def test_odd_below_range_blocks(self):
        snap = make_passing_snapshot(**{"value__bookie_odd": ODD_MIN - 0.01})
        result = evaluate_alert_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["odd_range"])

    def test_odd_above_range_blocks(self):
        snap = make_passing_snapshot(**{"value__bookie_odd": ODD_MAX + 0.01})
        result = evaluate_alert_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["odd_range"])

    def test_consensus_gap_above_15pp_blocks(self):
        snap = make_passing_snapshot(**{"consensus__gap": MAX_CONSENSUS_GAP_PP + 0.1})
        result = evaluate_alert_criteria(snap)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["consensus"])


class TestFormatAlertMessage(unittest.TestCase):

    def test_message_starts_with_required_brand_header(self):
        snap = make_passing_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = format_alert_message(snap, criteria)
        self.assertTrue(message.startswith(ALERT_BRAND_HEADER))
        self.assertEqual(ALERT_BRAND_HEADER, "🔥 FOOTBALL EDGE ENGINE")

    def test_message_never_reduces_to_just_bet(self):
        snap = make_passing_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = format_alert_message(snap, criteria)
        self.assertNotEqual(message.strip(), "Bet")
        self.assertFalse(message.strip().startswith("Bet\n"))

    def test_message_includes_all_expected_fields(self):
        snap = make_passing_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = format_alert_message(snap, criteria)

        self.assertIn(REQUIRED_DECISION_LABEL, message)
        self.assertIn("Liverpool vs Arsenal", message)
        self.assertIn("Minuto 67", message)
        self.assertIn(ALERT_MARKET_LABEL, message)
        self.assertIn("74%", message)
        self.assertIn("78%", message)
        self.assertIn("71%", message)
        self.assertIn("+9.4%", message)
        self.assertIn("+12%", message)
        self.assertIn("3.1%", message)
        self.assertIn("1.74", message)
        self.assertIn("Muito Forte", message)
        self.assertIn("pressão ofensiva muito elevada", message)
        self.assertIn("consenso entre todos os modelos", message)
        self.assertIn("odd acima da odd justa", message)
        self.assertIn("valor esperado positivo", message)
        self.assertIn("stake recomendada 3.1%", message)


class _MonitorTestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "live_alerts_test.db")
        self.sender = MagicMock(return_value=True)
        self.monitor = LiveAlertMonitor(db_path=self.db_path, sender=self.sender)
        self.now = datetime(2026, 8, 4, 20, 0, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self._tmpdir.cleanup()


class TestLiveAlertMonitorSendsOnlyWhenAllCriteriaPass(_MonitorTestCase):

    def test_sends_exactly_one_telegram_message_when_all_criteria_pass(self):
        outcome = self.monitor.evaluate_and_maybe_alert(make_passing_snapshot(), now=self.now)

        self.sender.assert_called_once()
        self.assertTrue(outcome.sent)
        self.assertTrue(outcome.telegram_sent)
        self.assertEqual(outcome.state, "ALERTA ENVIADO")
        self.assertEqual(self.monitor.alerts_sent_today(now=self.now), 1)

    def _assert_blocked(self, snapshot):
        outcome = self.monitor.evaluate_and_maybe_alert(snapshot, now=self.now)
        self.sender.assert_not_called()
        self.assertFalse(outcome.sent)
        self.assertEqual(outcome.state, "À ESPERA")
        self.assertEqual(self.monitor.alerts_sent_today(now=self.now), 0)

    def test_does_not_send_when_monte_carlo_below_70(self):
        self._assert_blocked(make_passing_snapshot(**{"models__monte_carlo__over_15": 69.9}))

    def test_does_not_send_when_goal_engine_below_70(self):
        self._assert_blocked(make_passing_snapshot(**{"models__goal_engine__probability": 69.9}))

    def test_does_not_send_when_decision_is_not_apostar_agora(self):
        self._assert_blocked(make_passing_snapshot(**{"decision__label": "🟡 AGUARDAR"}))

    def test_does_not_send_when_edge_at_or_below_5(self):
        self._assert_blocked(make_passing_snapshot(**{"value__edge_pct": 5.0 - 0.01}))

    def test_does_not_send_when_ev_at_or_below_0(self):
        self._assert_blocked(make_passing_snapshot(**{"value__ev_pct": 0.0}))

    def test_does_not_send_when_kelly_at_or_below_0(self):
        self._assert_blocked(make_passing_snapshot(**{"value__kelly_pct": 0.0}))

    def test_does_not_send_when_odd_outside_1_40_2_30(self):
        self._assert_blocked(make_passing_snapshot(**{"value__bookie_odd": 1.39}))
        self._assert_blocked(make_passing_snapshot(**{"value__bookie_odd": 2.31}))

    def test_does_not_send_when_consensus_diverges_more_than_15pp(self):
        self._assert_blocked(make_passing_snapshot(**{"consensus__gap": 15.1}))


class TestLiveAlertMonitorAntiSpam(_MonitorTestCase):

    def test_does_not_send_duplicate_alert_for_same_match_within_cooldown(self):
        snap = make_passing_snapshot()

        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)
        outcome2 = self.monitor.evaluate_and_maybe_alert(
            snap, now=self.now + timedelta(seconds=COOLDOWN_SECONDS - 1)
        )

        self.sender.assert_called_once()
        self.assertFalse(outcome2.sent)
        self.assertEqual(outcome2.state, "ATIVO")

    def test_does_not_resend_after_cooldown_if_nothing_material_changed(self):
        snap = make_passing_snapshot()

        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)
        outcome2 = self.monitor.evaluate_and_maybe_alert(
            snap, now=self.now + timedelta(seconds=COOLDOWN_SECONDS + 60)
        )

        self.sender.assert_called_once()
        self.assertFalse(outcome2.sent)

    def test_resends_after_cooldown_when_odd_changes_enough(self):
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        snap2 = make_passing_snapshot(**{"value__bookie_odd": 1.74 + MIN_ODD_DELTA})
        outcome2 = self.monitor.evaluate_and_maybe_alert(
            snap2, now=self.now + timedelta(seconds=COOLDOWN_SECONDS + 1)
        )

        self.assertEqual(self.sender.call_count, 2)
        self.assertTrue(outcome2.sent)

    def test_resends_after_cooldown_when_decision_changes(self):
        # Decisão diferente que ainda assim cumpre os 8 critérios não é
        # realista (o critério 3 exige sempre "🟢 APOSTAR AGORA"), mas o
        # anti-spam por si só deve permitir reenviar quando `last_decision`
        # muda, mesmo com a mesma odd — testado diretamente via
        # `_should_send` para isolar essa regra do critério 3.
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        later = self.now + timedelta(seconds=COOLDOWN_SECONDS + 1)
        should_send = self.monitor._should_send(
            str(snap["match_id"]), 70, "🟢 APOSTAR AGORA (variação)", snap["value"]["bookie_odd"], later
        )
        self.assertTrue(should_send)

    def test_small_odd_variation_alone_does_not_unlock_resend(self):
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        snap2 = make_passing_snapshot(**{"value__bookie_odd": 1.74 + (MIN_ODD_DELTA / 2.0)})
        outcome2 = self.monitor.evaluate_and_maybe_alert(
            snap2, now=self.now + timedelta(seconds=COOLDOWN_SECONDS + 1)
        )

        self.sender.assert_called_once()
        self.assertFalse(outcome2.sent)


class TestLiveAlertMonitorFinishedMatchCleanup(_MonitorTestCase):

    def test_finished_clears_state_and_allows_immediate_resend(self):
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        finished_outcome = self.monitor.evaluate_and_maybe_alert(
            snap, finished=True, now=self.now + timedelta(seconds=30)
        )
        self.assertEqual(finished_outcome.state, "FINALIZADO")
        self.assertFalse(finished_outcome.sent)

        # Imediatamente a seguir (dentro do que seria a janela de cooldown)
        # já deve poder voltar a enviar, porque o registo foi limpo.
        outcome_after = self.monitor.evaluate_and_maybe_alert(
            snap, now=self.now + timedelta(seconds=31)
        )
        self.assertTrue(outcome_after.sent)
        self.assertEqual(self.sender.call_count, 2)

    def test_sync_active_matches_clears_matches_no_longer_live(self):
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        # O jogo desapareceu da lista de jogos ao vivo (terminou) —
        # sync_active_matches deve limpar o registo interno dele.
        self.monitor.sync_active_matches(active_match_ids=[])

        outcome_after = self.monitor.evaluate_and_maybe_alert(
            snap, now=self.now + timedelta(seconds=5)
        )
        self.assertTrue(outcome_after.sent)
        self.assertEqual(self.sender.call_count, 2)

    def test_sync_active_matches_keeps_matches_still_live(self):
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        self.monitor.sync_active_matches(active_match_ids=[snap["match_id"]])

        outcome_after = self.monitor.evaluate_and_maybe_alert(
            snap, now=self.now + timedelta(seconds=5)
        )
        self.sender.assert_called_once()
        self.assertFalse(outcome_after.sent)


class TestLiveAlertMonitorLogging(_MonitorTestCase):

    def test_logs_alert_with_all_required_fields(self):
        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        rows = self.monitor.load_alerts()
        self.assertEqual(len(rows), 1)
        row = rows[0]

        for field in (
            "timestamp", "match_id", "home_team", "away_team", "minute", "market",
            "odd", "goal_engine_probability", "monte_carlo_probability", "ml_probability",
            "edge", "ev", "kelly", "decision", "telegram_sent",
        ):
            self.assertIn(field, row)

        self.assertEqual(row["home_team"], "Liverpool")
        self.assertEqual(row["away_team"], "Arsenal")
        self.assertEqual(row["minute"], 67)
        self.assertEqual(row["market"], ALERT_MARKET_LABEL)
        self.assertEqual(row["decision"], REQUIRED_DECISION_LABEL)
        self.assertEqual(row["telegram_sent"], 1)

    def test_does_not_log_when_criteria_are_not_met(self):
        snap = make_passing_snapshot(**{"value__ev_pct": -1.0})
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        self.assertEqual(self.monitor.load_alerts(), [])


class TestDashboardShowsHistoryCorrectly(_MonitorTestCase):
    """Painel '🚨 Live Alert Monitor' do Dashboard
    (src.report.dashboard_data.build_live_alert_monitor_rows)."""

    def test_rows_report_alerta_enviado_state_after_a_real_alert(self):
        from src.report import dashboard_data

        snap = make_passing_snapshot()
        self.monitor.evaluate_and_maybe_alert(snap, now=self.now)

        with patch.object(dashboard_data, "DEFAULT_ALERTS_DB_PATH", self.db_path):
            rows = dashboard_data.build_live_alert_monitor_rows([snap])
            alerts_today = dashboard_data.count_alerts_sent_today()

        self.assertEqual(len(rows), 1)
        self.assertIn("ALERTA ENVIADO", rows[0]["Estado"])
        self.assertEqual(rows[0]["Jogo"], "Liverpool vs Arsenal")
        self.assertEqual(rows[0]["Mercado"], ALERT_MARKET_LABEL)
        self.assertEqual(rows[0]["Odd"], 1.74)
        self.assertNotEqual(rows[0]["Hora do último alerta"], "—")
        self.assertGreaterEqual(alerts_today, 0)

    def test_rows_report_a_espera_state_and_reason_when_criteria_fail(self):
        from src.report import dashboard_data

        snap = make_passing_snapshot(**{"value__ev_pct": -2.0})

        with patch.object(dashboard_data, "DEFAULT_ALERTS_DB_PATH", self.db_path):
            rows = dashboard_data.build_live_alert_monitor_rows([snap])

        self.assertEqual(len(rows), 1)
        self.assertIn("À ESPERA", rows[0]["Estado"])
        self.assertIn("EV", rows[0]["Motivo"])
        self.assertEqual(rows[0]["Hora do último alerta"], "—")


if __name__ == "__main__":
    unittest.main()

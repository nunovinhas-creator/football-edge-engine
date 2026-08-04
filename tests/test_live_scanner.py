"""
Testes do Scanner Live autónomo (`src.alerts.live_scanner`).

Cobrem:
  - `ScannerAntiSpamGuard` — a camada adicional de anti-spam pedida:
    cooldown de 15 min, minuto máximo 88, "praticamente igual"
    (decisão/odd/confiança), limpeza quando o jogo termina;
  - `build_scanner_alert_message` — cabeçalho obrigatório "⚽ Football
    Edge Engine" e todos os campos pedidos, reutilizando
    `src.report.explainability` para o bloco de explicação;
  - `run_scanner_cycle` — reutilização do snapshot (sem chamadas HTTP
    duplicadas), deduplicação de jogos no mesmo ciclo, reavaliação
    inteligente (AGUARDAR -> APOSTAR AGORA envia; sem mudança material não
    reenvia), nunca envia depois do minuto 88, e o log de cada ciclo em
    `logs/live_scanner.log`.

Usa sempre mocks/injeção de dependências para o fetcher, odds provider e
para `build_premium_snapshot` — nunca contacta a BSD API nem o Telegram
reais. Não recalcula nenhuma probabilidade/edge/EV/Kelly: os
MatchSnapshots de teste são dicts construídos à mão, no mesmo formato que
`src.report.dashboard_data.build_match_snapshot` já produz (ver também
`tests/test_live_premium_alerts.py`).
"""

import logging
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.alerts.live_premium_alerts import (
    LiveAlertMonitor,
    REQUIRED_DECISION_LABEL,
    evaluate_alert_criteria,
)
from src.alerts.live_scanner import (
    SCANNER_ALERT_HEADER,
    SCANNER_COOLDOWN_SECONDS,
    SCANNER_MAX_ALERT_MINUTE,
    SCANNER_MIN_CONFIDENCE_DELTA,
    SCANNER_MIN_ODD_DELTA,
    ScannerAntiSpamGuard,
    build_scanner_alert_message,
    run_scanner_cycle,
)


def make_snapshot(**overrides):
    """MatchSnapshot mínimo (mesmo formato de `build_match_snapshot`) já
    construído para passar em todos os 8 critérios do Alerta Live
    Premium. `overrides` aceita chaves "dotted" (ver `_set_path`)."""
    snapshot = {
        "match_id": 555,
        "card": {
            "competition": "Premier League",
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "minute": 67,
        },
        "decision": {"label": REQUIRED_DECISION_LABEL, "confidence_score": 82.0},
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
        d = snapshot
        keys = dotted_key.split("__")
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value
    return snapshot


def _fake_event(match_id, minute=60):
    return {
        "id": match_id,
        "home_team": "Time A",
        "away_team": "Time B",
        "current_minute": minute,
        "home_score": 1,
        "away_score": 0,
    }


def _fake_fetcher(events):
    fetcher = MagicMock()
    fetcher.get_live_events.return_value = events
    fetcher.parse_live_metrics_for_engine.side_effect = lambda ev: {
        "match_id": ev["id"],
        "home_team": ev["home_team"],
        "away_team": ev["away_team"],
        "current_minute": ev["current_minute"],
        "home_score": ev["home_score"],
        "away_score": ev["away_score"],
    }
    return fetcher


def _fake_odds_provider(odd=1.74):
    provider = MagicMock()
    provider.get_live_odds.return_value = {"odds": {"over_15_goals": odd}}
    return provider


# ---------------------------------------------------------------------------
# ScannerAntiSpamGuard
# ---------------------------------------------------------------------------
class _GuardTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "scanner_guard_test.db")
        self.guard = ScannerAntiSpamGuard(db_path=self.db_path)
        self.now = datetime(2026, 8, 4, 20, 0, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self._tmpdir.cleanup()


class TestScannerAntiSpamGuardBasics(_GuardTestCase):

    def test_allows_first_send_for_new_match(self):
        allowed, _ = self.guard.should_allow_send(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        self.assertTrue(allowed)

    def test_blocks_after_minute_88(self):
        allowed, reason = self.guard.should_allow_send(1, 89, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        self.assertFalse(allowed)
        self.assertIn("88", reason)

    def test_allows_exactly_minute_88(self):
        allowed, _ = self.guard.should_allow_send(1, SCANNER_MAX_ALERT_MINUTE, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        self.assertTrue(allowed)


class TestScannerAntiSpamGuardCooldown(_GuardTestCase):

    def test_blocks_within_15_minute_cooldown(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        allowed, _ = self.guard.should_allow_send(
            1, 62, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS - 1)
        )
        self.assertFalse(allowed)

    def test_cooldown_is_15_minutes(self):
        self.assertEqual(SCANNER_COOLDOWN_SECONDS, 15 * 60)


class TestScannerAntiSpamGuardReevaluation(_GuardTestCase):

    def test_blocks_after_cooldown_when_nothing_material_changed(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
        allowed, _ = self.guard.should_allow_send(1, 65, "🟢 APOSTAR AGORA", 1.80, 80.0, later)
        self.assertFalse(allowed)

    def test_allows_after_cooldown_when_odd_changes_enough(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
        allowed, _ = self.guard.should_allow_send(
            1, 65, "🟢 APOSTAR AGORA", 1.80 + SCANNER_MIN_ODD_DELTA, 80.0, later
        )
        self.assertTrue(allowed)

    def test_small_odd_variation_alone_does_not_unlock_resend(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
        allowed, _ = self.guard.should_allow_send(
            1, 65, "🟢 APOSTAR AGORA", 1.80 + (SCANNER_MIN_ODD_DELTA / 2), 80.0, later
        )
        self.assertFalse(allowed)

    def test_allows_after_cooldown_when_decision_changes(self):
        self.guard.record_sent(1, 60, "🟡 AGUARDAR", 1.80, 80.0, self.now)
        later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
        allowed, _ = self.guard.should_allow_send(1, 65, "🟢 APOSTAR AGORA", 1.80, 80.0, later)
        self.assertTrue(allowed)

    def test_allows_after_cooldown_when_confidence_changes_enough(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
        allowed, _ = self.guard.should_allow_send(
            1, 65, "🟢 APOSTAR AGORA", 1.80, 80.0 + SCANNER_MIN_CONFIDENCE_DELTA, later
        )
        self.assertTrue(allowed)

    def test_small_confidence_variation_alone_does_not_unlock_resend(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
        allowed, _ = self.guard.should_allow_send(
            1, 65, "🟢 APOSTAR AGORA", 1.80, 80.0 + (SCANNER_MIN_CONFIDENCE_DELTA / 2), later
        )
        self.assertFalse(allowed)


class TestScannerAntiSpamGuardMatchLifecycle(_GuardTestCase):

    def test_clear_match_allows_immediate_resend(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        self.guard.clear_match(1)
        allowed, _ = self.guard.should_allow_send(
            1, 61, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now + timedelta(seconds=5)
        )
        self.assertTrue(allowed)

    def test_sync_active_matches_clears_matches_no_longer_live(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        self.guard.sync_active_matches([])
        allowed, _ = self.guard.should_allow_send(
            1, 61, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now + timedelta(seconds=5)
        )
        self.assertTrue(allowed)

    def test_sync_active_matches_keeps_matches_still_live(self):
        self.guard.record_sent(1, 60, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now)
        self.guard.sync_active_matches([1])
        allowed, _ = self.guard.should_allow_send(
            1, 61, "🟢 APOSTAR AGORA", 1.80, 80.0, self.now + timedelta(seconds=5)
        )
        self.assertFalse(allowed)


# ---------------------------------------------------------------------------
# Mensagem Telegram do Scanner
# ---------------------------------------------------------------------------
class TestBuildScannerAlertMessage(unittest.TestCase):

    def test_message_starts_with_required_header_exactly(self):
        snap = make_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = build_scanner_alert_message(snap, criteria)
        self.assertTrue(message.startswith(SCANNER_ALERT_HEADER))
        self.assertEqual(SCANNER_ALERT_HEADER, "⚽ Football Edge Engine")

    def test_message_never_starts_with_just_bet_or_alerta(self):
        snap = make_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = build_scanner_alert_message(snap, criteria)
        stripped = message.strip()
        self.assertFalse(stripped.startswith("BET"))
        self.assertFalse(stripped.startswith("ALERTA"))

    def test_message_includes_all_required_fields(self):
        snap = make_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = build_scanner_alert_message(snap, criteria)

        for expected in (
            "🏆 Premier League",
            "⚽ Liverpool vs Arsenal",
            "⏱ Minuto 67",
            "📈 Goal Engine",
            "🎲 Monte Carlo",
            "🤖 ML",
            "📊 Consenso",
            "💰 Odd",
            "💎 Edge",
            "📈 EV",
            "💵 Kelly",
            "🎯 Stake",
            "🚨 Decisão",
            "🧠",
            REQUIRED_DECISION_LABEL,
        ):
            self.assertIn(expected, message)

    def test_uses_explainability_module_for_explanation_block(self):
        # `format_explanation_block`/`generate_explanation` produzem
        # sempre a linha "🧠 *Porque esta decisão?*" — confirma que o
        # bloco de explicação vem mesmo de `src.report.explainability`.
        snap = make_snapshot()
        criteria = evaluate_alert_criteria(snap)
        message = build_scanner_alert_message(snap, criteria)
        self.assertIn("🧠 *Porque esta decisão?*", message)


# ---------------------------------------------------------------------------
# run_scanner_cycle
# ---------------------------------------------------------------------------
class _ScannerCycleTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "live_alerts_scanner_test.db")
        self.sender = MagicMock(return_value=True)
        self.alert_monitor = LiveAlertMonitor(
            db_path=self.db_path, sender=self.sender, message_formatter=build_scanner_alert_message
        )
        self.guard = ScannerAntiSpamGuard(db_path=self.db_path)
        self.now = datetime(2026, 8, 4, 20, 0, 0, tzinfo=timezone.utc)
        self.engine = MagicMock()
        self.ml_predictor = MagicMock()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run(self, fetcher, odds_provider, now=None, logger=None):
        return run_scanner_cycle(
            fetcher=fetcher,
            odds_provider=odds_provider,
            engine=self.engine,
            ml_predictor=self.ml_predictor,
            alert_monitor=self.alert_monitor,
            guard=self.guard,
            now=now or self.now,
            logger=logger,
        )


class TestRunScannerCycleSendsPremiumAlertOnly(_ScannerCycleTestCase):

    def test_sends_alert_and_reports_counts(self):
        fetcher = _fake_fetcher([_fake_event(1, minute=60)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=1, card__minute=60)

        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap):
            result = self._run(fetcher, odds_provider)

        self.assertEqual(result.analyzed, 1)
        self.assertEqual(result.alerts_sent, 1)
        self.assertEqual(result.errors, [])
        self.sender.assert_called_once()
        message = self.sender.call_args.args[0]
        self.assertTrue(message.startswith(SCANNER_ALERT_HEADER))

    def test_never_calls_the_plus_ev_notifier(self):
        # O Scanner só pode enviar o Alerta Live Premium — nunca o alerta
        # +EV (`src.live.value_alerts.notify_if_value`) usado por
        # `run_live_pipeline`.
        fetcher = _fake_fetcher([_fake_event(1, minute=60)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=1, card__minute=60)

        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap), \
             patch("src.live.value_alerts.notify_if_value") as mock_notify:
            self._run(fetcher, odds_provider)

        mock_notify.assert_not_called()


class TestRunScannerCyclePerformance(_ScannerCycleTestCase):

    def test_deduplicates_same_match_id_within_one_cycle(self):
        event = _fake_event(1, minute=60)
        fetcher = _fake_fetcher([event, dict(event)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=1, card__minute=60)

        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap) as mock_build:
            result = self._run(fetcher, odds_provider)

        self.assertEqual(mock_build.call_count, 1)
        odds_provider.get_live_odds.assert_called_once()
        self.assertEqual(result.analyzed, 1)

    def test_reuses_the_same_snapshot_for_guard_and_send(self):
        fetcher = _fake_fetcher([_fake_event(1, minute=60)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=1, card__minute=60)

        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap) as mock_build:
            self._run(fetcher, odds_provider)

        # Uma única construção de snapshot por jogo, mesmo precisando de
        # ser usada tanto pelo guard como pelo envio real.
        mock_build.assert_called_once()


class TestRunScannerCycleReevaluation(_ScannerCycleTestCase):

    def test_does_not_resend_when_nothing_material_changed(self):
        fetcher = _fake_fetcher([_fake_event(2, minute=60)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=2, card__minute=60)

        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap):
            self._run(fetcher, odds_provider, now=self.now)
            later = self.now + timedelta(seconds=SCANNER_COOLDOWN_SECONDS + 1)
            result2 = self._run(fetcher, odds_provider, now=later)

        self.sender.assert_called_once()
        self.assertEqual(result2.alerts_sent, 0)

    def test_resends_when_decision_flips_from_aguardar_to_apostar_agora(self):
        fetcher = _fake_fetcher([_fake_event(3, minute=60)])
        odds_provider = _fake_odds_provider()

        waiting_snap = make_snapshot(match_id=3, card__minute=60, decision__label="🟡 AGUARDAR")
        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=waiting_snap):
            result1 = self._run(fetcher, odds_provider, now=self.now)

        self.assertEqual(result1.alerts_sent, 0)
        self.sender.assert_not_called()

        betting_snap = make_snapshot(match_id=3, card__minute=61)
        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=betting_snap):
            result2 = self._run(fetcher, odds_provider, now=self.now + timedelta(seconds=30))

        self.assertEqual(result2.alerts_sent, 1)
        self.sender.assert_called_once()

    def test_never_sends_after_minute_88(self):
        fetcher = _fake_fetcher([_fake_event(4, minute=89)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=4, card__minute=89)

        with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap):
            result = self._run(fetcher, odds_provider)

        self.assertEqual(result.alerts_sent, 0)
        self.sender.assert_not_called()


class TestRunScannerCycleLogging(_ScannerCycleTestCase):

    def test_logs_cycle_summary_with_required_fields(self):
        fetcher = _fake_fetcher([_fake_event(1, minute=60)])
        odds_provider = _fake_odds_provider()
        snap = make_snapshot(match_id=1, card__minute=60)

        with tempfile.TemporaryDirectory() as log_dir:
            log_path = os.path.join(log_dir, "live_scanner_test.log")
            logger = logging.getLogger("live_scanner_test_logger")
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            logger.addHandler(handler)
            try:
                with patch("src.alerts.live_scanner.build_premium_snapshot", return_value=snap):
                    self._run(fetcher, odds_provider, logger=logger)
            finally:
                logger.removeHandler(handler)
                handler.close()

            with open(log_path, encoding="utf-8") as f:
                content = f.read()

        self.assertIn("Ciclo concluído", content)
        self.assertIn("analisado", content)
        self.assertIn("enviado", content)
        self.assertIn("erro", content)

    def test_logs_errors_when_processing_a_match_fails(self):
        fetcher = _fake_fetcher([_fake_event(1, minute=60)])
        odds_provider = MagicMock()
        odds_provider.get_live_odds.side_effect = RuntimeError("falha simulada da BSD API")

        with tempfile.TemporaryDirectory() as log_dir:
            log_path = os.path.join(log_dir, "live_scanner_error_test.log")
            logger = logging.getLogger("live_scanner_error_test_logger")
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            logger.addHandler(handler)
            try:
                result = self._run(fetcher, odds_provider, logger=logger)
            finally:
                logger.removeHandler(handler)
                handler.close()

            with open(log_path, encoding="utf-8") as f:
                content = f.read()

        self.assertEqual(len(result.errors), 1)
        self.assertIn("falha simulada da BSD API", content)


class TestScannerLoggerWritesToLogsLiveScannerLog(unittest.TestCase):

    def test_get_scanner_logger_points_to_logs_live_scanner_log(self):
        from src.alerts.live_scanner import LOG_PATH

        self.assertEqual(LOG_PATH.name, "live_scanner.log")
        self.assertEqual(LOG_PATH.parent.name, "logs")


if __name__ == "__main__":
    unittest.main()

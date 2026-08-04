"""
Live Scanner — camada de execução contínua e autónoma (24/7, via GitHub
Actions) do Alerta Live Premium.

Este módulo NÃO implementa nem altera nenhum algoritmo do motor:
Dixon-Coles, Monte Carlo (`src.engine.simulation`), Goal Engine
(`src.live.engine`), Machine Learning (`src.model.ml_predictor`), Kelly
(`src.engine.kelly`), Edge/EV (`src.engine.edge`), Decision Engine
(`src.engine.decision` / `src.engine.live_decision`) e os 8 critérios do
Alerta Live Premium (`src.alerts.live_premium_alerts`) permanecem
exatamente como estão.

Responsabilidades (apenas orquestração/execução contínua):
  - iniciar `BSDLiveFetcher` e obter os jogos em direto;
  - reutilizar `src.report.dashboard_data.build_match_snapshot` — que já
    orquestra `LivePipeline` (via `calculate_dynamic_lambda`) e todos os
    módulos oficiais do motor — para produzir o MatchSnapshot de cada
    jogo, exatamente como `src.engine.live_monitor.run_live_pipeline` já
    faz para o Alerta Live Premium;
  - reutilizar `build_premium_snapshot` (extraída de
    `src.engine.live_monitor`) e `LiveAlertMonitor.evaluate_and_maybe_alert`
    para decidir/enviar o alerta — nenhuma lógica de critérios/anti-spam é
    duplicada aqui;
  - reutilizar a infraestrutura de envio Telegram já existente
    (`LiveAlertMonitor.sender` -> `send_premium_alert`, que por sua vez
    reutiliza `src.api.http_retry.post_with_retry`, a mesma usada por
    `src.utils.telegram_notifier`);
  - reutilizar `src.report.explainability` para o bloco "🧠 Explicação".

Ao contrário de `run_live_pipeline` (que também dispara o alerta +EV via
`src.live.value_alerts.notify_if_value`), este Scanner envia **apenas** o
Alerta Live Premium para o Telegram — é essa a única finalidade da
execução contínua 24/7 pedida.

Camada adicional de anti-spam (`ScannerAntiSpamGuard`): mais estrita do
que o anti-spam já existente em `LiveAlertMonitor` (cooldown de 10 min +
variação mínima de odd de 0.05) — aqui: cooldown de 15 min, minuto máximo
88, e "praticamente igual" também considera a confiança do modelo, não só
odd/decisão. É um filtro adicional aplicado ANTES de chamar
`LiveAlertMonitor.evaluate_and_maybe_alert`: nunca afrouxa o anti-spam
existente, só o reforça. Persiste o seu próprio estado numa tabela nova
(`live_scanner_guard_state`) dentro do MESMO ficheiro
`data/live_alerts.db` já usado por `LiveAlertMonitor` — não cria nenhuma
outra base de dados.
"""

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.api.live_fetcher import BSDLiveFetcher
from src.live.providers.api_odds_provider import APIOddsProvider
from src.live.engine import LiveGoalEngine
from src.model.ml_predictor import LiveMLPredictor
from src.report.explainability import format_explanation_block, generate_explanation
from src.alerts.live_premium_alerts import (
    AlertCriteriaResult,
    DEFAULT_ALERTS_DB_PATH,
    LiveAlertMonitor,
)
from src.engine.live_monitor import build_premium_snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = REPO_ROOT / "logs" / "live_scanner.log"

# ---------------------------------------------------------------------------
# Guard de anti-spam adicional do Scanner (ver docstring do módulo).
# ---------------------------------------------------------------------------
SCANNER_COOLDOWN_SECONDS = 15 * 60
SCANNER_MIN_ODD_DELTA = 0.03
SCANNER_MIN_CONFIDENCE_DELTA = 2.0
SCANNER_MAX_ALERT_MINUTE = 88


@dataclass
class _ScannerMatchState:
    match_id: str
    last_minute: int
    last_decision: str
    last_odd: float
    last_confidence: float
    last_sent_at: datetime


class ScannerAntiSpamGuard:
    """
    Proteção adicional de anti-spam do Scanner — ver docstring do módulo.
    Não decide os 8 critérios do Alerta Live Premium (isso continua
    exclusivamente em `evaluate_alert_criteria`) nem envia nada: só decide
    SE vale a pena sequer chamar `LiveAlertMonitor.evaluate_and_maybe_alert`
    para este ciclo.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_ALERTS_DB_PATH,
        cooldown_seconds: int = SCANNER_COOLDOWN_SECONDS,
        min_odd_delta: float = SCANNER_MIN_ODD_DELTA,
        min_confidence_delta: float = SCANNER_MIN_CONFIDENCE_DELTA,
        max_alert_minute: int = SCANNER_MAX_ALERT_MINUTE,
    ):
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        self.min_odd_delta = min_odd_delta
        self.min_confidence_delta = min_confidence_delta
        self.max_alert_minute = max_alert_minute
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_scanner_guard_state (
                match_id TEXT PRIMARY KEY,
                last_minute INTEGER,
                last_decision TEXT,
                last_odd REAL,
                last_confidence REAL,
                last_sent_at DATETIME
            )
            """
        )
        conn.commit()
        conn.close()

    def _load_state(self, match_id: str) -> Optional[_ScannerMatchState]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT match_id, last_minute, last_decision, last_odd, last_confidence, last_sent_at "
                "FROM live_scanner_guard_state WHERE match_id = ?",
                (match_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _ScannerMatchState(
            match_id=row[0],
            last_minute=row[1],
            last_decision=row[2],
            last_odd=row[3],
            last_confidence=row[4],
            last_sent_at=datetime.fromisoformat(row[5]),
        )

    def should_allow_send(
        self,
        match_id: Any,
        minute: int,
        decision: str,
        odd: float,
        confidence: Optional[float],
        now: datetime,
    ) -> Tuple[bool, str]:
        """
        Devolve (permitido, motivo). Nunca sozinho decide enviar — apenas
        pode BLOQUEAR uma tentativa de envio antes de chegar a
        `LiveAlertMonitor.evaluate_and_maybe_alert`.
        """
        if minute is not None and minute > self.max_alert_minute:
            return False, f"Minuto {minute} > {self.max_alert_minute} (fora da janela do Scanner)."

        state = self._load_state(str(match_id))
        if state is None:
            return True, "Sem alerta anterior do Scanner para este jogo."

        elapsed = (now - state.last_sent_at).total_seconds()
        if elapsed < self.cooldown_seconds:
            return False, f"Cooldown do Scanner ({self.cooldown_seconds}s) ainda não expirou ({elapsed:.0f}s)."

        decision_changed = decision != state.last_decision
        odd_delta = abs(odd - state.last_odd)
        confidence_delta = abs((confidence or 0.0) - (state.last_confidence or 0.0))

        if not decision_changed and odd_delta < self.min_odd_delta and confidence_delta < self.min_confidence_delta:
            return False, (
                "Decisão, odd e confiança praticamente inalteradas desde o último alerta "
                f"(Δodd={odd_delta:.3f}, Δconfiança={confidence_delta:.1f})."
            )

        return True, "Alteração material (decisão/odd/confiança) detetada desde o último alerta."

    def record_sent(
        self,
        match_id: Any,
        minute: int,
        decision: str,
        odd: float,
        confidence: Optional[float],
        now: datetime,
    ) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO live_scanner_guard_state
                (match_id, last_minute, last_decision, last_odd, last_confidence, last_sent_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                last_minute = excluded.last_minute,
                last_decision = excluded.last_decision,
                last_odd = excluded.last_odd,
                last_confidence = excluded.last_confidence,
                last_sent_at = excluded.last_sent_at
            """,
            (str(match_id), minute, decision, odd, confidence or 0.0, now.isoformat()),
        )
        conn.commit()
        conn.close()

    def clear_match(self, match_id: Any) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM live_scanner_guard_state WHERE match_id = ?", (str(match_id),))
        conn.commit()
        conn.close()

    def sync_active_matches(self, active_match_ids: Iterable[Any]) -> None:
        """Limpa o estado de jogos que já terminaram (mesma estratégia de
        `LiveAlertMonitor.sync_active_matches`: já não aparecem em
        `events/live`)."""
        active = {str(m) for m in active_match_ids}
        conn = self._connect()
        try:
            rows = conn.execute("SELECT match_id FROM live_scanner_guard_state").fetchall()
            stale = [r[0] for r in rows if r[0] not in active]
            for match_id in stale:
                conn.execute("DELETE FROM live_scanner_guard_state WHERE match_id = ?", (match_id,))
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Mensagem Telegram do Scanner (formato pedido — cabeçalho obrigatório
# "⚽ Football Edge Engine", nunca "BET"/"ALERTA"). Reutiliza
# `src.report.explainability` para o bloco de explicação (obrigatório) —
# não cria nenhuma explicação alternativa.
# ---------------------------------------------------------------------------
SCANNER_ALERT_HEADER = "⚽ Football Edge Engine"


def build_scanner_alert_message(snapshot: Dict[str, Any], criteria: AlertCriteriaResult) -> str:
    card = snapshot["card"]
    value = snapshot["value"]
    consensus = snapshot["consensus"]
    v = criteria.values

    explanation = generate_explanation(snapshot)

    lines = [
        SCANNER_ALERT_HEADER,
        "",
        f"🏆 {card.get('competition', 'Competição não identificada')}",
        f"⚽ {card['home_team']} vs {card['away_team']}",
        f"⏱ Minuto {card['minute']}",
        "",
        f"📈 Goal Engine: {v['goal_engine_probability']:.0f}%",
        f"🎲 Monte Carlo: {v['monte_carlo_probability']:.0f}%",
        f"🤖 ML: {v['ml_probability']:.0f}%",
        f"📊 Consenso: {consensus['label']}",
        "",
        f"💰 Odd: {value['bookie_odd']:.2f}",
        f"💎 Edge: {v['edge_pct']:+.1f}%",
        f"📈 EV: {v['ev_pct']:+.1f}%",
        f"💵 Kelly: {v['kelly_pct']:.1f}%",
        f"🎯 Stake recomendada: {v['kelly_pct']:.1f}% da banca",
        "",
        f"🚨 Decisão: {v['decision_label']}",
        "",
        format_explanation_block(explanation),
    ]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Logging (logs/live_scanner.log) — hora, jogos analisados, alertas
# enviados, tempo de execução, erros.
# ---------------------------------------------------------------------------
_scanner_logger: Optional[logging.Logger] = None


def get_scanner_logger() -> logging.Logger:
    global _scanner_logger
    if _scanner_logger is not None:
        return _scanner_logger

    logger = logging.getLogger("live_scanner")
    logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(LOG_PATH) for h in logger.handlers):
        os.makedirs(LOG_PATH.parent, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    _scanner_logger = logger
    return logger


# ---------------------------------------------------------------------------
# Ciclo do Scanner
# ---------------------------------------------------------------------------
@dataclass
class ScannerCycleResult:
    analyzed: int
    alerts_sent: int
    elapsed_seconds: float
    errors: List[str]


def run_scanner_cycle(
    *,
    fetcher: Optional[BSDLiveFetcher] = None,
    odds_provider: Optional[APIOddsProvider] = None,
    engine: Optional[LiveGoalEngine] = None,
    ml_predictor: Optional[LiveMLPredictor] = None,
    alert_monitor: Optional[LiveAlertMonitor] = None,
    guard: Optional[ScannerAntiSpamGuard] = None,
    logger: Optional[logging.Logger] = None,
    now: Optional[datetime] = None,
) -> ScannerCycleResult:
    """
    Um ciclo completo do Scanner: obtém os jogos em direto, constrói o
    MatchSnapshot de cada um (via `build_premium_snapshot`, reutilizada de
    `src.engine.live_monitor`) e envia o Alerta Live Premium apenas para
    os que passam tanto os 8 critérios oficiais
    (`LiveAlertMonitor.evaluate_and_maybe_alert`) como o guard adicional
    acima. Nunca envia o alerta +EV (`notify_if_value`) — apenas o Alerta
    Live Premium, conforme o objetivo do Scanner autónomo.
    """
    logger = logger or get_scanner_logger()
    cycle_started = time.monotonic()
    errors: List[str] = []

    if fetcher is None:
        try:
            fetcher = BSDLiveFetcher()
        except Exception as e:
            msg = f"Erro ao inicializar BSDLiveFetcher: {e}"
            logger.error(msg)
            errors.append(msg)
            return ScannerCycleResult(analyzed=0, alerts_sent=0, elapsed_seconds=time.monotonic() - cycle_started, errors=errors)

    if odds_provider is None:
        try:
            odds_provider = APIOddsProvider()
        except Exception as e:
            msg = f"Odds provider indisponível (Alerta Live Premium desativado nesta run): {e}"
            logger.warning(msg)
            odds_provider = None

    engine = engine or LiveGoalEngine()
    ml_predictor = ml_predictor or LiveMLPredictor()
    alert_monitor = alert_monitor or LiveAlertMonitor(message_formatter=build_scanner_alert_message)
    guard = guard or ScannerAntiSpamGuard()
    now = now or datetime.now(timezone.utc)

    events = fetcher.get_live_events()

    # Nunca analisar o mesmo jogo duas vezes no mesmo ciclo.
    seen_ids = set()
    deduped_events = []
    for event in events:
        event_id = event.get("id")
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        deduped_events.append(event)

    active_ids = [event.get("id") for event in deduped_events]
    alert_monitor.sync_active_matches(active_ids)
    guard.sync_active_matches(active_ids)

    analyzed = 0
    alerts_sent = 0

    for event in deduped_events:
        match_id = event.get("id")
        try:
            match_data = fetcher.parse_live_metrics_for_engine(event)

            if match_data.get("home_score") is None or match_data.get("away_score") is None:
                continue

            if odds_provider is None:
                continue

            analyzed += 1

            odds_response = odds_provider.get_live_odds(match_data["match_id"])
            bookie_odd = odds_response["odds"]["over_15_goals"]

            # Reutiliza `build_premium_snapshot` (extraída de
            # `src.engine.live_monitor`, mesma função usada por
            # `run_live_pipeline`) — o snapshot é construído UMA única vez
            # e reutilizado tanto para a decisão do guard como para o
            # envio real, sem repetir nenhuma chamada HTTP nem recalcular
            # nenhum modelo.
            snapshot = build_premium_snapshot(
                match_data, bookie_odd, ml_predictor=ml_predictor, engine=engine
            )

            minute = snapshot["card"]["minute"]
            decision = snapshot["decision"]["label"]
            confidence = snapshot["decision"]["confidence_score"]

            allowed, reason = guard.should_allow_send(
                match_id=match_id,
                minute=minute,
                decision=decision,
                odd=bookie_odd,
                confidence=confidence,
                now=now,
            )
            if not allowed:
                logger.info("Jogo %s: alerta bloqueado pelo guard do Scanner (%s).", match_id, reason)
                continue

            outcome = alert_monitor.evaluate_and_maybe_alert(snapshot, now=now)
            if outcome.sent:
                alerts_sent += 1
                guard.record_sent(match_id, minute, decision, bookie_odd, confidence, now)
                logger.info(
                    "🔥 Alerta Live Premium enviado (Scanner): %s vs %s (match_id=%s).",
                    snapshot["card"]["home_team"], snapshot["card"]["away_team"], match_id,
                )
        except Exception as exc:
            msg = f"Erro ao processar match_id={match_id}: {exc}"
            errors.append(msg)
            logger.exception(msg)

    elapsed = time.monotonic() - cycle_started
    logger.info(
        "Ciclo concluído: %d jogo(s) analisado(s), %d alerta(s) enviado(s), "
        "%.2fs de execução, %d erro(s).",
        analyzed, alerts_sent, elapsed, len(errors),
    )

    return ScannerCycleResult(analyzed=analyzed, alerts_sent=alerts_sent, elapsed_seconds=elapsed, errors=errors)


if __name__ == "__main__":
    run_scanner_cycle()

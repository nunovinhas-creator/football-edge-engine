"""
Live Premium Alerts — camada de monitorização e notificação Telegram.

Este módulo NÃO implementa nenhum algoritmo/fórmula do motor: Dixon-Coles,
Monte Carlo (`src.engine.simulation`), Goal Engine (`src.live.engine`),
Machine Learning (`src.model.ml_predictor`), Kelly (`src.engine.kelly`),
Edge/EV (`src.engine.edge`) e Decision Engine (`src.engine.decision` /
`src.engine.live_decision`) permanecem exatamente como estão — nenhum
threshold do motor é alterado e nenhuma probabilidade/edge/EV/Kelly é
recalculada aqui.

Este módulo só LÊ os valores já produzidos por
`src.report.dashboard_data.build_match_snapshot` (o MatchSnapshot que já
alimenta o Dashboard Pro) e decide, com um conjunto FIXO de 8 critérios
(cada um já uma comparação simples sobre um valor existente, nenhum novo
cálculo), SE um alerta "Live Premium" deve ser enviado — e evita
reenviar o mesmo alerta em excesso (anti-spam persistido em SQLite, para
sobreviver a reinícios e a múltiplos processos: o script de monitorização
`src/engine/live_monitor.py` e o Dashboard Streamlit `scripts/app.py`).

Assunção de desenho (critério 1 — Monte Carlo):
`src.engine.simulation.MonteCarloSimulator` não produz uma probabilidade
"Over 0.5 Next Goal" isolada — apenas Over 1.5 / Over 2.5 / BTTS para o
resto do jogo (ver `SimulationResult`). Conforme o próprio requisito
autoriza explicitamente ("se existir apenas Over1.5, utilizar a
probabilidade já produzida pelo motor"), este módulo usa
`models.monte_carlo.over_15` — a única probabilidade de golo que o motor
Monte Carlo já produz e expõe no MatchSnapshot — como o "equivalente
usado atualmente" para este critério. Nenhuma probabilidade nova é
estimada.
"""

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

from src.api.http_retry import post_with_retry
from src.report.explainability import PRESSURE_HIGH

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ALERTS_DB_PATH = str(REPO_ROOT / "data" / "live_alerts.db")

# Cabeçalho obrigatório de todas as mensagens deste alerta (nunca "Bet").
ALERT_BRAND_HEADER = "🔥 FOOTBALL EDGE ENGINE"

# Nome de apresentação do mercado avaliado por este alerta — o mesmo
# mercado do Goal Engine (`next_goal_probability`) já usado por
# `evaluate_live_market`/`build_match_snapshot` ("Próximo Golo (15m)"),
# apenas com o rótulo pedido pela especificação do Alerta Live Premium.
ALERT_MARKET_LABEL = "Over 0.5 Next Goal"

# ---------------------------------------------------------------------------
# Critérios fixos do Alerta Live Premium (ver especificação) — todos
# comparações simples sobre valores já calculados pelo motor.
# ---------------------------------------------------------------------------
MONTE_CARLO_MIN_PROB = 70.0
GOAL_ENGINE_MIN_PROB = 70.0
REQUIRED_DECISION_LABEL = "🟢 APOSTAR AGORA"
MIN_EDGE_PCT = 5.0
ODD_MIN = 1.40
ODD_MAX = 2.30
MAX_CONSENSUS_GAP_PP = 15.0

# ---------------------------------------------------------------------------
# Anti-spam
# ---------------------------------------------------------------------------
COOLDOWN_SECONDS = 600
MIN_ODD_DELTA = 0.05


@dataclass
class AlertCriteriaResult:
    passed: bool
    failed_reasons: List[str]
    checks: Dict[str, bool]
    values: Dict[str, Any]


def evaluate_alert_criteria(snapshot: Dict[str, Any]) -> AlertCriteriaResult:
    """
    Avalia os 8 critérios do Alerta Live Premium sobre um MatchSnapshot já
    construído por `build_match_snapshot`. Pura — não tem efeitos
    secundários, não envia nada, não decide anti-spam.
    """
    models = snapshot["models"]
    value = snapshot["value"]
    consensus = snapshot["consensus"]
    decision_label = snapshot["decision"]["label"]

    monte_carlo_probability = models["monte_carlo"]["over_15"]
    goal_engine_probability = models["goal_engine"]["probability"]
    ml_probability = models["machine_learning"]["probability"]
    edge_pct = value["edge_pct"]
    ev_pct = value["ev_pct"]
    kelly_pct = value["kelly_pct"]
    odd = value["bookie_odd"]
    consensus_gap = consensus["gap"]

    checks = {
        "monte_carlo": monte_carlo_probability >= MONTE_CARLO_MIN_PROB,
        "goal_engine": goal_engine_probability >= GOAL_ENGINE_MIN_PROB,
        "decision": decision_label == REQUIRED_DECISION_LABEL,
        "edge": edge_pct >= MIN_EDGE_PCT,
        "ev": ev_pct > 0,
        "kelly": kelly_pct > 0,
        "odd_range": ODD_MIN <= odd <= ODD_MAX,
        "consensus": consensus_gap <= MAX_CONSENSUS_GAP_PP,
    }

    reasons = []
    if not checks["monte_carlo"]:
        reasons.append(f"Monte Carlo {monte_carlo_probability:.1f}% < {MONTE_CARLO_MIN_PROB:.0f}%")
    if not checks["goal_engine"]:
        reasons.append(f"Goal Engine {goal_engine_probability:.1f}% < {GOAL_ENGINE_MIN_PROB:.0f}%")
    if not checks["decision"]:
        reasons.append(f'Decisão "{decision_label}" != "{REQUIRED_DECISION_LABEL}"')
    if not checks["edge"]:
        reasons.append(f"Edge {edge_pct:+.1f}% < {MIN_EDGE_PCT:.0f}%")
    if not checks["ev"]:
        reasons.append(f"EV {ev_pct:+.1f}% <= 0%")
    if not checks["kelly"]:
        reasons.append(f"Kelly {kelly_pct:.2f}% <= 0%")
    if not checks["odd_range"]:
        reasons.append(f"Odd {odd:.2f} fora do intervalo [{ODD_MIN:.2f}, {ODD_MAX:.2f}]")
    if not checks["consensus"]:
        reasons.append(
            f"Consenso Goal Engine/ML: diferença {consensus_gap:.1f} p.p. > {MAX_CONSENSUS_GAP_PP:.0f} p.p."
        )

    values = {
        "monte_carlo_probability": monte_carlo_probability,
        "goal_engine_probability": goal_engine_probability,
        "ml_probability": ml_probability,
        "edge_pct": edge_pct,
        "ev_pct": ev_pct,
        "kelly_pct": kelly_pct,
        "odd": odd,
        "consensus_gap": consensus_gap,
        "decision_label": decision_label,
    }

    return AlertCriteriaResult(
        passed=all(checks.values()), failed_reasons=reasons, checks=checks, values=values
    )


@dataclass
class AlertOutcome:
    match_id: str
    sent: bool
    state: str
    reason: str
    criteria: Optional[AlertCriteriaResult]
    telegram_sent: Optional[bool] = None


@dataclass
class _MatchAlertState:
    match_id: str
    last_minute: int
    last_decision: str
    last_odd: float
    last_sent_at: datetime


def _build_explanation_bullets(snapshot: Dict[str, Any], criteria: AlertCriteriaResult) -> List[str]:
    """
    Bloco "Explicação" da mensagem — texto puramente interpretativo sobre
    valores já calculados (mesmo espírito de `src.report.explainability`),
    nenhum cálculo novo. Reutiliza `PRESSURE_HIGH` já definido em
    `src.report.explainability` em vez de inventar um novo limiar de
    apresentação.
    """
    bullets: List[str] = []
    live = snapshot["live"]
    value = snapshot["value"]

    if live["pressure"] >= PRESSURE_HIGH:
        bullets.append("pressão ofensiva muito elevada")

    bullets.append("consenso entre todos os modelos")

    fair_odd = value.get("fair_odd")
    if fair_odd is not None and criteria.values["odd"] > fair_odd:
        bullets.append("odd acima da odd justa")

    bullets.append("valor esperado positivo")
    bullets.append(f"stake recomendada {criteria.values['kelly_pct']:.1f}%")

    return bullets


def format_alert_message(snapshot: Dict[str, Any], criteria: AlertCriteriaResult) -> str:
    """
    Constrói a mensagem Telegram curta e legível do Alerta Live Premium.
    Começa sempre por `ALERT_BRAND_HEADER` ("🔥 FOOTBALL EDGE ENGINE"),
    nunca apenas "Bet". Usa exclusivamente valores já presentes no
    MatchSnapshot / `criteria.values`.
    """
    card = snapshot["card"]
    value = snapshot["value"]
    consensus = snapshot["consensus"]
    v = criteria.values

    lines = [
        ALERT_BRAND_HEADER,
        "",
        snapshot["decision"]["label"],
        "",
        f"{card['home_team']} vs {card['away_team']}",
        "",
        f"Minuto {card['minute']}",
        "",
        "Mercado:",
        ALERT_MARKET_LABEL,
        "",
        "Probabilidade Goal Engine:",
        f"{v['goal_engine_probability']:.0f}%",
        "",
        "Probabilidade Monte Carlo:",
        f"{v['monte_carlo_probability']:.0f}%",
        "",
        "ML:",
        f"{v['ml_probability']:.0f}%",
        "",
        "Edge:",
        f"{v['edge_pct']:+.1f}%",
        "",
        "EV:",
        f"{v['ev_pct']:+.0f}%",
        "",
        "Kelly:",
        f"{v['kelly_pct']:.1f}%",
        "",
        "Odd:",
        f"{value['bookie_odd']:.2f}",
        "",
        "Consenso:",
        consensus["label"],
        "",
        "Explicação:",
        "",
    ]

    for bullet in _build_explanation_bullets(snapshot, criteria):
        lines.append(f"• {bullet}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def send_premium_alert(message: str) -> bool:
    """
    Envia `message` tal como recebida (já contém `ALERT_BRAND_HEADER`) —
    ao contrário de `src.utils.telegram_notifier.send_telegram_alert`, NÃO
    antepõe nenhum prefixo próprio (esse prefixo é "⚽ FOOTBALL EDGE
    ENGINE", diferente do cabeçalho obrigatório deste alerta, e duplicaria
    o cabeçalho). Reutiliza `post_with_retry` (mesma infraestrutura de
    retries já usada por `send_telegram_alert`) em vez de reimplementar
    chamadas HTTP.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.error(
            "Live Premium Alert: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID em falta — alerta NÃO enviado."
        )
        print("⚠️ Alerta Live Premium omitido: Telegram não configurado.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        response = post_with_retry(url, json=payload, timeout=15)
    except (requests.Timeout, requests.ConnectionError) as e:
        logger.error("Live Premium Alert: falha de rede ao enviar: %s", e)
        print(f"❌ Erro de rede ao enviar Alerta Live Premium: {e}")
        return False
    except Exception:
        logger.exception("Live Premium Alert: erro inesperado ao enviar")
        return False

    if response.status_code == 200:
        logger.info("Live Premium Alert: enviado com sucesso (HTTP 200).")
        print("🔥 Alerta Live Premium enviado para o Telegram com sucesso!")
        return True

    logger.error(
        "Live Premium Alert: Telegram respondeu com erro HTTP %d: %s",
        response.status_code, response.text
    )
    print(f"❌ Erro da API do Telegram (Alerta Live Premium): HTTP {response.status_code}")
    return False


class LiveAlertMonitor:
    """
    Camada de monitorização com estado (anti-spam persistido em SQLite,
    `data/live_alerts.db` por omissão) — decide, para cada MatchSnapshot
    avaliado, SE o alerta deve realmente ser enviado, evitando duplicados
    para o mesmo jogo.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_ALERTS_DB_PATH,
        cooldown_seconds: int = COOLDOWN_SECONDS,
        min_odd_delta: float = MIN_ODD_DELTA,
        sender: Optional[Callable[[str], bool]] = None,
    ):
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        self.min_odd_delta = min_odd_delta
        self.sender = sender or send_premium_alert
        self._init_db()

    # -- infraestrutura SQLite -------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                match_id TEXT,
                home_team TEXT,
                away_team TEXT,
                minute INTEGER,
                market TEXT,
                odd REAL,
                goal_engine_probability REAL,
                monte_carlo_probability REAL,
                ml_probability REAL,
                edge REAL,
                ev REAL,
                kelly REAL,
                decision TEXT,
                telegram_sent BOOLEAN
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_alert_state (
                match_id TEXT PRIMARY KEY,
                last_minute INTEGER,
                last_decision TEXT,
                last_odd REAL,
                last_sent_at DATETIME
            )
            """
        )
        conn.commit()
        conn.close()

    # -- estado anti-spam --------------------------------------------------

    def _load_state(self, match_id: str) -> Optional[_MatchAlertState]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT match_id, last_minute, last_decision, last_odd, last_sent_at "
                "FROM live_alert_state WHERE match_id = ?",
                (match_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _MatchAlertState(
            match_id=row[0],
            last_minute=row[1],
            last_decision=row[2],
            last_odd=row[3],
            last_sent_at=datetime.fromisoformat(row[4]),
        )

    def _save_state(self, state: _MatchAlertState) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO live_alert_state (match_id, last_minute, last_decision, last_odd, last_sent_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                last_minute = excluded.last_minute,
                last_decision = excluded.last_decision,
                last_odd = excluded.last_odd,
                last_sent_at = excluded.last_sent_at
            """,
            (state.match_id, state.last_minute, state.last_decision, state.last_odd, state.last_sent_at.isoformat()),
        )
        conn.commit()
        conn.close()

    def clear_match(self, match_id) -> None:
        """Limpa o registo interno de anti-spam de um jogo (chamado quando o jogo termina)."""
        conn = self._connect()
        conn.execute("DELETE FROM live_alert_state WHERE match_id = ?", (str(match_id),))
        conn.commit()
        conn.close()

    def sync_active_matches(self, active_match_ids: Iterable[Any]) -> None:
        """
        Remove do registo interno qualquer jogo que já não esteja na lista
        de jogos ao vivo atual (`active_match_ids`) — a forma de detetar
        "jogo terminou" quando o provider de jogos ao vivo simplesmente
        deixa de o listar (sem um campo explícito de status "FT").
        """
        active = {str(m) for m in active_match_ids}
        conn = self._connect()
        try:
            rows = conn.execute("SELECT match_id FROM live_alert_state").fetchall()
            stale = [r[0] for r in rows if r[0] not in active]
            for match_id in stale:
                conn.execute("DELETE FROM live_alert_state WHERE match_id = ?", (match_id,))
            conn.commit()
        finally:
            conn.close()

    def _should_send(self, match_id: str, minute: int, decision: str, odd: float, now: datetime) -> bool:
        state = self._load_state(match_id)
        if state is None:
            return True

        elapsed = (now - state.last_sent_at).total_seconds()
        if elapsed < self.cooldown_seconds:
            return False

        odd_changed = abs(odd - state.last_odd) >= self.min_odd_delta
        decision_changed = decision != state.last_decision
        return odd_changed or decision_changed

    # -- log de alertas -----------------------------------------------------

    def _log_alert(
        self, snapshot: Dict[str, Any], criteria: AlertCriteriaResult, telegram_sent: bool, now: datetime
    ) -> None:
        card = snapshot["card"]
        value = snapshot["value"]
        v = criteria.values
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO live_alerts (
                timestamp, match_id, home_team, away_team, minute, market, odd,
                goal_engine_probability, monte_carlo_probability, ml_probability,
                edge, ev, kelly, decision, telegram_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now.isoformat(),
                str(snapshot.get("match_id")),
                card["home_team"],
                card["away_team"],
                card["minute"],
                ALERT_MARKET_LABEL,
                value["bookie_odd"],
                v["goal_engine_probability"],
                v["monte_carlo_probability"],
                v["ml_probability"],
                v["edge_pct"],
                v["ev_pct"],
                v["kelly_pct"],
                v["decision_label"],
                int(telegram_sent),
            ),
        )
        conn.commit()
        conn.close()

    def load_alerts(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM live_alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def alerts_sent_today(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        today = now.date().isoformat()
        if not os.path.exists(self.db_path):
            return 0
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT timestamp FROM live_alerts WHERE telegram_sent = 1"
            ).fetchall()
        finally:
            conn.close()
        return sum(1 for (ts,) in rows if isinstance(ts, str) and ts.startswith(today))

    # -- ponto de entrada principal ------------------------------------------

    def evaluate_and_maybe_alert(
        self,
        snapshot: Dict[str, Any],
        finished: bool = False,
        now: Optional[datetime] = None,
    ) -> AlertOutcome:
        """
        Avalia os 8 critérios sobre `snapshot` (MatchSnapshot já produzido
        por `build_match_snapshot`) e envia o Alerta Live Premium via
        Telegram apenas se TODOS passarem e o anti-spam permitir.

        `finished=True`: limpa o registo interno deste jogo (regra "se o
        jogo terminar limpar automaticamente") e não avalia critérios nem
        envia nada.
        """
        match_id = str(snapshot.get("match_id"))
        now = now or datetime.now(timezone.utc)

        if finished:
            self.clear_match(match_id)
            return AlertOutcome(
                match_id=match_id,
                sent=False,
                state="FINALIZADO",
                reason="Jogo terminado — registo de anti-spam limpo.",
                criteria=None,
            )

        criteria = evaluate_alert_criteria(snapshot)

        if not criteria.passed:
            return AlertOutcome(
                match_id=match_id,
                sent=False,
                state="À ESPERA",
                reason="; ".join(criteria.failed_reasons),
                criteria=criteria,
            )

        minute = snapshot["card"]["minute"]
        decision = snapshot["decision"]["label"]
        odd = snapshot["value"]["bookie_odd"]

        if not self._should_send(match_id, minute, decision, odd, now):
            return AlertOutcome(
                match_id=match_id,
                sent=False,
                state="ATIVO",
                reason="Critérios reunidos, mas alerta já enviado recentemente para este jogo (anti-spam).",
                criteria=criteria,
            )

        message = format_alert_message(snapshot, criteria)
        telegram_sent = bool(self.sender(message))

        self._log_alert(snapshot, criteria, telegram_sent, now)
        if telegram_sent:
            self._save_state(_MatchAlertState(match_id, minute, decision, odd, now))

        return AlertOutcome(
            match_id=match_id,
            sent=telegram_sent,
            state="ALERTA ENVIADO" if telegram_sent else "À ESPERA",
            reason="Todos os 8 critérios reunidos." if telegram_sent else "Critérios reunidos, mas o envio Telegram falhou.",
            criteria=criteria,
            telegram_sent=telegram_sent,
        )

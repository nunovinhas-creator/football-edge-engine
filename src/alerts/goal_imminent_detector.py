"""
Goal Imminent Detection — camada de observação adicional, totalmente
independente do motor de apostas.

Este módulo NÃO implementa nem recalcula nenhum algoritmo: Dixon-Coles,
Monte Carlo (`src.engine.simulation`), Goal Engine (`src.live.engine`),
Machine Learning (`src.model.ml_predictor`), Kelly (`src.engine.kelly`),
Edge/EV (`src.engine.edge`), Decision Engine (`src.engine.decision` /
`src.engine.live_decision`) e Lambda Estimator permanecem exatamente como
estão. Nenhuma probabilidade já calculada é alterada.

`evaluate_goal_imminent_criteria(snapshot)` recebe exclusivamente o
MatchSnapshot já produzido por
`src.report.dashboard_data.build_match_snapshot` (o mesmo snapshot que
alimenta o Dashboard Pro e o Alerta Live Premium) e só LÊ os valores já
lá calculados — nunca chama novamente Goal Engine, Monte Carlo, ML ou
Dixon-Coles.

Diferença face ao Alerta Live Premium (`src.alerts.live_premium_alerts`):
esta camada não decide "há valor para apostar" — decide "há uma
probabilidade excecionalmente elevada de golo iminente". É um alerta
distinto (`alert_type = "GOAL_IMMINENT"`), com os seus próprios 12
critérios (mais exigentes) e sem reavaliação: no máximo UM alerta deste
tipo por jogo (`data/goal_imminent_alerts.db`).

Origem dos limiares (nenhum é inventado):
  - `REQUIRED_DECISION_LABEL` e o Monte Carlo (Over 1.5 como "equivalente
    Next Goal") são reutilizados tal e qual de
    `src.alerts.live_premium_alerts` — mesma decisão do motor, mesma
    assunção de desenho já documentada lá para o Monte Carlo;
  - `PRESSURE_HIGH` é reutilizado de `src.report.explainability` — o
    mesmo limiar que já classifica a pressão como "muito elevada" no
    Alerta Live Premium;
  - `DANGEROUS_ATTACKS_HIGH_THRESHOLD` (15) e
    `SHOTS_ON_TARGET_HIGH_THRESHOLD` (8) são os mesmos denominadores de
    saturação já usados por `LiveGoalEngine.calculate_dominance_index`
    (`dangerous_attacks_10m / 15.0` e `shots_10m / 8.0`) — o ponto a
    partir do qual o próprio motor já considera este valor "no máximo"/
    elevado. Não é uma fórmula nova: é o mesmo número já existente no
    motor, apenas comparado diretamente (`>=`) em vez de normalizado.
"""

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from src.api.http_retry import post_with_retry
from src.alerts.live_premium_alerts import REQUIRED_DECISION_LABEL
from src.report.explainability import PRESSURE_HIGH

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GOAL_IMMINENT_DB_PATH = str(REPO_ROOT / "data" / "goal_imminent_alerts.db")

GOAL_IMMINENT_ALERT_TYPE = "GOAL_IMMINENT"
GOAL_IMMINENT_ALERT_HEADER = "⚽ Football Edge Engine"

# ---------------------------------------------------------------------------
# Critérios fixos do Goal Imminent Detection (12 critérios — ver docstring
# do módulo para a origem de cada limiar).
# ---------------------------------------------------------------------------
GOAL_ENGINE_MIN_PROB = 80.0
MONTE_CARLO_MIN_PROB = 75.0
ML_MIN_PROB = 70.0
EDGE_MIN_PCT = 5.0
REQUIRED_CONSENSUS_LABEL = "Muito Forte"
DANGEROUS_ATTACKS_HIGH_THRESHOLD = 15
SHOTS_ON_TARGET_HIGH_THRESHOLD = 8

FINISHED_STATUS_TOKENS = {
    "FT", "FINISHED", "TERMINADO", "ENCERRADO", "ENDED", "FULL TIME", "FULL-TIME",
}


@dataclass
class GoalImminentCriteriaResult:
    passed: bool
    failed_reasons: List[str]
    checks: Dict[str, bool]
    values: Dict[str, Any]


def evaluate_goal_imminent_criteria(snapshot: Dict[str, Any]) -> GoalImminentCriteriaResult:
    """
    Avalia os 12 critérios do Goal Imminent Detection sobre um
    MatchSnapshot já construído por `build_match_snapshot`. Pura — não
    tem efeitos secundários, não envia nada, não decide anti-spam.
    """
    card = snapshot["card"]
    models = snapshot["models"]
    value = snapshot["value"]
    consensus = snapshot["consensus"]
    live = snapshot["live"]
    decision_label = snapshot["decision"]["label"]

    goal_engine_probability = models["goal_engine"]["probability"]
    # Não existe probabilidade "Next Goal" separada no Monte Carlo — usa-se
    # exatamente `over_15`, a mesma assunção já documentada e usada por
    # `src.alerts.live_premium_alerts.evaluate_alert_criteria`.
    monte_carlo_probability = models["monte_carlo"]["over_15"]
    ml_probability = models["machine_learning"]["probability"]
    edge_pct = value["edge_pct"]
    ev_pct = value["ev_pct"]
    kelly_pct = value["kelly_pct"]
    consensus_label = consensus["label"]
    pressure = live["pressure"]
    dangerous_attacks = live["dangerous_attacks_10m"]
    shots_on_target = live["shots_on_target_10m"]
    status = str(card.get("status", "")).strip().upper()

    checks = {
        "decision": decision_label == REQUIRED_DECISION_LABEL,
        "goal_engine": goal_engine_probability >= GOAL_ENGINE_MIN_PROB,
        "monte_carlo": monte_carlo_probability >= MONTE_CARLO_MIN_PROB,
        "ml": ml_probability >= ML_MIN_PROB,
        "edge": edge_pct >= EDGE_MIN_PCT,
        "ev": ev_pct > 0,
        "kelly": kelly_pct > 0,
        "consensus": consensus_label == REQUIRED_CONSENSUS_LABEL,
        "pressure": pressure >= PRESSURE_HIGH,
        "dangerous_attacks": dangerous_attacks >= DANGEROUS_ATTACKS_HIGH_THRESHOLD,
        "shots_on_target": shots_on_target >= SHOTS_ON_TARGET_HIGH_THRESHOLD,
        "not_finished": status not in FINISHED_STATUS_TOKENS,
    }

    reasons: List[str] = []
    if not checks["decision"]:
        reasons.append(f'Decisão "{decision_label}" != "{REQUIRED_DECISION_LABEL}"')
    if not checks["goal_engine"]:
        reasons.append(f"Goal Engine {goal_engine_probability:.1f}% < {GOAL_ENGINE_MIN_PROB:.0f}%")
    if not checks["monte_carlo"]:
        reasons.append(f"Monte Carlo {monte_carlo_probability:.1f}% < {MONTE_CARLO_MIN_PROB:.0f}%")
    if not checks["ml"]:
        reasons.append(f"ML {ml_probability:.1f}% < {ML_MIN_PROB:.0f}%")
    if not checks["edge"]:
        reasons.append(f"Edge {edge_pct:+.1f}% < {EDGE_MIN_PCT:.0f}%")
    if not checks["ev"]:
        reasons.append(f"EV {ev_pct:+.1f}% <= 0%")
    if not checks["kelly"]:
        reasons.append(f"Kelly {kelly_pct:.2f}% <= 0%")
    if not checks["consensus"]:
        reasons.append(f'Consenso "{consensus_label}" != "{REQUIRED_CONSENSUS_LABEL}"')
    if not checks["pressure"]:
        reasons.append(f"Pressão {pressure:.0f} < {PRESSURE_HIGH:.0f} (Alta/Muito Alta)")
    if not checks["dangerous_attacks"]:
        reasons.append(f"Ataques perigosos {dangerous_attacks} < {DANGEROUS_ATTACKS_HIGH_THRESHOLD}")
    if not checks["shots_on_target"]:
        reasons.append(f"Remates à baliza {shots_on_target} < {SHOTS_ON_TARGET_HIGH_THRESHOLD}")
    if not checks["not_finished"]:
        reasons.append(f'Jogo já terminado (status="{status}")')

    values = {
        "competition": card.get("competition", "Competição não identificada"),
        "home_team": card.get("home_team", "Casa"),
        "away_team": card.get("away_team", "Fora"),
        "minute": card.get("minute"),
        "decision_label": decision_label,
        "goal_engine_probability": goal_engine_probability,
        "monte_carlo_probability": monte_carlo_probability,
        "ml_probability": ml_probability,
        "consensus_label": consensus_label,
        "edge_pct": edge_pct,
        "ev_pct": ev_pct,
        "kelly_pct": kelly_pct,
        "bookie_odd": value.get("bookie_odd"),
        "pressure": pressure,
        "dangerous_attacks": dangerous_attacks,
        "shots_on_target": shots_on_target,
        "estimated_xg_10m": live.get("estimated_xg_10m"),
        "momentum": live.get("momentum"),
        "status": status,
    }

    return GoalImminentCriteriaResult(
        passed=all(checks.values()), failed_reasons=reasons, checks=checks, values=values
    )


def build_goal_imminent_message(snapshot: Dict[str, Any], criteria: GoalImminentCriteriaResult) -> str:
    """
    Mensagem Telegram do Goal Imminent Detection — começa sempre por
    `GOAL_IMMINENT_ALERT_HEADER` ("⚽ Football Edge Engine"), para
    distinguir de outros bots Telegram. Usa exclusivamente valores já
    presentes em `criteria.values` (nenhum cálculo novo).
    """
    v = criteria.values

    lines = [
        GOAL_IMMINENT_ALERT_HEADER,
        "",
        "🚨 GOLO MUITO PROVÁVEL",
        "",
        "Liga:",
        str(v["competition"]),
        "",
        "Jogo:",
        f"{v['home_team']} vs {v['away_team']}",
        "",
        "Minuto:",
        str(v["minute"]),
        "",
        "Goal Engine:",
        f"{v['goal_engine_probability']:.0f}%",
        "",
        "Monte Carlo:",
        f"{v['monte_carlo_probability']:.0f}%",
        "",
        "Machine Learning:",
        f"{v['ml_probability']:.0f}%",
        "",
        "Consenso:",
        str(v["consensus_label"]),
        "",
        "Edge:",
        f"{v['edge_pct']:+.1f}%",
        "",
        "EV:",
        f"{v['ev_pct']:+.1f}%",
        "",
        "Kelly:",
        f"{v['kelly_pct']:.1f}%",
        "",
        "Odd Mercado:",
        f"{v['bookie_odd']:.2f}" if v.get("bookie_odd") is not None else "—",
        "",
        "Pressão:",
        f"{v['pressure']:.0f}/100",
        "",
        "Ataques perigosos:",
        str(v["dangerous_attacks"]),
        "",
        "Remates:",
        str(v["shots_on_target"]),
        "",
        "xG:",
        f"{v['estimated_xg_10m']:.2f}" if v.get("estimated_xg_10m") is not None else "—",
        "",
        "Momentum:",
        str(v.get("momentum") or "—"),
        "",
        "🔥 APOSTAR AGORA",
    ]

    return "\n".join(lines).rstrip() + "\n"


def send_goal_imminent_alert(message: str) -> bool:
    """
    Envia `message` tal como recebida (já contém `GOAL_IMMINENT_ALERT_HEADER`)
    — reutiliza `post_with_retry` (mesma infraestrutura de retries já usada
    por `src.utils.telegram_notifier` e `src.alerts.live_premium_alerts`)
    em vez de reimplementar chamadas HTTP.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.error(
            "Goal Imminent Detection: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID em falta — alerta NÃO enviado."
        )
        print("⚠️ Alerta Goal Imminent omitido: Telegram não configurado.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        response = post_with_retry(url, json=payload, timeout=15)
    except (requests.Timeout, requests.ConnectionError) as e:
        logger.error("Goal Imminent Detection: falha de rede ao enviar: %s", e)
        print(f"❌ Erro de rede ao enviar Alerta Goal Imminent: {e}")
        return False
    except Exception:
        logger.exception("Goal Imminent Detection: erro inesperado ao enviar")
        return False

    if response.status_code == 200:
        logger.info("Goal Imminent Detection: enviado com sucesso (HTTP 200).")
        print("🚨 Alerta Goal Imminent enviado para o Telegram com sucesso!")
        return True

    logger.error(
        "Goal Imminent Detection: Telegram respondeu com erro HTTP %d: %s",
        response.status_code, response.text
    )
    print(f"❌ Erro da API do Telegram (Alerta Goal Imminent): HTTP {response.status_code}")
    return False


@dataclass
class GoalImminentOutcome:
    match_id: str
    sent: bool
    state: str
    reason: str
    criteria: Optional[GoalImminentCriteriaResult]
    telegram_sent: Optional[bool] = None


class GoalImminentDetector:
    """
    Camada de observação com estado (anti-spam persistido em SQLite,
    `data/goal_imminent_alerts.db` por omissão) — decide, para cada
    MatchSnapshot avaliado, SE o Alerta Goal Imminent deve ser enviado.
    No máximo UM alerta deste tipo (`alert_type = "GOAL_IMMINENT"`) por
    `match_id`, para sempre — sem reavaliação/cooldown (ao contrário do
    Alerta Live Premium).
    """

    def __init__(
        self,
        db_path: str = DEFAULT_GOAL_IMMINENT_DB_PATH,
        sender: Optional[Callable[[str], bool]] = None,
        message_formatter: Optional[Callable[[Dict[str, Any], GoalImminentCriteriaResult], str]] = None,
    ):
        self.db_path = db_path
        self.sender = sender or send_goal_imminent_alert
        self.message_formatter = message_formatter or build_goal_imminent_message
        self._init_db()

    # -- infraestrutura SQLite -------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goal_imminent_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                alert_type TEXT,
                competition TEXT,
                home_team TEXT,
                away_team TEXT,
                minute INTEGER,
                goal_engine_probability REAL,
                monte_carlo_probability REAL,
                ml_probability REAL,
                consensus TEXT,
                edge REAL,
                ev REAL,
                kelly REAL,
                pressure REAL,
                dangerous_attacks INTEGER,
                shots_on_target INTEGER,
                xg REAL,
                market_odd REAL,
                decision TEXT,
                outcome TEXT,
                telegram_sent BOOLEAN,
                created_at DATETIME
            )
            """
        )
        conn.commit()
        conn.close()

    # -- anti-spam (no máximo um alerta enviado por match_id) -----------

    def has_already_alerted(self, match_id: Any) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM goal_imminent_alerts "
                "WHERE match_id = ? AND alert_type = ? AND telegram_sent = 1 LIMIT 1",
                (str(match_id), GOAL_IMMINENT_ALERT_TYPE),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    # -- log ---------------------------------------------------------------

    def _log_alert(
        self,
        snapshot: Dict[str, Any],
        criteria: GoalImminentCriteriaResult,
        telegram_sent: bool,
        now: datetime,
    ) -> None:
        v = criteria.values
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO goal_imminent_alerts (
                match_id, alert_type, competition, home_team, away_team, minute,
                goal_engine_probability, monte_carlo_probability, ml_probability,
                consensus, edge, ev, kelly, pressure, dangerous_attacks,
                shots_on_target, xg, market_odd, decision, outcome, telegram_sent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(snapshot.get("match_id")),
                GOAL_IMMINENT_ALERT_TYPE,
                v["competition"],
                v["home_team"],
                v["away_team"],
                v["minute"],
                v["goal_engine_probability"],
                v["monte_carlo_probability"],
                v["ml_probability"],
                v["consensus_label"],
                v["edge_pct"],
                v["ev_pct"],
                v["kelly_pct"],
                v["pressure"],
                v["dangerous_attacks"],
                v["shots_on_target"],
                v["estimated_xg_10m"],
                v["bookie_odd"],
                v["decision_label"],
                "ALERTA ENVIADO" if telegram_sent else "ENVIO TELEGRAM FALHOU",
                int(telegram_sent),
                now.isoformat(),
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
                "SELECT * FROM goal_imminent_alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # -- ponto de entrada principal ------------------------------------------

    def evaluate_and_maybe_alert(
        self,
        snapshot: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> GoalImminentOutcome:
        """
        Avalia os 12 critérios sobre `snapshot` (MatchSnapshot já
        produzido por `build_match_snapshot`) e envia o Alerta Goal
        Imminent via Telegram apenas se TODOS passarem e ainda não tiver
        sido enviado nenhum alerta deste tipo para este `match_id`.
        """
        match_id = str(snapshot.get("match_id"))
        now = now or datetime.now(timezone.utc)

        if self.has_already_alerted(match_id):
            return GoalImminentOutcome(
                match_id=match_id,
                sent=False,
                state="JÁ ENVIADO",
                reason="Já existe um Alerta Goal Imminent enviado para este jogo.",
                criteria=None,
            )

        criteria = evaluate_goal_imminent_criteria(snapshot)

        if not criteria.passed:
            return GoalImminentOutcome(
                match_id=match_id,
                sent=False,
                state="CRITÉRIOS NÃO REUNIDOS",
                reason="; ".join(criteria.failed_reasons),
                criteria=criteria,
            )

        message = self.message_formatter(snapshot, criteria)
        telegram_sent = bool(self.sender(message))

        self._log_alert(snapshot, criteria, telegram_sent, now)

        return GoalImminentOutcome(
            match_id=match_id,
            sent=telegram_sent,
            state="ALERTA ENVIADO" if telegram_sent else "ENVIO TELEGRAM FALHOU",
            reason="Todos os 12 critérios reunidos." if telegram_sent else "Critérios reunidos, mas o envio Telegram falhou.",
            criteria=criteria,
            telegram_sent=telegram_sent,
        )

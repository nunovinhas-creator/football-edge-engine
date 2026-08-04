"""
Camada de agregação de dados para o Dashboard Pro (`scripts/app.py`).

Este módulo é EXCLUSIVAMENTE de apresentação. Monta, a partir das saídas
já produzidas pelos módulos oficiais do motor — Goal Engine
(`src.live.engine`), Monte Carlo (`src.engine.simulation`), Dixon-Coles
(`src.engine.dixon_coles` / `src.engine.value`), Machine Learning
(`src.model.ml_predictor`), Edge/EV (`src.engine.edge`), Kelly
(`src.engine.kelly`), Decision Engine (`src.engine.decision`,
`src.engine.live_decision`) e o Backtesting/Evaluation Framework
(`src.backtest.historical`) — as estruturas que a UI usa para desenhar os
painéis do Dashboard Pro.

Não define nenhuma fórmula matemática nova, não recalcula nenhuma
probabilidade/edge/EV/Kelly/lambda e não substitui nenhum módulo do
motor: apenas invoca as funções oficiais já existentes, inalteradas, e
organiza os resultados.

Os únicos valores "derivados" aqui são etiquetas de apresentação (cores,
texto, agrupamento em faixas, médias simples de indicadores já 0-100)
sobre números que o motor já produziu — nunca uma nova estimativa de
probabilidade, edge, EV, Kelly ou lambda. Cada função que faz esse tipo
de agrupamento documenta explicitamente que é "apenas apresentação".
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.models.live_state import LiveMatchState
from src.live.engine import LiveGoalEngine
from src.live.features.goal_window import GoalWindowPredictor
from src.live.features.momentum import Momentum
from src.model.ml_predictor import LiveMLPredictor
from src.engine.live_pipeline import LivePipeline
from src.engine.simulation import MonteCarloSimulator
from src.engine.value import estimate_pregame_probabilities
from src.engine.decision import DecisionEngine
from src.engine.live_decision import evaluate_live_market
from src.engine.edge import calculate_ev
from src.backtest.logger import DB_PATH
from src.backtest.historical import BacktestEngine
from src.backtest.historical.dataset import load_historical_dataset
from src.alerts.live_premium_alerts import (
    ALERT_MARKET_LABEL,
    DEFAULT_ALERTS_DB_PATH,
    evaluate_alert_criteria,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_BACKTEST_DATASET = REPO_ROOT / "examples" / "backtest" / "sample_real_games.csv"

# Odd de mercado por omissão quando a API não fornece uma odd ao vivo —
# mesmo valor de fallback já usado antes desta reformulação em
# `scripts/app.py`.
DEFAULT_BOOKIE_ODD = 1.85

# Jogo de demonstração mostrado quando não há jogos em direto na BSD API
# (sem chave configurada, sem jogos a decorrer, ou API inacessível) —
# os mesmos valores de fallback já usados antes desta reformulação em
# `scripts/app.py`, apenas centralizados aqui.
DEMO_EVENT = {
    "id": 999,
    "home_team": "FC Porto",
    "away_team": "Sporting CP",
    "current_minute": 64,
    "home_score": 1,
    "away_score": 1,
}
DEMO_MATCH_DATA = {
    "match_id": 999,
    "home_team": "FC Porto",
    "away_team": "Sporting CP",
    "current_minute": 64,
    "home_score": 1,
    "away_score": 1,
    "home_xg_last5": 1.8,
    "away_conceded_xg_last5": 1.4,
    "home_style": "high_press",
    "away_style": "low_block_vulnerable",
    "dangerous_attacks_10m": 14,
    "shots_on_target_10m": 3,
    "shots_10m": 9,
    "corners_10m": 4,
    "home_possession": 57.0,
    "previous_pressure": 38.0,
    "red_cards": 0,
    "live_odd_over": 2.10,
}


# ---------------------------------------------------------------------------
# Estado do sistema (cabeçalho)
# ---------------------------------------------------------------------------

def get_bsd_status() -> Tuple[str, str]:
    """(label, cor) do estado da API BSD, a partir da chave já configurada
    em `src.config.settings` — não faz nenhum pedido de rede novo."""
    from src.config.settings import API_KEY

    if API_KEY:
        return "🟢 Configurada", "ok"
    return "🔴 Sem chave API", "off"


def get_telegram_status() -> Tuple[str, str]:
    """(label, cor) do estado do Telegram, a partir das variáveis de
    ambiente já usadas por `src.utils.telegram_notifier`."""
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        return "🟢 Ativo", "ok"
    return "🔴 Não configurado", "off"


def get_ml_status(ml_predictor: LiveMLPredictor) -> Tuple[str, str]:
    """(label, cor) do estado do Machine Learning, a partir do estado real
    do `LiveMLPredictor` já construído (qual modelo foi efetivamente
    carregado) — não adivinha nada, lê os atributos já preenchidos por
    `LiveMLPredictor.__init__`."""
    if ml_predictor.real_predictor is not None:
        return "🟢 LiveGoalModel (LightGBM)", "ok"
    if ml_predictor.model is not None:
        return "🟡 XGBoost (fallback)", "warn"
    return "🟠 Heurística (sem modelo treinado)", "warn"


# ---------------------------------------------------------------------------
# Extração de metadados do evento em direto
# ---------------------------------------------------------------------------

def extract_competition(event: Dict[str, Any]) -> str:
    for key in ("league_name", "league", "competition", "tournament_name", "tournament"):
        value = event.get(key)
        if value:
            return str(value)
    return "Competição não identificada"


def extract_status_label(event: Dict[str, Any], minute: int) -> str:
    status = event.get("status") or event.get("match_status")
    if status:
        return str(status)
    return "AO VIVO" if minute else "—"


# ---------------------------------------------------------------------------
# Rótulos de apresentação (buckets sobre valores já calculados)
# ---------------------------------------------------------------------------

def decision_badge(live_action: str) -> Tuple[str, str, str]:
    """
    Traduz a ação já decidida por `evaluate_live_market` (3 níveis:
    "🔥 BET VALUE" / "⚠️ WATCH" / "❄️ PASS", já calculados a partir do
    Edge oficial) para o rótulo grande do painel de Decisão. Não altera o
    critério de decisão — apenas o texto/emoji apresentado.
    """
    if "BET" in live_action:
        return "🟢 APOSTAR AGORA", "ok", "O motor identificou valor (+EV) neste momento do jogo."
    if "WATCH" in live_action:
        return "🟡 AGUARDAR", "warn", "Existe algum valor, mas ainda abaixo do limiar de entrada do motor."
    return "🔴 NÃO APOSTAR", "off", "Sem valor (+EV) identificado neste momento do jogo."


def confidence_badge(confidence_score: float) -> Tuple[str, str]:
    """
    Agrupa `MLPredictionResult.confidence_score` (0-100, já calculado por
    `LiveMLPredictor`) em 4 faixas de apresentação. Apenas apresentação —
    não altera o valor nem influencia nenhuma decisão.
    """
    if confidence_score >= 85:
        return "Muito Alta", "ok"
    if confidence_score >= 65:
        return "Alta", "ok"
    if confidence_score >= 40:
        return "Média", "warn"
    return "Baixa", "off"


def consensus_badge(gap: float) -> Tuple[str, str]:
    """
    Agrupa a diferença absoluta entre as probabilidades do Goal Engine e
    do Machine Learning (ambos estimam o mesmo mercado: P(golo nos
    próximos 15 min)) em 4 faixas de "consenso entre modelos". Quanto
    menor o afastamento entre os dois modelos, mais forte o consenso.
    Apenas apresentação — não é um novo modelo nem altera nenhuma
    probabilidade.
    """
    if gap <= 5:
        return "Muito Forte", "ok"
    if gap <= 15:
        return "Forte", "ok"
    if gap <= 30:
        return "Moderado", "warn"
    return "Fraco", "off"


def _norm(value: Optional[float], scale: float = 1.0, cap: float = 100.0) -> Optional[float]:
    """Escala/recorta `value` para [0, cap] — normalização de apresentação
    usada apenas para combinar indicadores heterogéneos no Engine Score."""
    if value is None:
        return None
    return round(min(max(value * scale, 0.0), cap), 1)


def compute_engine_score(components: Dict[str, Optional[float]]) -> float:
    """
    Engine Score (0-100): média aritmética simples dos indicadores já
    calculados pelos módulos oficiais (Goal Engine, ML, Monte Carlo,
    Dixon-Coles, Confiança, Consenso, Edge, EV, Kelly — cada um já
    normalizado para 0-100 por `_norm`). NÃO é uma métrica nova do motor,
    não introduz nenhum algoritmo de prevenção/otimização, e não
    influencia nenhuma decisão de aposta — a decisão BET/WATCH/PASS
    continua a vir exclusivamente de `evaluate_live_market`/
    `DecisionEngine`. É apenas uma bússola visual de 0 a 100 para o
    utilizador perceber, num único número, o quão alinhados estão todos
    os sinais já produzidos pelo motor.
    """
    values = [v for v in components.values() if v is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


def engine_score_badge(score: float) -> Tuple[str, str]:
    if score >= 75:
        return "Excelente", "ok"
    if score >= 55:
        return "Bom", "ok"
    if score >= 35:
        return "Moderado", "warn"
    return "Fraco", "off"


# ---------------------------------------------------------------------------
# Snapshot completo de um jogo em direto
# ---------------------------------------------------------------------------

def _dynamic_lambdas(pipeline: LivePipeline, match_state: LiveMatchState, live_result: Dict[str, Any]) -> Tuple[float, float]:
    """
    Reutiliza EXATAMENTE `LivePipeline.calculate_dynamic_lambda` (não
    reimplementa a fórmula) para obter (lambda_home, lambda_away) — a
    mesma orquestração já documentada em `LivePipeline.evaluate()`
    (mesma função reaproveitada duas vezes, mesmo ajuste de cartões
    vermelhos com o fator 0.15 já usado lá). `pipeline` é construído com
    providers "dummy" (ver `_build_pipeline`) só para evitar instanciar
    `APIMatchProvider`/`APIOddsProvider` (que exigem rede/chave), sem
    tocar em `LivePipeline` nem alterar o seu comportamento.
    """
    lambda_home = pipeline.calculate_dynamic_lambda(live_result)

    away_live_result = {
        "estimated_xg_10m": match_state.away_conceded_xg_last5,
        "pressure": live_result.get("pressure"),
    }
    lambda_away = pipeline.calculate_dynamic_lambda(away_live_result)

    if match_state.red_cards > 0:
        lambda_away = round(lambda_away * max(0.0, 1 - 0.15 * match_state.red_cards), 2)

    return lambda_home, lambda_away


def _build_pipeline() -> LivePipeline:
    # providers "dummy": qualquer valor truthy evita a construção por
    # omissão de APIMatchProvider()/APIOddsProvider() (que exigem
    # rede/API key) — não usamos `pipeline.evaluate()`, só o método puro
    # `calculate_dynamic_lambda`.
    return LivePipeline(match_provider="unused", odds_provider="unused")


def build_match_snapshot(
    match_data: Dict[str, Any],
    competition: str = "Competição não identificada",
    status_label: str = "AO VIVO",
    ml_predictor: Optional[LiveMLPredictor] = None,
    goal_engine: Optional[LiveGoalEngine] = None,
) -> Dict[str, Any]:
    """
    Constrói o snapshot completo (todos os painéis do Dashboard Pro) para
    um único jogo em direto, a partir de `match_data` (mesmo formato já
    devolvido por `BSDLiveFetcher.parse_live_metrics_for_engine`).
    """
    goal_engine = goal_engine or LiveGoalEngine()
    ml_predictor = ml_predictor or LiveMLPredictor()

    match_state = LiveMatchState(
        minute=match_data.get("current_minute", 0),
        home_score=match_data.get("home_score", 0),
        away_score=match_data.get("away_score", 0),
        home_xg_last5=match_data.get("home_xg_last5", 1.5),
        away_conceded_xg_last5=match_data.get("away_conceded_xg_last5", 1.2),
        home_style=match_data.get("home_style", "balanced"),
        dangerous_attacks_10m=match_data.get("dangerous_attacks_10m", 0),
        shots_on_target_10m=match_data.get("shots_on_target_10m", 0),
        shots_10m=match_data.get("shots_10m", 0),
        corners_10m=match_data.get("corners_10m", 0),
        possession=match_data.get("home_possession", 50.0),
        previous_pressure=match_data.get("previous_pressure", 0.0),
        goals_last_15=match_data.get("goals_last_15", 0),
        last_goal_minute=match_data.get("last_goal_minute"),
        red_cards=match_data.get("red_cards", 0),
        game_state=match_data.get("game_state", "unknown"),
    )

    bookie_odd = float(match_data.get("live_odd_over", DEFAULT_BOOKIE_ODD))

    # --- Goal Engine ---------------------------------------------------
    live_result = goal_engine.predict_next_goal_probability(match_state)
    goal_engine_prob = live_result["next_goal_probability"]

    # --- Lambda dinâmico (reutilizado pelo Monte Carlo e Dixon-Coles) --
    pipeline = _build_pipeline()
    lambda_home, lambda_away = _dynamic_lambdas(pipeline, match_state, live_result)

    # --- Monte Carlo -----------------------------------------------------
    simulator = MonteCarloSimulator()
    simulation = simulator.run_match_simulation(
        current_minute=match_state.minute,
        current_home_score=match_state.home_score,
        current_away_score=match_state.away_score,
        home_lambda=lambda_home,
        away_lambda=lambda_away,
        match_id=match_data.get("match_id"),
    )

    # --- Dixon-Coles (mesmo lambda dinâmico do Monte Carlo) ------------
    dc_probs = estimate_pregame_probabilities(lambda_home, lambda_away)

    # --- Machine Learning ------------------------------------------------
    ml_res = ml_predictor.predict(match_state, live_odd_over=bookie_odd)

    # --- Goal Window / Momentum -----------------------------------------
    goal_window = GoalWindowPredictor().predict_window(match_state, live_result["pressure"])
    momentum = Momentum().calculate(live_result["pressure"], match_state.previous_pressure)

    # --- Decisão (mercado "Próximo Golo", já o mercado do Goal Engine) -
    live_bet = evaluate_live_market(
        probability_pct=goal_engine_prob,
        bookie_odd=bookie_odd,
        market="NEXT GOAL (15m)",
    )

    # --- Decisão (mercado "Over 1.5", já o mercado do Monte Carlo) -----
    dec_engine = DecisionEngine()
    bet_rec = dec_engine.evaluate_bet("Over 1.5", simulation.over_15_prob, bookie_odd)
    ev_over15_pct = round(calculate_ev(simulation.over_15_prob / 100.0, bookie_odd) * 100, 2)

    fair_odd = round(100.0 / goal_engine_prob, 2) if goal_engine_prob > 0 else None

    # --- Consenso entre modelos (Goal Engine vs ML — mesmo mercado) ----
    consensus_gap = round(abs(goal_engine_prob - ml_res.goal_probability), 1)
    consensus_score = round(max(0.0, 100.0 - consensus_gap * 2.0), 1)
    consensus_label, consensus_color = consensus_badge(consensus_gap)

    # --- Engine Score (índice visual, ver compute_engine_score) --------
    engine_score_components = {
        "Goal Engine": _norm(goal_engine_prob),
        "Machine Learning": _norm(ml_res.goal_probability),
        "Monte Carlo (Over 1.5)": _norm(simulation.over_15_prob),
        "Dixon-Coles": _norm(max(dc_probs.values()) * 100),
        "Confiança (ML)": _norm(ml_res.confidence_score),
        "Consenso": _norm(consensus_score),
        "Edge": _norm(live_bet.edge, scale=4.0),
        "EV": _norm(ev_over15_pct, scale=3.0),
        "Kelly": _norm(bet_rec.kelly_stake_pct, scale=20.0),
    }
    engine_score = compute_engine_score(engine_score_components)
    engine_score_label, engine_score_color = engine_score_badge(engine_score)

    decision_label, decision_color, decision_reason = decision_badge(live_bet.action)
    confidence_label, confidence_color = confidence_badge(ml_res.confidence_score)

    # --- Explicação (só texto sobre valores já existentes) -------------
    explanation = build_explanation(
        live_bet=live_bet,
        ev_over15_pct=ev_over15_pct,
        consensus_label=consensus_label,
        consensus_gap=consensus_gap,
        recommendation=live_result["recommendation"],
        goal_window=goal_window,
        momentum=momentum,
        bookie_odd=bookie_odd,
        fair_odd=fair_odd,
        dominance_index=live_result["dominance_index"],
    )

    return {
        "match_id": match_data.get("match_id"),
        "card": {
            "competition": competition,
            "home_team": match_data.get("home_team", "Casa"),
            "away_team": match_data.get("away_team", "Fora"),
            "home_score": match_state.home_score,
            "away_score": match_state.away_score,
            "minute": match_state.minute,
            "elapsed": f"{min(match_state.minute, 90)}'",
            "status": status_label,
        },
        "decision": {
            "label": decision_label,
            "color": decision_color,
            "reason": decision_reason,
            "confidence_label": confidence_label,
            "confidence_color": confidence_color,
            "confidence_score": ml_res.confidence_score,
        },
        "engine_score": {
            "score": engine_score,
            "label": engine_score_label,
            "color": engine_score_color,
            "components": engine_score_components,
        },
        "models": {
            "goal_engine": {
                "probability": goal_engine_prob,
                "market": "Golo nos próximos 15 min",
                "status": live_result["recommendation"],
            },
            "machine_learning": {
                "probability": ml_res.goal_probability,
                "market": "Golo nos próximos 15 min",
                "status": ml_res.model_used,
                "confidence": ml_res.confidence_score,
            },
            "monte_carlo": {
                "over_15": simulation.over_15_prob,
                "over_25": simulation.over_25_prob,
                "btts": simulation.btts_prob,
                "expected_home_goals": simulation.expected_goals_home,
                "expected_away_goals": simulation.expected_goals_away,
                "market": "Resto do jogo (Over/BTTS)",
            },
            "dixon_coles": {
                "home": round(dc_probs["home"] * 100, 1),
                "draw": round(dc_probs["draw"] * 100, 1),
                "away": round(dc_probs["away"] * 100, 1),
                "market": "Resultado 1X2 (λ dinâmico)",
            },
        },
        "consensus": {
            "gap": consensus_gap,
            "score": consensus_score,
            "label": consensus_label,
            "color": consensus_color,
        },
        "value": {
            "market": "Próximo Golo (15m)",
            "bookie_odd": bookie_odd,
            "fair_odd": fair_odd,
            "edge_pct": live_bet.edge,
            "ev_pct": round(((goal_engine_prob / 100.0) * bookie_odd - 1.0) * 100, 2) if bookie_odd else 0.0,
            "kelly_pct": bet_rec.kelly_stake_pct,
            "over15_edge_pct": bet_rec.edge_pct,
            "over15_ev_pct": ev_over15_pct,
            "over15_kelly_pct": bet_rec.kelly_stake_pct,
            "over15_action": bet_rec.action,
        },
        "live": {
            "pressure": live_result["pressure"],
            "dominance_index": live_result["dominance_index"],
            "dangerous_attacks_10m": match_state.dangerous_attacks_10m,
            "shots_10m": match_state.shots_10m,
            "shots_on_target_10m": match_state.shots_on_target_10m,
            "corners_10m": match_state.corners_10m,
            "possession": match_state.possession,
            "estimated_xg_10m": live_result["estimated_xg_10m"],
            "momentum": momentum,
            "red_cards": match_state.red_cards,
            "goal_window": goal_window.predicted_window,
            "goal_window_intensity": goal_window.intensity,
        },
        "strength": {
            "home_lambda": lambda_home,
            "away_lambda": lambda_away,
            "tier": "N/D — modo live (sem histórico H2H carregado)",
            "effective_sample_size": None,
            "h2h_available": False,
        },
        "explanation": explanation,
    }


def build_explanation(
    live_bet,
    ev_over15_pct: float,
    consensus_label: str,
    consensus_gap: float,
    recommendation: str,
    goal_window,
    momentum: str,
    bookie_odd: float,
    fair_odd: Optional[float],
    dominance_index: float,
) -> List[str]:
    """
    Gera o painel textual de explicação exclusivamente a partir de valores
    já calculados pelo motor (sem IA externa, sem LLM, sem novo cálculo).
    """
    bullets = []

    if live_bet.edge > 0:
        bullets.append(f"Edge positivo no mercado 'Próximo Golo': {live_bet.edge:+.1f}%.")
    else:
        bullets.append(f"Edge não positivo no mercado 'Próximo Golo': {live_bet.edge:+.1f}%.")

    if ev_over15_pct > 0:
        bullets.append(f"EV positivo no mercado Over 1.5: {ev_over15_pct:+.1f}%.")
    else:
        bullets.append(f"EV do mercado Over 1.5 ainda não é positivo: {ev_over15_pct:+.1f}%.")

    bullets.append(f"Consenso entre Goal Engine e Machine Learning: {consensus_label} (diferença de {consensus_gap:.1f} p.p.).")
    bullets.append(f"Classificação do Goal Engine para esta fase do jogo: {recommendation}.")
    bullets.append(f"Janela de golo prevista: {goal_window.predicted_window} ({goal_window.intensity}).")
    bullets.append(f"Momentum da pressão ofensiva: {momentum}.")
    bullets.append(f"Índice de domínio territorial: {dominance_index:.1f}/100.")

    if fair_odd is not None:
        if bookie_odd > fair_odd:
            bullets.append(f"A odd de mercado ({bookie_odd:.2f}) é superior à odd justa do modelo ({fair_odd:.2f}).")
        else:
            bullets.append(f"A odd de mercado ({bookie_odd:.2f}) ainda não supera a odd justa do modelo ({fair_odd:.2f}).")

    return bullets


# ---------------------------------------------------------------------------
# Backtest (reutiliza o Backtesting Framework já existente, sem recalcular
# nenhum modelo — apenas corre o `BacktestEngine` inalterado sobre o
# pequeno dataset histórico real já incluído no repositório)
# ---------------------------------------------------------------------------

def run_demo_backtest():
    """
    Corre `BacktestEngine` (inalterado) sobre
    `examples/backtest/sample_real_games.csv` (jogos históricos reais já
    incluídos no repositório para o modo `--demo` de `run_backtest.py`).
    Devolve o `BacktestReport` oficial — nenhuma métrica é recalculada
    aqui, só invocada.
    """
    dataset = load_historical_dataset(str(DEMO_BACKTEST_DATASET))
    return BacktestEngine().run(dataset)


# ---------------------------------------------------------------------------
# Histórico (lê o mesmo `data/live_history.db` já escrito pelo Logger e
# pelos alertas Telegram — não escreve, não recalcula, só lê)
# ---------------------------------------------------------------------------

def load_live_history(limit: int = 300) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT * FROM match_snapshots ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def load_value_alerts(limit: int = 100) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT match_id, sent_at FROM telegram_value_alerts ORDER BY sent_at DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 🚨 Live Alert Monitor (lê o mesmo `data/live_alerts.db` já escrito por
# `src.alerts.live_premium_alerts.LiveAlertMonitor`, chamado pelo monitor
# real em `src.engine.live_monitor` — não escreve, não decide, não envia
# nenhum alerta a partir daqui)
# ---------------------------------------------------------------------------

def load_live_alerts(limit: int = 200) -> pd.DataFrame:
    if not os.path.exists(DEFAULT_ALERTS_DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DEFAULT_ALERTS_DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT * FROM live_alerts ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def count_alerts_sent_today() -> int:
    df = load_live_alerts(limit=100000)
    if df.empty or "timestamp" not in df.columns or "telegram_sent" not in df.columns:
        return 0
    today = datetime.utcnow().date().isoformat()
    sent = df[df["telegram_sent"] == 1]
    return int(sent["timestamp"].astype(str).str.startswith(today).sum())


def build_live_alert_monitor_rows(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Painel "🚨 Live Alert Monitor" (novo painel, não substitui nenhum
    existente). Cruza `evaluate_alert_criteria` (puro, sem efeitos
    secundários) com o histórico já gravado em `data/live_alerts.db` para
    apresentar, por jogo em direto: Estado (🚨 ALERTA ENVIADO / 🟢 ATIVO /
    🟡 À ESPERA), hora do último alerta, mercado, odd, probabilidade e
    motivo. Não recalcula nenhuma probabilidade/edge/EV/Kelly, não decide
    nem envia nenhum alerta — isso é feito exclusivamente por
    `LiveAlertMonitor.evaluate_and_maybe_alert`.
    """
    alerts_df = load_live_alerts(limit=500)
    rows: List[Dict[str, Any]] = []

    for snap in snapshots:
        match_id = str(snap.get("match_id"))
        criteria = evaluate_alert_criteria(snap)
        card = snap["card"]
        value = snap["value"]

        last_alert_at = None
        if not alerts_df.empty and "match_id" in alerts_df.columns:
            match_alerts = alerts_df[alerts_df["match_id"].astype(str) == match_id]
            if not match_alerts.empty:
                last_alert_at = match_alerts.iloc[0]["timestamp"]

        if last_alert_at is not None:
            state = "🚨 ALERTA ENVIADO"
        elif criteria.passed:
            state = "🟢 ATIVO"
        else:
            state = "🟡 À ESPERA"

        rows.append(
            {
                "Estado": state,
                "Jogo": f"{card['home_team']} vs {card['away_team']}",
                "Mercado": ALERT_MARKET_LABEL,
                "Odd": value["bookie_odd"],
                "Probabilidade (Goal Engine)": criteria.values["goal_engine_probability"],
                "Motivo": "Todos os critérios reunidos." if criteria.passed else "; ".join(criteria.failed_reasons),
                "Hora do último alerta": last_alert_at or "—",
            }
        )

    return rows

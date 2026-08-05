"""
🎯 Oportunidades das Próximas 24 Horas — camada de agregação (Sprint 2).

Painel NOVO e completamente independente do Dashboard Live
(`src.report.dashboard_data` / `scripts/app.py`). Não substitui, não
altera e não recalcula nenhum dos motores oficiais — Goal Engine
(`src.live.engine`), Monte Carlo (`src.engine.simulation`), Dixon-Coles
(`src.engine.dixon_coles` / `src.engine.value`), Machine Learning
(`src.model.ml_predictor`), Edge/EV (`src.engine.edge`), Kelly
(`src.engine.kelly`), Decision Engine (`src.engine.decision`,
`src.engine.live_decision`) ou o Backtesting Framework
(`src.backtest.historical`).

Reutiliza, sem alterações, `src.report.dashboard_data.build_match_snapshot`
— a MESMA função que já monta o snapshot completo de um jogo para o
Dashboard Live — para cada jogo agendado nas próximas 24 horas. A única
diferença face ao Dashboard Live é a origem dos dados de entrada: em vez
de estatísticas ao vivo (pressão, remates, posse — que só existem depois
do apito inicial), um jogo agendado usa golos esperados pré-jogo
(`src.engine.lambda_estimator.estimate_lambda`, o estimador pré-jogo
oficial já usado por `src.collector.client.EventCollector`) e valores
neutros/zero para todos os campos que só fazem sentido ao vivo. Nenhuma
fórmula nova é introduzida — apenas se escolhe QUE valores alimentar ao
mesmo `build_match_snapshot` já existente.

A pesquisa de "jogos históricos semelhantes" (bloco "📈 Histórico de Jogos
Semelhantes") reutiliza exatamente `src.report.historical_validation`
(`find_similar_bets`/`summarize_similar_bets`, já usados pela "Validação
Histórica da Aposta Atual" do Dashboard Live) e
`src.backtest.historical.statistics.brier_score` — nenhuma métrica nova é
inventada, nenhum dataset novo é carregado.

Este módulo é a única camada de lógica de negócio da Sprint 2: `scripts/
app.py` apenas invoca as funções abaixo e desenha o resultado (filtros,
pesquisa, cartões, cores, selo de estrelas) — não decide nada, não
recalcula nada.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.api.client import BzzoiroClient
from src.collector.odds import OddsCollector
from src.engine.lambda_estimator import estimate_lambda
from src.engine.pregame_lambda import estimate_pregame_lambdas
from src.live.engine import LiveGoalEngine
from src.model.ml_predictor import LiveMLPredictor
from src.report.dashboard_data import (
    DEFAULT_BOOKIE_ODD,
    build_match_snapshot,
    extract_competition,
)
from src.report.historical_validation import (
    build_current_bet_profile,
    find_similar_bets,
    summarize_similar_bets,
)
from src.backtest.historical.statistics import brier_score

# Janela oficial desta funcionalidade: agora -> agora + 24h (requisito da
# Sprint 2). Não confundir com a janela de 3 dias usada por
# `src.engine.predict_today` (script legado, não tocado por esta sprint).
WINDOW_HOURS = 24

# Campos "ao vivo" (pressão, remates, posse, minuto...) que um jogo ainda
# não iniciado não pode ter — zerados/neutros, exatamente como
# `LiveMatchState` já assume por omissão (ver `src/models/live_state.py`).
_DEFAULT_HOME_STYLE = "balanced"
_DEFAULT_AWAY_STYLE = "balanced"
_DEFAULT_POSSESSION = 50.0

# Ordem oficial de agrupamento do ranking (requisito da Sprint 2): 🟢
# primeiro, depois 🟡, depois 🔴. Mesmos rótulos já devolvidos por
# `src.report.dashboard_data.decision_badge` — não é uma nova decisão.
_DECISION_ORDER = {
    "🟢 APOSTAR AGORA": 0,
    "🟡 AGUARDAR": 1,
    "🔴 NÃO APOSTAR": 2,
}


# ---------------------------------------------------------------------------
# Janela temporal — busca e seleção de eventos futuros
# ---------------------------------------------------------------------------

def _extract_kickoff_dt(event: Dict[str, Any]) -> Optional[datetime]:
    """
    Data/hora do jogo, a partir das mesmas chaves já usadas por
    `src.engine.predict_today.fetch_enriched_data_from_bsd`
    ("event_date"/"date"/"start_time") — sem inventar um novo formato.
    Devolve `None` (nunca lança exceção) quando a data está ausente ou é
    inválida.
    """
    raw = event.get("event_date") or event.get("date") or event.get("start_time")
    if not raw:
        return None
    try:
        parsed = pd.to_datetime(raw, utc=True)
    except (ValueError, TypeError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def select_events_in_window(
    events: Sequence[Dict[str, Any]],
    hours: int = WINDOW_HOURS,
    now: Optional[datetime] = None,
) -> List[Tuple[Dict[str, Any], datetime]]:
    """
    Filtra `events` (formato já devolvido pelo endpoint `events/` da BSD
    API) para os que começam entre `now` e `now + hours`, ordenados
    cronologicamente. Função pura — `now` é injetável para testes.
    """
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(hours=hours)

    dated: List[Tuple[Dict[str, Any], datetime]] = []
    for event in events or []:
        kickoff = _extract_kickoff_dt(event)
        if kickoff is None:
            continue
        if now <= kickoff <= horizon:
            dated.append((event, kickoff))

    dated.sort(key=lambda pair: pair[1])
    return dated


def fetch_upcoming_events(
    hours: int = WINDOW_HOURS,
    client: Optional[BzzoiroClient] = None,
    limit: int = 200,
) -> List[Tuple[Dict[str, Any], datetime]]:
    """
    Vai buscar os eventos futuros à BSD API (`events/`, o mesmo endpoint já
    usado por `EventCollector`/`predict_today`) e devolve apenas os que
    caem na janela [agora, agora+`hours`], ordenados cronologicamente.
    """
    client = client or BzzoiroClient()
    response = client.get(f"events/?limit={limit}&ordering=event_date")
    events = response.get("results", []) if isinstance(response, dict) else (response or [])
    return select_events_in_window(events, hours=hours)


# ---------------------------------------------------------------------------
# Odd de mercado (melhor esforço — nunca falha a pipeline por falta de odd)
# ---------------------------------------------------------------------------

def fetch_odd_for_event(
    event_id: Any,
    odds_collector: Optional[OddsCollector] = None,
) -> Optional[float]:
    """
    Odd 1X2 (mercado "HOME", com fallback para "AWAY"/"DRAW") já disponível
    na BSD API para este jogo, via `OddsCollector` (inalterado). Devolve
    `None` quando não há odd disponível — o chamador aplica o mesmo
    fallback (`DEFAULT_BOOKIE_ODD`) já usado pelo Dashboard Live.
    """
    odds_collector = odds_collector or OddsCollector()
    try:
        market = odds_collector.get_event_odds(event_id)
    except Exception:
        return None
    if not market:
        return None
    for key in ("HOME", "AWAY", "DRAW"):
        value = market.get(key)
        if value and value > 1.0:
            return float(value)
    return None


# ---------------------------------------------------------------------------
# Adaptação pré-jogo -> mesmo formato de `match_data` já consumido por
# `build_match_snapshot` (ver `src.api.live_fetcher.BSDLiveFetcher.
# parse_live_metrics_for_engine` / `DEMO_MATCH_DATA`).
# ---------------------------------------------------------------------------

def build_pregame_match_data(
    event: Dict[str, Any],
    odd: Optional[float] = None,
    h2h: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constrói o `match_data` de um jogo AINDA NÃO iniciado, no mesmo formato
    que `build_match_snapshot` já espera. Golos esperados pré-jogo
    (`home_xg_last5`/`away_conceded_xg_last5`) vêm do estimador pré-jogo
    oficial (`estimate_lambda`, com fallback para o adaptador heurístico
    `estimate_pregame_lambdas` — a mesma cascata já usada por
    `EventCollector.get_matches`); todos os campos que só existem ao vivo
    (pressão, remates, posse dinâmica, minuto) ficam a zero/neutros, tal
    como `LiveMatchState` já assume por omissão.
    """
    h2h = h2h if h2h is not None else event.get("head_to_head")

    try:
        lambda_home, mu_away = estimate_lambda(h2h)
    except Exception:
        lambda_home, mu_away = estimate_pregame_lambdas(h2h)

    bookie_odd = odd if odd and odd > 1.0 else DEFAULT_BOOKIE_ODD

    return {
        "match_id": event.get("id"),
        "home_team": event.get("home_team", "Casa"),
        "away_team": event.get("away_team", "Fora"),
        "current_minute": 0,
        "home_score": 0,
        "away_score": 0,
        "home_xg_last5": lambda_home,
        "away_conceded_xg_last5": mu_away,
        "home_style": _DEFAULT_HOME_STYLE,
        "away_style": _DEFAULT_AWAY_STYLE,
        "dangerous_attacks_10m": 0,
        "shots_on_target_10m": 0,
        "shots_10m": 0,
        "corners_10m": 0,
        "home_possession": _DEFAULT_POSSESSION,
        "previous_pressure": 0.0,
        "goals_last_15": 0,
        "last_goal_minute": None,
        "red_cards": 0,
        "game_state": "scheduled",
        "live_odd_over": bookie_odd,
    }


def build_upcoming_snapshot(
    event: Dict[str, Any],
    kickoff_dt: Optional[datetime],
    ml_predictor: LiveMLPredictor,
    goal_engine: LiveGoalEngine,
    odds_collector: Optional[OddsCollector] = None,
) -> Dict[str, Any]:
    """
    Snapshot completo de um jogo agendado — construído por
    `build_match_snapshot` (inalterado, o MESMO usado pelo Dashboard
    Live), apenas com dados de entrada pré-jogo. Adiciona uma única chave
    nova ao dicionário devolvido (`"kickoff"`, hora do jogo) — nenhuma
    chave já existente é alterada, pelo que este snapshot pode ser
    passado, tal e qual, a qualquer painel do Dashboard Live (ex.
    `scripts.app.render_match`).
    """
    odd = fetch_odd_for_event(event.get("id"), odds_collector=odds_collector)
    match_data = build_pregame_match_data(event, odd=odd, h2h=event.get("head_to_head"))
    competition = extract_competition(event)

    snap = build_match_snapshot(
        match_data,
        competition=competition,
        status_label="🗓️ Agendado",
        ml_predictor=ml_predictor,
        goal_engine=goal_engine,
    )
    snap["kickoff"] = {
        "datetime": kickoff_dt,
        "hour_label": kickoff_dt.strftime("%H:%M") if kickoff_dt else "—",
        "iso": kickoff_dt.isoformat() if kickoff_dt else None,
    }
    return snap


# ---------------------------------------------------------------------------
# Classificação por estrelas — apenas apresentação sobre o Engine Score já
# calculado (mesmo espírito de `dashboard_data.engine_score_tier`, mas com
# as faixas exatas pedidas pela Sprint 2, distintas das do Dashboard Live).
# ---------------------------------------------------------------------------

def star_rating(engine_score: float) -> str:
    """
    ★★★★★ 90-100 · ★★★★☆ 80-89 · ★★★☆☆ 70-79 · ★★☆☆☆ 60-69 · ★☆☆☆☆ <60.
    Depende exclusivamente do Engine Score já calculado por
    `compute_engine_score` — não introduz nenhum novo algoritmo.
    """
    if engine_score >= 90:
        filled = 5
    elif engine_score >= 80:
        filled = 4
    elif engine_score >= 70:
        filled = 3
    elif engine_score >= 60:
        filled = 2
    else:
        filled = 1
    return "★" * filled + "☆" * (5 - filled)


def main_monte_carlo_market(monte_carlo: Dict[str, Any]) -> str:
    """
    "Monte Carlo principal" do cartão (ex.: "Over 2.5 = 78%") — escolhe,
    entre os três mercados que o Monte Carlo já simula
    (`snap["models"]["monte_carlo"]`), o de maior probabilidade. Não
    simula nada de novo: apenas seleciona qual dos três números já
    calculados por `MonteCarloSimulator.run_match_simulation` é mostrado
    em destaque.
    """
    candidates = [
        ("Over 1.5", monte_carlo["over_15"]),
        ("Over 2.5", monte_carlo["over_25"]),
        ("BTTS", monte_carlo["btts"]),
    ]
    label, prob = max(candidates, key=lambda pair: pair[1])
    return f"{label} = {prob:.0f}%"


# ---------------------------------------------------------------------------
# 📈 Histórico de Jogos Semelhantes — reutiliza exclusivamente o
# Backtesting Framework já existente (`historical_validation` +
# `statistics.brier_score`).
# ---------------------------------------------------------------------------

def build_similar_games_summary(snap: Dict[str, Any], all_bets: pd.DataFrame) -> Dict[str, Any]:
    """
    Número de jogos semelhantes, ROI histórico, Yield, Win Rate, CLV,
    Brier e Drawdown — TODOS calculados exclusivamente pelas funções
    oficiais do Backtesting Framework já existente
    (`src.report.historical_validation.find_similar_bets`/
    `summarize_similar_bets`, que por sua vez reaplicam
    `src.backtest.historical.metrics.summary_metrics`, e
    `src.backtest.historical.statistics.brier_score`). Nenhuma fórmula
    nova, nenhum dataset novo.
    """
    profile = build_current_bet_profile(snap)
    search = find_similar_bets(profile, all_bets)
    summary = summarize_similar_bets(search["matches"])

    return {
        "n_bets": summary.get("n_bets", 0),
        "roi_pct": summary.get("roi_pct", 0.0),
        "yield_pct": summary.get("yield_pct", 0.0),
        "hit_rate_pct": summary.get("hit_rate_pct", 0.0),
        "clv_pct": summary.get("avg_clv_percentage"),
        "brier_score": brier_score(search["matches"]),
        "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
    }


# ---------------------------------------------------------------------------
# Oportunidade completa de um jogo (cartão + detalhe + histórico) — cada
# peça é calculada exatamente UMA vez aqui; o Streamlit nunca deve voltar a
# invocar `build_match_snapshot`/`build_similar_games_summary` para o mesmo
# jogo ao expandir um cartão (item de performance da Sprint 2).
# ---------------------------------------------------------------------------

def build_opportunity(
    event: Dict[str, Any],
    kickoff_dt: Optional[datetime],
    ml_predictor: LiveMLPredictor,
    goal_engine: LiveGoalEngine,
    all_bets: pd.DataFrame,
    odds_collector: Optional[OddsCollector] = None,
) -> Dict[str, Any]:
    snap = build_upcoming_snapshot(event, kickoff_dt, ml_predictor, goal_engine, odds_collector=odds_collector)

    return {
        "match_id": snap["match_id"],
        "snapshot": snap,
        "kickoff": snap["kickoff"],
        "card": snap["card"],
        "decision": snap["decision"],
        "engine_score": snap["engine_score"],
        "value": snap["value"],
        "monte_carlo_headline": main_monte_carlo_market(snap["models"]["monte_carlo"]),
        "star_rating": star_rating(snap["engine_score"]["score"]),
        "similar_games": build_similar_games_summary(snap, all_bets),
    }


def _decision_rank(decision_label: str) -> int:
    return _DECISION_ORDER.get(decision_label, len(_DECISION_ORDER))


def sort_opportunities(opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranking oficial da Sprint 2: 🟢 APOSTAR AGORA primeiro, depois 🟡
    AGUARDAR, depois 🔴 NÃO APOSTAR; dentro de cada grupo, Engine Score
    descendente. Não recalcula nem altera nenhuma decisão ou score — só
    ordena os já calculados por `build_opportunity`.
    """
    return sorted(
        opportunities,
        key=lambda opp: (_decision_rank(opp["decision"]["label"]), -opp["engine_score"]["score"]),
    )


def list_upcoming_opportunities(
    hours: int = WINDOW_HOURS,
    client: Optional[BzzoiroClient] = None,
    ml_predictor: Optional[LiveMLPredictor] = None,
    goal_engine: Optional[LiveGoalEngine] = None,
    all_bets: Optional[pd.DataFrame] = None,
    odds_collector: Optional[OddsCollector] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Ponto de entrada único deste módulo: lista, já ordenada pelo ranking
    oficial, todas as oportunidades das próximas `hours` horas.

    `events`, quando fornecido, substitui a chamada à BSD API (usado pelos
    testes e por quem já tem os eventos carregados) — passa diretamente
    por `select_events_in_window`, sem tocar na rede.
    """
    ml_predictor = ml_predictor or LiveMLPredictor()
    goal_engine = goal_engine or LiveGoalEngine()
    all_bets = all_bets if all_bets is not None else pd.DataFrame()

    if events is None:
        dated_events = fetch_upcoming_events(hours=hours, client=client)
    else:
        dated_events = select_events_in_window(events, hours=hours)

    opportunities = [
        build_opportunity(event, kickoff, ml_predictor, goal_engine, all_bets, odds_collector=odds_collector)
        for event, kickoff in dated_events
    ]
    return sort_opportunities(opportunities)


# ---------------------------------------------------------------------------
# Filtros e pesquisa — puramente sobre a lista já construída, nunca
# recalculam nenhum snapshot.
# ---------------------------------------------------------------------------

def available_competitions(opportunities: List[Dict[str, Any]]) -> List[str]:
    return sorted({opp["card"]["competition"] for opp in opportunities})


def available_markets(opportunities: List[Dict[str, Any]]) -> List[str]:
    return sorted({opp["value"]["market"] for opp in opportunities})


def available_decisions(opportunities: List[Dict[str, Any]]) -> List[str]:
    present = {opp["decision"]["label"] for opp in opportunities}
    return [label for label in _DECISION_ORDER if label in present]


def _hour_in_range(dt: Optional[datetime], hour_from: int, hour_to: int) -> bool:
    if dt is None:
        return False
    hour = dt.hour
    if hour_from <= hour_to:
        return hour_from <= hour <= hour_to
    # Janela que atravessa a meia-noite (ex.: 22h -> 3h).
    return hour >= hour_from or hour <= hour_to


def filter_opportunities(
    opportunities: List[Dict[str, Any]],
    competition: Optional[str] = None,
    market: Optional[str] = None,
    min_engine_score: Optional[float] = None,
    decision: Optional[str] = None,
    hour_from: Optional[int] = None,
    hour_to: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filtros da Sprint 2: Competição, Mercado, Engine Score mínimo,
    Decisão, Hora — todos sobre campos já presentes em cada oportunidade,
    nenhum recálculo."""
    result = opportunities

    if competition:
        result = [o for o in result if o["card"]["competition"] == competition]
    if market:
        result = [o for o in result if o["value"]["market"] == market]
    if min_engine_score is not None:
        result = [o for o in result if o["engine_score"]["score"] >= min_engine_score]
    if decision:
        result = [o for o in result if o["decision"]["label"] == decision]
    if hour_from is not None and hour_to is not None:
        result = [o for o in result if _hour_in_range(o["kickoff"]["datetime"], hour_from, hour_to)]

    return result


def search_opportunities(opportunities: List[Dict[str, Any]], query: Optional[str]) -> List[Dict[str, Any]]:
    """Pesquisa por equipa ou competição (substring, sem distinção
    maiúsculas/minúsculas)."""
    if not query:
        return opportunities
    needle = query.strip().casefold()
    if not needle:
        return opportunities

    result = []
    for opp in opportunities:
        card = opp["card"]
        haystack = f"{card['home_team']} {card['away_team']} {card['competition']}".casefold()
        if needle in haystack:
            result.append(opp)
    return result

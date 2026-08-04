"""
Validação Histórica da Aposta Atual — camada de apresentação.

SEGUNDO painel do Dashboard Pro, completamente distinto do Backtesting
Global (`tab_backtest` em `scripts/app.py`, alimentado por
`src.report.dashboard_data.run_demo_backtest`). O Backtesting Global mede
o desempenho do motor sobre TODO o dataset histórico de demonstração;
este módulo isola, dentro desse MESMO `BacktestReport` já produzido pelo
`BacktestEngine` (`src.backtest.historical`), apenas o subconjunto de
jogos históricos "semelhantes" à aposta atualmente recomendada/selecionada
e reaplica exatamente as MESMAS funções oficiais de métricas
(`src.backtest.historical.metrics`) a esse subconjunto — nunca a um
dataset novo, nunca com uma fórmula diferente.

Não define nenhuma fórmula matemática nova de probabilidade, Edge, EV,
Kelly ou lambda: esses valores já vêm calculados por
`BacktestEngine`/`evaluate_bets` (para os jogos históricos, ver
`src.backtest.historical.evaluator`) e por
`src.report.dashboard_data.build_match_snapshot` (para a aposta atual,
que por sua vez só invoca Goal Engine, Monte Carlo, Dixon-Coles, Machine
Learning, `src.engine.edge`, `src.engine.kelly` e `src.engine.decision`,
todos inalterados). A única lógica genuinamente nova deste módulo é a
PESQUISA de jogos semelhantes — comparação/filtragem por proximidade de
odd, probabilidade e competição, todos campos que já existem no dataset
— e a formatação de comparação/veredicto/explicação sobre os números daí
resultantes. Nenhuma métrica é inventada: quando uma dimensão pedida
(força das equipas, λ, minuto, pressão, xG, estado do marcador) não
existe no dataset histórico de demonstração, é assinalada como
"indisponível" em vez de simulada.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest.historical.metrics import equity_curve, summary_metrics

# ---------------------------------------------------------------------------
# Critérios de pesquisa — bandas de tolerância progressivas. Relaxam
# automaticamente até atingir uma amostra mínima de jogos semelhantes, ou
# até esgotar as bandas definidas. Operam apenas sobre colunas que já
# existem no dataset avaliado pelo BacktestEngine (odd, probability,
# competition) — nenhuma métrica nova é inventada.
# ---------------------------------------------------------------------------

_ODD_TOLERANCE_STEPS_PCT = [15.0, 30.0, 50.0, 100.0]
_PROB_TOLERANCE_STEPS_PP = [8.0, 15.0, 25.0, 50.0]
MIN_SAMPLE_TARGET = 5
MIN_CONCLUSIVE_SAMPLE = 5

# Dimensões pedidas para a pesquisa histórica que o dataset de demonstração
# (`examples/backtest/sample_real_games.csv`) não contém — assinaladas ao
# utilizador como indisponíveis, nunca preenchidas com valores inventados.
_UNAVAILABLE_CRITERIA = [
    "Força das equipas",
    "λ (lambda) semelhantes",
    "Minuto semelhante (live)",
    "Pressão semelhante",
    "xG semelhante",
    "Estado do marcador semelhante",
]


@dataclass
class CurrentBetProfile:
    """
    Retrato da aposta atualmente recomendada/selecionada, montado
    EXCLUSIVAMENTE a partir dos valores já presentes no snapshot produzido
    por `src.report.dashboard_data.build_match_snapshot` — nenhum valor é
    recalculado aqui.
    """

    market: str
    odd: float
    probability_pct: float
    edge_pct: float
    ev_pct: float
    kelly_pct: float
    confidence_label: str
    confidence_score: float
    consensus_label: str
    consensus_gap: float
    competition: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_lambda: Optional[float] = None
    away_lambda: Optional[float] = None
    minute: Optional[int] = None
    pressure: Optional[float] = None
    xg_10m: Optional[float] = None
    score_state: Optional[str] = None


def _score_state(home_score: int, away_score: int) -> str:
    if home_score == away_score:
        return "Empate"
    return "Casa a vencer" if home_score > away_score else "Fora a vencer"


def build_current_bet_profile(snap: Dict[str, Any]) -> CurrentBetProfile:
    """
    Extrai o perfil da aposta atual a partir do `snap` já construído por
    `build_match_snapshot` — apenas leitura de valores já calculados
    (`snap["value"]`, `snap["decision"]`, `snap["consensus"]`,
    `snap["strength"]`, `snap["live"]`, `snap["card"]`).
    """
    v = snap["value"]
    d = snap["decision"]
    c = snap["consensus"]
    s = snap["strength"]
    live = snap["live"]
    card = snap["card"]

    return CurrentBetProfile(
        market=v["market"],
        odd=v["bookie_odd"],
        # `snap["value"]` avalia o mercado "Próximo Golo (15m)", cuja
        # probabilidade do motor é a do Goal Engine (ver
        # `build_match_snapshot`: `live_bet` é calculado a partir de
        # `goal_engine_prob`) — mesmo valor, não recalculado.
        probability_pct=snap["models"]["goal_engine"]["probability"],
        edge_pct=v["edge_pct"],
        ev_pct=v["ev_pct"],
        kelly_pct=v["kelly_pct"],
        confidence_label=d["confidence_label"],
        confidence_score=d["confidence_score"],
        consensus_label=c["label"],
        consensus_gap=c["gap"],
        competition=card.get("competition"),
        home_team=card.get("home_team"),
        away_team=card.get("away_team"),
        home_lambda=s.get("home_lambda"),
        away_lambda=s.get("away_lambda"),
        minute=card.get("minute"),
        pressure=live.get("pressure"),
        xg_10m=live.get("estimated_xg_10m"),
        score_state=_score_state(card.get("home_score", 0), card.get("away_score", 0)),
    )


# ---------------------------------------------------------------------------
# Pesquisa de jogos históricos semelhantes
# ---------------------------------------------------------------------------


def find_similar_bets(
    profile: CurrentBetProfile,
    historical_bets: pd.DataFrame,
    min_sample: int = MIN_SAMPLE_TARGET,
) -> Dict[str, Any]:
    """
    Filtra `historical_bets` (o `BacktestReport.all_bets` já avaliado por
    `evaluate_bets`/`BacktestEngine` — mesmas colunas odd/probability/
    edge/competition/won/profit/stake usadas pelo Backtesting Global) para
    os jogos "semelhantes" à aposta atual.

    Critérios aplicados (só os que existem no dataset — ver
    `_UNAVAILABLE_CRITERIA` para os que a lista pedida também contempla
    mas que este dataset de demonstração não fornece):
        - odd semelhante (banda de tolerância relativa)
        - intervalo de probabilidade semelhante (banda em pontos percentuais)
        - competição semelhante (preferencial: tentada primeiro; relaxada
          se não houver amostra suficiente)

    As bandas alargam progressivamente até atingir `min_sample`
    resultados, ou até esgotar as bandas definidas — nesse caso devolve a
    banda mais larga tentada (mesmo que abaixo de `min_sample`).
    """
    empty_result = {
        "matches": historical_bets.iloc[0:0] if historical_bets is not None else pd.DataFrame(),
        "criteria_applied": [],
        "criteria_unavailable": list(_UNAVAILABLE_CRITERIA),
        "odd_tolerance_pct": None,
        "prob_tolerance_pp": None,
        "competition_matched": False,
    }
    if historical_bets is None or historical_bets.empty:
        return empty_result

    base = historical_bets.copy()
    best_subset = base.iloc[0:0]
    best_meta = {"odd_tolerance_pct": None, "prob_tolerance_pp": None, "competition_matched": False}

    for require_competition in (True, False):
        if require_competition and not profile.competition:
            continue
        for odd_tol, prob_tol in zip(_ODD_TOLERANCE_STEPS_PCT, _PROB_TOLERANCE_STEPS_PP):
            mask = pd.Series(True, index=base.index)

            odd_low = profile.odd * (1.0 - odd_tol / 100.0)
            odd_high = profile.odd * (1.0 + odd_tol / 100.0)
            mask &= base["odd"].between(odd_low, odd_high)

            prob_low = max((profile.probability_pct - prob_tol) / 100.0, 0.0)
            prob_high = min((profile.probability_pct + prob_tol) / 100.0, 1.0)
            mask &= base["probability"].between(prob_low, prob_high)

            if require_competition:
                mask &= base["competition"].astype(str).str.casefold() == str(profile.competition).casefold()

            subset = base[mask]

            if len(subset) > len(best_subset):
                best_subset = subset
                best_meta = {
                    "odd_tolerance_pct": odd_tol,
                    "prob_tolerance_pp": prob_tol,
                    "competition_matched": require_competition,
                }

            if len(subset) >= min_sample:
                return _build_search_result(profile, subset, odd_tol, prob_tol, require_competition)

    if best_meta["odd_tolerance_pct"] is None:
        return empty_result

    return _build_search_result(
        profile,
        best_subset,
        best_meta["odd_tolerance_pct"],
        best_meta["prob_tolerance_pp"],
        best_meta["competition_matched"],
    )


def _build_search_result(
    profile: CurrentBetProfile,
    subset: pd.DataFrame,
    odd_tol: float,
    prob_tol: float,
    competition_matched: bool,
) -> Dict[str, Any]:
    criteria_applied = [
        f"Odd semelhante a {profile.odd:.2f} (± {odd_tol:.0f}%)",
        f"Probabilidade do motor semelhante a {profile.probability_pct:.1f}% (± {prob_tol:.0f} p.p.)",
    ]
    if competition_matched:
        criteria_applied.append(f"Competição semelhante ({profile.competition})")

    return {
        "matches": subset.sort_values("date") if "date" in subset.columns else subset,
        "criteria_applied": criteria_applied,
        "criteria_unavailable": list(_UNAVAILABLE_CRITERIA),
        "odd_tolerance_pct": odd_tol,
        "prob_tolerance_pp": prob_tol,
        "competition_matched": competition_matched,
    }


# ---------------------------------------------------------------------------
# Resultado histórico (reutiliza src.backtest.historical.metrics)
# ---------------------------------------------------------------------------


def summarize_similar_bets(subset: pd.DataFrame) -> Dict[str, Any]:
    """
    Aplica `summary_metrics` (a MESMA função usada pelo Backtesting
    Global) ao subconjunto de jogos semelhantes, e prepara as séries para
    os gráficos de distribuição (item 4) — todas derivadas de colunas já
    existentes em `subset` (profit, stake, odd, probability, won),
    nenhuma recalculada com fórmula diferente.
    """
    metrics = summary_metrics(subset)

    if subset.empty:
        return {
            **metrics,
            "equity_curve": pd.Series(dtype=float),
            "wl_sequence": [],
            "roi_per_bet_pct": pd.Series(dtype=float),
            "odds": pd.Series(dtype=float),
            "probabilities": pd.Series(dtype=float),
        }

    valid_stake = subset[subset["stake"] > 0]
    roi_per_bet_pct = (valid_stake["profit"] / valid_stake["stake"]) * 100.0

    return {
        **metrics,
        "equity_curve": equity_curve(subset),
        "wl_sequence": subset["won"].tolist(),
        "roi_per_bet_pct": roi_per_bet_pct,
        "odds": subset["odd"],
        "probabilities": subset["probability"] * 100.0,
    }


# ---------------------------------------------------------------------------
# Comparação lado a lado
# ---------------------------------------------------------------------------


def build_comparison(profile: CurrentBetProfile, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compara a aposta atual com a média do histórico semelhante — apenas
    nas dimensões que ambos os lados têm (probabilidade, edge, kelly,
    odd). Nenhum valor novo é calculado: `avg_edge_pct`/`avg_kelly_pct`
    já vêm de `summary_metrics`; a média de probabilidade é a média
    simples da mesma coluna `probability` já usada por `summary_metrics`.
    """
    n_bets = summary.get("n_bets", 0)
    avg_probability_pct = round(float(summary["probabilities"].mean()), 1) if n_bets else None
    avg_odd = summary.get("avg_odd") if n_bets else None

    return [
        {
            "label": "Probabilidade",
            "current": profile.probability_pct,
            "historical_avg": avg_probability_pct,
            "unit": "%",
        },
        {
            "label": "Edge",
            "current": profile.edge_pct,
            "historical_avg": summary.get("avg_edge_pct") if n_bets else None,
            "unit": "%",
        },
        {
            "label": "Kelly",
            "current": profile.kelly_pct,
            "historical_avg": summary.get("avg_kelly_pct") if n_bets else None,
            "unit": "%",
        },
        {
            "label": "Odd",
            "current": profile.odd,
            "historical_avg": avg_odd,
            "unit": "",
        },
    ]


# ---------------------------------------------------------------------------
# Veredicto e explicação
# ---------------------------------------------------------------------------


def build_verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classificação de apresentação (mesmo padrão de
    `src.report.dashboard_data.decision_badge`/`confidence_badge`: agrupa
    valores já calculados em rótulos, não introduz nenhum cálculo
    financeiro novo) sobre o ROI/Hit Rate já devolvidos por
    `summary_metrics` para o subconjunto de jogos semelhantes.
    """
    n_bets = summary.get("n_bets", 0)
    roi_pct = summary.get("roi_pct", 0.0)
    hit_rate_pct = summary.get("hit_rate_pct", 0.0)

    if n_bets < MIN_CONCLUSIVE_SAMPLE:
        return {
            "label": "🟡 DADOS HISTÓRICOS INSUFICIENTES",
            "color": "warn",
            "headline": (
                f"Foram encontrados apenas {n_bets} jogo(s) semelhante(s) no dataset de demonstração "
                "— amostra pequena de mais para uma validação estatisticamente conclusiva."
            ),
            "roi_pct": roi_pct,
            "hit_rate_pct": hit_rate_pct,
            "n_bets": n_bets,
        }

    if roi_pct > 0:
        return {
            "label": "🟢 VALIDAÇÃO HISTÓRICA POSITIVA",
            "color": "ok",
            "headline": (
                f"Esta aposta apresenta comportamento semelhante a {n_bets} apostas anteriores, "
                f"com ROI histórico de {roi_pct:+.1f}% e taxa de sucesso de {hit_rate_pct:.1f}%."
            ),
            "roi_pct": roi_pct,
            "hit_rate_pct": hit_rate_pct,
            "n_bets": n_bets,
        }

    return {
        "label": "🔴 VALIDAÇÃO HISTÓRICA NEGATIVA",
        "color": "off",
        "headline": (
            f"Apesar do Edge atual, {n_bets} apostas semelhantes apresentaram ROI histórico de "
            f"{roi_pct:+.1f}% e taxa de sucesso de {hit_rate_pct:.1f}%."
        ),
        "roi_pct": roi_pct,
        "hit_rate_pct": hit_rate_pct,
        "n_bets": n_bets,
    }


def build_validation_explanation(verdict: Dict[str, Any], summary: Dict[str, Any]) -> str:
    """Texto automático (só formatação sobre valores já calculados, sem IA/LLM)."""
    if verdict["color"] == "warn":
        return (
            "Ainda não existem jogos históricos suficientes semelhantes a esta aposta no dataset de "
            "demonstração para confirmar ou desaconfiar a recomendação atual — a decisão do motor "
            "mantém-se baseada apenas no Edge/EV/Kelly calculados agora."
        )
    if verdict["color"] == "ok":
        return (
            "A recomendação atual é reforçada porque apostas historicamente semelhantes (mesma odd, "
            f"probabilidade e competição na mesma vizinhança) apresentaram ROI positivo "
            f"({verdict['roi_pct']:+.1f}%) e taxa de sucesso de {verdict['hit_rate_pct']:.1f}%, com "
            f"drawdown máximo de {summary.get('max_drawdown_pct', 0.0):.1f}%."
        )
    return (
        "A recomendação atual deve ser interpretada com cautela: apostas historicamente semelhantes "
        f"(mesma odd, probabilidade e competição na mesma vizinhança) tiveram desempenho abaixo do "
        f"esperado (ROI de {verdict['roi_pct']:+.1f}%, taxa de sucesso de {verdict['hit_rate_pct']:.1f}%)."
    )


# ---------------------------------------------------------------------------
# Ponto de entrada único
# ---------------------------------------------------------------------------


def build_historical_validation(snap: Dict[str, Any], all_bets: pd.DataFrame) -> Dict[str, Any]:
    """
    Ponto de entrada único deste módulo: monta toda a estrutura consumida
    pelo painel "📈 VALIDAÇÃO HISTÓRICA DA APOSTA ATUAL" em `scripts/app.py`.

    `all_bets` é `BacktestReport.all_bets` — o MESMO relatório (mesmo
    `BacktestEngine`, mesmo dataset) já usado pelo painel de Backtesting
    Global; este módulo não carrega nem recalcula nenhum dataset novo.
    """
    profile = build_current_bet_profile(snap)
    search = find_similar_bets(profile, all_bets)
    summary = summarize_similar_bets(search["matches"])
    comparison = build_comparison(profile, summary)
    verdict = build_verdict(summary)
    explanation = build_validation_explanation(verdict, summary)

    return {
        "profile": profile,
        "search": search,
        "summary": summary,
        "comparison": comparison,
        "verdict": verdict,
        "explanation": explanation,
    }

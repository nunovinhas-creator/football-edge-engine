"""
Explainability Engine — Melhoria #13.

Camada de interpretação 100% determinística (SEM IA, SEM LLM) sobre os
resultados que o motor já produz — Goal Engine, Monte Carlo, Machine
Learning, Dixon-Coles, Edge, EV, Kelly, Consenso entre modelos, Lambda
Estimator/Strength e métricas live (pressão, momentum).

`generate_explanation(snapshot)` recebe o MatchSnapshot já existente
(o mesmo dict devolvido por `src.report.dashboard_data.build_match_snapshot`,
ver esse módulo) e apenas LÊ os valores já lá calculados. Este ficheiro:

  - NÃO recalcula nenhuma probabilidade, edge, EV, Kelly ou lambda;
  - NÃO altera nenhuma decisão (BET/WATCH/PASS) nem nenhum threshold do
    motor (Dixon-Coles, Monte Carlo, Goal Engine, ML, Kelly, Edge, EV,
    Lambda Estimator, Decision Engine permanecem exatamente como estão);
  - NÃO importa nem chama nenhum desses módulos — só lê o dict já
    construído por eles.

Os limiares usados abaixo (ex.: "Goal Engine > 70%") são puramente de
INTERPRETAÇÃO/apresentação (redação do texto explicativo), definidos por
este módulo — nunca os thresholds de decisão do motor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Explanation:
    decision: str
    confidence: str
    score: float
    positives: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    consensus: str = "—"
    summary: str = ""


# ---------------------------------------------------------------------------
# Limiares de interpretação (apenas texto — não são thresholds do motor)
# ---------------------------------------------------------------------------

GOAL_ENGINE_FAVORABLE = 70.0
ML_FAVORABLE = 65.0
MONTE_CARLO_FAVORABLE = 65.0
EDGE_FAVORABLE_PCT = 5.0
KELLY_LOW_PCT = 1.0
PRESSURE_HIGH = 60.0
PRESSURE_LOW = 40.0
CONFIDENCE_LOW = 40.0
CONSENSUS_GAP_DIVIDED = 15.0
EFFECTIVE_SAMPLE_LOW = 10
LIVE_MINUTE_EARLY = 15
ODDS_SHIFT_RELEVANT_PCT = 8.0

_RISING_MOMENTUM = {"SURGING", "RISING"}


def _dig(source: Any, *keys: str, default: Any = None) -> Any:
    """Navega `source[keys[0]][keys[1]]...` sem lançar exceção nem
    recalcular nada — devolve `default` assim que uma chave falta ou o
    valor é None."""
    current = source
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _join_reasons(reasons: List[str]) -> str:
    if not reasons:
        return ""
    if len(reasons) == 1:
        return reasons[0]
    return ", ".join(reasons[:-1]) + " e " + reasons[-1]


# ---------------------------------------------------------------------------
# Pontos positivos
# ---------------------------------------------------------------------------

def _positives(snapshot: Dict[str, Any]) -> List[str]:
    positives: List[str] = []

    goal_engine_prob = _dig(snapshot, "models", "goal_engine", "probability")
    if goal_engine_prob is not None and goal_engine_prob > GOAL_ENGINE_FAVORABLE:
        positives.append(f"Goal Engine muito favorável ({goal_engine_prob:.1f}%).")

    ml_prob = _dig(snapshot, "models", "machine_learning", "probability")
    if ml_prob is not None and ml_prob > ML_FAVORABLE:
        positives.append(f"Modelo ML confirma oportunidade ({ml_prob:.1f}%).")

    monte_carlo_prob = _dig(snapshot, "models", "monte_carlo", "over_15")
    if monte_carlo_prob is not None and monte_carlo_prob > MONTE_CARLO_FAVORABLE:
        positives.append(f"Monte Carlo confirma cenário ({monte_carlo_prob:.1f}%).")

    edge_pct = _dig(snapshot, "value", "edge_pct")
    if edge_pct is not None and edge_pct > EDGE_FAVORABLE_PCT:
        positives.append(f"Existe valor estatístico (Edge {edge_pct:+.1f}%).")

    ev_pct = _dig(snapshot, "value", "ev_pct")
    if ev_pct is not None and ev_pct > 0:
        positives.append(f"Valor esperado positivo (EV {ev_pct:+.2f}%).")

    kelly_pct = _dig(snapshot, "value", "kelly_pct")
    if kelly_pct is not None and kelly_pct > 0:
        positives.append(f"Gestão de banca recomenda entrada (Kelly {kelly_pct:.1f}%).")

    consensus_label = _dig(snapshot, "consensus", "label")
    if consensus_label == "Muito Forte":
        positives.append("Consenso Muito Forte — todos os modelos concordam.")

    pressure = _dig(snapshot, "live", "pressure")
    if pressure is not None and pressure > PRESSURE_HIGH:
        positives.append(f"Pressão ofensiva significativa ({pressure:.0f}/100).")

    momentum = _dig(snapshot, "live", "momentum")
    if momentum in _RISING_MOMENTUM:
        positives.append("Momentum crescente — equipa dominante nos últimos minutos.")

    return positives


# ---------------------------------------------------------------------------
# Pontos negativos
# ---------------------------------------------------------------------------

def _negatives(snapshot: Dict[str, Any]) -> List[str]:
    negatives: List[str] = []

    edge_pct = _dig(snapshot, "value", "edge_pct")
    if edge_pct is not None and edge_pct <= 0:
        negatives.append(f"Edge negativo ({edge_pct:+.1f}%) — sem valor estatístico neste mercado.")

    ev_pct = _dig(snapshot, "value", "ev_pct")
    if ev_pct is not None and ev_pct < 0:
        negatives.append(f"EV negativo ({ev_pct:+.2f}%) — valor esperado desfavorável.")

    goal_engine_prob = _dig(snapshot, "models", "goal_engine", "probability")
    ml_prob = _dig(snapshot, "models", "machine_learning", "probability")
    consensus_gap = _dig(snapshot, "consensus", "gap")
    if consensus_gap is None and goal_engine_prob is not None and ml_prob is not None:
        consensus_gap = abs(goal_engine_prob - ml_prob)
    if consensus_gap is not None and consensus_gap > CONSENSUS_GAP_DIVIDED:
        negatives.append(f"Goal Engine dividido do ML (diferença de {consensus_gap:.1f} p.p.).")

    pressure = _dig(snapshot, "live", "pressure")
    if pressure is not None and pressure < PRESSURE_LOW:
        negatives.append(f"Pouca pressão ofensiva ({pressure:.0f}/100).")

    confidence_score = _dig(snapshot, "decision", "confidence_score")
    if confidence_score is None:
        confidence_score = _dig(snapshot, "models", "machine_learning", "confidence")
    if confidence_score is not None and confidence_score < CONFIDENCE_LOW:
        negatives.append(f"Baixa confiança estatística ({confidence_score:.0f}/100).")

    kelly_pct = _dig(snapshot, "value", "kelly_pct")
    if kelly_pct is not None and 0 < kelly_pct < KELLY_LOW_PCT:
        negatives.append(f"Kelly muito reduzido ({kelly_pct:.2f}%) — stake recomendada marginal.")

    return negatives


# ---------------------------------------------------------------------------
# Avisos
# ---------------------------------------------------------------------------

def _warnings(snapshot: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    effective_sample_size = _dig(snapshot, "strength", "effective_sample_size")
    h2h_available = _dig(snapshot, "strength", "h2h_available")
    if effective_sample_size is not None and effective_sample_size < EFFECTIVE_SAMPLE_LOW:
        warnings.append(f"Modelo baseado em poucos jogos (amostra efetiva de {effective_sample_size}).")
    elif h2h_available is False:
        warnings.append("Modelo baseado em poucos jogos (sem histórico H2H carregado).")

    odds_shift_pct = _dig(snapshot, "live", "odds_shift_pct")
    if odds_shift_pct is None:
        odds_shift_pct = _dig(snapshot, "value", "odds_shift_pct")
    if odds_shift_pct is not None and abs(odds_shift_pct) >= ODDS_SHIFT_RELEVANT_PCT:
        warnings.append(f"Odds alteraram rapidamente ({odds_shift_pct:+.1f}%).")

    lambda_confidence = _dig(snapshot, "strength", "lambda_confidence")
    tier = _dig(snapshot, "strength", "tier")
    if lambda_confidence is not None and lambda_confidence < CONFIDENCE_LOW:
        warnings.append(f"Baixa confiança do Lambda ({lambda_confidence:.0f}/100).")
    elif lambda_confidence is None and isinstance(tier, str) and "N/D" in tier:
        warnings.append("Baixa confiança do Lambda (sem força de equipas pré-jogo carregada).")

    consensus_label = _dig(snapshot, "consensus", "label")
    if consensus_label == "Fraco":
        warnings.append("Sem consenso entre modelos (Goal Engine e ML divergem fortemente).")

    minute = _dig(snapshot, "card", "minute")
    dangerous_attacks = _dig(snapshot, "live", "dangerous_attacks_10m")
    shots = _dig(snapshot, "live", "shots_10m")
    corners = _dig(snapshot, "live", "corners_10m")
    if minute is not None and minute < LIVE_MINUTE_EARLY and not any([dangerous_attacks, shots, corners]):
        warnings.append("Poucos dados live disponíveis (jogo ainda em fase inicial).")

    return warnings


# ---------------------------------------------------------------------------
# Resumo automático
# ---------------------------------------------------------------------------

def _summary(
    decision_label: str,
    consensus_label: str,
    edge_pct: Optional[float],
    ev_pct: Optional[float],
) -> str:
    label = decision_label or ""

    if "NÃO APOSTAR" in label or "❌" in label:
        reasons = []
        if edge_pct is not None and edge_pct <= 0:
            reasons.append("não existe valor estatístico")
        if consensus_label in ("Fraco", "Moderado", None, "—"):
            reasons.append("não existe consenso suficiente entre os modelos")
        if not reasons:
            reasons.append("não existe valor estatístico nem consenso suficiente")
        return "A recomendação é não apostar porque " + _join_reasons(reasons) + "."

    if "AGUARDAR" in label:
        reasons = []
        if consensus_label in ("Moderado", "Fraco"):
            reasons.append("os modelos apresentam divergência")
        if edge_pct is None or edge_pct <= EDGE_FAVORABLE_PCT:
            reasons.append("o Edge ainda é insuficiente")
        if not reasons:
            reasons.append("o valor estatístico ainda não é suficientemente forte")
        return "A recomendação é aguardar porque " + _join_reasons(reasons) + "."

    if "APOSTAR" in label:
        reasons = []
        if consensus_label in ("Muito Forte", "Forte"):
            reasons.append(f"Goal Engine, Monte Carlo e ML apresentam consenso {consensus_label.lower()}")
        if edge_pct is not None and edge_pct > 0:
            reasons.append("existe Edge positivo")
        if ev_pct is not None and ev_pct > 0:
            reasons.append("o EV é positivo")
        if not reasons:
            reasons.append("os indicadores disponíveis são favoráveis")
        return "A recomendação é apostar porque " + _join_reasons(reasons) + "."

    return "Sem decisão suficientemente clara para gerar um resumo."


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def generate_explanation(snapshot: Dict[str, Any]) -> Explanation:
    """
    Gera a `Explanation` determinística de um MatchSnapshot já construído
    pelo motor (`src.report.dashboard_data.build_match_snapshot` ou
    qualquer dict com a mesma forma). Não recalcula nada: apenas lê
    `decision`, `models`, `value`, `consensus`, `live`, `strength` e
    `engine_score` já presentes no snapshot.
    """
    snapshot = snapshot or {}

    decision_label = _dig(snapshot, "decision", "label", default="—")
    confidence_label = _dig(snapshot, "decision", "confidence_label", default="—")
    score = _dig(snapshot, "engine_score", "score", default=0.0)
    consensus_label = _dig(snapshot, "consensus", "label", default="—")
    edge_pct = _dig(snapshot, "value", "edge_pct")
    ev_pct = _dig(snapshot, "value", "ev_pct")

    positives = _positives(snapshot)
    negatives = _negatives(snapshot)
    warnings = _warnings(snapshot)
    summary = _summary(decision_label, consensus_label, edge_pct, ev_pct)

    return Explanation(
        decision=decision_label,
        confidence=confidence_label,
        score=float(score) if score is not None else 0.0,
        positives=positives,
        negatives=negatives,
        warnings=warnings,
        consensus=consensus_label,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Formatação para Telegram (apenas texto — reutiliza os campos já gerados
# acima, não recalcula nem decide nada)
# ---------------------------------------------------------------------------

def format_explanation_block(explanation: Explanation) -> str:
    """Bloco '🧠 Porque esta decisão?' pronto a anexar a uma mensagem já
    existente (ex.: o alerta Telegram de +EV em `src.live.value_alerts`)."""
    lines = ["🧠 *Porque esta decisão?*"]
    for item in explanation.positives:
        lines.append(f"✅ {item}")
    for item in explanation.negatives:
        lines.append(f"⚠ {item}")
    for item in explanation.warnings:
        lines.append(f"🚨 {item}")
    lines.append("")
    lines.append(f"_{explanation.summary}_")
    return "\n".join(lines)


def build_telegram_message(snapshot: Dict[str, Any]) -> str:
    """
    Mensagem Telegram completa (cabeçalho + métricas já existentes no
    snapshot + bloco de explicação) no formato pedido pela Melhoria #13.
    Só lê `snapshot` — não recalcula probabilidade, odd justa, edge, EV
    ou Kelly.
    """
    explanation = generate_explanation(snapshot)

    decision_label = _dig(snapshot, "decision", "label", default=explanation.decision)
    goal_engine_prob = _dig(snapshot, "models", "goal_engine", "probability")
    fair_odd = _dig(snapshot, "value", "fair_odd")
    edge_pct = _dig(snapshot, "value", "edge_pct")
    ev_pct = _dig(snapshot, "value", "ev_pct")
    kelly_pct = _dig(snapshot, "value", "kelly_pct")

    lines = ["⚽ Football Edge Engine", "", decision_label or "—", ""]

    if goal_engine_prob is not None:
        lines.append(f"Probabilidade: {goal_engine_prob:.0f}%")
        lines.append("")
    if fair_odd is not None:
        lines.append(f"Odd Justa: {fair_odd:.2f}")
        lines.append("")
    if edge_pct is not None:
        lines.append(f"Edge: {edge_pct:+.1f}%")
        lines.append("")
    if ev_pct is not None:
        lines.append(f"EV: {ev_pct:+.2f}")
        lines.append("")
    if kelly_pct is not None:
        lines.append(f"Kelly: {kelly_pct:.1f}%")
        lines.append("")

    lines.append(format_explanation_block(explanation))

    return "\n".join(lines).rstrip()

"""
Orquestrador Principal do Engine (Full Engine).
Integra os módulos de Edge, Kelly, Confiança, Decisão e Explicabilidade.
"""

from typing import Dict, Any

try:
    from src.engine.edge import calculate_edge
except ImportError:
    from src.engine.edge import edge as calculate_edge

try:
    from src.engine.kelly import calculate_kelly
except ImportError:
    try:
        from src.engine.kelly import kelly_stake as calculate_kelly
    except ImportError:
        def calculate_kelly(p: float, o: float, fraction: float = 0.25) -> float:
            b = o - 1.0
            q = 1.0 - p
            f = (b * p - q) / b if b > 0 else 0.0
            return max(0.0, f * fraction)

from src.engine.confidence import calculate_confidence
from src.engine.decision import evaluate_decision
from src.engine.explanation import generate_explanation

def run_pipeline(
    prob_model: float,
    odd_house: float,
    bankroll: float = 1000.0,
    sample_size: int = 5,
    model_std: float = 0.03,
    match_info: str = "Jogo Desconhecido"
) -> Dict[str, Any]:
    """
    Executa o pipeline completo do engine para uma oportunidade de aposta.
    """
    # 1. Edge (EV)
    edge = calculate_edge(prob_model, odd_house)

    # 2. Kelly Stake
    raw_stake_pct = calculate_kelly(prob_model, odd_house)

    # 3. Confiança
    confidence = calculate_confidence(
        base_confidence=prob_model,
        model_std=model_std,
        sample_size=sample_size
    )

    # 4. Decisão Ponderada (aplica capping se necessário)
    decision_result = evaluate_decision(
        edge=edge,
        confidence=confidence,
        stake=raw_stake_pct
    )

    # 5. Stake Final Reajustada
    final_stake_pct = decision_result.recommended_stake
    stake_amount = round(bankroll * final_stake_pct, 2)

    # 6. Relatório de Explicabilidade
    explanation_text = generate_explanation(decision_result, match_info=match_info)

    return {
        "match_info": match_info,
        "prob_model": prob_model,
        "odd_house": odd_house,
        "edge": edge,
        "confidence": confidence,
        "stake_pct": final_stake_pct,
        "stake_amount": stake_amount,
        "decision": decision_result,
        "explanation": explanation_text
    }

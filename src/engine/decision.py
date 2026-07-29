"""
Módulo de Tomada de Decisão do Engine.
Avalia múltiplos fatores ponderados para aprovar (BET) ou rejeitar (PASS) uma oportunidade.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class DecisionFactor:
    name: str
    value: float
    threshold: float
    passed: bool
    weight: float

@dataclass
class DecisionResult:
    action: str             # "BET" ou "PASS"
    total_score: float      # Pontuação de 0.0 a 100.0
    edge: float
    recommended_stake: float
    factors: List[DecisionFactor] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

def evaluate_decision(
    edge: float,
    confidence: float,
    stake: float = 0.02,
    min_edge: float = 0.03,
    min_confidence: float = 0.50,
    max_stake: float = 0.05
) -> DecisionResult:
    factors = []
    reasons = []
    
    # 1. Avaliação do Edge (EV)
    edge_passed = edge >= min_edge
    factors.append(DecisionFactor(
        name="Edge (EV)",
        value=edge,
        threshold=min_edge,
        passed=edge_passed,
        weight=0.5
    ))
    if not edge_passed:
        reasons.append(f"Edge insuficiente ({edge:.1%} < mínimo de {min_edge:.1%})")
        
    # 2. Avaliação da Confiança
    conf_passed = confidence >= min_confidence
    factors.append(DecisionFactor(
        name="Confiança",
        value=confidence,
        threshold=min_confidence,
        passed=conf_passed,
        weight=0.3
    ))
    if not conf_passed:
        reasons.append(f"Confiança insuficiente ({confidence:.2f} < mínimo de {min_confidence:.2f})")
        
    # 3. Avaliação da Stake (Gestão de Risco)
    stake_passed = 0 < stake <= max_stake
    factors.append(DecisionFactor(
        name="Stake Recomendada",
        value=stake,
        threshold=max_stake,
        passed=stake_passed,
        weight=0.2
    ))
    if stake <= 0:
        reasons.append("Stake recomendada é zero")
    elif stake > max_stake:
        reasons.append(f"Stake excede o limite máximo permitido ({stake:.1%} > {max_stake:.1%})")

    # Cálculo da pontuação final (0 a 100)
    score_edge = min(max(edge / (min_edge * 2), 0.0), 1.0) * 50
    score_conf = confidence * 30
    score_stake = min(stake / max_stake, 1.0) * 20 if stake > 0 else 0
    
    total_score = round(score_edge + score_conf + score_stake, 2)
    
    action = "BET" if (edge_passed and conf_passed and 0 < stake <= max_stake) else "PASS"
    
    return DecisionResult(
        action=action,
        total_score=total_score,
        edge=edge,
        recommended_stake=stake,
        factors=factors,
        reasons=reasons
    )

def make_decision(edge: float, confidence: float, stake: float = 0.02, **kwargs) -> DecisionResult:
    """Wrapper para manter compatibilidade com os testes existentes."""
    return evaluate_decision(edge, confidence, stake, **kwargs)

def decide(edge: float, confidence: float, stake: float = 0.02, **kwargs) -> DecisionResult:
    """Wrapper de conveniência."""
    return evaluate_decision(edge, confidence, stake, **kwargs)

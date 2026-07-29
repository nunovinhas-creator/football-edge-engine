"""
Módulo de Explicabilidade do Engine.
Converte objetos DecisionResult em relatórios visuais e legíveis para CLI/Logs.
"""

from typing import Any

def generate_explanation(decision_result: Any, match_info: str = "Jogo") -> str:
    if isinstance(decision_result, bool):
        return f"[{'BET' if decision_result else 'PASS'}] Decisão simplificada."
    
    action = getattr(decision_result, 'action', 'PASS')
    score = getattr(decision_result, 'total_score', 0.0)
    edge = getattr(decision_result, 'edge', 0.0)
    stake = getattr(decision_result, 'recommended_stake', 0.0)
    factors = getattr(decision_result, 'factors', [])
    reasons = getattr(decision_result, 'reasons', [])

    status_symbol = "🟢 [APROVADO]" if action == "BET" else "🔴 [REJEITADO]"
    
    lines = []
    lines.append("=" * 50)
    lines.append(f" DECISÃO FINAL: {action} {status_symbol}")
    lines.append(f" Contexto: {match_info}")
    lines.append(f" Pontuação Geral (Score): {score:.1f}/100")
    lines.append("-" * 50)
    lines.append(" FACTORES AVALIADOS:")
    
    for f in factors:
        icon = "  ✓" if f.passed else "  ✗"
        if "Edge" in f.name:
            val_str = f"{f.value:+.1%}"
            thresh_str = f"{f.threshold:.1%}"
        elif "Stake" in f.name:
            val_str = f"{f.value:.1%}"
            thresh_str = f"{f.threshold:.1%}"
        else:
            val_str = f"{f.value:.2f}"
            thresh_str = f"{f.threshold:.2f}"
            
        lines.append(f"{icon} {f.name:<18}: {val_str} (Mín/Máx: {thresh_str})")
        
    lines.append("-" * 50)
    lines.append(f" STAKE RECOMENDADA: {stake:.1%}")
    
    if reasons:
        lines.append("-" * 50)
        lines.append(" MOTIVOS DE REJEIÇÃO / AVISOS:")
        for r in reasons:
            lines.append(f"  - {r}")
            
    lines.append("=" * 50)
    
    return "\n".join(lines)

def explain(decision_result: Any, match_info: str = "Jogo") -> str:
    return generate_explanation(decision_result, match_info)

"""
Módulo de cálculo de Confiança do Engine.
Pondera a incerteza do modelo e o volume de amostra dos dados.
"""

def calculate_confidence(
    base_confidence: float = 1.0,
    model_std: float = 0.0,
    sample_size: int = 5,
    min_sample_size: int = 5
) -> float:
    """
    Calcula um índice de confiança entre 0.0 e 1.0.
    
    :param base_confidence: Confiança base inicial (0.0 a 1.0).
    :param model_std: Desvio-padrão das previsões das árvores (incerteza do ensemble).
    :param sample_size: Número de jogos recentes analisados.
    :param min_sample_size: Mínimo de jogos desejado para amostra completa.
    :return: Valor numérico da confiança final.
    """
    uncertainty_penalty = min(model_std * 2.0, 0.4)
    sample_factor = min(sample_size / max(min_sample_size, 1), 1.0)
    adjusted_confidence = (base_confidence - uncertainty_penalty) * sample_factor
    
    return max(0.0, min(1.0, round(adjusted_confidence, 4)))


def confidence_level(base_confidence: float = 1.0, model_std: float = 0.0, sample_size: int = 5) -> float:
    """Wrapper para manter compatibilidade com os testes existentes."""
    return calculate_confidence(
        base_confidence=base_confidence,
        model_std=model_std,
        sample_size=sample_size
    )

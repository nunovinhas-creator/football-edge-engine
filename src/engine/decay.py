import numpy as np

def apply_exponential_decay(series: np.ndarray, decay_rate: float = 0.05) -> np.ndarray:
    """
    Aplica ponderação exponencial a uma série temporal de dados estatísticos.
    Os jogos mais recentes recebem peso mais próximo de 1.0, reduzindo gradualmente para os mais antigos.
    """
    n = len(series)
    if n == 0:
        return series
    
    # Pesos do mais antigo (0) ao mais recente (n-1)
    weights = np.exp(-decay_rate * np.arange(n)[::-1])
    weights /= np.sum(weights)  # Normalizar pesos
    
    return np.sum(series * weights)

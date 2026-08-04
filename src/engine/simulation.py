import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationResult:
    simulations: int
    over_15_prob: float
    over_25_prob: float
    btts_prob: float
    expected_goals_home: float
    expected_goals_away: float

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def run_match_simulation(
        self,
        current_minute: int,
        current_home_score: int,
        current_away_score: int,
        home_lambda: float,
        away_lambda: float,
        match_id=None,
    ) -> SimulationResult:
        """
        Simula o restante tempo de jogo usando Distribuição de Poisson + Monte Carlo.
        """
        remaining_ratio = max(0.0, (90.0 - current_minute) / 90.0)

        # Ajusta as taxas de golo esperadas para o tempo restante
        rem_home_lambda = home_lambda * remaining_ratio
        rem_away_lambda = away_lambda * remaining_ratio

        # Gerador local, semeado apenas com (match_id, minute) — reprodutível
        # para o mesmo jogo/minuto sem usar seeds globais (np.random.seed).
        seed = [int(match_id or 0) % (2**32), int(current_minute) % (2**32)]
        rng = np.random.default_rng(seed)

        # Simulação Monte Carlo (geração de golos via Poisson)
        simulated_home_goals = rng.poisson(rem_home_lambda, self.n_simulations)
        simulated_away_goals = rng.poisson(rem_away_lambda, self.n_simulations)
        
        final_home = current_home_score + simulated_home_goals
        final_away = current_away_score + simulated_away_goals
        total_goals = final_home + final_away
        
        # Cálculo das probabilidades
        over_15 = np.mean(total_goals > 1.5) * 100
        over_25 = np.mean(total_goals > 2.5) * 100
        btts = np.mean((final_home > 0) & (final_away > 0)) * 100
        
        return SimulationResult(
            simulations=self.n_simulations,
            over_15_prob=round(float(over_15), 1),
            over_25_prob=round(float(over_25), 1),
            btts_prob=round(float(btts), 1),
            expected_goals_home=round(float(np.mean(final_home)), 2),
            expected_goals_away=round(float(np.mean(final_away)), 2)
        )

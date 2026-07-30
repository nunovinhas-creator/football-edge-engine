"""
Backtester de Estratégia Live: Validação do Índice de Pressão e ROI
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto (/workspaces/football-edge-engine) está no sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from typing import List, Dict, Any

try:
    from src.live.engine import LiveGoalEngine
except ModuleNotFoundError:
    from live.engine import LiveGoalEngine

class LiveGoalBacktester:
    def __init__(self, initial_bankroll: float = 1000.0, stake_pct: float = 0.02):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.stake_pct = stake_pct
        self.engine = LiveGoalEngine()
        self.bets_history = []

    def run_simulation(self, historical_matches_data: List[Dict[str, Any]]):
        print(f"📊 A iniciar Backtest em {len(historical_matches_data)} partidas históricas...")
        
        for match in historical_matches_data:
            for minute in range(10, 85, 5):
                snapshot = self._extract_snapshot_at_minute(match, minute)
                p_goal_15m = self.engine.predict_next_goal_probability(snapshot)
                
                fair_odd = 1.0 / p_goal_15m if p_goal_15m > 0 else 99.0
                live_odd_bookie = snapshot.get('live_odd_over', 1.85)
                
                if p_goal_15m >= 0.65 and live_odd_bookie > fair_odd:
                    goal_occurred = snapshot.get('goal_in_next_15m', False)
                    stake = self.current_bankroll * self.stake_pct
                    
                    if goal_occurred:
                        profit = stake * (live_odd_bookie - 1.0)
                        self.current_bankroll += profit
                        result = 'WIN'
                    else:
                        profit = -stake
                        self.current_bankroll += profit
                        result = 'LOSS'

                    self.bets_history.append({
                        'match': f"{match['home_team']} vs {match['away_team']}",
                        'minute': minute,
                        'prob': round(p_goal_15m, 3),
                        'fair_odd': round(fair_odd, 2),
                        'bookie_odd': live_odd_bookie,
                        'goal_occurred': goal_occurred,
                        'result': result,
                        'stake': round(stake, 2),
                        'profit': round(profit, 2),
                        'bankroll_after': round(self.current_bankroll, 2)
                    })

        self._generate_report()

    def _extract_snapshot_at_minute(self, match: Dict[str, Any], minute: int) -> Dict[str, Any]:
        minute_stats = match.get('timeline', {}).get(minute, {})
        return {
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'current_minute': minute,
            'home_xg_last5': match.get('home_xg_last5', 1.50),
            'away_conceded_xg_last5': match.get('away_conceded_xg_last5', 1.20),
            'home_possession': minute_stats.get('home_possession', 60),
            'away_possession': minute_stats.get('away_possession', 40),
            'home_style': match.get('home_style', 'high_press'),
            'away_style': match.get('away_style', 'low_block_vulnerable'),
            'dangerous_attacks_10m': minute_stats.get('dangerous_attacks_10m', 12),
            'shots_on_target_10m': minute_stats.get('shots_on_target_10m', 2),
            'corners_10m': minute_stats.get('corners_10m', 3),
            'live_odd_over': minute_stats.get('live_odd_over', 1.95),
            'goal_in_next_15m': minute_stats.get('goal_in_next_15m', True)
        }

    def _generate_report(self):
        df = pd.DataFrame(self.bets_history)
        if df.empty:
            print("⚠️ Nenhuma entrada válida foi disparada durante o backtest.")
            return

        total_bets = len(df)
        wins = len(df[df['result'] == 'WIN'])
        losses = len(df[df['result'] == 'LOSS'])
        win_rate = (wins / total_bets) * 100
        total_profit = self.current_bankroll - self.initial_bankroll
        roi = (total_profit / (df['stake'].sum())) * 100

        print("\n" + "="*50)
        print("📈 RESULTADOS DO BACKTEST DE PRESSÃO LIVE")
        print("="*50)
        print(f"Total de Apostas Disparadas: {total_bets}")
        print(f"Taxa de Acerto (Win Rate):  {win_rate:.2f}% ({wins}W / {losses}L)")
        print(f"Banca Inicial:              €{self.initial_bankroll:.2f}")
        print(f"Banca Final:                €{self.current_bankroll:.2f}")
        print(f"Lucro/Prejuízo Líquido:     €{total_profit:+.2f}")
        print(f"ROI (Return on Investment): {roi:+.2f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    dummy_history = [
        {
            'home_team': 'Benfica', 'away_team': 'Braga',
            'home_xg_last5': 1.8, 'away_conceded_xg_last5': 1.4,
            'home_style': 'high_press', 'away_style': 'low_block_vulnerable',
            'timeline': {
                60: {'dangerous_attacks_10m': 15, 'shots_on_target_10m': 3, 'corners_10m': 4, 'live_odd_over': 2.10, 'goal_in_next_15m': True},
                75: {'dangerous_attacks_10m': 4, 'shots_on_target_10m': 0, 'corners_10m': 0, 'live_odd_over': 1.80, 'goal_in_next_15m': False}
            }
        },
        {
            'home_team': 'Porto', 'away_team': 'Boavista',
            'home_xg_last5': 1.6, 'away_conceded_xg_last5': 1.5,
            'home_style': 'high_press', 'away_style': 'low_block_vulnerable',
            'timeline': {
                55: {'dangerous_attacks_10m': 14, 'shots_on_target_10m': 2, 'corners_10m': 3, 'live_odd_over': 1.95, 'goal_in_next_15m': True},
                70: {'dangerous_attacks_10m': 11, 'shots_on_target_10m': 2, 'corners_10m': 2, 'live_odd_over': 2.05, 'goal_in_next_15m': False}
            }
        }
    ]

    backtester = LiveGoalBacktester(initial_bankroll=1000.0)
    backtester.run_simulation(dummy_history)

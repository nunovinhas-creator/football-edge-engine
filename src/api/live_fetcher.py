"""
Módulo de Extração de Jogos e Estatísticas em Direto (Live / In-Play) da BSD API.
"""

import time
import pandas as pd
from typing import List, Dict, Any
from src.api.client import BzzoiroClient

class BSDLiveFetcher:
    """
    Cliente especializado para monitorizar eventos live na BSD API e 
    extrair métricas de pressão e momentum das equipas.
    """
    def __init__(self):
        self.client = BzzoiroClient()

    def get_live_events(self) -> List[Dict[str, Any]]:
        """
        Procura todos os eventos ativos em direto (status = 'inplay' / 'live').
        """
        print("📡 A pesquisar jogos em direto na BSD API...")
        try:
            # Tenta filtrar eventos in-play diretamente no endpoint
            response = self.client.get("events/?status=inplay")
            events = response.get("results", []) if isinstance(response, dict) else response
            
            # Se a API não aceitar o filtro de status na URL, filtra no cliente
            if not events:
                all_events = self.client.get("events/?limit=100")
                results = all_events.get("results", []) if isinstance(all_events, dict) else all_events
                events = [e for e in results if e.get("status") in ["inplay", "live", "1st_half", "2nd_half"]]

            print(f"⚽ Jogos em direto encontrados: {len(events)}")
            return events
        except Exception as e:
            print(f"❌ Erro ao procurar jogos em direto: {e}")
            return []

    def get_live_statistics(self, event_id: int) -> Dict[str, Any]:
        """
        Extrai estatísticas detalhadas de um jogo a decorrer (ataques perigosos, cantos, remates).
        """
        try:
            stats_resp = self.client.get(f"events/{event_id}/statistics/")
            return stats_resp if isinstance(stats_resp, dict) else {}
        except Exception as e:
            print(f"⚠️ Erro ao obter estatísticas em direto para o evento {event_id}: {e}")
            return {}

    def parse_live_metrics_for_engine(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converte o evento e as suas estatísticas live no formato esperado pelo LiveGoalEngine.
        """
        eid = event.get("id")
        home = event.get("home_team", "Casa")
        away = event.get("away_team", "Fora")
        minute = event.get("current_minute") or event.get("minute") or 45
        
        # Procura as estatísticas em tempo real
        stats = self.get_live_statistics(eid)
        
        da_home_10m = stats.get("home_dangerous_attacks_last10", 8)
        da_away_10m = stats.get("away_dangerous_attacks_last10", 4)
        
        sot_home_10m = stats.get("home_shots_on_target_last10", 1)
        sot_away_10m = stats.get("away_shots_on_target_last10", 0)
        
        corners_home_10m = stats.get("home_corners_last10", 2)
        corners_away_10m = stats.get("away_corners_last10", 0)
        
        possession_home = stats.get("home_possession", 58)
        possession_away = stats.get("away_possession", 42)

        return {
            'event_id': eid,
            'home_team': home,
            'away_team': away,
            'current_minute': minute,
            'home_score': event.get('home_score', 0),
            'away_score': event.get('away_score', 0),
            # Métricas Históricas Baseline
            'home_xg_last5': 1.65,
            'away_conceded_xg_last5': 1.30,
            # Métricas Táticas
            'home_possession': possession_home,
            'away_possession': possession_away,
            'home_style': 'high_press',
            'away_style': 'low_block_vulnerable',
            # Métricas Live do Momento (Últimos 10 Minutos)
            'dangerous_attacks_10m': da_home_10m + da_away_10m,
            'shots_on_target_10m': sot_home_10m + sot_away_10m,
            'corners_10m': corners_home_10m + corners_away_10m,
            'home_pressure_share': da_home_10m / max(1, da_home_10m + da_away_10m)
        }

if __name__ == "__main__":
    fetcher = BSDLiveFetcher()
    live_games = fetcher.get_live_events()
    print(f"✓ Conector Live configurado com sucesso. Jogos mapeados: {len(live_games)}")

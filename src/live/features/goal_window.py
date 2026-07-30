from dataclasses import dataclass
from src.models.live_state import LiveMatchState

@dataclass
class GoalWindowResult:
    predicted_window: str
    intensity: str
    confidence_pct: float

class GoalWindowPredictor:
    def predict_window(self, match: LiveMatchState, pressure_index: float) -> GoalWindowResult:
        current_minute = match.minute
        
        if current_minute >= 85:
            return GoalWindowResult("85' - 90+'", "⚠️ Alta Urgência / Desespero", 60.0)

        # Se a pressão estiver muito alta, o golo é iminente (próximos 6 a 12 minutos)
        if pressure_index >= 60.0:
            start = current_minute + 2
            end = min(current_minute + 10, 90)
            return GoalWindowResult(f"{start}' - {end}'", "🔥 Pressão Sufocante", 85.0)
        elif pressure_index >= 40.0:
            start = current_minute + 5
            end = min(current_minute + 15, 90)
            return GoalWindowResult(f"{start}' - {end}'", "⚡ Pressão Moderada", 65.0)
        else:
            return GoalWindowResult("Sem Janela Detetada", "❄️ Jogo Morno", 20.0)

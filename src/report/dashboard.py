import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.models.live_state import LiveMatchState
from src.engine.live_pipeline import LivePipeline
from src.engine.simulation import MonteCarloSimulator
from src.engine.decision import DecisionEngine
from src.live.features.goal_window import GoalWindowPredictor
from src.model.ml_predictor import LiveMLPredictor

console = Console()

def render_live_dashboard(home_team: str, away_team: str, score: str, match_state: LiveMatchState, bookie_over15_odd: float):
    # Motores
    pipeline = LivePipeline()
    sim_engine = MonteCarloSimulator(n_simulations=1000)
    dec_engine = DecisionEngine()
    window_predictor = GoalWindowPredictor()
    ml_predictor = LiveMLPredictor()

    # Cálculos
    analysis = pipeline.evaluate(match_state)
    live_analysis = analysis["live"]
    pipeline_sim = analysis["simulation"]
    h_score, a_score = map(int, score.split("-"))
    sim_res = sim_engine.run_match_simulation(
        current_minute=match_state.minute,
        current_home_score=h_score,
        current_away_score=a_score,
        home_lambda=1.6,
        away_lambda=1.1
    )
    bet_rec = dec_engine.evaluate_bet("Over 1.5", sim_res.over_15_prob, bookie_over15_odd)
    goal_window = window_predictor.predict_window(match_state, live_analysis['pressure'])
    ml_res = ml_predictor.predict(match_state)

    # Tabela Live Match Engine
    metrics_table = Table(box=box.SIMPLE, show_header=False, expand=True)
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", style="bold yellow")
    
    metrics_table.add_row("⏱️ Minuto", f"{match_state.minute}'")
    metrics_table.add_row("🔥 Pressure Index", f"{live_analysis['pressure']} / 100")
    metrics_table.add_row("👑 Dominance Index", f"{live_analysis['dominance_index']} / 100")
    metrics_table.add_row("⚽ Estimated xG (10m)", f"{live_analysis['estimated_xg_10m']}")
    metrics_table.add_row("🤖 XGBoost Goal Prob", f"[bold green]{ml_res.goal_probability}%[/bold green] ({ml_res.model_used})")
    metrics_table.add_row("⏱️ GOAL WINDOW AI", f"[bold green]{goal_window.predicted_window}[/bold green] ({goal_window.intensity})")

    metrics_table.add_row(
        "🎯 Next Goal Probability",
        f"[bold green]{live_analysis['next_goal_probability']}%[/bold green]"
    )

    metrics_table.add_row(
        "🤖 Live Recommendation",
        f"[bold yellow]{live_analysis['recommendation']}[/bold yellow]"
    )

    # Tabela Simulação
    sim_table = Table(box=box.SIMPLE, show_header=False, expand=True)
    sim_table.add_column("Market", style="bold magenta")
    sim_table.add_column("Prob", style="bold white")
    
    sim_table.add_row("Over 1.5 Prob", f"{sim_res.over_15_prob}%")
    sim_table.add_row("Over 2.5 Prob", f"{sim_res.over_25_prob}%")
    sim_table.add_row("BTTS Prob", f"{sim_res.btts_prob}%")
    sim_table.add_row(
        "Expected Final xG",
        f"{pipeline_sim['expected_home_goals']} - {pipeline_sim['expected_away_goals']}"
    )

    sim_table.add_row(
        "Pipeline Over 1.5",
        f"{pipeline_sim['over_15']}%"
    )

    sim_table.add_row(
        "Pipeline Over 2.5",
        f"{pipeline_sim['over_25']}%"
    )

    sim_table.add_row(
        "Pipeline BTTS",
        f"{pipeline_sim['btts']}%"
    )

    # Tabela Decisão
    dec_table = Table(box=box.SIMPLE, show_header=False, expand=True)
    dec_table.add_column("Parameter", style="bold green")
    dec_table.add_column("Value", style="bold white")

    dec_table.add_row("Market Evaluated", bet_rec.market)
    dec_table.add_row("Bookie Odd", str(bet_rec.bookie_odd))
    dec_table.add_row("Model Edge", f"{bet_rec.edge_pct}%")
    dec_table.add_row("Kelly Stake", f"{bet_rec.kelly_stake_pct}%")
    dec_table.add_row("Action", f"[bold green]{bet_rec.action}[/bold green]" if "BET" in bet_rec.action else f"[bold red]{bet_rec.action}[/bold red]")

    # Layout
    console.print(Panel(f"[bold white]⚽ {home_team} {score} {away_team}[/bold white] | [italic yellow]Football Edge Engine Live v3.0 (ML Powered)[/italic yellow]", style="bold blue", box=box.DOUBLE))
    console.print(Panel(metrics_table, title="[bold cyan]1. Live Match Engine & ML Inference (v3.0)[/bold cyan]", box=box.ROUNDED))
    console.print(Panel(sim_table, title="[bold magenta]2. Monte Carlo Simulation Engine[/bold magenta]", box=box.ROUNDED))
    console.print(Panel(dec_table, title="[bold green]3. Decision Engine (Kelly & Edge)[/bold green]", box=box.ROUNDED))

if __name__ == "__main__":
    state = LiveMatchState(
        minute=75,
        possession=60.0,
        dangerous_attacks_10m=12,
        shots_on_target_10m=4,
        shots_10m=8,
        corners_10m=3,
        previous_pressure=55.0
    )
    render_live_dashboard("Benfica", "Porto", "1-0", state, bookie_over15_odd=2.15)

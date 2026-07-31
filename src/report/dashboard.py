import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.models.live_state import LiveMatchState
from src.engine.simulation import MonteCarloSimulator
from src.engine.decision import DecisionEngine
from src.live.features.goal_window import GoalWindowPredictor
from src.model.ml_predictor import LiveMLPredictor
from src.engine.live_decision import evaluate_live_market

console = Console()


def render_live_dashboard(
    home_team: str,
    away_team: str,
    score: str,
    match_state: LiveMatchState,
    bookie_over15_odd: float,
    analysis: dict
):

    sim_engine = MonteCarloSimulator(n_simulations=1000)
    dec_engine = DecisionEngine()
    window_predictor = GoalWindowPredictor()
    ml_predictor = LiveMLPredictor()

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

    bet_rec = dec_engine.evaluate_bet(
        "Over 1.5",
        sim_res.over_15_prob,
        bookie_over15_odd
    )

    goal_window = window_predictor.predict_window(
        match_state,
        live_analysis["pressure"]
    )

    ml_res = ml_predictor.predict(match_state)

    live_bet = evaluate_live_market(
        probability_pct=live_analysis["next_goal_probability"],
        bookie_odd=bookie_over15_odd,
        market="NEXT GOAL"
    )

    metrics_table = Table(box=box.SIMPLE, show_header=False, expand=True)
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", style="bold yellow")

    metrics_table.add_row("⏱️ Minuto", f"{match_state.minute}'")
    metrics_table.add_row("🔥 Pressure", str(live_analysis["pressure"]))
    metrics_table.add_row("🎯 Dominance", str(live_analysis["dominance_index"]))
    metrics_table.add_row("⚽ xG 10m", str(live_analysis["estimated_xg_10m"]))
    metrics_table.add_row("🤖 ML Goal", f"{ml_res.goal_probability}%")
    metrics_table.add_row("🪟 Goal Window", goal_window.predicted_window)
    metrics_table.add_row("🎯 Goal Prob", f"{live_analysis['next_goal_probability']}%")
    metrics_table.add_row("💰 Edge", f"{live_bet.edge}%")
    metrics_table.add_row("🤖 Decision", live_bet.action)

    sim_table = Table(box=box.SIMPLE, show_header=False, expand=True)
    sim_table.add_column("Market")
    sim_table.add_column("Probability")

    sim_table.add_row("Over 1.5", f"{sim_res.over_15_prob}%")
    sim_table.add_row("Over 2.5", f"{sim_res.over_25_prob}%")
    sim_table.add_row("BTTS", f"{sim_res.btts_prob}%")

    dec_table = Table(box=box.SIMPLE, show_header=False, expand=True)
    dec_table.add_column("Metric")
    dec_table.add_column("Value")

    dec_table.add_row("Bookie Odd", str(bookie_over15_odd))
    dec_table.add_row("Edge", f"{bet_rec.edge_pct}%")
    dec_table.add_row("Kelly", f"{bet_rec.kelly_stake_pct}%")
    dec_table.add_row("Action", bet_rec.action)

    console.print(
        Panel(
            f"[bold white]{home_team} {score} {away_team}[/bold white]",
            title="Football Edge Engine v4",
            box=box.DOUBLE
        )
    )

    console.print(Panel(metrics_table, title="Live Engine"))
    console.print(Panel(sim_table, title="Monte Carlo"))
    console.print(Panel(dec_table, title="Decision Engine"))

from src.engine.live_pipeline import LivePipeline
from src.report.dashboard import render_live_dashboard


def run_live():

    MATCH_ID = 219488

    pipeline = LivePipeline()

    analysis = pipeline.evaluate(MATCH_ID)

    match_state = pipeline.match_provider.get_live_match(MATCH_ID)

    render_live_dashboard(
        home_team="Home",
        away_team="Away",
        score="0-0",
        match_state=match_state,
        bookie_over15_odd=analysis["odds"]["odds"]["over_15_goals"],
        analysis=analysis
    )

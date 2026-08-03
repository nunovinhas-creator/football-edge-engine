import argparse
import sys

from src.cli.live import run_live
from src.cli.predict import run_predict
from src.cli.train import run_train
from src.cli.dashboard import run_dashboard


COMMANDS = {
    "live": run_live,
    "predict": run_predict,
    "train": run_train,
    "dashboard": run_dashboard,
}


def build_parser():

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Football Edge Engine — unified command-line entrypoint.",
        epilog=(
            "Examples:\n"
            "  python main.py live        Run the live match analysis dashboard\n"
            "  python main.py predict     Scan upcoming matches for value bets\n"
            "  python main.py train       Train the live goal probability model\n"
            "  python main.py dashboard   Launch the Streamlit live dashboard\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{live,predict,train,dashboard}"
    )

    subparsers.add_parser(
        "live",
        help="Run the live match analysis dashboard",
        description=(
            "Evaluate a live match through the LivePipeline "
            "(providers + engine + simulation) and render the console dashboard."
        )
    )

    subparsers.add_parser(
        "predict",
        help="Scan upcoming matches for value bets",
        description=(
            "Collect upcoming events, compute edge/EV per market "
            "and print the ranked value bets."
        )
    )

    subparsers.add_parser(
        "train",
        help="Train the live goal probability model",
        description=(
            "Train and persist the XGBoost live-goal probability model "
            "used by the live dashboard."
        )
    )

    subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit live dashboard",
        description="Launch the Streamlit web dashboard (scripts/app.py)."
    )

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    try:
        COMMANDS[args.command]()
    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Erro ao executar '{args.command}': {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

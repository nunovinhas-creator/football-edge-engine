import argparse

from src.cli.live import run_live
from src.cli.predict import run_predict
from src.cli.train import run_train
from src.cli.dashboard import run_dashboard


def build_parser():

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Football Edge Engine"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("live", help="Run the live match analysis dashboard")
    subparsers.add_parser("predict", help="Scan upcoming matches for value bets")
    subparsers.add_parser("train", help="Train the live goal probability model")
    subparsers.add_parser("dashboard", help="Launch the Streamlit live dashboard")

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "live":
        run_live()
    elif args.command == "predict":
        run_predict()
    elif args.command == "train":
        run_train()
    elif args.command == "dashboard":
        run_dashboard()


if __name__ == "__main__":
    main()

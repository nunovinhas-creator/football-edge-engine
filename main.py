import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Football Edge Engine CLI")
    parser.add_argument("--mode", choices=["predict", "monitor", "backtest"], required=True,
                        help="Modo de execução do motor")
    
    args = parser.parse_args()

    if args.mode == "predict":
        from src.engine.predict_today import run_predictions
        run_predictions()
    elif args.mode == "monitor":
        from src.engine.live_monitor import start_monitoring
        start_monitoring()
    elif args.mode == "backtest":
        from research.backtest_engine import run_backtest
        run_backtest()

if __name__ == "__main__":
    main()

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "app.py"


def run_dashboard():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
        check=True
    )

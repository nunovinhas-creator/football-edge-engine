# ⚽ Football Edge Engine

## Overview

Football Edge Engine is a quantitative football analytics platform focused on live match analysis, expected value detection and probabilistic betting models.

The project combines:

- BSD Sports API
- Monte Carlo Simulation
- Poisson Models
- Kelly Criterion
- Expected Value Engine
- Live Goal Engine
- Streamlit Dashboard
- Telegram Alerts
- GitHub Automation

---

# Current Status

Version

v0.9-dev

Implemented

- BSD Sports API Integration
- Live Matches
- Live Odds
- Live Match Incidents
- Match Statistics Provider
- Feature Adapter
- Live Pipeline
- Monte Carlo Simulation
- Poisson Engine
- Kelly Criterion
- Expected Value Engine
- Goal Probability Engine
- Streamlit Dashboard
- Telegram Integration
- GitHub Actions

---

# Architecture

BSD Sports API

↓

Providers

↓

Feature Adapter

↓

Live Pipeline

↓

Monte Carlo

↓

Poisson

↓

Goal Engine

↓

Expected Value Engine

↓

Kelly Criterion

↓

Decision Engine

↓

Dashboard

↓

Telegram

↓

Logger

---

# Project Structure

src/

cli/

engine/

live/

models/

backtest/

api/

utils/

research/

docs/

scripts/

---

# Live Providers

- APIMatchProvider
- APIOddsProvider
- StatsProvider
- IncidentsProvider
- BSDFeatureAdapter

---

# Live Engine

Current live engine calculates:

- Goal Probability
- Pressure Score
- Dominance Index
- Expected Goals
- Kelly Stake
- Expected Value

---

# Dashboard

Streamlit dashboard displays:

- Live Matches
- Goal Probability
- Pressure
- Odds
- Expected Value
- Betting Signals

---

# Telegram

Supports:

- Live Alerts
- Goal Alerts
- High EV Alerts
- Daily Predictions

---

# BSD Sports API

Integrated endpoints

/api/v2/events/live/

/api/v2/events/{id}/

 /api/v2/events/{id}/odds/

/api/v2/events/{id}/stats/

/api/v2/events/{id}/incidents/

---

# Installation

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

---

# Environment

BSD_API_KEY=YOUR_KEY

---

# Run

A single entrypoint (`main.py`) dispatches to every mode via argparse:

```
python main.py live        # Live match analysis dashboard (console)
python main.py predict     # Scan upcoming matches for value bets
python main.py train       # Train the live goal probability model
python main.py dashboard   # Launch the Streamlit live dashboard
```

`dashboard` launches the Streamlit app (`scripts/app.py`) under the hood, equivalent to running `streamlit run scripts/app.py` directly.

---

# Roadmap

Remaining work

- Real Pressure Model
- Automatic Decision Engine
- Historical Database
- Backtesting Improvements
- Portfolio Manager

---

# License

MIT

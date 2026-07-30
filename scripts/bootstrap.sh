#!/bin/bash

echo "📁 Creating project structure..."

mkdir -p docs
mkdir -p src
mkdir -p tests
mkdir -p notebooks
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/cache

########################################
# 01_project_scope.md
########################################

cat > docs/01_project_scope.md << 'EOT'
# Football Edge Engine

## Project Scope

### Vision

Football Edge Engine is a quantitative research platform designed to identify positive expected value (+EV) betting opportunities.

The objective is not to predict football matches but to estimate true probabilities and compare them with market prices.

---

## Philosophy

- Football is a dynamic system.
- Markets are not perfectly efficient.
- Features are more valuable than algorithms.
- Every model must be explainable.
- Success is measured by long-term profitability.

---

## Initial Markets

- Corners
- Shots
- Cards

Future:

- Goals
- BTTS
- Live Markets

---

## Data Source

- Bzzoiro Sports API

---

## Success Metrics

- ROI
- Yield
- CLV
- Expected Value
- Drawdown

Prediction accuracy is not the primary goal.

---

## Development Strategy

Each feature must have:

- Football hypothesis
- Mathematical definition
- Implementation
- Historical validation
EOT

########################################
# 02_architecture.md
########################################

cat > docs/02_architecture.md << 'EOT'
# System Architecture

## Philosophy

The Football Edge Engine is built around feature engineering.

Models may change.

Features remain.

---

## High-Level Architecture

Bzzoiro API

↓

Data Collector

↓

Raw Database

↓

Feature Store

↓

Market Models

↓

Edge Engine

↓

Backtesting

↓

Bet Recommendation

---

## Components

### Data Collector

Downloads and validates API data.

---

### Raw Database

Stores original API responses.

---

### Feature Store

Calculates football metrics.

Examples:

- Corner Pressure Index
- Territorial Dominance
- Transition Speed
- Referee Strictness
- Travel Fatigue

---

### Market Models

Independent models:

- Corners
- Shots
- Cards

---

### Edge Engine

Compares model probabilities with bookmaker odds.

Calculates Expected Value.

---

### Backtesting

Evaluates:

- ROI
- Yield
- CLV
- Drawdown

---

## Principle

We do not predict football matches.

We identify betting value.
EOT

########################################
# README
########################################

cat > README.md << 'EOT'
# Football Edge Engine

Quantitative football betting research platform focused on:

- Feature Engineering
- Game State Modelling
- Expected Value Detection

Current Markets:

- Corners
- Shots
- Cards
EOT

echo "requests
pandas
numpy" > requirements.txt

echo "__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
.env" > .gitignore

echo "✅ Bootstrap complete!"

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

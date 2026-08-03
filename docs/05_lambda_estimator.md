# Pre-Match Lambda Estimator

**Files:** `src/engine/lambda_estimator.py` (new, primary estimator),
`src/engine/pregame_lambda.py` (unchanged — kept as the legacy heuristic and
as an internal building block, see below), `src/collector/client.py`
(integration point).

**Scope:** this document covers ONLY how `lambda_home`/`mu_away` are
computed before being passed into `dixon_coles_simulate_match()`
(`src/engine/dixon_coles.py`). The Dixon-Coles equations themselves
(`tau()`, matrix construction, normalization), the Monte Carlo engine, Kelly
staking, the Goal Engine and the backtest framework are **not** touched by
this change — see `docs/AUDIT_MATEMATICA.md` for those.

---

## 1. Football hypothesis

> *"A team's expected goals against a specific opponent are best predicted
> by how that fixture has actually played out historically — weighted more
> heavily by recent meetings than old ones — and should be pulled toward a
> league-average baseline when the head-to-head sample is too small to
> trust on its own."*

This follows directly from `docs/01_project_scope.md`'s principle #1
("football is a dynamic system, not a static average") and principle #3
("feature engineering is more important than complex algorithms"): the
improvement here is not a new goals model, it is using *more of the data
the project already has* about a specific fixture, with statistically
sound handling of small samples.

---

## 2. What data is actually available (and what is not)

Before designing the estimator, the existing data-collection layer was
audited end-to-end (collector, API schema, live-history database, research
datasets) to establish what is genuinely collected today, since the task
explicitly disallows inventing data. Summary:

| Requested feature | Available today? | Source |
|---|---|---|
| Head-to-head aggregate (`avg_total_goals`, `home_win_rate`, `away_win_rate`, `total_matches`) | **Yes** | `event["head_to_head"]`, `src/collector/client.py` (already used by the old heuristic) |
| Head-to-head per-team goal totals (`home_goals`, `away_goals`) | **Yes, but unused until now** | Same `head_to_head` object — documented in `schema.yaml` (`head_to_head` field description) but never read by `pregame_lambda.py` or `predict_probability()` |
| Head-to-head individual recent matches (`recent_matches`) | **Yes, but unused until now** | Same `head_to_head` object, per `schema.yaml` — never read anywhere in the repo before this change |
| League average goals | **Only as a static constant** (`DEFAULT_AVG_TOTAL_GOALS = 2.5`) | No endpoint is called anywhere in the repo to fetch a real per-league average. `/api/v2/leagues/{id}/standings/` exists in `schema.yaml` but its v2 response schema is undocumented ("No response body") and no API key is available in this environment to inspect it live |
| Team attacking/defensive strength (season-long, not H2H-specific) | **No** | No endpoint call exists anywhere in the collector for team-level season stats. `TeamDetailV2Schema` (`/api/v2/teams/{id}/`) only exposes `id/name/short_name/country/venue_id` — no goals, no xG, no ELO for football teams (an `elo_rating` field exists in shared schema components but is not part of the football team schema actually returned) |
| xG / xGA (pre-match, per team) | **No** | The only xG-like fields in the repo (`home_xg_last5`, `away_conceded_xg_last5` in `src/models/live_state.py`) are **live in-game** defaults/estimates for the Goal Engine, unrelated to pre-match team season data, and are not fetched from any endpoint for the `predict` pipeline |
| Rolling last-N form (independent of H2H) | **No** | `research/pressure_shots` has rolling-5 features, but for a *shots* market, computed from a CSV/build pipeline not wired into `predict`; not team goals data |
| Recent-match weighting | **Yes, now used** | `head_to_head.recent_matches` (new) combined with `src/engine/decay.py::apply_exponential_decay` — an existing, already-implemented, previously **unused** utility (flagged in `docs/AUDIT_MATEMATICA.md` §12.2 as "ready but not wired to Dixon-Coles") |
| Home advantage | **Yes, as a constant** | Same fixed assumption already used by `pregame_lambda.py` (`HOME_ADVANTAGE_SHARE`), reused as-is |

**Conclusion:** the only genuinely available, match-specific pre-game
signal is the `head_to_head` object returned per event — but the project
was only using one field of it (`avg_total_goals`) plus the two win rates.
This estimator uses the *entire* object (per-team goal totals, individual
recent matches) instead of inventing new data sources. Team-level
season-wide attack/defense/xG splits, requested in the task description,
are **not implemented** because no code path in this repository collects
that data today — see "Limitations" below for exactly what would be needed.

---

## 3. Methodology

### 3.1 Tiered data selection

For each match, the richest available signal is used, falling back
gracefully:

```
Tier A: head_to_head.recent_matches present, >=2 usable entries
        -> recency-weighted average goals per team (apply_exponential_decay)

Tier B: head_to_head.total_matches, home_goals, away_goals present
        -> direct empirical average goals per team (home_goals/total_matches)

Tier C/D: only head_to_head.avg_total_goals (or nothing at all)
        -> delegates to the existing pregame_lambda.py::estimate_pregame_lambdas()
           (home-advantage + win-rate tilt split of the aggregate total)
```

Tiers A and B are genuinely new: they use per-team goal data that was
already present in the API response but previously discarded. Tier C/D
intentionally **reuses** the existing, tested heuristic instead of
re-implementing the same win-rate-tilt logic a second time (`No duplicated
logic`).

### 3.2 Empirical-Bayes shrinkage toward a league prior

Whatever raw (home_avg, away_avg, sample_size) tier produced, the estimate
is pulled toward a league-average prior in proportion to how much data
backs it:

```
shrunk = (raw_value * n + prior * K) / (n + K)
```

- `n = 0` → shrunk equals the prior exactly (no data, trust the league average).
- `n → ∞` → shrunk approaches the raw observed value (enough data to trust it fully).
- `n == K` → prior and observed data are weighted equally.

`K = SHRINKAGE_K = 4.0` (four "pseudo-matches" of prior weight). The prior
itself (`LEAGUE_PRIOR_HOME_GOALS`/`LEAGUE_PRIOR_AWAY_GOALS`) is **not** a
new invented number — it is the existing `DEFAULT_AVG_TOTAL_GOALS` (2.5)
split by the existing `HOME_ADVANTAGE_SHARE` (0.06), the same constants
`pregame_lambda.py` already used.

This is the standard statistical technique for estimating a rate from a
small sample (e.g. batting averages, click-through rates): the maximum
likelihood estimate from 1-2 observations is not trustworthy on its own,
and shrinking it toward a population-level prior reduces expected error —
this is exactly what was missing from the original heuristic, which used
`total_matches` only as an unused/ignored variable rather than as a
confidence signal.

### 3.3 Effective sample size for recency-weighted data (Tier A)

Exponentially-decayed weights do not use all `n` matches with equal
information — a strongly recency-weighted average of 50 matches carries far
less "effective" information than a plain average of 50 matches. The
design effect formula is used to compute this:

```
n_eff = 1 / sum(normalized_weight_i^2)
```

For the decay rate used here (`RECENT_MATCH_DECAY_RATE = 0.35`), `n_eff`
saturates around ~5.8 no matter how large the raw match count is. `n_eff`
(not the raw count) is what feeds the shrinkage step above — this was a
correction made *during* validation (see below): without it, the benchmark
showed the new estimator's mean-squared error getting *worse* than the old
heuristic for large synthetic samples, because the recency weighting was
being trusted as if it had as much information as a plain average.

### 3.4 Home advantage

Tiers A and B derive the home/away split directly from real head-to-head
goal counts, which already embed home advantage implicitly (one team really
was playing at home in each of those matches) — no separate home-advantage
term is applied on top, to avoid double-counting the same signal. Tier C/D
(no per-team split available) still applies the existing fixed home-advantage
share, exactly as before, via the reused legacy function.

### 3.5 Bounds

- `MIN_LAMBDA = 0.35` (unchanged, from `pregame_lambda.py`) — a Poisson with
  λ≈0 degenerates to "0 goals almost certainly", which is never realistic.
- `MAX_LAMBDA = 6.0` (new) — `dixon_coles_simulate_match()` truncates its
  score matrix at `max_goals=8`; a λ far above that starts losing real
  probability mass outside the matrix. This only activates in pathological
  input scenarios (e.g. a single 9-0 head-to-head result), never in normal
  data.

---

## 4. Public API

```python
from src.engine.lambda_estimator import estimate_lambda, estimate_lambda_detailed

lambda_home, mu_away = estimate_lambda(h2h)          # drop-in for pregame_lambda.estimate_pregame_lambdas
detail = estimate_lambda_detailed(h2h)               # + tier used, raw values, effective sample size
```

Same contract as the function it replaces as the pipeline's default:
accepts `None`/`{}`/malformed input without raising, always returns two
floats `> 0`.

**Backward compatibility:** `src/engine/pregame_lambda.py::estimate_pregame_lambdas()`
is **unchanged** — same file, same behavior, same test coverage
(`tests/test_dixon_coles_pipeline.py`). It is still directly importable and
is now used two ways: (1) as the Tier C/D building block inside the new
estimator, and (2) as the collector's defensive fallback if the new
estimator ever raises an unexpected exception (it should not, by design,
but the fallback follows the same fail-safe pattern already used elsewhere
in this pipeline, e.g. `LivePipeline.calculate_dynamic_lambda()`).

`src/collector/client.py::EventCollector.get_matches()` now calls
`estimate_lambda()` first, falling back to `estimate_pregame_lambdas()` only
on exception:

```python
try:
    lambda_home, mu_away = estimate_lambda(h2h)
except Exception:
    lambda_home, mu_away = estimate_pregame_lambdas(h2h)
```

Nothing downstream changed: `Match.xg_home`/`xg_away`,
`dixon_coles_probabilities`, and `src/cli/predict.py` continue to consume
these two floats exactly as before — the only thing that changed is how
they are computed.

---

## 5. Assumptions

1. **`home_win_rate`/`away_win_rate`/`home_goals`/`away_goals` (and, by
   extension, `recent_matches` entries) are team-identity-oriented relative
   to the *upcoming* fixture**, not venue-of-that-specific-past-match. This
   mirrors the assumption already implicit in `predict_probability()` and
   the original `pregame_lambda.py` (both compare `home_rate` vs
   `away_rate` as if higher `home_rate` favors the *current* home team,
   regardless of where each historical meeting was actually played). If the
   real API instead orients these fields strictly by the venue of each past
   match, the interpretation of Tier A/B would need revisiting — this
   cannot be verified without a live API key.
2. **`recent_matches` ordering**: if entries carry a `date` field, they are
   sorted chronologically. Otherwise, the list is assumed to be returned
   most-recent-first (the common REST convention) and is reversed before
   applying decay. This is also unverified against a live response.
3. **The league prior is a single global constant**, not a per-league
   dynamic value — see Limitations.

---

## 6. Limitations

- **No team-level attack/defense/xG estimation.** A full Dixon-Coles-style
  MLE fit (attack/defense parameters per team over a full league season,
  with time decay) — the improvement already flagged as missing in
  `docs/AUDIT_MATEMATICA.md` §12.2/§13 — would require a dataset of every
  team's full match history across a competition, which no collector in
  this repository fetches today. Building that would mean adding a new data
  pipeline (e.g. `/api/v2/teams/{id}/fixtures/` for every team in a league,
  aggregated over a season with time-decay), which is out of scope for "the
  goal is not to modify the Dixon-Coles algorithm itself... only improve
  the lambda estimation stage" using *already collected* data.
- **League average is static**, not fetched per competition. The `standings`
  endpoint exists in the API schema but its v2 response is undocumented in
  `schema.yaml` and could not be verified without API credentials (none are
  configured in this environment). Using it without confirming its actual
  field names would risk silently mis-parsing live data.
- **Tier A's recency weighting trades consistency for responsiveness** —
  see §3.3/Validation: its effective sample size saturates around ~6
  "equivalent" matches, so it will not keep improving indefinitely as more
  head-to-head history accumulates. This is a deliberate football-motivated
  choice (react to squad/manager changes) rather than an oversight, but it
  means Tier A is not asymptotically optimal for a genuinely stationary
  fixture (see Validation, Section 2 of the benchmark).
- **Orientation assumptions (§5) are unverified** against a live API
  response, since no API key is available in this environment.

---

## 7. Validation

### 7.1 What was run

`scripts/benchmark_lambda_estimator.py` — deterministic, reproducible
(`numpy` seed 42), reuses existing formulas (`src/backtest/historical/statistics.py::brier_score/log_loss`)
rather than re-implementing them. Two sections:

**Section 1 — Scenario comparison** (illustrative, no ground truth):
representative H2H inputs run through both estimators side by side. Example
finding: with only 1 historical meeting (5-0 home win), the old heuristic
ignores the actual scoreline entirely (`avg_total_goals` was not supplied
in that scenario) and falls back to the pure league default (λ_home=1.40),
while the new estimator uses the real result but shrinks it substantially
(λ_home=2.12, not 5.0) rather than either extreme.

**Section 2 — Synthetic recovery simulation.** This is a **statistical
validation of the estimator's mechanics**, using a fully synthetic,
known-by-construction ground truth (`numpy.random.default_rng(42).poisson(...)`)
— **it is not a claim about real-world predictive performance**, and is
labeled as such directly in the script's output. Method: for several true
`(lambda_home, mu_away)` pairs, synthetic head-to-head samples of varying
size (`n = 1..50`) are drawn, both estimators are run on each sample, and
their output is scored two ways: (a) mean-squared error against the known
true λ, and (b) Brier score / log loss of the resulting Dixon-Coles 1X2
probability against **held-out** synthetic matches (drawn independently
from the same true λ, never used to fit the estimate).

Aggregate result over all scenarios/sample sizes (400 trials each, one
run's raw numbers — see the script for the full per-scenario table):

| Metric | Old heuristic | New estimator |
|---|---|---|
| MSE(λ) | 0.6550 | 0.3308 |
| Brier score | 0.2219 | 0.2203 |
| Log loss | 0.6351 | 0.6312 |

The improvement is concentrated where it matters most in practice: most
real head-to-head samples are small (`n = 1-5`, a handful of historical
meetings between two specific clubs), and that is exactly where shrinkage
has the largest effect — e.g. for the "balanced" true scenario at `n=1`,
MSE(λ) drops from 1.26 (old) to 0.11 (new).

An honest negative/neutral finding, kept in rather than hidden: for large
synthetic samples (`n >= 20`) of a stationary process, the new estimator's
MSE(λ) plateaus rather than continuing to shrink toward zero the way the
old heuristic's does — a direct, expected consequence of Tier A's bounded
effective sample size (§3.3/§6). Brier score and log loss (what actually
drives the Dixon-Coles probabilities used downstream) remain flat-to-better
than the old heuristic across virtually every scenario and sample size
tested, including the large-`n` cases.

### 7.2 Real-world backtest — not performed, and why

The task asked for Brier score, log loss, calibration, and ROI **on
historical backtests**, with an explicit instruction to document missing
data rather than fabricate results. This repository does not currently
have what that requires:

- `src/backtest/historical/` expects a dataset of `(date, competition,
  home_team, away_team, market, odd, model_prob, result)` — see
  `src/backtest/historical/dataset.py`. The only sample data committed is
  `examples/backtest/sample_real_games.csv`: **9 rows**, no two of which
  share a team pairing, and it carries no `head_to_head` snapshot at all
  (it stores a pre-computed `model_prob` from an unrelated source, not raw
  H2H inputs). It cannot be used to compare two *lambda estimators*, only
  to backtest an already-computed probability.
- `data/live_history.db` contains 1,625 **live in-game** snapshots
  (`match_snapshots` table) for the Goal Engine's next-goal-in-15-minutes
  model — pressure/xG/possession features mid-match, not pre-match H2H
  data, and not linked to final scores in a form usable here.
- No API credentials are configured in this environment (`BSD_API_KEY` /
  equivalents are all unset — confirmed via `src/config/settings.py`), so a
  fresh historical dataset could not be pulled from the live API either.

**What would be needed to run this properly:** a dataset of past matches
where, for each row, the `head_to_head` object is captured **as it would
have been available before that match was played** (not recomputed with
hindsight — that would leak the outcome being predicted into the H2H
aggregate), together with the actual final score and the pre-match market
odds. Concretely: `date, competition, home_team, away_team, head_to_head_snapshot (json), home_goals, away_goals, odd_home, odd_draw, odd_away`.
With that, both estimators could be fed the identical snapshot, and Brier
score / log loss / calibration (`src/backtest/historical/statistics.py`,
already implemented, reused directly) and ROI (`src/backtest/historical/metrics.py::roi`,
already implemented) could be computed and compared without writing any new
metric code. This is recorded here as a concrete, actionable requirement
for whoever owns data collection next, per the project's own principle
("every feature must satisfy... historical validation" —
`docs/01_project_scope.md`).

---

## 8. Test coverage

- `tests/test_lambda_estimator.py` — unit tests per tier (`_split_from_*`),
  the shrinkage function, the effective-sample-size helper, and
  `estimate_lambda`/`estimate_lambda_detailed` end to end (never raises,
  bounds respected, tier selection priority, integration with
  `dixon_coles_simulate_match`/`estimate_pregame_probabilities`).
- `tests/test_dixon_coles_pipeline.py` — updated in one place (the
  collector integration test's exact expected λ values, since the
  collector now uses the new estimator by default — the change is
  documented inline in the test); all other tests in that file, including
  full coverage of the untouched `pregame_lambda.py` heuristic and of
  `dixon_coles.py` itself, are unmodified and still pass.

# Daily Volatility Forecasting Tracker — Phase 1

## Research question

Can a simple HAR-RV regression forecast next-day S&P 500 variance better than naive
benchmarks, when evaluated honestly out of sample with a walk-forward procedure and
no look-ahead?

Phase 1 builds the data pipeline, the baselines and the evaluation. Later phases
(not built yet) add VIX and economic-calendar features, a live daily logger, and
news sentiment.

## Data

- Daily OHLC for the S&P 500 index (`^GSPC`) from 2000-01-03 to 2026-11-04 (6,751 trading days), downloaded free with `yfinance` (no API keys) and cached in `data/raw/gspc.csv`.
- Validation: no duplicate dates, sorted index, no missing High/Low/Close, High >= Low. No rows were dropped.

## Method

**Variance proxy (Parkinson).** `RV_t = (ln(High_t / Low_t))^2 / (4 ln 2)`. A range-based estimate uses the whole day's price path, so it is much less noisy than a squared close-to-close return. It is still only a proxy for true realized variance (see Limitations).

**HAR features** (all known at the end of day t):
`rv_d` = RV_t, `rv_w` = mean RV over the last 5 days, `rv_m` = mean RV over the last 22 days. Target = RV_{t+1}.

**Models** (one-step-ahead):
1. Naive: tomorrow = today (`rv_d`)
2. Rolling mean: tomorrow = `rv_m`
3. HAR-RV (levels): OLS of target on `rv_d`, `rv_w`, `rv_m`; forecasts floored at 1e-8 (the floor was never needed: 0 of 5,729 forecasts)
4. HAR-RV (log): OLS in logs, back-transformed with `exp(pred + 0.5 * residual_variance)`

**Walk-forward setup.** Expanding training window, at least 1,000 trading days before the first forecast, regressions refit every 21 trading days. The model sees only data up to day t to forecast day t+1; the training set at origin t excludes row t because its target (RV_{t+1}) is not yet known. This is checked by automated tests (`tests/`).

**Scoring.** MSE on variance, and QLIKE = actual/forecast − ln(actual/forecast) − 1 (lower is better for both). Because 1,000 days of history are required first, the forecasts start on 2004-01-28, so the "2000-2007" sub-period actually covers 2004-2007 (989 forecast days).

**Significance.** Diebold-Mariano test on daily QLIKE loss differences, using Newey-West (HAC) standard errors with 9 lags.

## Results

Out-of-sample forecasts: 5,729 days (2004-01-28 to 2026-11-03).

### Overall

| Model | MSE | QLIKE |
|---|---|---|
| Naive | 3.511e-08 | 0.7667 |
| Rolling mean | 3.507e-08 | 0.4756 |
| HAR-RV (levels) | 2.702e-08 | 0.3990 |
| HAR-RV (log) | 2.759e-08 | 0.3961 |

### QLIKE by sub-period

| Model | 2000-2007* | 2008-2009 | 2010-2019 | 2020-present |
|---|---|---|---|---|
| Naive | 0.7868 | 0.6895 | 0.7943 | 0.7373 |
| Rolling mean | 0.3743 | 0.3737 | 0.5144 | 0.5071 |
| HAR-RV (levels) | 0.3865 | 0.3252 | 0.4246 | 0.3904 |
| HAR-RV (log) | 0.3710 | 0.3215 | 0.4173 | 0.4013 |

\* effectively 2004-2007. In this sub-period the Rolling mean (0.3743) beat HAR-RV (levels) (0.3865); only HAR-RV (log) beat it. Full MSE-by-period numbers are in `outputs/results_by_period.csv`.

### Diebold-Mariano tests (QLIKE; negative = model A is better)

| Model A | Model B | Mean loss diff | DM stat | p-value |
|---|---|---|---|---|
| HAR-RV (levels) | Naive | -0.3677 | -17.54 | 7.3e-69 |
| HAR-RV (levels) | Rolling mean | -0.0766 | -6.27 | 3.7e-10 |
| HAR-RV (log) | Naive | -0.3706 | -18.43 | 7.9e-76 |
| HAR-RV (log) | Rolling mean | -0.0795 | -8.01 | 1.2e-15 |

Both HAR variants beat both baselines on average QLIKE, and the differences are statistically significant. The MSE gap between the Rolling mean and the Naive model is tiny (3.507e-08 vs 3.511e-08) even though their QLIKE scores differ a lot; MSE is dominated by a few extreme-volatility days, which is one reason QLIKE is preferred.

### Charts

![HAR-RV forecast vs actual, last 2 years](outputs/forecast_vs_actual.png)

![QLIKE by sub-period](outputs/qlike_by_period.png)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_phase1           # add --refresh to re-download the data
python -m pytest                   # run the tests
```

Outputs are written to `outputs/`.

## Limitations

- **The Parkinson estimator is a proxy, not true realized variance.** True realized variance needs intraday (e.g. 5-minute) returns, which are not used here. Parkinson also ignores overnight gaps, so it understates variance on gap days, and the scores are measured against this noisy proxy.
- **One index, one history.** Only the S&P 500 since 2000, which contains a small number of crisis episodes (2008, 2020, 2025). Results may not generalize to other assets or regimes.
- **Results depend on design choices:** the 1,000-day minimum window, the 21-day refit interval, and the 5/22-day HAR windows. They were fixed in advance and not tuned, but other choices could change the numbers.
- **The DM test is approximate.** HAR nests the Rolling-mean model with estimated parameters, so the Normal approximation is not exact; read the p-values as approximate.
- **Forecast accuracy only.** No trading, transaction costs or portfolio decisions are involved.

# Daily Volatility Forecasting Tracker (Phases 1 and 2)

## Research question

Can a simple HAR-RV regression forecast next-day S&P 500 variance better than naive
benchmarks, evaluated out of sample with a walk-forward procedure and no look-ahead?
And does adding information known at the close of day t (the VIX, and a flag for
scheduled Fed decision days) improve it?

Phase 1 built the data pipeline, the baselines and the evaluation. Phase 2 adds VIX and
an FOMC-tomorrow flag. Not built yet: a live daily logger and news sentiment.

## Data

- Daily OHLC for the S&P 500 index (`^GSPC`) from 2000-01-03 to 2026-11-04 (6,751 trading days) and the VIX index (`^VIX`), downloaded free with `yfinance` (no API keys) and cached in `data/raw/`.
- Validation: no duplicate dates, sorted index, no missing High/Low/Close, High >= Low. No rows were dropped. Every S&P trading day has a VIX close.
- Scheduled FOMC decision dates, 2000 to 2027 (223 dates), read from federalreserve.gov by `scripts/fetch_fomc_dates.py` and saved in `data/events/`. Conference calls, unscheduled meetings, notation votes and cancelled meetings are excluded because they were not known in advance.

## Method

**Variance proxy (Parkinson).** `RV_t = (ln(High_t / Low_t))^2 / (4 ln 2)`. A range-based estimate uses the whole day's price path, so it is much less noisy than a squared close-to-close return. It is still only a proxy for true realized variance (see Limitations).

**Features** (all known at the end of day t):
- `rv_d` = RV_t, `rv_w` = mean RV over the last 5 days, `rv_m` = mean RV over the last 22 days.
- `vix_var` = (VIX close / 100)^2 / 252, the VIX converted to a daily variance so it is in the same units as RV.
- `fomc_next` = 1 if the next trading day is a scheduled FOMC decision day. The calendar is published in advance, so this is known at the close of day t.
- Target = RV_{t+1}.

**Models** (one-step-ahead):
1. Naive: tomorrow = today (`rv_d`).
2. Rolling mean: tomorrow = `rv_m`.
3. HAR-RV (levels / log): regression on `rv_d`, `rv_w`, `rv_m`. The log version is back-transformed with `exp(pred + 0.5 * residual_variance)`.
4. HAR+VIX (levels / log): adds `vix_var`.
5. HAR+VIX+FOMC (levels / log): adds `fomc_next`. In the log models only the variance-type inputs are logged; the 0/1 flag is not.

Levels forecasts are floored at 1e-8 because a variance cannot be negative. The share of floored forecasts is reported for every levels model.

**Walk-forward setup.** Expanding training window, at least 1,000 trading days before the first forecast, regressions refit every 21 trading days. The model sees only data up to day t to forecast day t+1; the training set at origin t excludes row t because its target (RV_{t+1}) is not yet known. Automated tests check that changing future prices, future VIX values or future FOMC dates never changes a forecast made earlier.

**Scoring.** MSE on variance, and QLIKE = actual/forecast − ln(actual/forecast) − 1 (lower is better for both). Because 1,000 days of history are required first, the forecasts start on 2004-01-28, so the "2000-2007" sub-period actually covers 2004-2007 (989 forecast days).

**Significance.** Diebold-Mariano test on daily QLIKE loss differences, using Newey-West (HAC) standard errors with 9 lags.

## Results

Out-of-sample forecasts: 5,729 days (2004-01-28 to 2026-11-03).

### Overall

| Model | MSE | QLIKE | Forecasts floored |
|---|---|---|---|
| Naive | 3.511e-08 | 0.7667 | |
| Rolling mean | 3.507e-08 | 0.4756 | |
| HAR-RV (levels) | 2.702e-08 | 0.3990 | 0.0% |
| HAR-RV (log) | 2.759e-08 | 0.3961 | |
| HAR+VIX (levels) | 2.484e-08 | 76.5235 | 5.2% |
| HAR+VIX (log) | 2.463e-08 | 0.3613 | |
| HAR+VIX+FOMC (levels) | 2.482e-08 | 78.8294 | 5.4% |
| HAR+VIX+FOMC (log) | 2.462e-08 | 0.3574 | |

### QLIKE by sub-period

| Model | 2000-2007* | 2008-2009 | 2010-2019 | 2020-present |
|---|---|---|---|---|
| Naive | 0.7868 | 0.6895 | 0.7943 | 0.7373 |
| Rolling mean | 0.3743 | 0.3737 | 0.5144 | 0.5071 |
| HAR-RV (levels) | 0.3865 | 0.3252 | 0.4246 | 0.3904 |
| HAR-RV (log) | 0.3710 | 0.3215 | 0.4173 | 0.4013 |
| HAR+VIX (log) | 0.3429 | 0.3216 | 0.3746 | 0.3641 |
| HAR+VIX+FOMC (log) | 0.3418 | 0.3205 | 0.3722 | 0.3557 |
| HAR+VIX (levels) | 19.4601 | 0.3345 | 143.8474 | 33.1986 |
| HAR+VIX+FOMC (levels) | 18.4481 | 0.3343 | 151.3993 | 30.4126 |

\* effectively 2004-2007. MSE by period is in `outputs/results_by_period.csv`.

### Diebold-Mariano tests (QLIKE; negative = model A is better)

| Model A | Model B | Mean loss diff | DM stat | p-value |
|---|---|---|---|---|
| HAR-RV (levels) | Naive | -0.3677 | -17.54 | 7.3e-69 |
| HAR-RV (levels) | Rolling mean | -0.0766 | -6.27 | 3.7e-10 |
| HAR-RV (log) | Naive | -0.3706 | -18.43 | 7.9e-76 |
| HAR-RV (log) | Rolling mean | -0.0795 | -8.01 | 1.2e-15 |
| HAR+VIX (levels) | HAR-RV (levels) | 76.12 | 6.33 | 2.5e-10 |
| HAR+VIX (log) | HAR-RV (log) | -0.0348 | -7.21 | 5.6e-13 |
| HAR+VIX+FOMC (levels) | HAR+VIX (levels) | 2.306 | 1.23 | 0.219 |
| HAR+VIX+FOMC (log) | HAR+VIX (log) | -0.0038 | -3.02 | 0.0025 |

### What the results say

- **Phase 1:** both HAR variants beat both baselines on QLIKE, and the differences are statistically significant. The MSE gap between the Rolling mean and the Naive model is tiny even though their QLIKE scores differ a lot; MSE is dominated by a few extreme-volatility days, which is one reason QLIKE is preferred.
- **VIX helps in the log model:** HAR+VIX (log) improves QLIKE from 0.3961 to 0.3613 (DM statistic -7.2). In 2008-2009 it makes no difference (0.3216 vs 0.3215).
- **The FOMC flag helps a little:** QLIKE improves from 0.3613 to 0.3574, statistically significant but small. The estimated effect is that variance on a decision day is about 1.6 times higher, holding the other inputs fixed.
- **The levels models with VIX fail on QLIKE.** Their MSE is fine (2.48e-08, better than plain HAR), but 5% of their forecasts are floored at 1e-8, nearly all in calm markets (161 of 300 in 2017). In calm markets the fitted intercept is negative and the monthly term has a negative weight, so the linear forecast goes to zero or below, and QLIKE punishes near-zero forecasts very heavily. This is a weakness of the levels form with VIX, not a data error. I did not change the floor after seeing it, since that would be tuning on the results. The two models are left out of the charts and shown in the tables above.
- **In the VIX models the monthly realized term gets almost no weight** (coefficient 0.004 in the log version), because VIX already carries that information. See `outputs/coefficients_full_sample.csv`.

### Charts

![Actual vs HAR forecasts, without and with VIX](outputs/forecast_vs_actual.png)

![QLIKE by sub-period](outputs/qlike_by_period.png)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_phase1           # add --refresh to re-download the data
python -m pytest                   # run the tests
python scripts/fetch_fomc_dates.py # only to rebuild the FOMC date list (needs internet and curl)
```

Outputs are written to `outputs/`. `python -m src.run_phase1` also writes `outputs/sheet_payload.json`, the rows used to refresh the project's Google Sheet.

## Limitations

- **The Parkinson estimator is a proxy, not true realized variance.** True realized variance needs intraday (e.g. 5-minute) returns, which are not used here. Parkinson also ignores overnight gaps, so it understates variance on gap days, and the scores are measured against this noisy proxy.
- **One index, one history.** Only the S&P 500 since 2000, which contains a small number of crisis episodes (2008, 2020, 2025). Results may not generalize to other assets or regimes.
- **Results depend on design choices:** the 1,000-day minimum window, the 21-day refit interval, the 5/22-day HAR windows, and the 1e-8 floor. They were fixed in advance and not tuned, but other choices could change the numbers. The 1e-8 floor is the reason the levels models with VIX score so badly.
- **The DM tests are approximate.** HAR nests the Rolling mean, and each added input nests the model before it, so the Normal approximation is not exact. Several tests are run without a multiple-testing adjustment. The tests involving the levels VIX models are distorted by the floored forecasts.
- **The log VIX models' slopes sum to 1.17 (above 1).** The inputs are on different scales and are highly correlated, so this is not necessarily a problem, but I have not investigated whether it matters in extreme markets.
- **Only scheduled FOMC meetings are flagged.** Emergency meetings are not known in advance and are excluded. CPI and jobs-report dates are not used.
- **Forecast accuracy only.** No trading, transaction costs or portfolio decisions are involved.

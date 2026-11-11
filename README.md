# Daily Volatility Forecasting Tracker (Phases 1 to 6)

## Research question

Can a simple HAR-RV regression forecast next-day S&P 500 variance better than naive
benchmarks, evaluated out of sample with a walk-forward procedure and no look-ahead?
And does adding information known at the close of day t (the VIX, and a flag for
scheduled Fed decision days) improve it?

Phase 1 built the data pipeline, the baselines and the evaluation. Phase 2 adds VIX and
an FOMC-tomorrow flag. Not built yet: a live daily logger and news sentiment.

## Data

- Daily OHLC for the S&P 500 index (`^GSPC`) from 2000-01-03 to 2026-11-09 (6,754 trading days) and the VIX index (`^VIX`), downloaded free with `yfinance` (no API keys) and cached in `data/raw/`.
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

Out-of-sample forecasts: 5,732 days (2004-01-28 to 2026-11-06).

### Overall

| Model | MSE | QLIKE | Forecasts floored |
|---|---|---|---|
| Naive | 3.525e-08 | 0.7694 | |
| Rolling mean | 3.513e-08 | 0.4788 | |
| HAR-RV (levels) | 2.710e-08 | 0.4009 | 0.0% |
| HAR-RV (log) | 2.766e-08 | 0.3980 | |
| HAR+VIX (levels) | 2.492e-08 | 76.4858 | 5.2% |
| HAR+VIX (log) | 2.469e-08 | 0.3633 | |
| HAR+VIX+FOMC (levels) | 2.490e-08 | 78.7905 | 5.4% |
| HAR+VIX+FOMC (log) | 2.469e-08 | 0.3595 | |

### QLIKE by sub-period

| Model | 2000-2007* | 2008-2009 | 2010-2019 | 2020-present |
|---|---|---|---|---|
| Naive | 0.7868 | 0.6895 | 0.7943 | 0.7464 |
| Rolling mean | 0.3743 | 0.3737 | 0.5144 | 0.5177 |
| HAR-RV (levels) | 0.3865 | 0.3252 | 0.4246 | 0.3968 |
| HAR-RV (log) | 0.3710 | 0.3215 | 0.4173 | 0.4077 |
| HAR+VIX (log) | 0.3429 | 0.3216 | 0.3746 | 0.3707 |
| HAR+VIX+FOMC (log) | 0.3418 | 0.3205 | 0.3722 | 0.3624 |
| HAR+VIX (levels) | 19.4601 | 0.3345 | 143.8474 | 33.1485 |
| HAR+VIX+FOMC (levels) | 18.4481 | 0.3343 | 151.3993 | 30.3675 |

\* effectively 2004-2007. MSE by period is in `outputs/results_by_period.csv`.

### Diebold-Mariano tests (QLIKE; negative = model A is better)

| Model A | Model B | Mean loss diff | DM stat | p-value |
|---|---|---|---|---|
| HAR-RV (levels) | Naive | -0.3685 | -17.57 | 4.2e-69 |
| HAR-RV (levels) | Rolling mean | -0.0779 | -6.34 | 2.3e-10 |
| HAR-RV (log) | Naive | -0.3714 | -18.46 | 4.3e-76 |
| HAR-RV (log) | Rolling mean | -0.0808 | -8.08 | 6.5e-16 |
| HAR+VIX (levels) | HAR-RV (levels) | 76.08 | 6.33 | 2.5e-10 |
| HAR+VIX (log) | HAR-RV (log) | -0.0347 | -7.20 | 5.8e-13 |
| HAR+VIX+FOMC (levels) | HAR+VIX (levels) | 2.305 | 1.23 | 0.219 |
| HAR+VIX+FOMC (log) | HAR+VIX (log) | -0.0038 | -2.99 | 0.0028 |

### What the results say

- **Phase 1:** both HAR variants beat both baselines on QLIKE, and the differences are statistically significant. The MSE gap between the Rolling mean and the Naive model is tiny even though their QLIKE scores differ a lot; MSE is dominated by a few extreme-volatility days, which is one reason QLIKE is preferred.
- **VIX helps in the log model:** HAR+VIX (log) improves QLIKE from 0.3981 to 0.3633 (DM statistic -7.2). In 2008-2009 it makes no difference (0.3216 vs 0.3215).
- **The FOMC flag helps a little:** QLIKE improves from 0.3633 to 0.3595, statistically significant but small. The estimated effect is that variance on a decision day is about 1.6 times higher, holding the other inputs fixed.
- **The levels models with VIX fail on QLIKE.** Their MSE is fine (2.49e-08, better than plain HAR), but 5% of their forecasts are floored at 1e-8, nearly all in calm markets (161 of 300 in 2017). In calm markets the fitted intercept is negative and the monthly term has a negative weight, so the linear forecast goes to zero or below, and QLIKE punishes near-zero forecasts very heavily. This is a weakness of the levels form with VIX, not a data error. I did not change the floor after seeing it, since that would be tuning on the results. The two models are left out of the charts and shown in the tables above.
- **In the VIX models the monthly realized term gets almost no weight** (coefficient 0.005 in the log version), because VIX already carries that information. See `outputs/coefficients_full_sample.csv`.

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

## Phase 4: VIX term-structure slope

`python -m src.run_phase4` adds one input, `vix_slope = ln(VIX3M / VIX)`, known at the close. VIX3M starts 2006-07, so the forecasts start in 2010-07 and all models are scored on the same 4,111 rows. The candidate was fixed in advance, picked on 2010-2015 and checked on 2016 onward.

| Model | QLIKE (all) | 2010-2015 | 2016+ (holdout) |
|---|---|---|---|
| Naive | 0.7646 | 0.7739 | 0.7599 |
| HAR-RV (log) | 0.4055 | 0.3876 | 0.4146 |
| HAR+VIX+FOMC (log) | 0.3597 | 0.3409 | 0.3692 |
| HAR+VIX+FOMC+Slope (log) | **0.3460** | 0.3293 | 0.3544 |

The slope model beats HAR+VIX+FOMC with DM statistic -4.97 on all rows and -4.04 on the holdout alone. Tested and rejected (no holdout gain): overnight gap, daily VIX change, down-day flag, SKEW, VVIX, day of week, options-expiry flag, fitting directly on QLIKE. VIX9D slope helped slightly on a shorter sample (2015+) and was not added. CPI and jobs-report dates were not tested (no reliable date source loaded). The Google Sheet has a "Slope model" tab with these numbers; the daily refresh task runs `python -m src.run_phase4 --refresh` and updates that tab from `outputs/slope_sheet_payload.json`.

## Phase 5: volatility-managed SPY

`python -m src.volmanaged` tests whether the forecasts make money. Hold SPY with weight `min(1, c_t / sigma_t)`, where `sigma_t` is the volatility implied by each model's forecast and `c_t` is its expanding median; no leverage, the rest in T-bills (`^IRX`), 2 bps cost, order at the close. The rule was fixed before looking at results. All models use the same 4,091 days (2010-08 to 2026-11).

| | CAGR | Volatility | Sharpe | Max drawdown | Worst day |
|---|---|---|---|---|---|
| Buy & hold SPY | 14.3% | 16.9% | 0.78 | -33.7% | -10.9% |
| Managed, HAR-RV (log) | 11.6% | 11.8% | 0.87 | -18.1% | -4.2% |
| Managed, HAR+VIX+FOMC+Slope (log) | 10.7% | 11.4% | 0.81 | -20.0% | -4.3% |

- **Risk falls a lot:** volatility drops by about a third, and the maximum drawdown and worst day roughly halve, at the cost of some return.
- **Sharpe is not clearly better.** Differences vs buy & hold are +0.03 to +0.08 with 95% bootstrap intervals that include zero (`outputs/volmanaged_sharpe_diff_lag0.csv`). A one-day execution delay (`--lag 1`) gives the same picture.
- **Better QLIKE did not give better Sharpe.** The most accurate forecaster (slope) is not the best managed portfolio; plain HAR is. The differences between models are within noise, so QLIKE gains should not be assumed to translate into trading gains.
- **VXX/SVXY backtest (Phase 3) caveat:** the 10-year numbers mix regimes. VXX only exists from 2018-01 (earlier days use 2x UVXY returns), and SVXY was -1x until February 2018 and -0.5x after; its -83% day on 2018-02-06 is the strategy's worst day. Treat those results as unreliable.

## Phase 6: earning more than buy & hold SPY

`python -m src.higher_return --lag 1` tests three strategies fixed in advance (no fitted parameters), all with orders placed one trading day after the signal, which is the realistic case (VIX and VIX3M publish after most ETF order deadlines). Days: 2011-10 to 2026-11.

| | CAGR | Volatility | Sharpe | Max drawdown | Worst day |
|---|---|---|---|---|---|
| Buy & hold SPY | 15.3% | 16.7% | 0.84 | -33.7% | -10.9% |
| Managed SPY, dial k=1.5 (HAR-RV) | 17.4% | 19.3% | 0.84 | -27.6% | -7.5% |
| Managed SPY, dial k=1.25 (HAR-RV) | 14.8% | 16.6% | 0.81 | -23.4% | -6.3% |
| SVXY in contango, else cash | 11.6% | 31.0% | 0.46 | -58.3% | -21.4% |
| 50% SPY + 50% SVXY in contango | 14.3% | 21.8% | 0.65 | -37.9% | -12.2% |

- **Managed SPY with a risk dial is the one that works.** The Phase 5 weight is multiplied by k (k=1.5 means an average exposure of about 1.5x, capped at 2x; borrowing at T-bills + 0.5%). It earns more than SPY with a smaller drawdown. Sharpe is the same as SPY's, so this is more return for more volatility, not free money.
- **It holds up through 2008.** On a longer sample (2004-2026, plain HAR-RV forecasts) k=1.5 gives 13.2% a year against 10.6% for SPY, with a -39% max drawdown against -55%, and Sharpe 0.64 against 0.54 (`outputs/higher_return_long_sample_lag1.csv`).
- **The SVXY strategies do not work once orders are delayed a day.** With same-close execution (`--lag 0`, not realistic) B and C looked better than SPY; with a one-day delay both fall below it. The contango filter was on 93% of days, so B is close to just holding SVXY.
- **Assumptions to watch:** leverage costs more than T-bills + 0.5% in most retail accounts (futures or margin can be cheaper or dearer; a leveraged ETF adds its own decay). SVXY before 2018-02-28 is modeled as 0.5 x actual (a -0.5x approximation). The sample is dominated by a strong bull market. Results are history, not a promise.

## Limitations

- **The Parkinson estimator is a proxy, not true realized variance.** True realized variance needs intraday (e.g. 5-minute) returns, which are not used here. Parkinson also ignores overnight gaps, so it understates variance on gap days, and the scores are measured against this noisy proxy.
- **One index, one history.** Only the S&P 500 since 2000, which contains a small number of crisis episodes (2008, 2020, 2025). Results may not generalize to other assets or regimes.
- **Results depend on design choices:** the 1,000-day minimum window, the 21-day refit interval, the 5/22-day HAR windows, and the 1e-8 floor. They were fixed in advance and not tuned, but other choices could change the numbers. The 1e-8 floor is the reason the levels models with VIX score so badly.
- **The DM tests are approximate.** HAR nests the Rolling mean, and each added input nests the model before it, so the Normal approximation is not exact. Several tests are run without a multiple-testing adjustment. The tests involving the levels VIX models are distorted by the floored forecasts.
- **The log VIX models' slopes sum to 1.17 (above 1).** The inputs are on different scales and are highly correlated, so this is not necessarily a problem, but I have not investigated whether it matters in extreme markets.
- **Only scheduled FOMC meetings are flagged.** Emergency meetings are not known in advance and are excluded. CPI and jobs-report dates are not used.
- **Forecast accuracy only.** No trading, transaction costs or portfolio decisions are involved.

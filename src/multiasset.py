"""Phase 7: multi-asset risk parity with HAR-RV volatility forecasts. More return per unit of risk?

Assets: SPY (US stocks), IEF (7-10y Treasuries), GLD (gold). Fixed in advance, not chosen on results.

Each asset gets its own walk-forward HAR-RV (log) forecast of next-day variance from its own daily
High/Low range (Parkinson). MIN_TRAIN is 500 days (not the 1,000 used for the S&P index) so that the
test starts in late 2006 and includes 2008-09; HAR has only 4 coefficients, so 2 years is plenty.

Strategies (rules fixed in advance, nothing fitted to returns):
  1. Buy & hold SPY                         benchmark
  2. 60/40 SPY/IEF                          the standard "balanced" benchmark
  3. Equal weight SPY/IEF/GLD               naive diversification, no forecasts
  4. Risk parity                            w_i proportional to 1 / forecast vol_i, sums to 1
  5. Risk parity, managed                   4 scaled by min(1, typical / forecast portfolio vol);
                                            "typical" = expanding median of its own forecast vol
  6. Risk parity, levered to SPY risk       4 scaled to SPY's typical forecast vol, gross cap 2x,
                                            borrowing at T-bills + 0.5%
Forecast portfolio vol = sqrt(w' D C D w): D = forecast vols, C = 252-day correlation of daily
returns through day t (both known at the close of day t).

Constant-weight strategies (2, 3) are modelled as rebalanced daily at no cost (a small flattery).
Signals at close t apply to day t+1 (lag 0) or t+2 (lag 1, the headline).
Samples: All, 2008-2009, 2016+ and 2025+ (the untouched final check; only ~1.9 years, low power).

Usage: python -m src.multiasset --lag 1 [--refresh]   (lag 1 also writes the Sheet payload)
"""

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.data import load_data
from src.evaluate import walk_forward
from src.features import build_features
from src.higher_return import BORROW_SPREAD, LEVER_CAP, run
from src.models import HARLogModel
from src.volmanaged import OUT_DIR, RAW_DIR, load_inputs, sharpe_diff_ci, sigma_from_forecast, stats

ASSETS = ["SPY", "IEF", "GLD"]
COSTS = {"SPY": 2.0, "IEF": 2.0, "GLD": 3.0}   # bps per unit of turnover
MIN_TRAIN = 500
CORR_WINDOW, CORR_MIN = 252, 126
SAMPLES = [("All", slice(None)), ("2008-2009", slice("2008", "2009")),
           ("2016+", slice("2016", None)), ("2025+ (final check)", slice("2025", None))]
CI_SAMPLES = ["All", "2016+"]
SHEET_RANGE = "'Multi-asset'!A1"


def forecast_vols(refresh: bool = False) -> pd.DataFrame:
    """Annualized HAR-RV (log) forecast vol per asset, indexed by origin day t (forecast for t+1)."""
    out = {}
    for a in ASSETS:
        ohlc = load_data(refresh=refresh, verbose=False, ticker=a, cache_path=RAW_DIR / f"ohlc_{a.lower()}.csv")
        fc = walk_forward(build_features(ohlc), [HARLogModel(name=a)], min_train=MIN_TRAIN)[a]
        out[a] = sigma_from_forecast(fc)
    return pd.DataFrame(out).dropna()


def asset_returns(refresh: bool = False) -> pd.DataFrame:
    from src.backtest import load_yahoo_close
    return pd.DataFrame({a: load_yahoo_close(a, refresh).pct_change(fill_method=None) for a in ASSETS})


def inverse_vol_weights(sigma: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / sigma
    return inv.div(inv.sum(axis=1), axis=0)


def rolling_corr(rets: pd.DataFrame, window: int = CORR_WINDOW, min_periods: int = CORR_MIN) -> pd.Series:
    """MultiIndex (date, asset) x asset correlation matrices using returns through each date."""
    return rets.rolling(window, min_periods=min_periods).corr()


def portfolio_vol(w: pd.DataFrame, sigma: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """sqrt(w' D C D w) per day; days without a full correlation matrix are dropped."""
    out = {}
    cols = list(w.columns)
    for t in w.index:
        if t not in corr.index.get_level_values(0):
            continue
        c = corr.loc[t].loc[cols, cols].to_numpy()
        if np.isnan(c).any():
            continue
        x = w.loc[t].to_numpy() * sigma.loc[t, cols].to_numpy()
        out[t] = float(np.sqrt(x @ c @ x))
    return pd.Series(out)


def build_weights(sigma: pd.DataFrame, rets: pd.DataFrame) -> dict:
    """{strategy: DataFrame of weights per asset}, all on the same days (set at the close of each day)."""
    rp = inverse_vol_weights(sigma)
    pvol = portfolio_vol(rp, sigma, rolling_corr(rets))
    days = pvol.index
    rp = rp.loc[days]
    typical_own = pvol.expanding(min_periods=20).median()
    typical_spy = sigma["SPY"].loc[days].expanding(min_periods=20).median()
    managed = (typical_own / pvol).clip(upper=1.0)
    levered = (typical_spy / pvol).clip(upper=LEVER_CAP)
    const = lambda d: pd.DataFrame({a: d.get(a, 0.0) for a in ASSETS}, index=days)
    w = {"Buy & hold SPY": const({"SPY": 1.0}),
         "60/40 SPY/IEF": const({"SPY": 0.6, "IEF": 0.4}),
         "Equal weight SPY/IEF/GLD": const({a: 1 / 3 for a in ASSETS}),
         "Risk parity": rp,
         "Risk parity, managed": rp.mul(managed, axis=0),
         "Risk parity, levered to SPY risk": rp.mul(levered, axis=0)}
    return {k: v.dropna() for k, v in w.items()}


COMPARISONS = [("Equal weight SPY/IEF/GLD", "Buy & hold SPY"), ("Risk parity", "Buy & hold SPY"),
               ("Risk parity, managed", "Buy & hold SPY"), ("Risk parity, levered to SPY risk", "Buy & hold SPY"),
               ("Risk parity", "Equal weight SPY/IEF/GLD"), ("Risk parity, managed", "Risk parity"),
               ("Risk parity", "60/40 SPY/IEF")]


def build_sheet_rows(res, ci, corr_avg, first, last, n) -> list:
    blank = [""]
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "WorstDay", "Calmar", "avg_weight"]
    head = ["Strategy", "CAGR", "Volatility", "Sharpe (excess of cash)", "Max drawdown", "Worst day",
            "Calmar", "Average gross exposure"]
    rows = [["Phase 7: multi-asset risk parity (SPY, IEF, GLD) with HAR-RV forecasts"],
            [f"Orders one trading day after the signal. Same days for every row: {first} to {last} ({n:,} days). "
             "Rules fixed in advance. Costs 2 bps SPY/IEF, 3 bps GLD per unit of turnover; borrowing at T-bills "
             "+ 0.5%. Risk parity: weight proportional to 1 / forecast vol. 2025+ is the untouched final check."]]
    for label, _ in SAMPLES:
        rows += [blank, [label], head]
        for (m, s), r in res.iterrows():
            if s == label:
                rows.append([m] + [round(float(r[c]), 4) for c in cols])
    rows += [blank, ["Sharpe difference A - B (block bootstrap, 95% interval)"],
             ["A", "B", "Sample", "Sharpe difference", "Low 95%", "High 95%"]]
    for r in ci.itertuples(index=False):
        rows.append([r.A, r.B, r.sample, round(r.sharpe_diff, 3), round(r.lo95, 3), round(r.hi95, 3)])
    rows += [blank, ["Average 252-day correlation of daily returns"], ["Pair", "Correlation"]]
    rows += [[p, round(v, 3)] for p, v in corr_avg.items()]
    rows += [blank,
             ["Reading it (as of the first run): diversifying did the work. Plain equal weight had the best Sharpe "
              "(about 0.77 vs 0.54 for SPY) and much smaller drawdowns, though the bootstrap interval still includes "
              "zero. Weighting by HAR-RV forecasts (risk parity) did not beat equal weight; it held about 57% bonds on "
              "average, and bonds earned little after 2020. Vol-timing the whole portfolio added almost nothing. The levered version sat at the "
              "2x cap about 90% of the time."],
             [f"Last refreshed {date.today().isoformat()}. Source: outputs/multiasset_*.csv "
                     "(python -m src.multiasset --lag 1)."]]
    return rows


def main(lag: int = 1, refresh: bool = False) -> None:
    _, _, cash = load_inputs()
    sigma, rets = forecast_vols(refresh), asset_returns(refresh)
    weights = build_weights(sigma, rets)
    rdict = {a: rets[a] for a in ASSETS}
    runs = {}
    for name, w in weights.items():
        costs = {a: 0.0 for a in ASSETS} if name in ("Buy & hold SPY", "60/40 SPY/IEF",
                                                       "Equal weight SPY/IEF/GLD") else COSTS
        runs[name] = run({a: w[a] for a in ASSETS}, rdict, cash, costs, lag, BORROW_SPREAD)
    idx = None
    for r in runs.values():
        idx = r.index if idx is None else idx.intersection(r.index)
    rows = []
    for name, r in runs.items():
        for label, sl in SAMPLES:
            rr = r.loc[idx].loc[sl]
            s = stats(rr["ret"], rr["cash"]); s.update(model=name, sample=label, avg_weight=float(rr["w"].mean()))
            rows.append(s)
    res = pd.DataFrame(rows).set_index(["model", "sample"])
    ci = []
    for a, b in COMPARISONS:
        for label, sl in SAMPLES:
            if label in CI_SAMPLES:
                ra, rb = runs[a].loc[idx].loc[sl], runs[b].loc[idx].loc[sl]
                ci.append((a, b, label) + sharpe_diff_ci(ra["ret"], rb["ret"], ra["cash"]))
    ci = pd.DataFrame(ci, columns=["A", "B", "sample", "sharpe_diff", "lo95", "hi95"])
    corr = rets.loc[idx].rolling(CORR_WINDOW).corr()
    corr_avg = {f"{x}-{y}": float(corr.xs(x, level=1)[y].mean())
                for i, x in enumerate(ASSETS) for y in ASSETS[i + 1:]}
    res.to_csv(OUT_DIR / f"multiasset_stats_lag{lag}.csv")
    ci.to_csv(OUT_DIR / f"multiasset_sharpe_diff_lag{lag}.csv", index=False)
    if lag == 1:
        wt = pd.concat({k: v.loc[idx] for k, v in weights.items()}, axis=1)
        wt.to_csv(OUT_DIR / "multiasset_weights.csv")
        sheet = build_sheet_rows(res, ci, corr_avg, idx.min().date(), idx.max().date(), len(idx))
        with open(OUT_DIR / "multiasset_sheet_payload.json", "w") as f:
            json.dump({"range": SHEET_RANGE, "data": sheet}, f)
    pd.set_option("display.width", 200)
    print(f"lag={lag}  {idx.min().date()} to {idx.max().date()} ({len(idx)} days)")
    print(res.round(3).to_string()); print(ci.round(3).to_string(index=False)); print(corr_avg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    main(a.lag, a.refresh)

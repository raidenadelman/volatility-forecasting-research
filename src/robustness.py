"""Phase 8: is the equal-weight SPY/IEF/GLD result robust? Realistic rebalancing, a trend filter,
sub-periods, and a Sharpe ratio deflated for how many strategies this project has tried.

Rules fixed in advance (written before running):
  * Portfolios now DRIFT with prices between rebalances, and every trade pays costs
    (2 bps SPY/IEF, 3 bps GLD per unit of turnover). Phase 7 assumed free daily rebalancing.
  * Rebalancing rules for equal weight (1/3 each):
        daily    back to 1/3 at every close
        monthly  back to 1/3 at the last close of each month
        band     back to 1/3 whenever any weight is more than 5 percentage points from 1/3
  * Trend filter (Faber 2007 style): at each month-end close, hold an asset's 1/3 only if its
    (dividend-adjusted) price is above its 200-day average; otherwise that third sits in T-bills.
  * Lag: a decision at the close of day t is traded at the close of day t+lag (lag 1 = headline).
  * Same days as Phase 7 (2007-01-17 onward). Fixed sub-periods: 2007-2012, 2013-2019, 2020+.
  * Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): the probability that the true Sharpe is
    above what the best of N useless strategies would show by luck. N counts every strategy variant
    this project has backtested (Phases 3, 5, 6, 7, 8); a conservative N = 50 is shown as well.

Usage: python -m src.robustness --lag 1   (lag 1 also writes the Sheet payload)
"""

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.multiasset import ASSETS, COSTS, asset_returns
from src.volmanaged import OUT_DIR, load_inputs, sharpe_diff_ci, stats

START = "2007-01-17"
BAND = 0.05
SMA_DAYS = 200
PERIODS = [("2007-2012", slice("2007", "2012")), ("2013-2019", slice("2013", "2019")),
           ("2020+", slice("2020", None))]
SHEET_RANGE = "'Robustness'!A1"
EULER = 0.5772156649


# ----------------------------------------------------------------------------
# Drifting-portfolio simulator
# ----------------------------------------------------------------------------
def month_end_flags(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month in `index`."""
    s = pd.Series(index.month, index=index)
    return s.ne(s.shift(-1)).fillna(True)


def simulate(target: pd.DataFrame, rets: pd.DataFrame, cash: pd.Series, costs: dict, lag: int = 1,
             rule: str = "daily", band: float = BAND) -> pd.DataFrame:
    """Daily returns of a portfolio whose weights drift with prices.

    target.loc[t] = desired weights decided at the close of day t (gross <= 1; the rest is cash).
    rule 'daily': trade every day; 'monthly': trade on month-end decisions (and whenever the
    target changes); 'band': trade when any drifted weight is more than `band` from target (or
    the target changes). A decision taken at the close of day t is executed at the close of day
    t + lag, at the target decided on day t. The first day starts fully invested at target.
    """
    days = target.index.intersection(rets.dropna().index)
    target, rets = target.loc[days, ASSETS], rets.loc[days, ASSETS]
    cash = cash.reindex(days).fillna(0.0)
    month_end = month_end_flags(days)
    c = np.array([costs[a] for a in ASSETS]) / 1e4
    tgt, r, cs = target.to_numpy(), rets.to_numpy(), cash.to_numpy()
    w = tgt[0].copy()
    pending = {}        # execution day index -> target vector
    out_ret, out_turn = np.zeros(len(days)), np.zeros(len(days))
    for i in range(len(days)):
        if i > 0:       # earn day i's return on weights held from the previous close
            port = w @ r[i] + (1 - w.sum()) * cs[i]
            w = w * (1 + r[i]) / (1 + port)
            out_ret[i] = port
        # decision at close of day i, using drifted weights now
        changed = i > 0 and not np.allclose(tgt[i], tgt[i - 1])
        if rule == "daily":
            trade = True
        elif rule == "monthly":
            trade = bool(month_end.iloc[i]) or changed
        elif rule == "band":
            trade = bool(np.abs(w - tgt[i]).max() > band) or changed
        else:
            raise ValueError(rule)
        if trade and i > 0:
            pending[i + lag] = tgt[i].copy()
        if i in pending:  # execute an order decided `lag` days ago at today's close
            new = pending.pop(i)
            turn = np.abs(new - w)
            out_turn[i] = turn.sum()
            out_ret[i] -= turn @ c
            w = new
    out = pd.DataFrame({"ret": out_ret, "cash": cs, "turnover": out_turn}, index=days)
    return out.iloc[1:]


def trend_target(prices: pd.DataFrame, sma_days: int = SMA_DAYS) -> pd.DataFrame:
    """1/3 per asset if its price is above its 200-day average at the latest month-end close, else 0.
    Uses closes up to and including each month-end; held unchanged until the next month end."""
    above = (prices > prices.rolling(sma_days).mean()).astype(float).where(prices.rolling(sma_days).count() == sma_days)
    me = month_end_flags(prices.index)
    sig = above.where(me).ffill()
    return (sig / len(ASSETS)).dropna()


# ----------------------------------------------------------------------------
# Deflated Sharpe ratio
# ----------------------------------------------------------------------------
def expected_max_sharpe(sr_var: float, n_trials: int) -> float:
    """Expected maximum of n_trials Sharpe ratios (true Sharpe 0) with variance sr_var (same units)."""
    return np.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                              + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(excess: pd.Series, sr_var_daily: float, n_trials: int) -> float:
    """P(true Sharpe > expected best-of-N luck), adjusting for skew and fat tails. Daily units inside."""
    x = excess.dropna()
    sr = x.mean() / x.std(ddof=1)
    sr0 = expected_max_sharpe(sr_var_daily, n_trials)
    skew, kurt = x.skew(), x.kurt() + 3.0  # pandas kurt() is excess kurtosis
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / denom))


def trial_sharpes(new: dict) -> pd.Series:
    """Annualized Sharpe of every strategy variant backtested so far (lag 1, 'All' samples), excluding
    the buy-and-hold benchmarks. Samples differ by phase; this is only used for their spread."""
    vals = {}
    for f in ["volmanaged_stats_lag1.csv", "higher_return_stats_lag1.csv", "multiasset_stats_lag1.csv"]:
        d = pd.read_csv(OUT_DIR / f)
        d = d[(d["sample"] == "All") & ~d["model"].str.startswith("Buy & hold")]
        vals.update({f"{f.split('_stats')[0]}: {m}": s for m, s in zip(d["model"], d["Sharpe"])})
    bt = pd.read_csv(OUT_DIR / "backtest_stats_lag1_10y.csv", index_col=0)
    vals["phase3: VXX/SVXY signals"] = bt.loc["Strategy (signals)", "sharpe"]
    vals.update({f"robustness: {k}": v for k, v in new.items()})
    return pd.Series(vals)


# ----------------------------------------------------------------------------
def build_sheet_rows(res, ci, dsr, n_trials, sr_sd, first, last, n) -> list:
    blank = [""]
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "WorstDay", "Calmar", "turnover"]
    head = ["Strategy", "CAGR", "Volatility", "Sharpe (excess of cash)", "Max drawdown", "Worst day",
            "Calmar", "Turnover per year"]
    rows = [["Phase 8: robustness of equal weight SPY/IEF/GLD"],
            [f"Weights drift with prices between rebalances; every trade pays 2 bps (SPY, IEF) or 3 bps (GLD). "
             f"Orders one trading day after the decision. Same days for every row: {first} to {last} ({n:,} days). "
             "Rules fixed before running. Trend filter: hold each third only if price > 200-day average at month end."]]
    for label in ["All"] + [p for p, _ in PERIODS] + ["2025+ (final check)"]:
        rows += [blank, [label], head]
        for (m, s), r in res.iterrows():
            if s == label:
                rows.append([m] + [round(float(r[c]), 4) for c in cols])
    rows += [blank, ["Sharpe difference vs buy & hold SPY (block bootstrap, 95% interval)"],
             ["Strategy", "Sample", "Sharpe difference", "Low 95%", "High 95%"]]
    for r in ci.itertuples(index=False):
        rows.append([r.model, r.sample, round(r.sharpe_diff, 3), round(r.lo95, 3), round(r.hi95, 3)])
    rows += [blank, [f"Deflated Sharpe ratio: probability the true Sharpe beats the luckiest of N useless strategies "
                     f"(spread of Sharpe across trials: {sr_sd:.2f} a year). 0.95 or more is the usual bar."],
             ["Strategy", "Sharpe", f"N = {n_trials} (strategies tried here)", "N = 50 (conservative)"]]
    for r in dsr.itertuples(index=False):
        rows.append([r.model, round(r.Sharpe, 3), round(r.dsr_n, 3), round(r.dsr_50, 3)])
    rows += [blank,
             ["Reading it (as of the first run): realistic rebalancing barely matters (Sharpe about 0.76 either "
              "way; a 5-point band cuts turnover more than 10x). But the edge over SPY is not steady: it came from "
              "2007-2012 (the only period whose interval excludes zero); SPY won 2013-2019 and 2020+ was about even. "
              "The trend filter roughly halved the maximum drawdown at the cost of return, not a higher Sharpe. "
              "The deflated Sharpe says equal weight's Sharpe is very likely above zero after counting every "
              "strategy tried; it does NOT say it beats SPY (SPY itself scores about 0.89)."],
             [f"Last refreshed {date.today().isoformat()}. Source: outputs/robustness_*.csv "
                     "(python -m src.robustness --lag 1)."]]
    return rows


def main(lag: int = 1) -> None:
    from src.backtest import load_yahoo_close
    _, _, cash = load_inputs()
    rets = asset_returns()
    prices = pd.DataFrame({a: load_yahoo_close(a) for a in ASSETS}).dropna()
    days = rets.dropna().index
    ew = pd.DataFrame(1 / 3, index=days, columns=ASSETS)
    spy = pd.DataFrame({"SPY": 1.0, "IEF": 0.0, "GLD": 0.0}, index=days)
    runs = {"Buy & hold SPY": simulate(spy, rets, cash, {a: 0.0 for a in ASSETS}, lag, "monthly"),
            "Equal weight, daily rebalance": simulate(ew, rets, cash, COSTS, lag, "daily"),
            "Equal weight, monthly rebalance": simulate(ew, rets, cash, COSTS, lag, "monthly"),
            "Equal weight, 5-point band": simulate(ew, rets, cash, COSTS, lag, "band"),
            "Equal weight + 200-day trend filter": simulate(trend_target(prices), rets, cash, COSTS, lag, "monthly")}
    idx = None
    for r in runs.values():
        idx = r.index if idx is None else idx.intersection(r.index)
    idx = idx[idx >= START]
    samples = [("All", slice(None))] + PERIODS + [("2025+ (final check)", slice("2025", None))]
    rows, ci = [], []
    bh = runs["Buy & hold SPY"]
    for name, r in runs.items():
        for label, sl in samples:
            rr = r.loc[idx].loc[sl]
            s = stats(rr["ret"], rr["cash"])
            s.update(model=name, sample=label, turnover=float(rr["turnover"].sum() / (len(rr) / 252)))
            rows.append(s)
            if name != "Buy & hold SPY" and label in ("All", "2007-2012", "2013-2019", "2020+"):
                ci.append((name, label) + sharpe_diff_ci(rr["ret"], bh.loc[idx].loc[sl, "ret"], rr["cash"]))
    res = pd.DataFrame(rows).set_index(["model", "sample"])
    ci = pd.DataFrame(ci, columns=["model", "sample", "sharpe_diff", "lo95", "hi95"])

    new = {m: res.loc[(m, "All"), "Sharpe"] for m in runs if m != "Buy & hold SPY"}
    trials = trial_sharpes(new)
    n_trials, sr_var_daily = len(trials), float(trials.var(ddof=1)) / 252
    dsr = []
    for m in runs:
        ex = runs[m].loc[idx, "ret"] - runs[m].loc[idx, "cash"]
        dsr.append((m, res.loc[(m, "All"), "Sharpe"], deflated_sharpe(ex, sr_var_daily, n_trials),
                    deflated_sharpe(ex, sr_var_daily, 50)))
    dsr = pd.DataFrame(dsr, columns=["model", "Sharpe", "dsr_n", "dsr_50"])

    res.to_csv(OUT_DIR / f"robustness_stats_lag{lag}.csv")
    ci.to_csv(OUT_DIR / f"robustness_sharpe_diff_lag{lag}.csv", index=False)
    dsr.to_csv(OUT_DIR / f"robustness_deflated_sharpe_lag{lag}.csv", index=False)
    if lag == 1:
        trials.rename("Sharpe").to_csv(OUT_DIR / "robustness_trials.csv")
        sheet = build_sheet_rows(res, ci, dsr, n_trials, float(trials.std(ddof=1)),
                                 idx.min().date(), idx.max().date(), len(idx))
        with open(OUT_DIR / "robustness_sheet_payload.json", "w") as f:
            json.dump({"range": SHEET_RANGE, "data": sheet}, f)
    pd.set_option("display.width", 200)
    print(f"lag={lag}  {idx.min().date()} to {idx.max().date()} ({len(idx)} days); trials N={n_trials}, "
          f"sd of trial Sharpe {trials.std(ddof=1):.3f}")
    print(res.round(3).to_string()); print(ci.round(3).to_string(index=False)); print(dsr.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lag", type=int, default=1)
    main(ap.parse_args().lag)

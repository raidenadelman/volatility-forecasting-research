"""Phase 6: can we earn MORE than buy & hold SPY? Three strategies, fixed in advance (no fitted parameters).

A. "Managed SPY, risk dial k": weight = min(2, k * typical vol / forecast vol), k in {1.25, 1.5} (k=1 is
   Phase 5). Leverage is allowed up to 2x; borrowed money costs the T-bill rate + 0.5% a year. This is
   a risk dial, not a tuned parameter: it shows what each extra unit of return costs in drawdown.
B. "SVXY in contango": hold SVXY (short volatility) when VIX3M > VIX at the close, else cash.
C. "50% SPY + 50% SVXY in contango": B mixed with buy & hold SPY.

SVXY was -1x short-term VIX futures until 2018-02-27 and -0.5x after. Daily-rebalanced leverage is
linear in the daily return, so before 2018-02-28 I use 0.5 * the actual SVXY return to get a
consistent -0.5x series (an approximation: it ignores fee differences). The -83% day of 2018-02-06
therefore becomes about -41%, i.e. what a -0.5x fund would have lost.

Signals are formed at the close of day t and applied to day t+1 (lag 0) or t+2 (lag 1).
Costs: 2 bps per unit of SPY turnover, 10 bps per unit of SVXY turnover.

Usage: python -m src.higher_return --lag 1   (lag 1 is the headline; it also writes the Sheet payload)
"""

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.data import ROOT, load_data, load_vix
from src.evaluate import walk_forward
from src.features import build_features
from src.models import HARLogModel
from src.volmanaged import (HOLD, OUT_DIR, RAW_DIR, load_inputs, managed_weight, sharpe_diff_ci,
                            sigma_from_forecast, stats)

SPY_COST_BPS, SVXY_COST_BPS = 2.0, 10.0
BORROW_SPREAD = 0.005          # annual spread over T-bills on borrowed money
LEVER_CAP = 2.0
DIAL = (1.25, 1.5)
SVXY_SWITCH = pd.Timestamp("2018-02-28")


def svxy_half_x(svxy_close: pd.Series) -> pd.Series:
    """Consistent -0.5x daily returns: 0.5 * actual before the 2018-02-28 leverage change."""
    r = svxy_close.pct_change(fill_method=None)
    return r.where(r.index >= SVXY_SWITCH, 0.5 * r)


def contango_signal(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """1.0 if VIX3M > VIX at the close of day t (contango: calm), else 0.0."""
    return (vix3m.reindex(vix.index) > vix).astype(float)


def run(weights: dict, rets: dict, cash: pd.Series, costs: dict, lag: int,
        borrow_spread: float = BORROW_SPREAD) -> pd.DataFrame:
    """weights/rets: {asset: Series}. Weight set at close t is held on day t+1+lag.
    Cash earns T-bills on the unallocated part; leverage above 100% pays T-bills + spread."""
    w = pd.DataFrame({a: s.shift(1 + lag) for a, s in weights.items()}).dropna()
    r = pd.DataFrame({a: rets[a].reindex(w.index) for a in weights}).dropna()
    w = w.reindex(r.index)
    cash = cash.reindex(r.index).fillna(0.0)
    gross = w.sum(axis=1)
    carry = (1 - gross).clip(lower=0) * cash - (gross - 1).clip(lower=0) * (cash + borrow_spread / 252)
    cost = sum(w[a].diff().abs().fillna(w[a].abs()) * costs[a] / 1e4 for a in weights)
    out = pd.DataFrame({"ret": (w * r).sum(axis=1) + carry - cost, "cash": cash, "w": gross})
    return out


SHEET_RANGE = "'Higher returns'!A1"
FOMC_PATH = ROOT / "data" / "events" / "fomc_decision_dates.csv"


def long_sample(cash_ret: pd.Series, spy_ret: pd.Series, lag: int = 1) -> pd.DataFrame:
    """Plain HAR-RV forecasts go back to 2004, so this tests managed SPY (k = 1, 1.25, 1.5) through 2008-09.
    Same rules and costs as strategy A; only HAR-RV (log) is used because the VIX models need more history."""
    df, vix = load_data(verbose=False), load_vix(verbose=False)
    fomc = pd.read_csv(FOMC_PATH, parse_dates=["date"])["date"]
    fc = walk_forward(build_features(df, vix, fomc), [HARLogModel(name="har")])["har"]
    days = fc.index
    runs = {"Buy & hold SPY": run({"SPY": pd.Series(1.0, index=days)}, {"SPY": spy_ret}, cash_ret,
                                  {"SPY": 0.0}, lag)}
    sig = sigma_from_forecast(fc)
    for k in (1.0,) + DIAL:
        w = k * managed_weight(sig, cap=LEVER_CAP / k)
        runs[f"Managed SPY k={k:g}: HAR-RV (log)"] = run({"SPY": w}, {"SPY": spy_ret}, cash_ret,
                                                         {"SPY": SPY_COST_BPS}, lag)
    idx = runs["Buy & hold SPY"].index
    for r in runs.values():
        idx = idx.intersection(r.index)
    rows = []
    for name, r in runs.items():
        for label, sl in [(f"{idx.min().year}-{idx.max().year}", slice(None)), ("2008-2009", slice("2008", "2009"))]:
            rr = r.loc[idx].loc[sl]
            s = stats(rr["ret"], rr["cash"]); s.update(model=name, window=label, avg_weight=float(rr["w"].mean()))
            rows.append(s)
    return pd.DataFrame(rows)


def build_sheet_rows(res: pd.DataFrame, long: pd.DataFrame, first, last, n) -> list:
    blank = [""]
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "WorstDay", "Calmar", "avg_weight"]
    head = ["Strategy", "CAGR", "Volatility", "Sharpe (excess of cash)", "Max drawdown", "Worst day",
            "Calmar", "Average exposure"]
    rows = [["Phase 6: higher returns than buy & hold SPY"],
            [f"All orders placed one trading day after the signal (the realistic case). Same days for every row: "
             f"{first} to {last} ({n:,} days). Rules fixed in advance. Costs: 2 bps SPY, 10 bps SVXY per unit of "
             "turnover; borrowing at T-bills + 0.5%. SVXY before 2018-02-28 is modeled as -0.5x (0.5 x actual)."]]
    for label, title in [("All", "All days"), (f"{HOLD}+", "2016 onward (holdout)")]:
        rows += [blank, [title], head]
        for (m, s), r in res.iterrows():
            if s == label:
                rows.append([m] + [round(float(r[c]), 4) for c in cols])
    rows += [blank, ["Long sample including 2008-2009 (plain HAR-RV forecasts, one-day delay)"],
             ["Strategy", "Window", "CAGR", "Volatility", "Sharpe (excess of cash)", "Max drawdown", "Worst day",
              "Average exposure"]]
    for r in long.itertuples(index=False):
        rows.append([r.model, r.window, round(r.CAGR, 4), round(r.Vol, 4), round(r.Sharpe, 4),
                     round(r.MaxDD, 4), round(r.WorstDay, 4), round(r.avg_weight, 4)])
    rows += [blank,
             ["Reading it (as of the first run): managed SPY with a risk dial of 1.5 (average exposure about 1.5x) "
              "earned more than SPY with a smaller maximum drawdown, including through 2008-09, but it needs "
              "leverage, which costs more for most retail accounts than assumed here. The SVXY strategies lost "
              "their edge with a one-day delay."],
             [f"Last refreshed {date.today().isoformat()}. Source: outputs/higher_return_*.csv (python -m src.higher_return --lag 1)."]]
    return rows


def main(lag: int = 0) -> None:
    fc, spy_ret, cash_ret = load_inputs()
    vix = load_vix(verbose=False)
    vix3m = pd.read_csv(RAW_DIR / "close_vix3m.csv", index_col=0, parse_dates=True)["Close"]
    svxy = svxy_half_x(pd.read_csv(RAW_DIR / "close_svxy.csv", index_col=0, parse_dates=True)["Close"])
    rets = {"SPY": spy_ret, "SVXY": svxy}
    costs = {"SPY": SPY_COST_BPS, "SVXY": SVXY_COST_BPS}
    days = fc.index
    cont = contango_signal(vix, vix3m).reindex(days)
    one = pd.Series(1.0, index=days)

    runs = {"Buy & hold SPY": run({"SPY": one}, rets, cash_ret, {"SPY": 0.0}, lag)}
    for k in DIAL:
        for m in ["HAR-RV (log)", "HAR+VIX+FOMC+Slope (log)"]:
            w = k * managed_weight(sigma_from_forecast(fc[m]), cap=LEVER_CAP / k)
            runs[f"A. Managed SPY k={k}: {m}"] = run({"SPY": w}, rets, cash_ret, costs, lag)
    runs["B. SVXY in contango, else cash"] = run({"SVXY": cont}, rets, cash_ret, costs, lag)
    runs["C. 50% SPY + 50% SVXY in contango"] = run({"SPY": 0.5 * one, "SVXY": 0.5 * cont},
                                                    rets, cash_ret, costs, lag)
    bh = runs["Buy & hold SPY"]
    idx = bh.index
    for r in runs.values():
        idx = idx.intersection(r.index)
    rows, ci = [], []
    for name, r in runs.items():
        for label, sl in [("All", slice(None)), (f"{HOLD}+", slice(HOLD, None))]:
            rr = r.loc[idx].loc[sl]
            s = stats(rr["ret"], rr["cash"]); s.update(model=name, sample=label, avg_weight=float(rr["w"].mean()))
            rows.append(s)
            if name != "Buy & hold SPY":
                ci.append((name, label) + sharpe_diff_ci(rr["ret"], bh.loc[idx].loc[sl, "ret"], rr["cash"]))
    res = pd.DataFrame(rows).set_index(["model", "sample"])
    ci = pd.DataFrame(ci, columns=["model", "sample", "sharpe_diff", "lo95", "hi95"])
    res.to_csv(OUT_DIR / f"higher_return_stats_lag{lag}.csv")
    ci.to_csv(OUT_DIR / f"higher_return_sharpe_diff_lag{lag}.csv", index=False)
    if lag == 1:
        long = long_sample(cash_ret, spy_ret, lag)
        long.to_csv(OUT_DIR / "higher_return_long_sample_lag1.csv", index=False)
        rows = build_sheet_rows(res, long, idx.min().date(), idx.max().date(), len(idx))
        with open(OUT_DIR / "higher_return_sheet_payload.json", "w") as f:
            json.dump({"range": SHEET_RANGE, "data": rows}, f)
        print(long.round(3).to_string())
    pd.set_option("display.width", 200)
    print(f"lag={lag}  {idx.min().date()} to {idx.max().date()} ({len(idx)} days)")
    print(res.round(3).to_string()); print(ci.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lag", type=int, default=0)
    main(ap.parse_args().lag)

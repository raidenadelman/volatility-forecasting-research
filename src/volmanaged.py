"""Phase 5: volatility-managed SPY. Does a better variance forecast make money?

Rule (fixed in advance, not tuned on results). At the close of day t, with sigma_t = annualized
volatility implied by the model's forecast of day t+1's variance:
    weight_t = min(1, c_t / sigma_t),   c_t = expanding MEDIAN of sigma_s over s <= t
so exposure is 1.0 in a typical-or-calmer market and shrinks as forecast volatility rises. No
leverage. The weight applies to SPY's return on day t+1 (lag=0: order at the close) or t+2
(lag=1: one extra day to execute). The rest sits in cash earning the 13-week T-bill rate (^IRX).
Cost: COST_BPS per unit of turnover. Sharpe is on returns in excess of cash.

Every model is run on the SAME forecast days (the Phase 4 sample, 2010-07 onward).

Usage: python -m src.volmanaged [--lag 1] [--refresh]   (needs outputs/slope_forecasts.csv from src.run_phase4)
"""

import json
from datetime import date

import numpy as np
import pandas as pd

from src.data import ROOT

OUT_DIR = ROOT / "outputs"
RAW_DIR = ROOT / "data" / "raw"
COST_BPS = 2.0      # SPY spread + slippage, per unit of turnover
HOLD = "2016"       # same holdout start as Phase 4
N_BOOT, BLOCK = 2000, 20


def sigma_from_forecast(var_forecast: pd.Series) -> pd.Series:
    """Daily variance forecast -> annualized volatility (fraction, e.g. 0.16)."""
    return np.sqrt(252 * var_forecast)


def managed_weight(sigma: pd.Series, cap: float = 1.0) -> pd.Series:
    """min(cap, c_t / sigma_t), c_t = expanding median of sigma up to and including t."""
    c = sigma.expanding(min_periods=20).median()
    return (c / sigma).clip(upper=cap).dropna()


def backtest(weight: pd.Series, spy_ret: pd.Series, cash_ret: pd.Series, lag: int = 0,
             cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Daily returns. weight[t] (set at close t) is held on day t+1+lag; cash earns the rest."""
    w = weight.shift(1 + lag).dropna()
    out = pd.DataFrame({"w": w})
    out["spy"] = spy_ret.reindex(out.index)
    out["cash"] = cash_ret.reindex(out.index).fillna(0.0)
    out = out.dropna(subset=["spy"])
    turnover = out["w"].diff().abs().fillna(out["w"].abs())
    out["ret"] = out["w"] * out["spy"] + (1 - out["w"]) * out["cash"] - turnover * cost_bps / 1e4
    out["excess"] = out["ret"] - out["cash"]
    return out


def stats(ret: pd.Series, cash: pd.Series) -> dict:
    ex = ret - cash.reindex(ret.index).fillna(0.0)
    eq = (1 + ret).cumprod()
    years = len(ret) / 252
    vol = ret.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    return {"CAGR": cagr, "Vol": vol, "Sharpe": ex.mean() * 252 / (ex.std() * np.sqrt(252)),
            "MaxDD": dd, "WorstDay": ret.min(), "Calmar": cagr / abs(dd)}


def sharpe_diff_ci(a: pd.Series, b: pd.Series, cash: pd.Series, n_boot: int = N_BOOT,
                   block: int = BLOCK, seed: int = 0):
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b) on excess returns; (diff, lo95, hi95)."""
    d = pd.concat([a - cash, b - cash], axis=1).dropna().to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    sh = lambda x: x.mean(0) / x.std(0, ddof=1) * np.sqrt(252)
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        s = np.concatenate([d[i:i + block] for i in starts])[:n]
        v = sh(s)
        diffs.append(v[0] - v[1])
    point = sh(d)
    return float(point[0] - point[1]), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def load_inputs(refresh: bool = False):
    if refresh:  # re-download SPY and the T-bill rate (ETF/index closes, cached in data/raw)
        from src.backtest import load_yahoo_close
        load_yahoo_close("SPY", True)
        load_yahoo_close("^IRX", True)
    fc = pd.read_csv(OUT_DIR / "slope_forecasts.csv", index_col=0, parse_dates=True)
    spy = pd.read_csv(RAW_DIR / "close_spy.csv", index_col=0, parse_dates=True)["Close"]
    irx = pd.read_csv(RAW_DIR / "close_irx.csv", index_col=0, parse_dates=True)["Close"]
    spy_ret = spy.pct_change(fill_method=None)
    cash_ret = (irx.reindex(spy.index).ffill() / 100 / 252)  # IRX is in percent per year
    return fc, spy_ret, cash_ret


SHEET_RANGE = "'Vol-managed SPY'!A1"


def build_sheet_rows(res: pd.DataFrame, ci: pd.DataFrame, first, last, n) -> list:
    """Rows for the Sheet tab 'Vol-managed SPY' (written at SHEET_RANGE)."""
    blank = [""]
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "WorstDay", "Calmar"]
    head = ["Model", "CAGR", "Volatility", "Sharpe (excess of cash)", "Max drawdown", "Worst day", "Calmar"]
    rows = [["Phase 5: volatility-managed SPY"],
            [f"Hold SPY with weight min(1, typical vol / forecast vol), no leverage, rest in T-bills; "
             f"order placed at the close. Same days for every model: {first} to {last} ({n:,} days). "
             "Rules fixed in advance. Cost 2 bps per unit of turnover."]]
    for label in ["All", f"{HOLD}+"]:
        rows += [blank, [("All days" if label == "All" else f"{HOLD} onward (holdout)")], head]
        for (m, s), r in res.iterrows():
            if s == label:
                rows.append([m] + [round(float(r[c]), 4) for c in cols])
    rows += [blank, ["Sharpe difference vs buy & hold SPY (block bootstrap, 95% interval)"],
             ["Model", "Sample", "Sharpe difference", "Low 95%", "High 95%"]]
    for r in ci.itertuples(index=False):
        rows.append([r.model, r.sample, round(r.sharpe_diff, 3), round(r.lo95, 3), round(r.hi95, 3)])
    rows += [blank,
             ["Reading it: managing exposure cut volatility by about a third and roughly halved the "
              "maximum drawdown and worst day, at the cost of some return. Sharpe differences are not "
              "statistically distinguishable from zero, and the model with the best QLIKE did not give "
              "the best Sharpe."],
             [f"Last refreshed {date.today().isoformat()}. Source: outputs/volmanaged_*.csv (python -m src.volmanaged)."]]
    return rows


def main(lag: int = 0, refresh: bool = False) -> None:
    fc, spy_ret, cash_ret = load_inputs(refresh)
    models = [c for c in fc.columns if c != "actual"]
    days = fc.index
    runs = {}
    bh = backtest(pd.Series(1.0, index=days), spy_ret, cash_ret, lag, cost_bps=0)
    runs["Buy & hold SPY"] = bh
    for m in models:
        runs[f"Managed: {m}"] = backtest(managed_weight(sigma_from_forecast(fc[m])), spy_ret, cash_ret, lag)
    idx = bh.index.intersection(runs[f"Managed: {models[-1]}"].index)
    rows = []
    for name, r in runs.items():
        for label, sl in [("All", slice(None)), (f"{HOLD}+", slice(HOLD, None))]:
            rr = r.loc[idx].loc[sl]
            s = stats(rr["ret"], rr["cash"])
            s.update(model=name, sample=label, avg_weight=float(rr["w"].mean()))
            rows.append(s)
    res = pd.DataFrame(rows).set_index(["model", "sample"])
    # Sharpe difference vs buy & hold, with bootstrap CI (All sample)
    ci = []
    for name, r in runs.items():
        if name == "Buy & hold SPY":
            continue
        for label, sl in [("All", slice(None)), (f"{HOLD}+", slice(HOLD, None))]:
            a, b = r.loc[idx].loc[sl], bh.loc[idx].loc[sl]
            ci.append((name, label) + sharpe_diff_ci(a["ret"], b["ret"], a["cash"]))
    ci = pd.DataFrame(ci, columns=["model", "sample", "sharpe_diff", "lo95", "hi95"])
    suffix = f"_lag{lag}"
    res.to_csv(OUT_DIR / f"volmanaged_stats{suffix}.csv")
    ci.to_csv(OUT_DIR / f"volmanaged_sharpe_diff{suffix}.csv", index=False)
    if lag == 0:
        rows = build_sheet_rows(res, ci[ci["model"].ne("")], idx.min().date(), idx.max().date(), len(idx))
        with open(OUT_DIR / "volmanaged_sheet_payload.json", "w") as f:
            json.dump({"range": SHEET_RANGE, "data": rows}, f)
    pd.set_option("display.width", 170)
    print(f"lag={lag}  days {idx.min().date()} to {idx.max().date()} ({len(idx)})")
    print(res.round(4).to_string()); print(ci.round(3).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    main(a.lag, a.refresh)

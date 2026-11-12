"""Phase 9: does the volatility-managed rule work outside the US? An out-of-sample test on 15 country funds.

PRE-REGISTERED RULES. Written and committed before any international data was downloaded or run.
Nothing below may be changed after seeing results.

Question: is "cut exposure when the HAR-RV forecast says volatility is high" (Phase 5) a real, general
effect, or something that happened to work on US stocks? Phases 5-8 only ever used US data.

Funds (selection rule: every developed-market fund in iShares' original March 1996 "WEBS" country
launch that still trades; Malaysia EWM and Mexico EWW are emerging markets and excluded):
    Asia-Pacific  EWJ Japan, EWA Australia, EWH Hong Kong, EWS Singapore
    Americas      EWC Canada
    Europe        EWU UK, EWG Germany, EWQ France, EWL Switzerland, EWN Netherlands, EWI Italy,
                  EWP Spain, EWD Sweden, EWK Belgium, EWO Austria
SPY is run the same way as a reference row; it is NOT counted in the pass count or the pooled result.

Data: daily OHLC (for the Parkinson range) and dividend-adjusted closes (for returns) from Yahoo.
  * Start-date rule: early years traded thinly, with zero-range days (High == Low). Each fund starts on
    the trading day after the last 252-day window in which zero-range days were 1% or more of days.
  * Any zero-range day after the start gets the smallest positive Parkinson variance seen so far for
    that fund (an expanding minimum, so no look-ahead), because log(0) is undefined.
  * A fund with fewer than MIN_TRAIN + 252 days after its start is reported as excluded, not dropped.

Forecast: HAR-RV (log) on each fund's own Parkinson variance; 500-day minimum training window, refit
every 21 days (same walk-forward code as Phases 1-8).

Trading rule (identical to Phase 5): weight_t = min(1, c_t / sigma_t), c_t = expanding median of the
fund's own forecast vol; no leverage; the rest in T-bills (^IRX). Signal at the close of day t is
held on day t+2 (one-day execution delay). Cost 10 bps per unit of turnover (country ETFs have wider
spreads than SPY; SPY reference row uses 2 bps as before). Benchmark: buy & hold the same fund.
Evaluation window: 2000-01-03 (first day of the cached T-bill series) to the latest data.

Judging:
  * Per fund: change in max drawdown, Sharpe difference with a 95% block-bootstrap interval, worst day,
    and max drawdown inside the crisis windows 2000-2002, 2008-2009, 2020, 2022.
  * PASS = managed max drawdown smaller than buy & hold AND Sharpe difference >= 0.
  * Headline: average Sharpe difference across the 15 funds, with one 95% interval from a block
    bootstrap that draws the SAME dates for every fund (so funds that crash together are not treated
    as independent evidence). Also reported by region.
  * Decision (fixed now):
        KEEP volatility scaling in the final strategy   if passes >= 10 of 15 AND pooled low 95% > 0
        CRASH INSURANCE ONLY                             if not KEEP, but drawdown smaller in >= 10 of 15
        DROP it (final strategy = plain equal weight)    otherwise

Usage: python -m src.international [--refresh]
"""

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.data import load_data
from src.evaluate import walk_forward
from src.features import MONTH, WEEK, parkinson_variance
from src.models import HARLogModel
from src.volmanaged import (BLOCK, N_BOOT, OUT_DIR, RAW_DIR, backtest, load_inputs, managed_weight,
                            sharpe_diff_ci, sigma_from_forecast, stats)

FUNDS = {"EWJ": ("Japan", "Asia-Pacific"), "EWA": ("Australia", "Asia-Pacific"),
         "EWH": ("Hong Kong", "Asia-Pacific"), "EWS": ("Singapore", "Asia-Pacific"),
         "EWC": ("Canada", "Americas"),
         "EWU": ("United Kingdom", "Europe"), "EWG": ("Germany", "Europe"), "EWQ": ("France", "Europe"),
         "EWL": ("Switzerland", "Europe"), "EWN": ("Netherlands", "Europe"), "EWI": ("Italy", "Europe"),
         "EWP": ("Spain", "Europe"), "EWD": ("Sweden", "Europe"), "EWK": ("Belgium", "Europe"),
         "EWO": ("Austria", "Europe")}
REFERENCE = {"SPY": ("United States (reference)", "Reference")}
COST_BPS, SPY_COST_BPS = 10.0, 2.0
LAG = 1
MIN_TRAIN = 500
ZERO_WINDOW, ZERO_MAX_SHARE = 252, 0.01
EVAL_START = "2000-01-03"
CRISES = [("2000-2002", slice("2000", "2002")), ("2008-2009", slice("2008", "2009")),
          ("2020", slice("2020", "2020")), ("2022", slice("2022", "2022"))]
PASS_NEEDED = 10
SHEET_RANGE = "'International'!A1"


# ----------------------------------------------------------------------------
# Data rules
# ----------------------------------------------------------------------------
def data_start(ohlc: pd.DataFrame, window: int = ZERO_WINDOW, max_share: float = ZERO_MAX_SHARE):
    """Trading day after the last `window`-day window whose share of zero-range days is >= max_share."""
    zero = (ohlc["High"] <= ohlc["Low"]).astype(float)
    share = zero.rolling(window).mean()
    bad = share[share >= max_share]
    if bad.empty:
        return ohlc.index[0]
    later = ohlc.index[ohlc.index > bad.index[-1]]
    return later[0] if len(later) else None


def rv_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """HAR features from Parkinson variance; zero-range days get the expanding minimum positive value."""
    rv = parkinson_variance(ohlc["High"], ohlc["Low"])
    floor = rv.where(rv > 0).expanding().min()
    rv = rv.where(rv > 0, floor)
    out = pd.DataFrame({"rv_d": rv, "rv_w": rv.rolling(WEEK).mean(), "rv_m": rv.rolling(MONTH).mean(),
                        "target": rv.shift(-1)})
    return out.dropna()


# ----------------------------------------------------------------------------
# Pooled bootstrap
# ----------------------------------------------------------------------------
def pooled_sharpe_diff(managed_ex: pd.DataFrame, bh_ex: pd.DataFrame, n_boot: int = N_BOOT,
                       block: int = BLOCK, seed: int = 0, min_obs: int = 252):
    """Average across funds of Sharpe(managed) - Sharpe(buy & hold), and a 95% interval from a moving-block
    bootstrap that resamples the same dates for every fund. Funds may start on different dates (NaN before)."""
    m, b = managed_ex.to_numpy(float), bh_ex.to_numpy(float)
    n = len(m)

    def avg_diff(mm, bb):
        ok = ~np.isnan(mm) & ~np.isnan(bb)
        cnt = ok.sum(0)
        mm, bb = np.where(ok, mm, np.nan), np.where(ok, bb, np.nan)
        sh = lambda x: np.nanmean(x, 0) / np.nanstd(x, 0, ddof=1) * np.sqrt(252)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = sh(mm) - sh(bb)
        return float(np.nanmean(np.where(cnt >= min_obs, d, np.nan)))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boots.append(avg_diff(m[idx], b[idx]))
    return avg_diff(m, b), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def decide(passes: int, dd_better: int, pooled_lo: float, needed: int = PASS_NEEDED) -> str:
    if passes >= needed and pooled_lo > 0:
        return "KEEP volatility scaling"
    if dd_better >= needed:
        return "CRASH INSURANCE ONLY"
    return "DROP volatility scaling"


# ----------------------------------------------------------------------------
def run_fund(ticker: str, cash: pd.Series, refresh: bool = False):
    """Returns (info dict, managed backtest, buy & hold backtest) or (info, None, None) if excluded."""
    from src.backtest import load_yahoo_close
    ohlc = load_data(refresh=refresh, verbose=False, ticker=ticker,
                     cache_path=RAW_DIR / f"ohlc_{ticker.lower()}.csv")
    close = load_yahoo_close(ticker, refresh)
    start = data_start(ohlc)
    info = {"fund": ticker, "first_data": ohlc.index[0].date(), "start": start.date() if start is not None else None,
            "zero_range_days": int((ohlc["High"] <= ohlc["Low"]).sum())}
    if start is None or (ohlc.index >= start).sum() < MIN_TRAIN + 252:
        info["excluded"] = "too little clean data"
        return info, None, None
    ohlc = ohlc.loc[start:]
    fc = walk_forward(rv_features(ohlc), [HARLogModel(name=ticker)], min_train=MIN_TRAIN)[ticker]
    ret = close.pct_change(fill_method=None)
    w = managed_weight(sigma_from_forecast(fc))
    cost = SPY_COST_BPS if ticker == "SPY" else COST_BPS
    man = backtest(w, ret, cash, LAG, cost)
    bh = backtest(pd.Series(1.0, index=w.index), ret, cash, LAG, 0.0)
    idx = man.index.intersection(bh.index)
    idx = idx[idx >= EVAL_START]
    info["excluded"] = ""
    return info, man.loc[idx], bh.loc[idx]


def weekly_corr(a: pd.Series, b: pd.Series) -> float:
    wk = lambda r: (1 + r).resample("W-FRI").prod() - 1
    d = pd.concat([wk(a), wk(b)], axis=1, sort=True).dropna()
    return float(d.corr().iloc[0, 1])


def build_sheet_rows(table, crises, pooled, regions, verdict, passes, dd_better, n_funds, first, last) -> list:
    blank = [""]
    rows = [["Phase 9: volatility-managed rule on 15 international country funds (out-of-sample)"],
            ["Rules were committed to git before any international data was downloaded. Same rule as Phase 5: "
             "weight = min(1, typical vol / forecast vol), no leverage, one-day delay, 10 bps per unit of turnover, "
             f"rest in T-bills. Benchmark: buy & hold the same fund. Evaluation from {first} (earliest, after the 500-day "
             f"warm-up; later for funds with a later data start) to {last}. "
             "PASS = smaller max drawdown AND Sharpe difference >= 0."],
            blank,
            [f"Decision: {verdict}", f"Passes: {passes} of {n_funds}", f"Drawdown smaller: {dd_better} of {n_funds}",
             f"Average Sharpe difference: {pooled[0]:+.3f}", f"95% interval: {pooled[1]:+.3f} to {pooled[2]:+.3f}"],
            blank,
            ["Fund", "Country", "Region", "First evaluation day", "Weekly corr. with SPY", "Sharpe (buy & hold)", "Sharpe (managed)",
             "Sharpe difference", "Low 95%", "High 95%", "Max DD (buy & hold)", "Max DD (managed)",
             "Worst day (buy & hold)", "Worst day (managed)", "Average exposure", "PASS"]]
    for r in table.itertuples(index=False):
        if r.excluded:
            rows.append([r.fund, r.country, r.region, f"clean data from {r.start}", f"EXCLUDED: {r.excluded}"])
            continue
        rows.append([r.fund, r.country, r.region, str(r.eval_start), round(r.corr_spy, 3), round(r.sharpe_bh, 3),
                     round(r.sharpe_man, 3), round(r.sharpe_diff, 3), round(r.lo95, 3), round(r.hi95, 3),
                     round(r.maxdd_bh, 4), round(r.maxdd_man, 4), round(r.worst_bh, 4), round(r.worst_man, 4),
                     round(r.avg_weight, 3), "PASS" if r.passed else "fail"])
    rows += [blank, ["Average Sharpe difference by region (same-date block bootstrap)"],
             ["Region", "Funds", "Average Sharpe difference", "Low 95%", "High 95%"]]
    for r in regions.itertuples(index=False):
        rows.append([r.region, int(r.n), round(r.diff, 3), round(r.lo95, 3), round(r.hi95, 3)])
    rows += [blank, ["Max drawdown inside crisis windows, averaged over funds with data"],
             ["Window", "Funds", "Buy & hold", "Managed", "Funds where managed was smaller"]]
    for r in crises.itertuples(index=False):
        rows.append([r.window, int(r.n), round(r.dd_bh, 4), round(r.dd_man, 4), int(r.n_better)])
    rows += [blank,
             ["Reading it (as of the first run): under the pre-registered rule the verdict is CRASH INSURANCE ONLY. "
              "Managed exposure cut the maximum drawdown in all 14 funds and in every fund in 2008-09 and 2020 (11 of 14 in 2022), but it did not "
              "raise Sharpe: 3 of 14 passed, and the pooled Sharpe difference is about zero. SPY (in-sample) "
              "looked better than any foreign fund, so the US Sharpe gain was probably partly luck."],
             ["Disclosures: (1) the project's data loader starts Yahoo data in 2000, not 1996, so after the 500-day "
              "warm-up evaluation starts in 2002 at the earliest and the 2000-02 crash is mostly missed. "
              "(2) Belgium (EWK) had 371 zero-range days; the start-date rule left too little data, so it is "
              "excluded and 14 funds are counted. (3) A post-hoc check (not part of the rules): with 0 bps costs "
              "the pooled Sharpe difference is still only +0.005, so costs are not the reason. Turnover averages "
              "about 5.5x a year."],
             [f"Last refreshed {date.today().isoformat()}. Source: outputs/international_*.csv "
                     "(python -m src.international)."]]
    return rows


def main(refresh: bool = False) -> None:
    _, spy_ret, cash = load_inputs()
    rows, man_ex, bh_ex, crisis_rows = [], {}, {}, []
    for t, (country, region) in {**FUNDS, **REFERENCE}.items():
        info, man, bh = run_fund(t, cash, refresh)
        info.update(country=country, region=region)
        if man is not None:
            sm, sb = stats(man["ret"], man["cash"]), stats(bh["ret"], bh["cash"])
            d, lo, hi = sharpe_diff_ci(man["ret"], bh["ret"], man["cash"])
            info.update(eval_start=man.index[0].date(), n_days=len(man), corr_spy=weekly_corr(bh["ret"], spy_ret),
                        sharpe_bh=sb["Sharpe"], sharpe_man=sm["Sharpe"], sharpe_diff=d, lo95=lo, hi95=hi,
                        cagr_bh=sb["CAGR"], cagr_man=sm["CAGR"], vol_bh=sb["Vol"], vol_man=sm["Vol"],
                        maxdd_bh=sb["MaxDD"], maxdd_man=sm["MaxDD"], worst_bh=sb["WorstDay"],
                        worst_man=sm["WorstDay"], avg_weight=float(man["w"].mean()))
            info["dd_better"] = bool(sm["MaxDD"] > sb["MaxDD"])
            info["passed"] = bool(info["dd_better"] and d >= 0)
            if t in FUNDS:
                man_ex[t], bh_ex[t] = man["ret"] - man["cash"], bh["ret"] - bh["cash"]
            for label, sl in CRISES:
                mm, bb = man.loc[sl], bh.loc[sl]
                if len(mm) > 20:
                    ddm = stats(mm["ret"], mm["cash"])["MaxDD"]; ddb = stats(bb["ret"], bb["cash"])["MaxDD"]
                    crisis_rows.append({"fund": t, "window": label, "dd_bh": ddb, "dd_man": ddm})
        rows.append(info)
    table = pd.DataFrame(rows)
    table["excluded"] = table["excluded"].fillna("")
    funds = table[table["fund"].isin(FUNDS) & table["excluded"].eq("")]
    passes, dd_better = int(funds["passed"].sum()), int(funds["dd_better"].sum())
    M, B = pd.DataFrame(man_ex), pd.DataFrame(bh_ex)
    pooled = pooled_sharpe_diff(M, B)
    regions = []
    for reg in ["Asia-Pacific", "Americas", "Europe"]:
        cols = [t for t in M.columns if FUNDS[t][1] == reg]
        regions.append((reg, len(cols)) + pooled_sharpe_diff(M[cols], B[cols]))
    regions = pd.DataFrame(regions, columns=["region", "n", "diff", "lo95", "hi95"])
    cr = pd.DataFrame(crisis_rows)
    crf = cr[cr["fund"].isin(FUNDS)]
    crises = (crf.assign(better=crf["dd_man"] > crf["dd_bh"]).groupby("window", sort=False)
              .agg(n=("fund", "size"), dd_bh=("dd_bh", "mean"), dd_man=("dd_man", "mean"), n_better=("better", "sum"))
              .reset_index())
    verdict = decide(passes, dd_better, pooled[1])
    n_funds = len(funds)

    table.to_csv(OUT_DIR / "international_by_fund.csv", index=False)
    cr.to_csv(OUT_DIR / "international_crises.csv", index=False)
    pd.DataFrame([{"avg_sharpe_diff": pooled[0], "lo95": pooled[1], "hi95": pooled[2], "passes": passes,
                   "dd_better": dd_better, "n_funds": n_funds, "decision": verdict}]).to_csv(
        OUT_DIR / "international_pooled.csv", index=False)
    regions.to_csv(OUT_DIR / "international_regions.csv", index=False)
    first, last = M.index.min().date(), max(M.index.max(), B.index.max()).date()
    sheet = build_sheet_rows(table, crises, pooled, regions, verdict, passes, dd_better, n_funds, first, last)
    with open(OUT_DIR / "international_sheet_payload.json", "w") as f:
        json.dump({"range": SHEET_RANGE, "data": sheet}, f)

    pd.set_option("display.width", 250)
    cols = ["fund", "country", "start", "corr_spy", "sharpe_bh", "sharpe_man", "sharpe_diff", "lo95", "hi95",
            "maxdd_bh", "maxdd_man", "avg_weight", "passed", "excluded"]
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
    print(regions.round(3).to_string(index=False)); print(crises.round(3).to_string(index=False))
    print(f"Passes {passes}/{n_funds}; drawdown smaller {dd_better}/{n_funds}; pooled Sharpe diff "
          f"{pooled[0]:+.3f} [{pooled[1]:+.3f}, {pooled[2]:+.3f}] -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)

"""Front-of-book tabs for the Google Sheet: Overview, Catalogue, Vol signals (Phase 3) and Methods.

Numbers are read from the result CSVs in outputs/, so re-running the phases and then this module keeps
the tabs current. Writes outputs/overview_sheet_payload.json = {"tabs": [{"range": ..., "data": rows}, ...]}.
The refresh task copies each tab's rows to its range with update_values (no Google API calls here).

Usage: python -m src.sheet_overview
"""

import json
from datetime import date

import pandas as pd

from src.data import ROOT

OUT = ROOT / "outputs"
B = [""]


def _csv(name, **kw):
    return pd.read_csv(OUT / name, **kw)


def _pick(df, model, sample="All", col="Sharpe"):
    return float(df[(df["model"] == model) & (df["sample"] == sample)][col].iloc[0])


def _pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def load() -> dict:
    """Every number the front tabs quote, in one place."""
    n = {}
    slope = _csv("slope_results.csv").set_index("model")
    n["q_naive"], n["q_har"] = slope.loc["Naive", "QLIKE"], slope.loc["HAR-RV (log)", "QLIKE"]
    n["q_full"], n["q_slope"] = slope.loc["HAR+VIX+FOMC (log)", "QLIKE"], slope.loc["HAR+VIX+FOMC+Slope (log)", "QLIKE"]
    n["q_slope_hold"] = slope.loc["HAR+VIX+FOMC+Slope (log)", "QLIKE 2016+ (holdout)"]
    sdm = _csv("slope_dm_tests.csv")
    n["dm_slope"] = float(sdm[sdm["model_B"] == "HAR+VIX+FOMC (log)"]["DM_stat"].iloc[0])
    ro = _csv("results_overall.csv").set_index("model")["QLIKE"]
    n["p1"] = ro
    dm = _csv("dm_tests.csv")
    n["dm_min"], n["dm_max"] = dm[dm["model_B"].isin(["Naive", "Rolling mean"])]["DM_stat"].agg(["max", "min"])

    vm1 = _csv("volmanaged_stats_lag1.csv")
    n["vm_bh"] = _pick(vm1, "Buy & hold SPY")
    n["vm"] = {m: _pick(vm1, f"Managed: {m}") for m in
               ["Naive", "HAR-RV (log)", "HAR+VIX+FOMC (log)", "HAR+VIX+FOMC+Slope (log)"]}
    vm0 = _csv("volmanaged_stats_lag0.csv")
    n["vm0_dd_bh"], n["vm0_dd_har"] = _pick(vm0, "Buy & hold SPY", col="MaxDD"), _pick(vm0, "Managed: HAR-RV (log)", col="MaxDD")
    d0 = _csv("volmanaged_sharpe_diff_lag0.csv")
    n["vm0_diff_lo"], n["vm0_diff_hi"] = d0[d0["sample"] == "All"]["sharpe_diff"].agg(["min", "max"])

    hr = _csv("higher_return_stats_lag1.csv")
    n["hr_spy_cagr"] = _pick(hr, "Buy & hold SPY", col="CAGR")
    n["hr_k15_cagr"] = _pick(hr, "A. Managed SPY k=1.5: HAR-RV (log)", col="CAGR")

    ma = _csv("multiasset_stats_lag1.csv")
    n["ma"] = ma[ma["sample"] == "All"].set_index("model")
    rb = _csv("robustness_stats_lag1.csv")
    n["rb"] = rb.set_index(["model", "sample"])
    n["rb_ci"] = _csv("robustness_sharpe_diff_lag1.csv")
    n["dsr"] = _csv("robustness_deflated_sharpe_lag1.csv").set_index("model")

    intl = _csv("international_by_fund.csv")
    n["intl"] = intl
    n["intl_pool"] = _csv("international_pooled.csv").iloc[0]
    cr = _csv("international_crises.csv")
    c08 = cr[(cr["window"] == "2008-2009") & (cr["fund"] != "SPY")]
    n["intl_dd08_bh"], n["intl_dd08_man"] = c08["dd_bh"].mean(), c08["dd_man"].mean()
    ok = intl[intl["excluded"].fillna("").eq("") & intl["fund"].ne("SPY")]
    n["intl_cagr_cost"] = float((ok["cagr_bh"] - ok["cagr_man"]).mean())

    n["p3_5y"] = _csv("backtest_stats_lag1.csv", index_col=0)
    n["p3_10y"] = _csv("backtest_stats_lag1_10y.csv", index_col=0)
    n["p3_cal"] = _csv("backtest_calibration_lag1_10y.csv")
    n["p3_year"] = _csv("backtest_yearly_lag1_10y.csv", index_col=0)
    return n


def overview(n) -> list:
    rb, ma, pool = n["rb"], n["ma"], n["intl_pool"]
    ew = rb.loc[("Equal weight, 5-point band", "All")]
    spy = rb.loc[("Buy & hold SPY", "All")]
    rows = [["Forecasting market volatility, and what it is actually worth to an investor"],
            ["A nine-phase research project. Walk-forward models forecast next-day S&P 500 volatility well. Better "
             "forecasts did not raise return per unit of risk, but they reliably cut losses in crashes. Diversifying "
             "across stocks, bonds and gold did more for risk-adjusted return than any forecast. Safeguards: walk-forward "
             "fitting, one-day execution delay, trading costs, bootstrap confidence intervals, a Sharpe ratio deflated "
             "for the number of strategies tried, and a final test whose rules were committed before its data was "
             "downloaded."],
            B, ["Key findings"],
            ["Finding", "Evidence", "Headline number", "Where to look"],
            ["Next-day volatility is predictable, and each added input helped.",
             "HAR-RV, then VIX, a Fed-decision flag and the VIX term-structure slope; every step significant "
             "(Diebold-Mariano, p < 0.01); the slope model also won on a 2016+ holdout.",
             f"QLIKE {n['q_naive']:.3f} (naive) to {n['q_slope']:.3f} (best model)", "Forecast summary, Slope model"],
            ["Past a basic model, better forecasts did not make a better portfolio.",
             "Volatility-managed SPY beat buy & hold on Sharpe with every HAR model, but the most accurate model gave "
             "the lowest of the three, and every difference is inside its 95% bootstrap interval.",
             f"Sharpe {n['vm']['HAR-RV (log)']:.2f} (HAR) vs {n['vm']['HAR+VIX+FOMC+Slope (log)']:.2f} "
             f"(most accurate) vs {n['vm_bh']:.2f} (SPY)", "Vol-managed SPY"],
            ["Protection has a price, and diversification is the cheapest source.",
             "Equal weight SPY/IEF/GLD, rebalanced on a 5-point band, 2007-2026 with costs. The edge over SPY came "
             "mostly from 2007-2012.",
             f"Sharpe {ew['Sharpe']:.2f} vs {spy['Sharpe']:.2f}; max drawdown {_pct(ew['MaxDD'], 0)} vs "
             f"{_pct(spy['MaxDD'], 0)}; return {_pct(ew['CAGR'])} vs {_pct(spy['CAGR'])} a year",
             "Multi-asset, Robustness"],
            ["Out of sample, the volatility rule cut drawdowns everywhere but did not raise Sharpe.",
             "Pre-registered test on 14 international country funds (rules committed before the data was downloaded).",
             f"Drawdown smaller in {int(pool['dd_better'])} of {int(pool['n_funds'])}; Sharpe change "
             f"{pool['avg_sharpe_diff']:+.3f} (95%: {pool['lo95']:+.3f} to {pool['hi95']:+.3f})", "International"],
            B, ["Bottom line"],
            [f"For an investor who wants a smoother ride, the evidence supports a diversified core (equal-weight stocks, "
             f"bonds and gold) with optional volatility scaling, priced at roughly {n['intl_cagr_cost'] * 100:.1f} points "
             f"of return a year for about a third smaller crashes (average 2008-09 drawdown "
             f"{_pct(n['intl_dd08_bh'], 0)} to {_pct(n['intl_dd08_man'], 0)} across 14 countries). It does not support "
             "a claim of beating the market. This is research on historical data, not investment advice."],
            B, ["Headline comparison: 2007-01 to 2026-11, one-day execution delay, trading costs included"],
            ["Strategy", "Return per year", "Volatility", "Sharpe (excess of cash)", "Max drawdown"]]
    pts = [("Buy & hold SPY", spy), ("60/40 SPY/IEF", ma.loc["60/40 SPY/IEF"]),
           ("Equal weight SPY/IEF/GLD, 5-point band", ew),
           ("Equal weight + 200-day trend filter", rb.loc[("Equal weight + 200-day trend filter", "All")]),
           ("Risk parity, volatility-managed", ma.loc["Risk parity, managed"])]
    rows += [[k, round(float(r["CAGR"]), 4), round(float(r["Vol"]), 4), round(float(r["Sharpe"]), 3),
              round(float(r["MaxDD"]), 4)] for k, r in pts]
    rows += [B, ["Tab guide"], ["Tab", "Phase", "What it shows"],
             ["Catalogue", "All", "Every idea tested, including the ones that failed, with its result and verdict"],
             ["Forecast summary", "1-2", "Scoreboard of the 8 forecasting models, best model by period, charts"],
             ["Scores", "1-2", "QLIKE and MSE overall and by sub-period"],
             ["Significance", "1-2", "Diebold-Mariano tests between models"],
             ["Coefficients", "1-2", "Fitted HAR coefficients on the full sample"],
             ["Slope model", "4", "VIX term-structure slope: the most accurate forecaster, with a 2016+ holdout"],
             ["Vol signals", "3", "Long/short VXX/SVXY signals from the forecasts (rejected)"],
             ["Vol-managed SPY", "5", "Scaling SPY exposure by forecast volatility"],
             ["Higher returns", "6", "Leverage dial and short-volatility strategies"],
             ["Multi-asset", "7", "SPY/IEF/GLD: 60/40, equal weight, risk parity"],
             ["Robustness", "8", "Realistic rebalancing, trend filter, sub-periods, deflated Sharpe"],
             ["International", "9", "Pre-registered out-of-sample test on 14 country funds"],
             ["Methods", "All", "Safeguards, limitations and how to reproduce"],
             ["Daily forecasts", "1-2", "Last 252 days of forecasts vs actual volatility (refreshed each weekday)"],
             ["Run info", "1-2", "Data ranges and settings of the latest run"],
             B, [f"Last built {date.today().isoformat()} from outputs/*.csv (python -m src.sheet_overview)."]]
    return rows


def catalogue(n) -> list:
    p1, ma, rb, pool = n["p1"], n["ma"], n["rb"], n["intl_pool"]
    s5, s10 = n["p3_5y"], n["p3_10y"]
    ew = rb.loc[("Equal weight, 5-point band", "All")]
    rows = [["Catalogue of everything tried"],
            ["Every idea tested in the project, including the ones that failed. Lag 1 = orders placed one trading day "
             "after the signal."], B,
            ["Phase", "Question", "What was tested", "Result", "Verdict", "Tab"],
            ["1", "Can a simple model forecast next-day S&P 500 variance?",
             "Naive, rolling mean, HAR-RV (levels and log); walk-forward from 2004",
             f"HAR beats both baselines (DM {n['dm_min']:.0f} to {n['dm_max']:.0f})",
             "Kept: HAR-RV (log) is the base model", "Forecast summary"],
            ["2", "Does information known at the close help?", "VIX as implied variance; scheduled-FOMC-tomorrow flag",
             f"VIX: QLIKE {p1['HAR-RV (log)']:.4f} to {p1['HAR+VIX (log)']:.4f}; FOMC flag: to "
             f"{p1['HAR+VIX+FOMC (log)']:.4f} (p = 0.003)", "Kept (log form)", "Significance"],
            ["2", "Levels HAR with VIX", "Same inputs, unlogged",
             "About 5% of forecasts hit the 1e-8 floor in calm markets; QLIKE explodes",
             "Rejected: log form only", "Scores"],
            ["3", "Can the forecasts drive long/short volatility trades?",
             "Walk-forward logistic signals on VXX/SVXY, confidence-based sizing, 5 bps costs, lag 1",
             f"5 years: $10k to ${s5.loc['Strategy (signals)', 'final_value'] / 1e3:.1f}k vs "
             f"${s5.loc['Buy & hold SPY', 'final_value'] / 1e3:.1f}k SPY. 10 years: "
             f"${s10.loc['Strategy (signals)', 'final_value'] / 1e3:.1f}k vs "
             f"${s10.loc['Buy & hold SPY', 'final_value'] / 1e3:.1f}k SPY, "
             f"{_pct(s10.loc['Strategy (signals)', 'max_drawdown'], 0)} max drawdown", "Rejected", "Vol signals"],
            ["4", "Does the VIX term structure help?", "ln(VIX3M / VIX); chosen on 2010-2015, checked on 2016+",
             f"QLIKE {n['q_full']:.3f} to {n['q_slope']:.3f}; DM {n['dm_slope']:.2f}", "Kept: best forecaster",
             "Slope model"],
            ["4", "Other candidate inputs", "Overnight gap, daily VIX change, down-day flag, SKEW, VVIX, day of week, "
             "options expiry, fitting on QLIKE, VIX9D slope", "No holdout gain (VIX9D only on a short sample)",
             "Rejected", "Slope model"],
            ["5", "Do the forecasts make money?", "SPY weight = min(1, typical vol / forecast vol), each model",
             f"Max drawdown {_pct(n['vm0_dd_bh'], 0)} to {_pct(n['vm0_dd_har'], 0)}; Sharpe "
             f"{n['vm0_diff_lo']:+.2f} to {n['vm0_diff_hi']:+.2f}, all intervals include zero",
             "Risk reduction only", "Vol-managed SPY"],
            ["6", "Can we earn more than SPY?", "Managed SPY with leverage dial k = 1.25 and 1.5",
             f"k = 1.5: {_pct(n['hr_k15_cagr'])} vs {_pct(n['hr_spy_cagr'])} a year, smaller drawdown, same Sharpe",
             "More return for more risk; needs leverage", "Higher returns"],
            ["6", "Short-volatility carry", "SVXY when VIX3M > VIX; 50/50 with SPY",
             "Good with same-close orders; below SPY with a one-day delay", "Rejected", "Higher returns"],
            ["7", "Does diversifying beat timing?", "SPY/IEF/GLD: 60/40, equal weight, risk parity, managed, levered",
             f"Equal weight Sharpe {ma.loc['Equal weight SPY/IEF/GLD', 'Sharpe']:.2f} vs SPY "
             f"{ma.loc['Buy & hold SPY', 'Sharpe']:.2f}; risk parity no better than equal weight",
             "Kept: equal weight", "Multi-asset"],
            ["8", "Is equal weight robust?", "Realistic drift and costs; daily/monthly/5-point-band rebalancing; "
             "200-day trend filter; sub-periods; deflated Sharpe",
             f"Sharpe about {ew['Sharpe']:.2f} under every rule; edge concentrated in 2007-2012; trend filter halves "
             "drawdown but not Sharpe", "Kept: 5-point band", "Robustness"],
            ["9", "Does the volatility rule work abroad? (pre-registered)",
             "15 iShares country funds (14 usable), same rule, 10 bps costs",
             f"Drawdown smaller {int(pool['dd_better'])} of {int(pool['n_funds'])}; Sharpe "
             f"{pool['avg_sharpe_diff']:+.3f} (95%: {pool['lo95']:+.3f} to {pool['hi95']:+.3f})",
             "Crash insurance only", "International"],
            B, [f"Last built {date.today().isoformat()} from outputs/*.csv (python -m src.sheet_overview)."]]
    return rows


def vol_signals(n) -> list:
    head = ["Strategy", "Final value of $10,000", "Total return", "Return per year", "Volatility", "Sharpe",
            "Max drawdown", "Worst day"]
    cols = ["final_value", "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day"]

    def table(df):
        return [head] + [[i] + [round(float(df.loc[i, c]), 4) for c in cols] for i in df.index]

    rows = [["Phase 3: long/short volatility signals (rejected)"],
            ["Tradable stand-ins: VXX (long volatility) and SVXY (short volatility). A walk-forward logistic regression "
             "on the gap between VIX and our forecast, VIX vs its 1-year mean, VIX momentum, the VIX/VIX3M term "
             "structure and the FOMC flag gives P(volatility rises). Sizing fixed in advance: full size at 65%+ "
             "confidence, half size at 55-65%, flat below. 5 bps costs, one-day execution delay."],
            B, ["Last 5 years (to 2026-11)"]] + table(n["p3_5y"])
    rows += [B, ["Last 10 years (to 2026-11)"]] + table(n["p3_10y"])
    cal = n["p3_cal"]
    rows += [B, ["Were the stated confidences honest? (10 years)"],
             ["Signal", "Confidence bucket", "Days", "Stated confidence", "Actual hit rate"]]
    rows += [[r.signal, f"'{r.bucket}", int(r.days), round(r.stated_confidence, 4), round(r.actual_hit_rate, 4)]
             for r in cal.itertuples(index=False)]
    yr = n["p3_year"]
    rows += [B, ["Calendar-year returns (10 years)"], ["Year"] + list(yr.columns)]
    rows += [[f"'{int(y)}"] + [round(float(v), 4) for v in yr.loc[y]] for y in yr.index]
    rows += [B, ["Reading it: the model was almost always short volatility, and its stated confidence was roughly "
                 "honest. It still lost badly to simply holding SPY over 10 years, and the February 2018 volatility "
                 "spike dominates. VXX only exists from 2018 (earlier days use 2x UVXY returns) and SVXY changed from "
                 "-1x to -0.5x in 2018, so these numbers mix regimes. Verdict: rejected."],
             [f"Last built {date.today().isoformat()}. Source: outputs/backtest_*_lag1*.csv (python -m src.backtest)."]]
    return rows


def methods() -> list:
    return [["Methods, safeguards and limitations"], B,
            ["Safeguards"], ["Item", "Detail"],
            ["Variance proxy", "Parkinson range estimator RV = (ln(High/Low))^2 / (4 ln 2): uses the whole day's range, "
             "far less noisy than a squared close-to-close return."],
            ["Walk-forward fitting", "Expanding window (1,000 days for the S&P index, 500 for ETFs), refit every 21 "
             "days. A forecast made at day t never uses data after day t."],
            ["No-look-ahead tests", "Automated tests change future prices, VIX values and FOMC dates and confirm earlier "
             "forecasts and weights do not move (58 tests in total)."],
            ["Scoring", "QLIKE (standard loss for variance forecasts) and MSE; Diebold-Mariano tests with Newey-West "
             "standard errors."],
            ["Execution delay", "Orders execute one trading day after the signal (lag 1) in every headline result."],
            ["Costs", "2 bps per unit of turnover for SPY/IEF, 3 bps GLD, 5-10 bps volatility products and country "
             "funds; cash earns T-bills, borrowing costs T-bills + 0.5%; weights drift between rebalances (Phase 8+)."],
            ["Uncertainty", "95% moving-block bootstrap intervals (20-day blocks, 2,000 draws) on every Sharpe "
             "difference; Phase 9 resamples the same dates for every fund."],
            ["Multiple testing", "Deflated Sharpe ratio (Bailey and Lopez de Prado 2014) against the ~20 strategy "
             "variants tried, and against a conservative 50."],
            ["Pre-registration", "Phase 9's rules were committed to git (1e353f7) before its data was downloaded; "
             "results in a separate commit (1187584)."],
            B, ["Limitations"], ["Item", "Detail"],
            ["Proxy, not true realized variance", "Intraday data is not used; Parkinson ignores overnight gaps."],
            ["One history", "20-25 years with a handful of crises; 20 years cannot confirm a 0.2 Sharpe advantage."],
            ["Design choices", "Training windows, refit interval, HAR windows and the 1e-8 floor were fixed in advance "
             "but other choices could change numbers."],
            ["Approximate DM tests", "Models are nested and several tests are not adjusted for multiple testing."],
            ["Backtest assumptions", "Simplified costs and borrowing; Phases 7-8 rely on a period when bonds mostly "
             "hedged stocks (until 2022) and gold rallied after 2024."],
            ["Many strategies tried", "About 20 variants; the deflated Sharpe and Phase 9 address this, but the 2025+ "
             "final check is under two years long."],
            ["Not tested live", "All results are historical."],
            B, ["How to reproduce"], ["Step", "Command"],
            ["Install", "pip install -r requirements.txt"], ["Tests", "python -m pytest"],
            ["Phases 1-2", "python -m src.run_phase1 --refresh"], ["Phase 3", "python -m src.backtest --lag 1 --years 10"],
            ["Phase 4", "python -m src.run_phase4"], ["Phase 5", "python -m src.volmanaged --lag 1"],
            ["Phase 6", "python -m src.higher_return --lag 1"], ["Phase 7", "python -m src.multiasset --lag 1"],
            ["Phase 8", "python -m src.robustness --lag 1"], ["Phase 9", "python -m src.international"],
            ["Sheet front tabs", "python -m src.sheet_overview"],
            B, [f"Last built {date.today().isoformat()}."]]


TABS = {"Overview": overview, "Catalogue": catalogue, "Vol signals": vol_signals}


def main() -> None:
    n = load()
    tabs = [{"range": f"'{name}'!A1", "data": fn(n)} for name, fn in TABS.items()]
    tabs.append({"range": "'Methods'!A1", "data": methods()})
    with open(OUT / "overview_sheet_payload.json", "w") as f:
        json.dump({"tabs": tabs}, f)
    for t in tabs:
        print(t["range"], len(t["data"]), "rows x", max(len(r) for r in t["data"]), "cols")


if __name__ == "__main__":
    main()

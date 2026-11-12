"""Figures for the README's key findings. Reads only the result CSVs in outputs/ (run the phases first).

Usage: python -m src.make_charts   -> outputs/figures/*.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import ROOT

OUT = ROOT / "outputs"
FIG = OUT / "figures"

# Reference palette (light mode): two categorical slots plus neutral chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK,
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy():
    """1. Forecast accuracy ladder (same 4,112 rows, lower QLIKE is better)."""
    d = pd.read_csv(OUT / "slope_results.csv").sort_values("QLIKE", ascending=False)
    labels = {"Naive": "Naive (tomorrow = today)", "HAR-RV (log)": "HAR-RV",
              "HAR+VIX+FOMC (log)": "HAR + VIX + FOMC", "HAR+VIX+FOMC+Slope (log)": "HAR + VIX + FOMC + slope"}
    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(d))
    ax.barh(y, d["QLIKE"], height=0.55, color=BLUE)
    for yi, v in zip(y, d["QLIKE"]):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [labels.get(m, m) for m in d["model"]], color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, d["QLIKE"].max() * 1.15)
    ax.set_xlabel("QLIKE loss, next-day S&P 500 variance (lower is better)")
    ax.set_title("Each added input made the forecast more accurate")
    fig.text(0.0, -0.14, "Walk-forward, 2010-07 to 2026-11 (4,112 days). Each step's gain is significant at p < 0.01 "
             "(Diebold-Mariano; the VIX and FOMC steps on the longer 2004-2026 sample).", color=MUTED, fontsize=8)
    _save(fig, "1_forecast_accuracy.png")


def fig_accuracy_vs_value():
    """2. Better forecasts did not mean a better managed portfolio."""
    q = pd.read_csv(OUT / "slope_results.csv").set_index("model")["QLIKE"]
    s = pd.read_csv(OUT / "volmanaged_stats_lag1.csv")
    s = s[s["sample"] == "All"].set_index("model")["Sharpe"]
    names = {"Naive": "Naive", "HAR-RV (log)": "HAR-RV", "HAR+VIX+FOMC (log)": "HAR+VIX+FOMC",
             "HAR+VIX+FOMC+Slope (log)": "+ slope (most accurate)"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m, lab in names.items():
        x, yv = q[m], s[f"Managed: {m}"]
        ax.scatter(x, yv, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(lab, (x, yv), xytext=(7, 4), textcoords="offset points", fontsize=9, color=INK2)
    bh = s["Buy & hold SPY"]
    ax.axhline(bh, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.31, bh, "Buy & hold SPY ", va="bottom", ha="left", fontsize=9, color=MUTED)
    ax.set_xlim(0.80, 0.25)  # reversed: more accurate to the right
    ax.set_ylim(0.70, 0.90)  # wide enough that small differences look small
    ax.set_xlabel("Forecast loss, QLIKE (more accurate to the right)")
    ax.set_ylabel("Sharpe of the managed SPY portfolio")
    ax.set_title("Past a basic model, better forecasts did not make a better portfolio")
    fig.text(0.0, -0.1, "Volatility-managed SPY, one-day execution delay, 2010-08 to 2026-11. Sharpe differences vs "
             "buy & hold are all inside their 95% bootstrap intervals.", color=MUTED, fontsize=8)
    _save(fig, "2_accuracy_vs_value.png")


def fig_price_of_protection():
    """3. Return given up vs worst loss avoided, 2007-2026 (one-day delay, costs included)."""
    r = pd.read_csv(OUT / "robustness_stats_lag1.csv")
    r = r[r["sample"] == "All"].set_index("model")
    m = pd.read_csv(OUT / "multiasset_stats_lag1.csv")
    m = m[m["sample"] == "All"].set_index("model")
    # (label, row, label offset in points, horizontal alignment)
    pts = [("Buy & hold SPY", r.loc["Buy & hold SPY"], (-8, -6), "right"),
           ("60/40 SPY/IEF", m.loc["60/40 SPY/IEF"], (8, -6), "left"),
           ("Equal weight SPY/IEF/GLD\n(5-point band)", r.loc["Equal weight, 5-point band"], (-8, 30), "left"),
           ("Equal weight + trend filter", r.loc["Equal weight + 200-day trend filter"], (8, 26), "left"),
           ("Risk parity, vol-managed", m.loc["Risk parity, managed"], (8, -6), "left")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, row, off, ha in pts:
        is_pick = lab.startswith("Equal weight SPY")
        ax.scatter(-row["MaxDD"] * 100, row["CAGR"] * 100, s=90 if is_pick else 60,
                   color=ORANGE if is_pick else BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{lab}\nSharpe {row['Sharpe']:.2f}", (-row["MaxDD"] * 100, row["CAGR"] * 100),
                    xytext=off, textcoords="offset points", fontsize=8.5, color=INK2, va="top", ha=ha)
    ax.set_xlim(5, 62); ax.set_ylim(4, 12.5)
    ax.set_xlabel("Worst peak-to-trough loss, % (smaller is better)")
    ax.set_ylabel("Return per year, %")
    ax.set_title("The price of protection: a little return buys much smaller crashes")
    fig.text(0.0, -0.06, "2007-01 to 2026-11, one-day execution delay, trading costs included. Orange = the most "
             "robust option found (Phase 8).", color=MUTED, fontsize=8)
    _save(fig, "3_price_of_protection.png")


def fig_international():
    """4. Out-of-sample: drawdown cut in every country, Sharpe unchanged."""
    t = pd.read_csv(OUT / "international_by_fund.csv")
    t = t[t["excluded"].fillna("").eq("") & t["fund"].ne("SPY")].copy()
    t = t.sort_values("maxdd_bh")
    lab = [f"{c} ({f})" for c, f in zip(t["country"], t["fund"])]
    y = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True, gridspec_kw={"wspace": 0.08})
    a1.hlines(y, -t["maxdd_bh"] * 100, -t["maxdd_man"] * 100, color=AXIS, linewidth=2)
    a1.scatter(-t["maxdd_bh"] * 100, y, s=50, color=MUTED, edgecolor=SURFACE, linewidth=2, zorder=3, label="Buy & hold")
    a1.scatter(-t["maxdd_man"] * 100, y, s=50, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="Volatility-managed")
    a1.set_yticks(y, lab, color=INK2); a1.tick_params(axis="y", length=0); a1.grid(axis="y", visible=False)
    a1.set_xlabel("Maximum drawdown, %"); a1.set_xlim(30, 80)
    a1.set_title("Drawdown smaller in 14 of 14")
    a1.legend(loc="upper right", frameon=False, fontsize=9)
    a2.axvline(0, color=AXIS, linewidth=1)
    a2.hlines(y, t["lo95"], t["hi95"], color=BLUE, alpha=0.35, linewidth=3)
    a2.scatter(t["sharpe_diff"], y, s=40, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    a2.tick_params(axis="y", length=0); a2.grid(axis="y", visible=False)
    a2.set_xlabel("Sharpe change vs buy & hold (dot) and 95% interval")
    a2.set_title("Sharpe: no gain (average −0.02)")
    fig.suptitle("Out-of-sample test on 14 international funds (rules committed before the data)",
                 x=0.0, ha="left", fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.0, -0.03, "iShares country ETFs, 2002-2026, one-day delay, 10 bps costs. Belgium excluded by the "
             "pre-registered data rule.", color=MUTED, fontsize=8)
    _save(fig, "4_international.png")


def main():
    fig_accuracy(); fig_accuracy_vs_value(); fig_price_of_protection(); fig_international()
    print(f"Saved 4 figures to {FIG}")


if __name__ == "__main__":
    main()

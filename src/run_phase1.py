"""Run the whole project end to end:  python -m src.run_phase1  [--refresh]

(The file keeps its Phase 1 name because the scheduled Sheet refresh calls it;
it now also runs the Phase 2 models that add VIX and the FOMC-tomorrow flag.)

Writes to outputs/:
    results_overall.csv, results_by_period.csv, dm_tests.csv, coefficients_full_sample.csv,
    forecast_vs_actual.png, qlike_by_period.png, sheet_payload.json
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT, load_data, load_vix
from src.evaluate import (MIN_TRAIN, REFIT_EVERY, dm_pairs, score_by_period,
                          score_overall, walk_forward)
from src.features import annualized_vol_pct, build_features
from src.models import HARLevelsModel, make_models
from src.sheet_export import build_payload, write_sheet_payload

OUT_DIR = ROOT / "outputs"
FOMC_PATH = ROOT / "data" / "events" / "fomc_decision_dates.csv"

# Significance tests: (model A, model B). A negative DM statistic means A is better.
DM_PAIRS = [
    # Phase 1: HAR against the two baselines
    ("HAR-RV (levels)", "Naive"), ("HAR-RV (levels)", "Rolling mean"),
    ("HAR-RV (log)", "Naive"), ("HAR-RV (log)", "Rolling mean"),
    # Phase 2: does each added input help? (HAR -> +VIX -> +FOMC)
    ("HAR+VIX (levels)", "HAR-RV (levels)"), ("HAR+VIX (log)", "HAR-RV (log)"),
    ("HAR+VIX+FOMC (levels)", "HAR+VIX (levels)"), ("HAR+VIX+FOMC (log)", "HAR+VIX (log)"),
]
# These two have QLIKE scores of 19 to 150 (not 0.3 to 0.5), caused by forecasts hitting
# the 1e-8 floor in calm markets. They are left out of the bar chart so the other bars stay readable.
CHART_EXCLUDED = ["HAR+VIX (levels)", "HAR+VIX+FOMC (levels)"]


def plot_forecast_vs_actual(fc: pd.DataFrame, path) -> None:
    """Annualized volatility: actual vs log-HAR forecasts with and without VIX, last 2 years."""
    recent = fc[fc.index >= fc.index.max() - pd.DateOffset(years=2)]
    # Index = origin date t; both the forecast and the actual are for day t+1,
    # so the lines are aligned with each other (x-axis shows the origin date).
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(recent.index, annualized_vol_pct(recent["actual"]), color="#c3cad2",
            lw=1, label="Actual (Parkinson proxy)")
    ax.plot(recent.index, annualized_vol_pct(recent["HAR-RV (log)"]), color="#1f5fbf",
            lw=1.6, label="HAR-RV (log) forecast")
    ax.plot(recent.index, annualized_vol_pct(recent["HAR+VIX (log)"]), color="#d9730d",
            lw=1.6, label="HAR+VIX (log) forecast")
    ax.set_title("S&P 500 next-day volatility: actual vs HAR forecasts, without and with VIX (last 2 years)")
    ax.set_xlabel("Forecast origin date (forecast is for the following trading day)")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qlike_by_period(by_period: pd.DataFrame, path) -> None:
    """Grouped bar chart: QLIKE per model per sub-period (lower is better)."""
    table = by_period.pivot(index="period", columns="model", values="QLIKE")
    table = table.loc[by_period["period"].unique()]  # keep chronological order
    order = [m for m in by_period["model"].unique() if m not in CHART_EXCLUDED]
    table = table[order]
    ax = table.plot(kind="bar", figsize=(12, 5.5), width=0.85, rot=0)
    ax.set_title("QLIKE loss by model and sub-period (lower is better)")
    ax.set_xlabel("Sub-period (by forecast origin date)")
    ax.set_ylabel("Average QLIKE")
    ax.set_ylim(0, table.to_numpy().max() * 1.3)  # headroom so the legend never covers a bar
    ax.legend(title="Model", loc="upper center", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.text(0.01, 0.01, "Not shown: HAR+VIX (levels) and HAR+VIX+FOMC (levels). Their QLIKE is 19 to 150 "
                   "because forecasts hit the 1e-8 floor in calm markets (see results_by_period.csv).",
                   fontsize=8, color="#555555")
    ax.figure.tight_layout(rect=(0, 0.03, 1, 1))
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def coefficient_table(features: pd.DataFrame) -> pd.DataFrame:
    """Fit each HAR model once on the FULL sample and list its coefficients.

    This is for interpretation only (what weight does each input get?). It is
    in-sample, so it is NOT used for any forecast score.
    """
    rows = []
    for m in make_models(extended=True)[2:]:
        m.fit(features)
        for term, value in zip(["const"] + m.features, m.params):
            rows.append({"model": m.name, "term": term, "coefficient": float(value)})
    return pd.DataFrame(rows)


def main(refresh: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    df = load_data(refresh=refresh)
    vix = load_vix(refresh=refresh)
    fomc_dates = pd.read_csv(FOMC_PATH, parse_dates=["date"])["date"]
    features = build_features(df, vix, fomc_dates)
    print(f"\nFeature rows: {len(features)}  |  min training window: {MIN_TRAIN} days  "
          f"|  refit every {REFIT_EVERY} days")

    models = make_models(extended=True)
    fc = walk_forward(features, models)
    print(f"Out-of-sample forecasts: {len(fc)} "
          f"({fc.index.min().date()} to {fc.index.max().date()})")

    for m in models:
        if isinstance(m, HARLevelsModel):
            print(f"{m.name} forecasts floored at 1e-8: {m.n_floored} of {m.n_predictions} "
                  f"({100 * m.n_floored / m.n_predictions:.2f}%)")

    overall = score_overall(fc)
    by_period = score_by_period(fc)
    dm = dm_pairs(fc, DM_PAIRS)

    overall.to_csv(OUT_DIR / "results_overall.csv")
    by_period.to_csv(OUT_DIR / "results_by_period.csv", index=False)
    dm.to_csv(OUT_DIR / "dm_tests.csv", index=False)
    coefficient_table(features).to_csv(OUT_DIR / "coefficients_full_sample.csv", index=False)
    plot_forecast_vs_actual(fc, OUT_DIR / "forecast_vs_actual.png")
    plot_qlike_by_period(by_period, OUT_DIR / "qlike_by_period.png")
    # Rows laid out for the Google Sheet (used by the scheduled refresh task).
    write_sheet_payload(build_payload(fc, overall, by_period, dm, df.index),
                        OUT_DIR / "sheet_payload.json")

    pd.set_option("display.width", 160)
    fmt = {"MSE": "{:.3e}".format, "QLIKE": "{:.4f}".format}
    print("\n=== Overall out-of-sample scores (lower is better) ===")
    print(overall.to_string(formatters=fmt))
    print("\n=== QLIKE by sub-period ===")
    print(by_period.pivot(index="model", columns="period", values="QLIKE")
          [list(by_period["period"].unique())].round(4).to_string())
    print("\n=== Diebold-Mariano tests on QLIKE (A vs B; negative = A better) ===")
    print(dm.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nFiles written to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pipeline (Phase 1 and Phase 2 models).")
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    main(refresh=parser.parse_args().refresh)

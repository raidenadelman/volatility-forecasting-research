"""Run all of Phase 1 end to end:  python -m src.run_phase1  [--refresh]

Writes to outputs/:
    results_overall.csv, results_by_period.csv, dm_tests.csv,
    forecast_vs_actual.png, qlike_by_period.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT, load_data
from src.evaluate import (MIN_TRAIN, REFIT_EVERY, dm_table, score_by_period,
                          score_overall, walk_forward)
from src.features import annualized_vol_pct, build_features
from src.models import make_models

OUT_DIR = ROOT / "outputs"


def plot_forecast_vs_actual(fc: pd.DataFrame, path) -> None:
    """Annualized volatility: actual vs HAR-RV (levels) forecast, last 2 years."""
    recent = fc[fc.index >= fc.index.max() - pd.DateOffset(years=2)]
    # Index = origin date t; both the forecast and the actual are for day t+1,
    # so the two lines are aligned with each other (x-axis shows the origin date).
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(recent.index, annualized_vol_pct(recent["actual"]), color="#9aa5b1",
            lw=1, label="Actual (Parkinson proxy)")
    ax.plot(recent.index, annualized_vol_pct(recent["HAR-RV (levels)"]), color="#1f5fbf",
            lw=1.8, label="HAR-RV (levels) forecast")
    ax.set_title("S&P 500 next-day volatility: actual vs HAR-RV forecast (last 2 years)")
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
    ax = table.plot(kind="bar", figsize=(10, 5), width=0.8, rot=0)
    ax.set_title("QLIKE loss by model and sub-period (lower is better)")
    ax.set_xlabel("Sub-period (by forecast origin date)")
    ax.set_ylabel("Average QLIKE")
    ax.set_ylim(0, table.to_numpy().max() * 1.25)  # headroom so the legend never covers a bar
    ax.legend(title="Model", loc="upper center", ncol=len(table.columns))
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def main(refresh: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    df = load_data(refresh=refresh)
    features = build_features(df)
    print(f"\nFeature rows: {len(features)}  |  min training window: {MIN_TRAIN} days  "
          f"|  refit every {REFIT_EVERY} days")

    models = make_models()
    fc = walk_forward(features, models)
    print(f"Out-of-sample forecasts: {len(fc)} "
          f"({fc.index.min().date()} to {fc.index.max().date()})")

    har_levels = models[2]
    print(f"HAR-RV (levels) forecasts floored at 1e-8: {har_levels.n_floored} of "
          f"{har_levels.n_predictions} ({100 * har_levels.n_floored / har_levels.n_predictions:.2f}%)")

    overall = score_overall(fc)
    by_period = score_by_period(fc)
    dm = dm_table(fc)

    overall.to_csv(OUT_DIR / "results_overall.csv")
    by_period.to_csv(OUT_DIR / "results_by_period.csv", index=False)
    dm.to_csv(OUT_DIR / "dm_tests.csv", index=False)
    plot_forecast_vs_actual(fc, OUT_DIR / "forecast_vs_actual.png")
    plot_qlike_by_period(by_period, OUT_DIR / "qlike_by_period.png")

    pd.set_option("display.width", 120)
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
    parser = argparse.ArgumentParser(description="Run Phase 1 end to end.")
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    main(refresh=parser.parse_args().refresh)

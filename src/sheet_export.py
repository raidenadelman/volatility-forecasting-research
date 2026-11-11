"""Turn the results into the exact rows the Google Sheet expects.

`run_phase1` calls `write_sheet_payload` at the end of every run, producing
outputs/sheet_payload.json. The scheduled refresh task then copies each block
into the Sheet range named in RANGES below. This module does no Google API
calls itself (that needs credentials, which this project does not use).
"""

import json
from datetime import date

import pandas as pd

from src.features import annualized_vol_pct

# Row/column order used in the Sheet. The two levels-with-VIX models go last so the
# Sheet's charts can use the first six rows/columns and leave them out (their QLIKE
# of 19 to 150 would flatten every other bar).
SHEET_ORDER = ["Naive", "Rolling mean", "HAR-RV (levels)", "HAR-RV (log)",
               "HAR+VIX (log)", "HAR+VIX+FOMC (log)",
               "HAR+VIX (levels)", "HAR+VIX+FOMC (levels)"]
COEF_TERMS = ["const", "rv_d", "rv_w", "rv_m", "vix_var", "fomc_next"]
# Period labels that look like numbers/dates get a leading apostrophe so Sheets
# keeps them as text. "*" marks the period whose start is cut off (see Sheet note).
PERIOD_LABELS = {"2000-2007": "'2000-2007*", "2008-2009": "'2008-2009",
                 "2010-2019": "'2010-2019", "2020-present": "2020-present"}
DAILY_ROWS = 252  # about one trading year shown in the Sheet

# Where each payload block goes in the Sheet (top-left cell of each block).
RANGES = {
    "scores_overall": "Scores!A3",
    "scores_qlike": "Scores!A14",
    "scores_mse": "Scores!A21",
    "significance": "Significance!A2",
    "coefficients": "Coefficients!A2",
    "daily": "'Daily forecasts'!A1",
    "run_info": "'Run info'!B2",
}


def build_payload(fc: pd.DataFrame, overall: pd.DataFrame, by_period: pd.DataFrame,
                  dm: pd.DataFrame, price_index: pd.DatetimeIndex,
                  floor_rates: dict, coefficients: pd.DataFrame) -> dict:
    """Return {block name: 2D list of rows} matching RANGES.

    floor_rates:  {model name: share of forecasts floored} for the levels models.
    coefficients: long table with columns model, term, coefficient (full-sample fits).
    """
    scores_overall = [[m, float(overall.loc[m, "MSE"]), round(float(overall.loc[m, "QLIKE"]), 4),
                       floor_rates.get(m)]  # None (blank cell) for models that cannot be floored
                      for m in SHEET_ORDER]

    qlike_rows, mse_rows = [], []
    for period, label in PERIOD_LABELS.items():
        sub = by_period[by_period["period"] == period].set_index("model")
        qlike_rows.append([label] + [round(float(sub.loc[m, "QLIKE"]), 4) for m in SHEET_ORDER]
                          + [int(sub["n_days"].iloc[0])])
        mse_rows.append([label] + [float(sub.loc[m, "MSE"]) for m in SHEET_ORDER])

    significance = [[r.model_A, r.model_B, round(r.mean_loss_diff, 4), round(r.DM_stat, 2),
                     float(f"{r.p_value:.3g}"), int(r.lags), int(r.n)]
                    for r in dm.itertuples(index=False)]

    wide = coefficients.pivot(index="model", columns="term", values="coefficient")
    coef_rows = []
    for m in SHEET_ORDER[2:]:  # the six regression models
        coef_rows.append([m] + [None if pd.isna(wide.loc[m].get(t)) else float(f"{wide.loc[m][t]:.6g}")
                                for t in COEF_TERMS])

    daily = [["Forecast origin date", "Actual vol (%, annualized)"] + [f"{m} (%)" for m in SHEET_ORDER]]
    for d, row in fc.tail(DAILY_ROWS).iterrows():
        daily.append([str(d.date())] + [round(float(annualized_vol_pct(row[c])), 2)
                                        for c in ["actual"] + SHEET_ORDER])

    run_info = [["'" + date.today().isoformat()],  # Last refreshed (local date of the run)
                ["Yahoo Finance via yfinance, tickers ^GSPC and ^VIX"],
                [f"{price_index.min().date()} to {price_index.max().date()}"],
                [f"{fc.index.min().date()} to {fc.index.max().date()}"],
                [len(fc)]]

    return {"scores_overall": scores_overall, "scores_qlike": qlike_rows,
            "scores_mse": mse_rows, "significance": significance,
            "coefficients": coef_rows, "daily": daily, "run_info": run_info}


def write_sheet_payload(payload: dict, path) -> None:
    """Save the payload (plus the target ranges) as JSON for the refresh task."""
    with open(path, "w") as f:
        json.dump({"ranges": RANGES, "data": payload}, f)


SLOPE_RANGE = "'Slope model'!A1"


def build_slope_payload(overall: pd.DataFrame, dm_all: pd.DataFrame, dm_hold: pd.DataFrame,
                        fc: pd.DataFrame) -> list:
    """Rows for the Sheet's "Slope model" tab (18 rows, 6 columns, written at SLOPE_RANGE)."""
    blank = [""]
    rows = [["Phase 4: VIX term-structure slope, ln(VIX3M / VIX)"],
            [f"Same rows for every model: {fc.index.min().date()} to {fc.index.max().date()} "
             f"({len(fc):,} forecasts). VIX3M only starts 2006-07, so this is shorter than the "
             "main 8-model run. Lower QLIKE is better."],
            blank,
            ["Model", "MSE", "QLIKE (all)", "QLIKE 2010-2015 (selection)", "QLIKE 2016+ (holdout)"]]
    for m, r in overall.iterrows():
        rows.append([m, float(r["MSE"]), round(float(r["QLIKE"]), 4),
                     round(float(r["QLIKE 2010-2015"]), 4), round(float(r["QLIKE 2016+ (holdout)"]), 4)])
    rows += [blank, ["Diebold-Mariano test on QLIKE (negative = A better)"],
             ["Model A", "Model B", "Sample", "Mean loss diff", "DM stat", "p-value"]]
    for label, dm in [("All", dm_all), ("2016+ only", dm_hold)]:
        for r in dm.itertuples(index=False):
            rows.append([r.model_A, r.model_B, label, round(r.mean_loss_diff, 4),
                         round(r.DM_stat, 2), float(f"{r.p_value:.3g}")])
    rows += [blank,
             ["Tested and rejected (no gain in the 2016+ holdout): overnight gap, daily VIX change, "
              "down-day flag, SKEW, VVIX, day of week, options-expiry flag, fitting directly on QLIKE. "
              "VIX9D slope: small gain on a shorter sample (2015+), not added."],
             [f"Last refreshed {date.today().isoformat()}. Source: outputs/slope_results.csv, "
              "outputs/slope_dm_tests.csv (python -m src.run_phase4)."]]
    return rows


def write_slope_payload(rows: list, path) -> None:
    with open(path, "w") as f:
        json.dump({"range": SLOPE_RANGE, "data": rows}, f)

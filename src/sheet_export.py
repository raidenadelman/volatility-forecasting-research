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

"""Turn the Phase 1 results into the exact rows the Google Sheet expects.

`run_phase1` calls `write_sheet_payload` at the end of every run, producing
outputs/sheet_payload.json. The scheduled refresh task then copies each block
into the Sheet range named in RANGES below. This module does no Google API
calls itself (that needs credentials, which this project does not use).
"""

import json
from datetime import date

import pandas as pd

from src.features import annualized_vol_pct

MODEL_ORDER = ["Naive", "Rolling mean", "HAR-RV (levels)", "HAR-RV (log)"]
# Period labels that look like numbers/dates get a leading apostrophe so Sheets
# keeps them as text. "*" marks the period whose start is cut off (see Sheet note).
PERIOD_LABELS = {"2000-2007": "'2000-2007*", "2008-2009": "'2008-2009",
                 "2010-2019": "'2010-2019", "2020-present": "2020-present"}
DAILY_ROWS = 252  # about one trading year shown in the Sheet

# Where each payload block goes in the Sheet (top-left cell of each block).
RANGES = {
    "scores_overall": "Scores!A3",
    "scores_qlike": "Scores!A10",
    "scores_mse": "Scores!A17",
    "significance": "Significance!A2",
    "daily": "'Daily forecasts'!A1",
    "run_info": "'Run info'!B2",
}


def build_payload(fc: pd.DataFrame, overall: pd.DataFrame, by_period: pd.DataFrame,
                  dm: pd.DataFrame, price_index: pd.DatetimeIndex) -> dict:
    """Return {block name: 2D list of rows} matching RANGES."""
    scores_overall = [[m, float(overall.loc[m, "MSE"]), round(float(overall.loc[m, "QLIKE"]), 4)]
                      for m in MODEL_ORDER]

    qlike_rows, mse_rows = [], []
    for period, label in PERIOD_LABELS.items():
        sub = by_period[by_period["period"] == period].set_index("model")
        qlike_rows.append([label] + [round(float(sub.loc[m, "QLIKE"]), 4) for m in MODEL_ORDER]
                          + [int(sub["n_days"].iloc[0])])
        mse_rows.append([label] + [float(sub.loc[m, "MSE"]) for m in MODEL_ORDER])

    significance = [[r.model_A, r.model_B, round(r.mean_loss_diff, 4), round(r.DM_stat, 2),
                     float(f"{r.p_value:.3g}"), int(r.lags), int(r.n)]
                    for r in dm.itertuples(index=False)]

    daily = [["Forecast origin date", "Actual vol (%, annualized)", "Naive (%)",
              "Rolling mean (%)", "HAR-RV levels (%)", "HAR-RV log (%)"]]
    for d, row in fc.tail(DAILY_ROWS).iterrows():
        daily.append([str(d.date())] + [round(float(annualized_vol_pct(row[c])), 2)
                                        for c in ["actual"] + MODEL_ORDER])

    run_info = [["'" + date.today().isoformat()],  # Last refreshed (local date of the run)
                ["Yahoo Finance via yfinance, ticker ^GSPC"],
                [f"{price_index.min().date()} to {price_index.max().date()}"],
                [f"{fc.index.min().date()} to {fc.index.max().date()}"],
                [len(fc)]]

    return {"scores_overall": scores_overall, "scores_qlike": qlike_rows,
            "scores_mse": mse_rows, "significance": significance,
            "daily": daily, "run_info": run_info}


def write_sheet_payload(payload: dict, path) -> None:
    """Save the payload (plus the target ranges) as JSON for the refresh task."""
    with open(path, "w") as f:
        json.dump({"ranges": RANGES, "data": payload}, f)

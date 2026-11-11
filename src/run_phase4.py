"""Phase 4: does the VIX term-structure slope, ln(VIX3M / VIX), improve the forecast?

Fixed in advance (not tuned on the results): one candidate feature, the same
walk-forward settings as Phase 1, scored on the SAME rows for every model.
VIX3M starts 2006-07, so with the 1,000-day minimum window the forecasts start in 2010.
The slope was picked on 2010-2015 and then checked on 2016 onward ("holdout").

Candidates that were tested and rejected (no gain in the holdout): overnight gap,
daily VIX change, down-day flag, SKEW, VVIX, day of week, options-expiry flag,
fitting directly on QLIKE. VIX9D slope gave a small gain on a shorter sample and was not added.

Usage: python -m src.run_phase4 [--refresh]   (writes outputs/slope_sheet_payload.json for the Sheet)
"""

import argparse

import pandas as pd

from src.backtest import load_yahoo_close
from src.data import ROOT, load_data, load_vix
from src.evaluate import (MIN_TRAIN, REFIT_EVERY, dm_pairs, qlike, score_overall, walk_forward)
from src.features import build_features
from src.models import make_slope_models
from src.sheet_export import build_slope_payload, write_slope_payload

OUT_DIR = ROOT / "outputs"
FOMC_PATH = ROOT / "data" / "events" / "fomc_decision_dates.csv"
SPLIT = "2015"  # selection period ends here (inclusive)
HOLD = "2016"   # the holdout starts here
PAIRS = [("HAR+VIX+FOMC+Slope (log)", "HAR+VIX+FOMC (log)"),
         ("HAR+VIX+FOMC+Slope (log)", "HAR-RV (log)")]


def main(refresh: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data(refresh=refresh, verbose=False)
    vix = load_vix(refresh=refresh, verbose=False)
    vix3m = load_yahoo_close("^VIX3M", refresh)
    fomc = pd.read_csv(FOMC_PATH, parse_dates=["date"])["date"]
    features = build_features(df, vix, fomc, vix3m).dropna()
    fc = walk_forward(features, make_slope_models())
    a = fc["actual"]
    models = [c for c in fc.columns if c != "actual"]

    overall = score_overall(fc)
    overall["QLIKE 2010-2015"] = [qlike(a[:SPLIT], fc[m][:SPLIT]) for m in overall.index]
    overall["QLIKE 2016+ (holdout)"] = [qlike(a[HOLD:], fc[m][HOLD:]) for m in overall.index]
    dm = dm_pairs(fc, PAIRS)
    # Same test on the holdout only
    dm_hold = dm_pairs(fc[HOLD:], PAIRS)
    overall.to_csv(OUT_DIR / "slope_results.csv")
    dm.to_csv(OUT_DIR / "slope_dm_tests.csv", index=False)
    fc.to_csv(OUT_DIR / "slope_forecasts.csv")
    write_slope_payload(build_slope_payload(overall, dm, dm_hold, fc), OUT_DIR / "slope_sheet_payload.json")

    pd.set_option("display.width", 160)
    print(f"Forecasts: {len(fc)} ({fc.index.min().date()} to {fc.index.max().date()}), "
          f"min window {MIN_TRAIN}, refit every {REFIT_EVERY}")
    print(overall.to_string(float_format=lambda v: f"{v:.4g}"))
    print(dm.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("holdout only:"); print(dm_hold.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: VIX slope comparison.")
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    main(refresh=parser.parse_args().refresh)

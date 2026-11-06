"""Tests for src/sheet_export.py: the payload must match the Sheet layout."""

import pandas as pd

from src.evaluate import dm_table, score_by_period, score_overall, walk_forward
from src.features import build_features
from src.sheet_export import DAILY_ROWS, MODEL_ORDER, build_payload
from tests.test_features import make_ohlc


def test_payload_shapes_match_sheet_layout():
    df = make_ohlc(n=1500)
    fc = walk_forward(build_features(df), min_train=100, refit_every=21)
    # Spread the synthetic dates over 2005-2021 so every sub-period has data.
    fc.index = pd.date_range("2005-01-03", "2021-12-31", periods=len(fc)).normalize()
    p = build_payload(fc, score_overall(fc), score_by_period(fc), dm_table(fc), df.index)

    assert [r[0] for r in p["scores_overall"]] == MODEL_ORDER
    assert all(len(r) == 3 for r in p["scores_overall"])        # Scores!A3:C6
    assert len(p["scores_qlike"]) == 4 and all(len(r) == 6 for r in p["scores_qlike"])  # A10:F13
    assert len(p["scores_mse"]) == 4 and all(len(r) == 5 for r in p["scores_mse"])      # A17:E20
    assert len(p["significance"]) == 4 and all(len(r) == 7 for r in p["significance"])  # A2:G5
    assert len(p["daily"]) == DAILY_ROWS + 1 and all(len(r) == 6 for r in p["daily"])   # A1:F253
    assert len(p["run_info"]) == 5                                                      # B2:B6

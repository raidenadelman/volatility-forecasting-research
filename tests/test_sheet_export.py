"""Tests for src/sheet_export.py: the payload must match the Sheet layout."""

import pandas as pd

from src.evaluate import dm_pairs, score_by_period, score_overall, walk_forward
from src.features import build_features
from src.models import HARLevelsModel, make_models
from src.run_phase1 import DM_PAIRS, coefficient_table
from src.sheet_export import DAILY_ROWS, SHEET_ORDER, build_payload
from tests.test_features import make_ohlc
from tests.test_phase2 import make_vix


def test_payload_shapes_match_sheet_layout():
    df = make_ohlc(n=1500)
    feats = build_features(df, make_vix(df.index), list(df.index[30::35]))
    models = make_models(extended=True)
    fc = walk_forward(feats, models, min_train=100, refit_every=21)
    # Spread the synthetic dates over 2005-2021 so every sub-period has data.
    fc.index = pd.date_range("2005-01-03", "2021-12-31", periods=len(fc)).normalize()
    floor_rates = {m.name: m.n_floored / m.n_predictions for m in models if isinstance(m, HARLevelsModel)}
    p = build_payload(fc, score_overall(fc), score_by_period(fc), dm_pairs(fc, DM_PAIRS),
                      df.index, floor_rates, coefficient_table(feats))

    assert [r[0] for r in p["scores_overall"]] == SHEET_ORDER
    assert all(len(r) == 4 for r in p["scores_overall"])                                # Scores!A3:D10
    assert len(p["scores_qlike"]) == 4 and all(len(r) == 10 for r in p["scores_qlike"])  # A14:J17
    assert len(p["scores_mse"]) == 4 and all(len(r) == 9 for r in p["scores_mse"])       # A21:I24
    assert len(p["significance"]) == 8 and all(len(r) == 7 for r in p["significance"])   # A2:G9
    assert len(p["coefficients"]) == 6 and all(len(r) == 7 for r in p["coefficients"])   # A2:G7
    assert len(p["daily"]) == DAILY_ROWS + 1 and all(len(r) == 10 for r in p["daily"])   # A1:J253
    assert len(p["run_info"]) == 5                                                       # B2:B6
    # Only levels models can be floored; the others have a blank (None) floor rate.
    assert [r[3] is None for r in p["scores_overall"]] == [True, True, False, True, True, True, False, False]
    # The plain HAR has no VIX or FOMC coefficient (blank cells), the full model has both.
    assert p["coefficients"][0][5] is None and p["coefficients"][0][6] is None
    assert p["coefficients"][5][5] is not None and p["coefficients"][5][6] is not None

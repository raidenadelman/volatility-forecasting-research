"""Tests for src/evaluate.py (the walk-forward loop and the loss functions)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import walk_forward, qlike, qlike_loss, mse
from src.features import build_features, parkinson_variance, FEATURE_COLUMNS
from src.models import make_models
from tests.test_features import make_ohlc

MIN_TRAIN = 60
REFIT_EVERY = 10


class SpyModel:
    """A fake model that records exactly what the walk-forward loop gives it."""
    name = "Spy"

    def __init__(self):
        self.train_frames = []   # every train_df passed to fit()
        self.test_rows = []      # every row passed to predict()
        self.fit_count_at_predict = []

    def fit(self, train_df):
        self.train_frames.append(train_df.copy())
        return self

    def predict(self, test_row):
        self.test_rows.append(test_row.copy())
        self.fit_count_at_predict.append(len(self.train_frames))
        return float(test_row["rv_d"])


def test_walk_forward_never_gives_model_data_after_origin():
    df = make_ohlc(n=300)
    feats = build_features(df)
    rv = parkinson_variance(df["High"], df["Low"])
    spy = SpyModel()
    out = walk_forward(feats, [spy], min_train=MIN_TRAIN, refit_every=REFIT_EVERY)

    # 1. Each prediction row is for exactly the origin date, with NO target column.
    origin_dates = list(out.index)
    assert [r.name for r in spy.test_rows] == origin_dates
    for row in spy.test_rows:
        assert list(row.index) == FEATURE_COLUMNS

    # 2. Each fit's training data ends strictly before the origin it is first used at.
    first_origin_of_fit = origin_dates[::REFIT_EVERY]
    assert len(spy.train_frames) == len(first_origin_of_fit)
    for train, origin in zip(spy.train_frames, first_origin_of_fit):
        assert train.index.max() < origin               # origin row itself excluded
        assert len(train) >= MIN_TRAIN                  # minimum window respected
        # The last training target is RV on the origin day itself (known at the
        # end of day t) -- and nothing later than that.
        assert train["target"].iloc[-1] == pytest.approx(rv.loc[origin])

    # 3. Between refits, the model keeps the same fit (21 -> here 10 origins).
    assert spy.fit_count_at_predict[:REFIT_EVERY] == [1] * REFIT_EVERY
    assert spy.fit_count_at_predict[REFIT_EVERY] == 2

    # 4. First forecast uses exactly MIN_TRAIN training rows (expanding window).
    assert len(spy.train_frames[0]) == MIN_TRAIN
    assert len(spy.train_frames[1]) == MIN_TRAIN + REFIT_EVERY


def test_forecasts_at_t_unchanged_when_future_prices_change():
    df = make_ohlc(n=300)
    cutoff = df.index[200]

    altered = df.copy()
    future = altered.index > cutoff
    altered.loc[future, "High"] *= 1.7
    altered.loc[future, "Low"] *= 0.6

    base = walk_forward(build_features(df), make_models(),
                        min_train=MIN_TRAIN, refit_every=REFIT_EVERY)
    changed = walk_forward(build_features(altered), make_models(),
                           min_train=MIN_TRAIN, refit_every=REFIT_EVERY)

    model_cols = [c for c in base.columns if c != "actual"]
    upto = base.index[base.index <= cutoff]
    # Forecasts made on or before the cutoff are bit-for-bit identical...
    pd.testing.assert_frame_equal(base.loc[upto, model_cols], changed.loc[upto, model_cols])
    # ...while forecasts after the cutoff do differ (so the test has teeth).
    later = base.index[base.index > cutoff]
    assert not np.allclose(base.loc[later, "Naive"], changed.loc[later, "Naive"])


def test_qlike_is_zero_when_forecast_equals_actual():
    actual = np.array([1e-5, 3e-5, 2e-4])
    assert qlike(actual, actual) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(qlike_loss(actual, actual), 0.0)


def test_qlike_is_positive_otherwise():
    actual = np.array([1e-4, 1e-4])
    over = np.array([2e-4, 2e-4])      # forecast too high
    under = np.array([0.5e-4, 0.5e-4])  # forecast too low
    assert qlike(actual, over) > 0
    assert qlike(actual, under) > 0
    # Known values: ratio 0.5 -> 0.5 - ln(0.5) - 1 = 0.19315; ratio 2 -> 2 - ln 2 - 1 = 0.30685
    assert qlike(actual, over) == pytest.approx(0.5 - np.log(0.5) - 1)
    assert qlike(actual, under) == pytest.approx(2 - np.log(2) - 1)


def test_mse_basic():
    assert mse([1.0, 2.0], [1.0, 4.0]) == pytest.approx(2.0)

"""Tests for the Phase 2 additions: VIX, the FOMC-tomorrow flag, and the extended models."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import walk_forward
from src.features import (FULL_COLUMNS, VIX_COLUMNS, build_features, fomc_tomorrow_flag,
                          vix_to_daily_variance)
from src.models import (FULL_FEATURES, VARIANCE_FLOOR, HARLevelsModel, HARLogModel,
                        make_models)
from tests.test_features import make_ohlc

MIN_TRAIN = 60
REFIT_EVERY = 10


def make_vix(index, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(15 + np.abs(rng.normal(0, 4, len(index))), index=index, name="vix_close")


def test_vix_to_daily_variance_hand_computed():
    # VIX 20 -> 20% a year -> annual variance 0.04 -> daily 0.04 / 252
    assert vix_to_daily_variance(pd.Series([20.0])).iloc[0] == pytest.approx(0.04 / 252)
    assert vix_to_daily_variance(pd.Series([20.0])).iloc[0] == pytest.approx(1.5873e-4, rel=1e-3)


def test_fomc_flag_looks_at_the_next_trading_day():
    # Fri 2024-01-05, Mon 01-08, Tue 01-09, Wed 01-10, Thu 01-11
    index = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
    flag = fomc_tomorrow_flag(index, ["2024-01-08"])  # decision on Monday
    # The Friday row is flagged (next trading day is Monday, weekend skipped);
    # the decision day itself and later days are not.
    assert flag.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    # A decision on Wednesday flags Tuesday only.
    assert fomc_tomorrow_flag(index, ["2024-01-10"]).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_build_features_without_extras_is_unchanged():
    df = make_ohlc()
    assert list(build_features(df).columns) == ["rv_d", "rv_w", "rv_m", "target"]


def test_build_features_with_extras_has_expected_columns():
    df = make_ohlc()
    feats = build_features(df, make_vix(df.index), [df.index[100]])
    assert list(feats.columns) == FULL_COLUMNS + ["target"]
    assert list(build_features(df, make_vix(df.index)).columns) == VIX_COLUMNS + ["target"]
    # fomc_next is 1 on the row just before the decision day, and nowhere else
    assert feats["fomc_next"].sum() == 1.0
    assert feats.loc[df.index[99], "fomc_next"] == 1.0


def _future_altered(df, vix, fomc_dates, cutoff):
    """Same inputs, but VIX after `cutoff` and FOMC dates after the day following `cutoff` changed."""
    altered_df = df.copy()
    altered_vix = vix.copy()
    altered_vix[altered_vix.index > cutoff] *= 3.0
    day_after = df.index[df.index.get_loc(cutoff) + 1]  # known at `cutoff` (calendar is public)
    kept = [d for d in fomc_dates if d <= day_after]
    extra = [d for d in df.index[df.index > day_after][::7]]  # brand-new future events
    return altered_df, altered_vix, kept + extra


def test_features_at_t_do_not_depend_on_future_vix_or_future_fomc_dates():
    df = make_ohlc()
    vix = make_vix(df.index)
    fomc = list(df.index[30::35])
    cutoff = df.index[200]
    base = build_features(df, vix, fomc)
    adf, avix, afomc = _future_altered(df, vix, fomc, cutoff)
    changed = build_features(adf, avix, afomc)

    cols = FULL_COLUMNS
    upto = base.index[base.index <= cutoff]
    pd.testing.assert_frame_equal(base.loc[upto, cols], changed.loc[upto, cols])
    # sanity: the change really did alter later rows
    later = base.index[base.index > cutoff]
    assert not np.allclose(base.loc[later, "vix_var"], changed.loc[later, "vix_var"])


def test_forecasts_at_t_unchanged_when_future_vix_and_fomc_dates_change():
    df = make_ohlc()
    vix = make_vix(df.index)
    fomc = list(df.index[30::35])
    cutoff = df.index[200]
    adf, avix, afomc = _future_altered(df, vix, fomc, cutoff)

    base = walk_forward(build_features(df, vix, fomc), make_models(extended=True),
                        min_train=MIN_TRAIN, refit_every=REFIT_EVERY)
    changed = walk_forward(build_features(adf, avix, afomc), make_models(extended=True),
                           min_train=MIN_TRAIN, refit_every=REFIT_EVERY)
    cols = [c for c in base.columns if c != "actual"]
    upto = base.index[base.index <= cutoff]
    pd.testing.assert_frame_equal(base.loc[upto, cols], changed.loc[upto, cols])
    later = base.index[base.index > cutoff]
    assert not np.allclose(base.loc[later, "HAR+VIX (log)"], changed.loc[later, "HAR+VIX (log)"])


def test_extended_models_have_expected_names():
    names = [m.name for m in make_models(extended=True)]
    assert names == ["Naive", "Rolling mean", "HAR-RV (levels)", "HAR-RV (log)",
                     "HAR+VIX (levels)", "HAR+VIX (log)",
                     "HAR+VIX+FOMC (levels)", "HAR+VIX+FOMC (log)"]
    assert len(make_models()) == 4  # Phase 1 default is unchanged


def test_log_model_does_not_take_the_log_of_the_zero_one_flag():
    # Build data where tomorrow's variance is exactly doubled on flagged days:
    # ln(target) = ln(rv_d) + ln(2)*flag. A correct model recovers a flag
    # coefficient near ln(2) = 0.693. Logging the flag would produce -inf/NaN instead.
    rng = np.random.default_rng(3)
    n = 800
    rv = np.exp(rng.normal(-9, 0.5, n))
    flag = (rng.random(n) < 0.2).astype(float)
    noisy = lambda: rv * np.exp(rng.normal(0, 0.3, n))  # related to rv_d but not identical to it
    train = pd.DataFrame({"rv_d": rv, "rv_w": noisy(), "rv_m": noisy(), "vix_var": noisy(), "fomc_next": flag,
                          "target": rv * np.where(flag == 1, 2.0, 1.0) * np.exp(rng.normal(0, 0.01, n))})
    model = HARLogModel(FULL_FEATURES, "test").fit(train)
    assert np.all(np.isfinite(model.params))
    assert model.params[-1] == pytest.approx(np.log(2), abs=0.05)  # coefficient on fomc_next
    row = train.iloc[0]
    assert model.predict(row) > 0


def test_levels_model_floors_negative_forecasts_and_counts_them():
    # A target that FALLS as rv_d rises, with a negative intercept: for a large rv_d the
    # linear forecast is negative, which is impossible for a variance.
    x = np.linspace(1e-5, 1e-4, 200)
    train = pd.DataFrame({"rv_d": x, "rv_w": x ** 0.9, "rv_m": np.sqrt(x) * 0.01,  # not collinear with rv_d
                          "target": 2e-4 - 1.5 * x})
    model = HARLevelsModel().fit(train)
    low = model.predict(pd.Series({"rv_d": 1e-3, "rv_w": 1e-3, "rv_m": 1e-3}))   # negative raw
    high = model.predict(pd.Series({"rv_d": 5e-5, "rv_w": 5e-5, "rv_m": 5e-5}))  # positive raw
    assert low == VARIANCE_FLOOR
    assert high > VARIANCE_FLOOR
    assert model.n_predictions == 2 and model.n_floored == 1

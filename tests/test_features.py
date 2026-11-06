"""Tests for src/features.py."""

import math

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, parkinson_variance, FEATURE_COLUMNS


def make_ohlc(n=300, seed=0) -> pd.DataFrame:
    """Synthetic daily prices with High > Low on every day (no network needed)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    half_range = np.abs(rng.normal(0.01, 0.004, n)) + 0.001  # always > 0
    return pd.DataFrame(
        {"Open": close, "High": close * (1 + half_range), "Low": close * (1 - half_range),
         "Close": close},
        index=pd.bdate_range("2020-01-01", periods=n, name="Date"))


def test_parkinson_hand_computed_example():
    # High=110, Low=100  ->  ln(1.1) = 0.0953102, squared = 0.0090840,
    # divided by 4*ln(2) = 2.7725887  ->  0.0032764 (computed by hand/calculator).
    rv = parkinson_variance(pd.Series([110.0]), pd.Series([100.0]))
    assert rv.iloc[0] == pytest.approx(0.0032764, abs=1e-7)
    # Cross-check the same number from the formula written out explicitly.
    assert rv.iloc[0] == pytest.approx(math.log(1.1) ** 2 / (4 * math.log(2)))


def test_parkinson_special_cases():
    # If High/Low = e then ln(High/Low) = 1, so RV = 1 / (4 ln 2).
    rv = parkinson_variance(pd.Series([math.e]), pd.Series([1.0]))
    assert rv.iloc[0] == pytest.approx(1 / (4 * math.log(2)))
    # No range at all -> zero variance.
    assert parkinson_variance(pd.Series([50.0]), pd.Series([50.0])).iloc[0] == 0.0


def test_target_equals_next_day_rv():
    df = make_ohlc()
    rv = parkinson_variance(df["High"], df["Low"])
    feats = build_features(df)
    for t in feats.index[::17]:  # spot-check many rows
        pos = df.index.get_loc(t)
        assert feats.loc[t, "target"] == pytest.approx(rv.iloc[pos + 1])
        assert feats.loc[t, "rv_d"] == pytest.approx(rv.iloc[pos])


def test_rolling_windows_match_manual_means():
    df = make_ohlc()
    rv = parkinson_variance(df["High"], df["Low"])
    feats = build_features(df)
    t = feats.index[50]
    pos = df.index.get_loc(t)
    assert feats.loc[t, "rv_w"] == pytest.approx(rv.iloc[pos - 4: pos + 1].mean())    # 5 days
    assert feats.loc[t, "rv_m"] == pytest.approx(rv.iloc[pos - 21: pos + 1].mean())   # 22 days


def test_features_at_t_do_not_depend_on_future_prices():
    df = make_ohlc()
    cutoff = df.index[200]
    base = build_features(df)

    # Wildly change every price AFTER the cutoff date.
    altered = df.copy()
    future = altered.index > cutoff
    altered.loc[future, "High"] *= 1.7
    altered.loc[future, "Low"] *= 0.6
    changed = build_features(altered)

    # Features (not the target!) for every row up to and including the cutoff
    # must be identical.
    upto = base.index[base.index <= cutoff]
    pd.testing.assert_frame_equal(base.loc[upto, FEATURE_COLUMNS],
                                  changed.loc[upto, FEATURE_COLUMNS])
    # Sanity check that the alteration really did change something later on.
    assert not np.allclose(base.loc[base.index > cutoff, "rv_d"],
                           changed.loc[changed.index > cutoff, "rv_d"])

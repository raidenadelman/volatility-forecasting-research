import numpy as np
import pandas as pd

from src.evaluate import walk_forward
from src.features import build_features, vix_term_slope
from src.models import SLOPE_FEATURES, make_slope_models


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close}, index=idx)
    vix = pd.Series(rng.uniform(12, 30, n), index=idx)
    vix3m = pd.Series(rng.uniform(14, 28, n), index=idx)
    return df, vix, vix3m, pd.DatetimeIndex([idx[50]])


def test_slope_formula():
    assert np.isclose(vix_term_slope(20.0, 22.0), np.log(1.1))


def test_slope_column_present_only_when_requested():
    df, vix, vix3m, fomc = _data()
    assert "vix_slope" not in build_features(df, vix, fomc).columns
    assert "vix_slope" in build_features(df, vix, fomc, vix3m).columns


def test_slope_has_no_look_ahead():
    df, vix, vix3m, fomc = _data()
    f1 = build_features(df, vix, fomc, vix3m)
    vix3m2 = vix3m.copy()
    vix3m2.iloc[300:] *= 5  # change only the FUTURE
    f2 = build_features(df, vix, fomc, vix3m2)
    pd.testing.assert_series_equal(f1["vix_slope"].iloc[:250], f2["vix_slope"].iloc[:250])


def test_slope_models_walk_forward():
    df, vix, vix3m, fomc = _data()
    f = build_features(df, vix, fomc, vix3m)
    fc = walk_forward(f, make_slope_models(), min_train=150)
    assert "HAR+VIX+FOMC+Slope (log)" in fc.columns
    assert (fc.drop(columns="actual") > 0).all().all()
    assert set(SLOPE_FEATURES) <= set(f.columns)


def test_slope_sheet_payload_shape():
    from src.sheet_export import build_slope_payload
    models = ["Naive", "HAR-RV (log)", "HAR+VIX+FOMC (log)", "HAR+VIX+FOMC+Slope (log)"]
    overall = pd.DataFrame({"MSE": [1e-8] * 4, "QLIKE": [0.5] * 4, "QLIKE 2010-2015": [0.5] * 4,
                            "QLIKE 2016+ (holdout)": [0.5] * 4}, index=models)
    dm = pd.DataFrame({"model_A": ["a", "a"], "model_B": ["b", "c"], "mean_loss_diff": [-0.01, -0.06],
                       "DM_stat": [-5.0, -7.0], "p_value": [1e-6, 1e-9], "lags": [9, 9], "n": [10, 10]})
    fc = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    rows = build_slope_payload(overall, dm, dm, fc)
    assert len(rows) == 18
    assert rows[3][0] == "Model" and rows[4][0] == "Naive" and rows[10][0] == "Model A"

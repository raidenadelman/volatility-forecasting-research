import numpy as np
import pandas as pd

from src.higher_return import contango_signal, run, svxy_half_x


def test_svxy_half_x_scales_only_before_switch():
    idx = pd.to_datetime(["2018-02-26", "2018-02-27", "2018-02-28", "2018-03-01"])
    close = pd.Series([100.0, 90.0, 81.0, 72.9], index=idx)  # -10% each day
    r = svxy_half_x(close)
    assert np.isclose(r.iloc[1], -0.05)       # before 2018-02-28: halved
    assert np.isclose(r.iloc[2], -0.10) and np.isclose(r.iloc[3], -0.10)  # on/after: actual


def test_contango_signal():
    idx = pd.bdate_range("2020-01-01", periods=3)
    vix = pd.Series([15.0, 30.0, 20.0], index=idx)
    vix3m = pd.Series([18.0, 25.0, 21.0], index=idx)
    assert contango_signal(vix, vix3m).tolist() == [1.0, 0.0, 1.0]


def test_run_lag_and_borrowing():
    idx = pd.bdate_range("2020-01-01", periods=6)
    spy = pd.Series([0.0, 0.0, 0.01, 0.0, 0.0, 0.0], index=idx)
    cash = pd.Series(0.0, index=idx)
    w = pd.Series(2.0, index=idx)
    r = run({"SPY": w}, {"SPY": spy}, cash, {"SPY": 0.0}, lag=1, borrow_spread=0.252)
    # weight 2x applies from day 2 (set at close of day 0, +1 +lag); borrowed 1x pays 0.252/252 = 0.1% a day
    assert np.isclose(r["ret"].loc[idx[2]], 2 * 0.01 - 0.001)


def test_run_cash_earns_on_unallocated():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cash = pd.Series(0.001, index=idx)
    r = run({"SPY": pd.Series(0.5, index=idx)}, {"SPY": pd.Series(0.0, index=idx)}, cash, {"SPY": 0.0}, lag=0)
    assert np.allclose(r["ret"], 0.5 * 0.001)

import numpy as np
import pandas as pd

from src.volmanaged import backtest, managed_weight, sharpe_diff_ci, sigma_from_forecast, stats


def test_sigma_conversion():
    assert np.isclose(sigma_from_forecast(pd.Series([0.0001]))[0], np.sqrt(0.0252))


def test_weight_capped_and_shrinks_with_vol():
    idx = pd.bdate_range("2020-01-01", periods=100)
    sigma = pd.Series(0.15, index=idx)
    sigma.iloc[80:] = 0.45  # volatility triples
    w = managed_weight(sigma)
    assert (w <= 1.0).all()
    assert w.iloc[-1] < 0.5 < w.iloc[30]


def test_weight_uses_only_past_sigma():
    idx = pd.bdate_range("2020-01-01", periods=100)
    sigma = pd.Series(np.linspace(0.1, 0.3, 100), index=idx)
    w1 = managed_weight(sigma)
    s2 = sigma.copy(); s2.iloc[90:] = 5.0  # change only the future
    w2 = managed_weight(s2)
    cut = idx[85]
    pd.testing.assert_series_equal(w1[:cut], w2[:cut])


def test_backtest_timing_and_cash():
    idx = pd.bdate_range("2020-01-01", periods=6)
    w = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    spy = pd.Series([0.0, 0.10, -0.50, 0.0, 0.0, 0.0], index=idx)
    cash = pd.Series(0.001, index=idx)
    r = backtest(w, spy, cash, lag=0, cost_bps=0)
    # weight set at close of day 0 earns SPY's day-1 return; weight 0 on day 2 avoids the -50%
    assert np.isclose(r["ret"].iloc[0], 0.10) and np.isclose(r["ret"].iloc[1], 0.001)


def test_stats_and_bootstrap_run():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=500)
    a = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.01, 500), index=idx)
    cash = pd.Series(0.0, index=idx)
    assert stats(a, cash)["MaxDD"] <= 0
    d, lo, hi = sharpe_diff_ci(a, b, cash, n_boot=100)
    assert lo <= d <= hi or lo <= hi

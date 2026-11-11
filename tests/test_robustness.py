import numpy as np
import pandas as pd

from src.robustness import (deflated_sharpe, expected_max_sharpe, month_end_flags, simulate,
                            trend_target)

A = ["SPY", "IEF", "GLD"]
ZERO = {a: 0.0 for a in A}


def _flat(n, idx=None):
    idx = idx if idx is not None else pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(0.0, index=idx, columns=A), pd.Series(0.0, index=idx)


def test_month_end_flags():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28"])
    assert month_end_flags(idx).tolist() == [False, True, False, True]


def test_weights_drift_without_rebalancing():
    rets, cash = _flat(5)
    rets.loc[rets.index[1], "SPY"] = 1.0   # SPY doubles on day 1
    tgt = pd.DataFrame(1 / 3, index=rets.index, columns=A)
    out = simulate(tgt, rets, cash, ZERO, lag=1, rule="band", band=0.99)  # never trades
    assert np.isclose(out["ret"].iloc[0], 1 / 3)
    assert np.isclose(out["turnover"].sum(), 0.0)


def test_daily_rebalance_pays_cost_with_lag():
    rets, cash = _flat(5)
    rets.loc[rets.index[1], "SPY"] = 1.0
    tgt = pd.DataFrame(1 / 3, index=rets.index, columns=A)
    out = simulate(tgt, rets, cash, {a: 100.0 for a in A}, lag=1, rule="daily")
    # drift after day 1: SPY 0.5, others 0.25 each; traded back at close of day 2 -> turnover 1/3
    assert np.isclose(out["turnover"].iloc[0], 0.0)
    assert np.isclose(out["turnover"].iloc[1], 1 / 3)
    assert np.isclose(out["ret"].iloc[1], -1 / 3 * 0.01)


def test_trend_target_uses_month_end_only_and_past_prices():
    idx = pd.bdate_range("2019-01-01", periods=300)
    prices = pd.DataFrame({a: np.linspace(100, 200, 300) for a in A}, index=idx)
    t1 = trend_target(prices)
    p2 = prices.copy(); p2.iloc[280:] = 1.0     # crash only in the future
    t2 = trend_target(p2)
    cut = idx[279]
    pd.testing.assert_frame_equal(t1.loc[:cut], t2.loc[:cut])
    assert set(np.unique(t1.to_numpy())) <= {0.0, 1 / 3}


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1.0, 50) > expected_max_sharpe(1.0, 10) > 0


def test_deflated_sharpe_high_for_strong_low_for_noise():
    rng = np.random.default_rng(0)
    strong = pd.Series(rng.normal(0.002, 0.01, 2500))
    noise = pd.Series(rng.normal(0.0, 0.01, 2500))
    noise -= noise.mean()          # true and sample Sharpe exactly 0: must fall below the luck bar
    v = 0.01 / 252
    assert deflated_sharpe(strong, v, 20) > 0.99
    assert deflated_sharpe(noise, v, 20) < 0.5

import numpy as np
import pandas as pd

from src.backtest import make_signals, run_backtest, walk_forward_probability


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_signal_tiers_and_direction():
    s = make_signals(pd.Series([0.30, 0.42, 0.50, 0.58, 0.70]))
    assert s.loc[0, "signal"] == "SHORT" and s.loc[0, "size"] == 1.0      # P(down)=0.70
    assert s.loc[1, "signal"] == "SHORT" and s.loc[1, "size"] == 0.5      # P(down)=0.58
    assert s.loc[2, "signal"] == "FLAT" and s.loc[2, "size"] == 0.0
    assert s.loc[3, "signal"] == "LONG" and s.loc[3, "size"] == 0.5
    assert s.loc[4, "signal"] == "LONG" and s.loc[4, "size"] == 1.0


def test_engine_applies_signal_to_later_day_and_charges_costs():
    idx = _idx(6)
    sig = pd.DataFrame({"direction": [1, 1, 1, 1, 1, 1], "size": 1.0}, index=idx)
    r = pd.Series([0.0, 0.01, 0.02, -0.01, 0.0, 0.0], index=idx)
    bt = run_backtest(sig, r, r * 0, lag=1, initial=1000, cost_bps=0)
    # signal at day t is held over day t+2 (lag 1): first return earned is r[2]
    assert bt.index[0] == idx[2]
    assert np.isclose(bt["equity"].iloc[-1], 1000 * 1.02 * 0.99)
    bt_cost = run_backtest(sig, r, r * 0, lag=1, initial=1000, cost_bps=10)
    assert np.isclose(bt_cost["turnover"].sum(), 1.0)  # one entry, no further trading


def test_probability_has_no_lookahead():
    rng = np.random.default_rng(0)
    idx = _idx(900)
    X = pd.DataFrame(rng.normal(size=(900, 3)), index=idx, columns=list("abc"))
    y = pd.Series((rng.random(900) < 0.4).astype(float), index=idx)
    y.iloc[-2:] = np.nan
    base = walk_forward_probability(X, y, horizon=2, start=idx[600], min_train=300)
    X2, y2 = X.copy(), y.copy()
    X2.iloc[700:] += 5.0                     # change the future after day 700
    y2.iloc[700:] = 1 - y2.iloc[700:]
    alt = walk_forward_probability(X2, y2, horizon=2, start=idx[600], min_train=300)
    cut = idx[697]                           # labels at j>=700 are unknown at origins <= 697
    assert np.allclose(base.loc[:cut, "p_up"], alt.loc[:cut, "p_up"])

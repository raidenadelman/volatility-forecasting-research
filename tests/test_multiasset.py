import numpy as np
import pandas as pd

from src.multiasset import build_weights, inverse_vol_weights, portfolio_vol, rolling_corr


def _toy(n=400, seed=0):
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=["SPY", "IEF", "GLD"])
    sigma = pd.DataFrame({"SPY": 0.2, "IEF": 0.05, "GLD": 0.15}, index=idx)
    return rets, sigma


def test_inverse_vol_weights_sum_to_one_and_equalize_risk():
    sigma = pd.DataFrame({"SPY": [0.2], "IEF": [0.05], "GLD": [0.1]})
    w = inverse_vol_weights(sigma)
    assert np.isclose(w.sum(axis=1).iloc[0], 1.0)
    risk = (w * sigma).iloc[0]
    assert np.allclose(risk, risk.iloc[0])  # each asset contributes the same standalone vol


def test_portfolio_vol_matches_formula():
    idx = pd.bdate_range("2020-01-01", periods=1)
    w = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=idx)
    sigma = pd.DataFrame({"A": [0.2], "B": [0.1]}, index=idx)
    corr = pd.DataFrame([[1.0, -0.5], [-0.5, 1.0]], index=pd.MultiIndex.from_product([idx, ["A", "B"]]),
                        columns=["A", "B"])
    expected = np.sqrt(0.1**2 + 0.05**2 + 2 * -0.5 * 0.1 * 0.05)
    assert np.isclose(portfolio_vol(w, sigma, corr).iloc[0], expected)


def test_weights_use_only_past_data():
    rets, sigma = _toy()
    w1 = build_weights(sigma, rets)
    r2, s2 = rets.copy(), sigma.copy()
    cut = rets.index[300]
    r2.loc[cut:] *= 10.0           # change only the future
    s2.loc[cut:] *= 3.0
    w2 = build_weights(s2, r2)
    for k in w1:
        before = w1[k].index[w1[k].index < cut]
        pd.testing.assert_frame_equal(w1[k].loc[before], w2[k].loc[before])


def test_managed_never_levers_and_levered_respects_cap():
    rets, sigma = _toy()
    w = build_weights(sigma, rets)
    assert (w["Risk parity, managed"].sum(axis=1) <= 1.0 + 1e-12).all()
    assert (w["Risk parity, levered to SPY risk"].sum(axis=1) <= 2.0 + 1e-12).all()
    assert np.allclose(w["Risk parity"].sum(axis=1), 1.0)


def test_rolling_corr_shape():
    rets, _ = _toy()
    c = rolling_corr(rets)
    last = c.loc[rets.index[-1]]
    assert last.shape == (3, 3) and np.allclose(np.diag(last), 1.0)

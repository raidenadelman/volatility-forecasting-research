import numpy as np
import pandas as pd

from src.international import FUNDS, data_start, decide, pooled_sharpe_diff, rv_features


def _ohlc(n=600, zero_days=()):
    idx = pd.bdate_range("1996-03-18", periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high, low = close * 1.01, close * 0.99
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close}, index=idx)
    for i in zero_days:
        df.iloc[i, df.columns.get_loc("High")] = df.iloc[i]["Low"]
    return df


def test_fund_list_is_the_fixed_fifteen():
    assert len(FUNDS) == 15 and "EWM" not in FUNDS and "EWW" not in FUNDS


def test_data_start_skips_thin_early_period():
    df = _ohlc(zero_days=range(0, 100, 5))       # 20 zero-range days early on
    start = data_start(df)
    assert start > df.index[99]
    assert data_start(_ohlc()) == _ohlc().index[0]  # clean data starts on day one


def test_rv_features_fill_zero_days_without_look_ahead():
    df = _ohlc(zero_days=[300])
    f = rv_features(df)
    assert (f[["rv_d", "rv_w", "rv_m"]] > 0).all().all()
    d = df.index[300]
    rv = np.log(df["High"] / df["Low"]) ** 2 / (4 * np.log(2))
    assert np.isclose(f.loc[d, "rv_d"], rv.iloc[:300].min())   # minimum of the PAST only


def test_rv_features_target_is_next_day():
    f = rv_features(_ohlc())
    assert np.allclose(f["target"].iloc[:-1].to_numpy(), f["rv_d"].iloc[1:].to_numpy())


def test_pooled_matches_average_of_single_funds_point_estimate():
    idx = pd.bdate_range("2000-01-03", periods=1500)
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 1500)
    M = pd.DataFrame({"A": common + rng.normal(0.0004, 0.003, 1500), "B": common + rng.normal(0.0002, 0.003, 1500)}, index=idx)
    B = pd.DataFrame({"A": common, "B": common}, index=idx)
    M.iloc[:300, 1] = np.nan; B.iloc[:300, 1] = np.nan      # fund B starts later
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    expected = np.mean([sh(M["A"]) - sh(B["A"]), sh(M["B"].dropna()) - sh(B["B"].dropna())])
    d, lo, hi = pooled_sharpe_diff(M, B, n_boot=200)
    assert np.isclose(d, expected) and lo < d < hi


def test_decision_rule():
    assert decide(12, 13, 0.01) == "KEEP volatility scaling"
    assert decide(12, 13, -0.01) == "CRASH INSURANCE ONLY"
    assert decide(4, 11, 0.05) == "CRASH INSURANCE ONLY"
    assert decide(4, 6, 0.05) == "DROP volatility scaling"

"""Volatility estimate (Parkinson) and HAR features.

Everything in `build_features` at row t uses only information available at the
END of day t, except the `target` column, which is (by design) tomorrow's value.
"""

import numpy as np
import pandas as pd

WEEK = 5    # trading days in the "weekly" window
MONTH = 22  # trading days in the "monthly" window

FEATURE_COLUMNS = ["rv_d", "rv_w", "rv_m"]


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily variance proxy: the Parkinson (1980) range-based estimator.

        RV_t = (ln(High_t / Low_t))^2 / (4 * ln 2)

    Why a range-based estimate?
      The squared close-to-close return uses only two prices per day (yesterday's
      close and today's close), so it is a very noisy measure of that day's
      variance: a day can swing 3% intraday and still close flat, giving a
      squared return of ~0. The High-Low range uses information about the whole
      price path during the day, so for the same underlying volatility it has a
      much smaller sampling error. (Under a driftless random walk, Parkinson is
      about 5 times more statistically efficient.)

    Caveat: this is still only a PROXY for true realized variance. True realized
    variance is built from many intraday returns (e.g. 5-minute data), which I
    don't have. Parkinson also assumes continuous trading with no overnight
    jumps, so it understates variance on days with big opening gaps.
    """
    return np.log(high / low) ** 2 / (4 * np.log(2))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the HAR feature table from an OHLC DataFrame (indexed by date).

    Columns returned (all at row t):
        rv_d   = RV_t                           daily variance
        rv_w   = mean(RV_{t-4} ... RV_t)        weekly (5-day) average
        rv_m   = mean(RV_{t-21} ... RV_t)       monthly (22-day) average
        target = RV_{t+1}                       NEXT day's variance (the thing we forecast)

    Rows with NaN (the first 21 days, which lack a full monthly window, and the
    final day, which has no "next day" yet) are dropped.
    """
    rv = parkinson_variance(df["High"], df["Low"])

    out = pd.DataFrame(index=df.index)
    out["rv_d"] = rv
    # rolling(n).mean() at row t averages rows t-n+1 .. t, i.e. only the past
    # and today. It never looks forward, so there is no look-ahead here.
    out["rv_w"] = rv.rolling(WEEK).mean()
    out["rv_m"] = rv.rolling(MONTH).mean()
    # shift(-1) pulls the NEXT row's value up to row t. This is the only place
    # the future is used, and only to build the label we want to predict.
    out["target"] = rv.shift(-1)

    return out.dropna()


def annualized_vol_pct(variance):
    """Convert daily variance to annualized volatility in percent.

    sqrt(252 * variance) * 100. Used for charts only (percent volatility is
    easier to read than tiny variance numbers); all models and scores work in
    variance.
    """
    return np.sqrt(252 * np.asarray(variance, dtype=float)) * 100

"""Volatility estimate (Parkinson) and HAR features.

Everything in `build_features` at row t uses only information available at the
END of day t, except the `target` column, which is (by design) tomorrow's value.
"""

import numpy as np
import pandas as pd

WEEK = 5    # trading days in the "weekly" window
MONTH = 22  # trading days in the "monthly" window

FEATURE_COLUMNS = ["rv_d", "rv_w", "rv_m"]  # the three HAR features
VIX_COLUMNS = FEATURE_COLUMNS + ["vix_var"]  # HAR + implied variance from VIX
FULL_COLUMNS = VIX_COLUMNS + ["fomc_next"]   # HAR + VIX + "FOMC decision tomorrow"


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


def vix_to_daily_variance(vix_close) -> pd.Series:
    """Convert VIX points to an implied DAILY variance: (VIX/100)^2 / 252.

    VIX is quoted as annualized volatility in percent (18.5 means 18.5% a year).
    Dividing by 100 gives a fraction, squaring gives annualized variance, and
    dividing by 252 trading days gives daily variance, the same units as the
    Parkinson variance, so the two can sit side by side in one regression.
    """
    return (vix_close / 100) ** 2 / 252


def fomc_tomorrow_flag(index: pd.DatetimeIndex, fomc_dates) -> pd.Series:
    """1.0 at row t if the NEXT trading day in `index` is a scheduled FOMC decision day.

    Why this is allowed at the close of day t: FOMC meeting dates are published
    well in advance, so "a Fed decision lands tomorrow" is known today. This is
    NOT look-ahead; the flag uses only the published calendar.

    The last row has no known next trading day in the data, so it gets 0 (that
    row is dropped anyway because its target is unknown).
    """
    fomc = pd.DatetimeIndex(fomc_dates)
    next_day = pd.Series(index, index=index).shift(-1)  # the next trading day for each row
    return next_day.isin(fomc).astype(float).rename("fomc_next")


def build_features(df: pd.DataFrame, vix=None, fomc_dates=None) -> pd.DataFrame:
    """Build the feature table from an OHLC DataFrame (indexed by date).

    Columns returned (all at row t):
        rv_d   = RV_t                           daily variance
        rv_w   = mean(RV_{t-4} ... RV_t)        weekly (5-day) average
        rv_m   = mean(RV_{t-21} ... RV_t)       monthly (22-day) average
        target = RV_{t+1}                       NEXT day's variance (the thing we forecast)
    Optional extra columns:
        vix_var   (if `vix` is given)        VIX close on day t as a daily variance
        fomc_next (if `fomc_dates` is given) 1 if tomorrow is an FOMC decision day, else 0

    Rows with NaN (the first 21 days, which lack a full monthly window, the
    final day, which has no "next day" yet, and any day with no VIX close) are dropped.
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

    # VIX on day t is the closing level on day t, so it is known at the end of day t.
    if vix is not None:
        out["vix_var"] = vix_to_daily_variance(vix.reindex(df.index))
    if fomc_dates is not None:
        out["fomc_next"] = fomc_tomorrow_flag(df.index, fomc_dates)

    # Put target last so the order is: features..., target.
    out = out[[c for c in out.columns if c != "target"] + ["target"]]
    return out.dropna()


def annualized_vol_pct(variance):
    """Convert daily variance to annualized volatility in percent.

    sqrt(252 * variance) * 100. Used for charts only (percent volatility is
    easier to read than tiny variance numbers); all models and scores work in
    variance.
    """
    return np.sqrt(252 * np.asarray(variance, dtype=float)) * 100

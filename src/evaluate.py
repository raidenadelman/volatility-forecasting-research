"""Walk-forward evaluation, loss functions, sub-period scoring and the Diebold-Mariano test."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features import FEATURE_COLUMNS
from src.models import make_models

MIN_TRAIN = 1000    # trading days of history required before the first forecast
REFIT_EVERY = 21    # refit the regressions every 21 trading days (~1 month)

# (label, first date, last date). None = open-ended. Assigned by forecast ORIGIN date.
PERIODS = {
    "2000-2007": ("2000-01-01", "2007-12-31"),
    "2008-2009": ("2008-01-01", "2009-12-31"),
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "2020-present": ("2020-01-01", None),
}


# ----------------------------------------------------------------------------
# Walk-forward loop
# ----------------------------------------------------------------------------
def walk_forward(features: pd.DataFrame, models=None,
                 min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """Expanding-window, one-step-ahead forecasts with NO look-ahead.

    Timeline for origin day t (row i of `features`):
      * End of day t: we know prices through day t, hence rv_d/rv_w/rv_m at row i.
      * Row i's `target` is RV_{t+1}: NOT known yet. So it is never shown to a model.
      * Training data = rows 0..i-1. Row i-1's target is RV_t, which IS known.
        (Row i itself is excluded because its target is tomorrow's variance.)
      * The forecast made at origin t is for day t+1; the realized value
        (`actual`) is stored for scoring only, and is never passed to a model.

    The first origin is row `min_train`, so the first fit uses `min_train` rows.
    Models are refit every `refit_every` origins; in between they reuse their
    latest coefficients on each new day's features.

    Returns a DataFrame indexed by origin date, with one column per model
    (named model.name) plus an `actual` column.
    """
    if models is None:
        models = make_models()

    X = features[FEATURE_COLUMNS]  # models only ever see these three columns
    n = len(features)
    dates, actuals = [], []
    forecasts = {m.name: [] for m in models}

    for i in range(min_train, n):
        if (i - min_train) % refit_every == 0:
            train = features.iloc[:i]  # rows strictly before origin row i
            for m in models:
                m.fit(train)

        row = X.iloc[i]  # today's features only (no target column)
        for m in models:
            forecasts[m.name].append(m.predict(row))
        dates.append(features.index[i])
        actuals.append(features["target"].iloc[i])  # scoring only

    out = pd.DataFrame(forecasts, index=pd.DatetimeIndex(dates, name="Date"))
    out["actual"] = actuals
    return out


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------
def mse(actual, forecast) -> float:
    """Mean squared error on variance."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean((actual - forecast) ** 2))


def qlike_loss(actual, forecast) -> np.ndarray:
    """QLIKE loss for each day:  actual/forecast - ln(actual/forecast) - 1.

    Why QLIKE is the preferred loss for volatility forecasts:
      True variance is never observed; we score against a noisy proxy (here the
      Parkinson estimate). Patton (2011) showed that only a family of losses
      (which includes MSE and QLIKE) ranks forecasts the same way when scored
      against a noisy-but-unbiased proxy as against the true variance, so the
      choice of loss is not distorted by proxy noise. Within that family QLIKE
      is preferred because it depends on the RATIO actual/forecast: an error
      matters in proportion to the volatility level, so a few crisis days
      (e.g. 2008) do not swamp the whole score the way they do under MSE, and
      it has more statistical power to tell models apart. It is also zero only
      when forecast == actual, and it penalises under-forecasting more heavily
      than over-forecasting by the same amount.
    """
    ratio = np.asarray(actual, float) / np.asarray(forecast, float)
    return ratio - np.log(ratio) - 1.0


def qlike(actual, forecast) -> float:
    """Average QLIKE loss over all days."""
    return float(np.mean(qlike_loss(actual, forecast)))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _model_columns(forecast_df: pd.DataFrame) -> list:
    return [c for c in forecast_df.columns if c != "actual"]


def score_overall(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """MSE and QLIKE per model over the whole out-of-sample period."""
    rows = {}
    for name in _model_columns(forecast_df):
        rows[name] = {"MSE": mse(forecast_df["actual"], forecast_df[name]),
                      "QLIKE": qlike(forecast_df["actual"], forecast_df[name])}
    return pd.DataFrame(rows).T.rename_axis("model")


def score_by_period(forecast_df: pd.DataFrame, periods: dict = PERIODS) -> pd.DataFrame:
    """MSE and QLIKE per model within each sub-period (by forecast origin date)."""
    rows = []
    for label, (start, end) in periods.items():
        sub = forecast_df.loc[start:end]
        if sub.empty:
            continue
        for name in _model_columns(forecast_df):
            rows.append({"period": label, "model": name, "n_days": len(sub),
                         "MSE": mse(sub["actual"], sub[name]),
                         "QLIKE": qlike(sub["actual"], sub[name])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Diebold-Mariano test
# ----------------------------------------------------------------------------
def diebold_mariano(actual, forecast_a, forecast_b, lags: int | None = None) -> dict:
    """Diebold-Mariano test on QLIKE loss differentials: model A vs model B.

    d_t = QLIKE_t(A) - QLIKE_t(B).  H0: E[d_t] = 0 (equal predictive accuracy).
    Negative mean d means A has LOWER loss, i.e. A is more accurate.

    Implementation: regress d_t on a constant; the constant is mean(d). Its
    standard error is the Newey-West (HAC) one, which allows d_t to be
    autocorrelated (it is, because volatility is persistent). The t-statistic
    is compared with the standard Normal distribution (two-sided p-value).
    Bandwidth: Newey-West rule of thumb floor(4 * (T/100)^(2/9)) lags unless
    `lags` is given.

    Caveats (documented, not hidden):
      * The textbook h-step DM uses h-1 lags (zero for one-step forecasts), but
        daily volatility loss differentials are autocorrelated anyway, so I use
        the Newey-West bandwidth instead of 0.
      * HAR nests the Rolling-mean model (HAR with weights 0,0,1 and no
        intercept is rolling mean), and DM's Normal approximation is not exact
        for nested models with estimated parameters (Clark-West is the usual
        alternative). With ~5,700 out-of-sample days and expanding windows the
        distortion is small, but read p-values as approximate.
    """
    d = qlike_loss(actual, forecast_a) - qlike_loss(actual, forecast_b)
    T = len(d)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    res = sm.OLS(d, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    dm_stat = float(res.tvalues[0])
    p_value = float(res.pvalues[0])  # two-sided, Normal-based (statsmodels default for HAC)
    return {"mean_loss_diff": float(d.mean()), "DM_stat": dm_stat,
            "p_value": p_value, "lags": lags, "n": T}


def dm_table(forecast_df: pd.DataFrame,
             challengers=("HAR-RV (levels)", "HAR-RV (log)"),
             benchmarks=("Naive", "Rolling mean")) -> pd.DataFrame:
    """DM tests of each HAR variant (A) against each baseline (B), on QLIKE."""
    rows = []
    for a in challengers:
        for b in benchmarks:
            r = diebold_mariano(forecast_df["actual"], forecast_df[a], forecast_df[b])
            rows.append({"model_A": a, "model_B": b, **r})
    return pd.DataFrame(rows)

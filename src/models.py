"""One-step-ahead variance forecasters.

Every model has the same two methods:
    fit(train_df)      learn from a DataFrame holding the feature columns plus target
    predict(test_row)  forecast next-day variance from ONE row (a pandas Series)
                       holding the feature columns for the current day
Each model only reads the columns it was built to use (rv_d/rv_w/rv_m, and
optionally vix_var and fomc_next), so extra columns in the row are ignored.

The two baselines have nothing to learn, so their fit() does nothing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES = ["rv_d", "rv_w", "rv_m"]
VIX_FEATURES = FEATURES + ["vix_var"]
FULL_FEATURES = VIX_FEATURES + ["fomc_next"]
SLOPE_FEATURES = FULL_FEATURES + ["vix_slope"]
LOG_COLUMNS = {"rv_d", "rv_w", "rv_m", "vix_var"}  # variance-type columns that get logged
VARIANCE_FLOOR = 1e-8  # variance can't be negative (or zero), so floor forecasts here


class NaiveModel:
    """Tomorrow's variance = today's variance (rv_d)."""
    name = "Naive"

    def fit(self, train_df: pd.DataFrame) -> "NaiveModel":
        return self  # nothing to learn

    def predict(self, test_row: pd.Series) -> float:
        return float(test_row["rv_d"])


class RollingMeanModel:
    """Tomorrow's variance = the 22-day average variance (rv_m)."""
    name = "Rolling mean"

    def fit(self, train_df: pd.DataFrame) -> "RollingMeanModel":
        return self  # nothing to learn

    def predict(self, test_row: pd.Series) -> float:
        return float(test_row["rv_m"])


class HARLevelsModel:
    """HAR-RV in levels:  target = b0 + b1*rv_d + b2*rv_w + b3*rv_m + error.

    Fitted by ordinary least squares (OLS) with statsmodels.

    Flooring: a linear model can in principle output a negative variance, which
    is impossible, so predictions are floored at VARIANCE_FLOOR. The counters
    `n_predictions` and `n_floored` record how often that happens; the run
    script reports the share. (Negative forecasts only happen if the fitted
    intercept is negative AND all recent variances are tiny.)
    """
    def __init__(self, features=FEATURES, name="HAR-RV (levels)"):
        self.features = list(features)  # which columns this model is allowed to use
        self.name = name
        self.params = None  # [const, one coefficient per feature, in `features` order]
        self.n_predictions = 0
        self.n_floored = 0

    def fit(self, train_df: pd.DataFrame) -> "HARLevelsModel":
        X = sm.add_constant(train_df[self.features], has_constant="add")
        result = sm.OLS(train_df["target"], X).fit()
        self.params = result.params.to_numpy()
        return self

    def predict(self, test_row: pd.Series) -> float:
        x = np.r_[1.0, [test_row[c] for c in self.features]]
        raw = float(x @ self.params)
        self.n_predictions += 1
        if raw < VARIANCE_FLOOR:
            self.n_floored += 1
        return max(raw, VARIANCE_FLOOR)


class HARLogModel:
    """HAR-RV in logs:  ln(target) = b0 + b1*ln(rv_d) + b2*ln(rv_w) + b3*ln(rv_m) + error.

    Working in logs guarantees positive forecasts and tames the extreme skew of
    variance. But there is a catch: if the log-forecast is unbiased for
    ln(variance), then exp(log-forecast) is BIASED LOW for variance itself
    (Jensen's inequality: E[exp(X)] > exp(E[X])). If the log error is roughly
    Normal with variance s^2, then E[exp(error)] = exp(0.5 * s^2). So we
    multiply back by that factor, i.e. forecast = exp(prediction + 0.5 * s^2),
    where s^2 is the residual variance of the log regression.
    """
    def __init__(self, features=FEATURES, name="HAR-RV (log)"):
        self.features = list(features)
        self.name = name
        self.params = None
        self.resid_var = 0.0

    def _transform(self, values: dict) -> np.ndarray:
        """Log the variance-type columns; leave 0/1 flags (like fomc_next) as they are."""
        return np.array([np.log(values[c]) if c in LOG_COLUMNS else values[c]
                         for c in self.features])

    def fit(self, train_df: pd.DataFrame) -> "HARLogModel":
        X = train_df[self.features].copy()
        for c in self.features:
            if c in LOG_COLUMNS:
                X[c] = np.log(X[c])
        X = sm.add_constant(X, has_constant="add")
        y = np.log(train_df["target"])
        result = sm.OLS(y, X).fit()
        self.params = result.params.to_numpy()
        self.resid_var = float(result.mse_resid)  # s^2, residual variance of the log regression
        return self

    def predict(self, test_row: pd.Series) -> float:
        x = np.r_[1.0, self._transform(test_row)]
        log_pred = float(x @ self.params)
        return float(np.exp(log_pred + 0.5 * self.resid_var))


def make_models(extended: bool = False) -> list:
    """Fresh, unfitted models.

    extended=False: the four Phase 1 models (Naive, Rolling mean, HAR-RV levels/log).
    extended=True : adds HAR+VIX and HAR+VIX+FOMC, each in levels and log form,
                    so the effect of each added input can be isolated:
                    HAR-RV -> +VIX -> +FOMC.
    """
    models = [NaiveModel(), RollingMeanModel(), HARLevelsModel(), HARLogModel()]
    if extended:
        models += [HARLevelsModel(VIX_FEATURES, "HAR+VIX (levels)"),
                   HARLogModel(VIX_FEATURES, "HAR+VIX (log)"),
                   HARLevelsModel(FULL_FEATURES, "HAR+VIX+FOMC (levels)"),
                   HARLogModel(FULL_FEATURES, "HAR+VIX+FOMC (log)")]
    return models


def make_slope_models() -> list:
    """Phase 4 comparison set (log models only; the levels forms can floor, see README).

    Run all four on the SAME rows: VIX3M only exists from 2006-07, so the sample is shorter
    than the main run's."""
    return [NaiveModel(), HARLogModel(name="HAR-RV (log)"),
            HARLogModel(FULL_FEATURES, "HAR+VIX+FOMC (log)"),
            HARLogModel(SLOPE_FEATURES, "HAR+VIX+FOMC+Slope (log)")]

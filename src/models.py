"""One-step-ahead variance forecasters.

Every model has the same two methods:
    fit(train_df)      learn from a DataFrame with columns rv_d, rv_w, rv_m, target
    predict(test_row)  forecast next-day variance from ONE row (a pandas Series)
                       holding rv_d, rv_w, rv_m for the current day

The two baselines have nothing to learn, so their fit() does nothing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES = ["rv_d", "rv_w", "rv_m"]
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
    name = "HAR-RV (levels)"

    def __init__(self):
        self.params = None
        self.n_predictions = 0
        self.n_floored = 0

    def fit(self, train_df: pd.DataFrame) -> "HARLevelsModel":
        X = sm.add_constant(train_df[FEATURES], has_constant="add")
        result = sm.OLS(train_df["target"], X).fit()
        self.params = result.params.to_numpy()  # [const, rv_d, rv_w, rv_m]
        return self

    def predict(self, test_row: pd.Series) -> float:
        x = np.array([1.0, test_row["rv_d"], test_row["rv_w"], test_row["rv_m"]])
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
    name = "HAR-RV (log)"

    def __init__(self):
        self.params = None
        self.resid_var = 0.0

    def fit(self, train_df: pd.DataFrame) -> "HARLogModel":
        X = sm.add_constant(np.log(train_df[FEATURES]), has_constant="add")
        y = np.log(train_df["target"])
        result = sm.OLS(y, X).fit()
        self.params = result.params.to_numpy()
        self.resid_var = float(result.mse_resid)  # s^2, residual variance of the log regression
        return self

    def predict(self, test_row: pd.Series) -> float:
        x = np.array([1.0,
                      np.log(test_row["rv_d"]),
                      np.log(test_row["rv_w"]),
                      np.log(test_row["rv_m"])])
        log_pred = float(x @ self.params)
        return float(np.exp(log_pred + 0.5 * self.resid_var))


def make_models() -> list:
    """Fresh, unfitted instances of all four models."""
    return [NaiveModel(), RollingMeanModel(), HARLevelsModel(), HARLogModel()]

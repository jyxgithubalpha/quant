"""
QuantileRegressor configuration.

Predicts a conditional quantile of the response variable.
Defaults to the median (quantile=0.5).
"""

from sklearn.linear_model import QuantileRegressor

MODEL_CLS = QuantileRegressor

DEFAULT_PARAMS = {
    "quantile": 0.5,
    "alpha": 1.0,
    "fit_intercept": True,
    "solver": "highs",
}

SEARCH_SPACE = {
    "quantile": ("float", 0.1, 0.9),
    "alpha": ("log_float", 1e-4, 100.0),
}

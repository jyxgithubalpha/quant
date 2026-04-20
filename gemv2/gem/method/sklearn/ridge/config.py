"""
Ridge regression configuration.
"""

from sklearn.linear_model import Ridge

MODEL_CLS = Ridge

DEFAULT_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
}

SEARCH_SPACE = {
    "alpha": ("log_float", 1e-4, 100.0),
    "solver": ("categorical", ["auto", "svd", "cholesky", "lsqr", "sag"]),
}

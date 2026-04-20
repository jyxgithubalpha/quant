"""
Lasso regression configuration.
"""

from sklearn.linear_model import Lasso

MODEL_CLS = Lasso

DEFAULT_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
    "max_iter": 1000,
}

SEARCH_SPACE = {
    "alpha": ("log_float", 1e-4, 100.0),
    "selection": ("categorical", ["cyclic", "random"]),
}

"""
HuberRegressor configuration.
"""

from sklearn.linear_model import HuberRegressor

MODEL_CLS = HuberRegressor

DEFAULT_PARAMS = {
    "epsilon": 1.35,
    "alpha": 0.0001,
    "max_iter": 100,
}

SEARCH_SPACE = {
    "epsilon": ("float", 1.01, 3.0),
    "alpha": ("log_float", 1e-6, 1.0),
}

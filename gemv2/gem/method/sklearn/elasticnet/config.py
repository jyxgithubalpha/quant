"""
ElasticNet regression configuration.
"""

from sklearn.linear_model import ElasticNet

MODEL_CLS = ElasticNet

DEFAULT_PARAMS = {
    "alpha": 1.0,
    "l1_ratio": 0.5,
    "fit_intercept": True,
    "max_iter": 1000,
}

SEARCH_SPACE = {
    "alpha": ("log_float", 1e-4, 100.0),
    "l1_ratio": ("float", 0.0, 1.0),
    "selection": ("categorical", ["cyclic", "random"]),
}

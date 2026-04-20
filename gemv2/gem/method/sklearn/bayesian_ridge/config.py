"""
BayesianRidge configuration.
"""

from sklearn.linear_model import BayesianRidge

MODEL_CLS = BayesianRidge

DEFAULT_PARAMS = {
    "max_iter": 300,
    "tol": 1e-3,
    "fit_intercept": True,
}

SEARCH_SPACE = {
    "alpha_1": ("log_float", 1e-8, 1e-3),
    "alpha_2": ("log_float", 1e-8, 1e-3),
    "lambda_1": ("log_float", 1e-8, 1e-3),
    "lambda_2": ("log_float", 1e-8, 1e-3),
}

"""
GaussianProcessRegressor configuration.

Note: GP scales as O(n^3) with training samples; use only on small datasets
or after dimensionality reduction.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

MODEL_CLS = GaussianProcessRegressor

DEFAULT_PARAMS = {
    "kernel": RBF() + WhiteKernel(),
    "alpha": 1e-10,
    "normalize_y": True,
    "n_restarts_optimizer": 3,
}

SEARCH_SPACE = {
    "alpha": ("log_float", 1e-10, 1e-2),
    "n_restarts_optimizer": ("int", 0, 10),
}

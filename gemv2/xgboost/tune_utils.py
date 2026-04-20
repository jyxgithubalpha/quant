"""
Ray Tune HPO utilities for XGBoost hyperparameter optimization.
Uses native xgb.train() — no dependency on xgboost_ray.

Usage (called from backtest_ablation.py when TUNE_CONFIG["enabled"] is True):

    from tune_utils import tune_xgb_params
    best_params = tune_xgb_params(
        X_train, y_train, groups_train,
        X_val, y_val, groups_val,
        n_trials=TUNE_CONFIG["n_trials"],
        num_boost_round=TUNE_CONFIG["num_boost_round"],
    )
    params = {**DEFAULT_XGB_PARAMS, **best_params}
"""

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from ray import tune
from ray.tune.schedulers import ASHAScheduler


TUNE_SEARCH_SPACE = {
    "eta": tune.loguniform(0.01, 0.3),
    "max_depth": tune.randint(4, 10),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "min_child_weight": tune.randint(5, 50),
    "lambda": tune.loguniform(0.1, 10.0),
    "alpha": tune.loguniform(0.01, 1.0),
}


def tune_xgb_params(
    X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, groups_val: np.ndarray,
    n_trials: int = 20,
    num_boost_round: int = 500,
) -> dict:
    """
    Run Ray Tune hyperparameter search and return the best XGBoost config.

    Parameters
    ----------
    X_train, y_train, groups_train : Training data + group sizes
    X_val, y_val, groups_val       : Validation data + group sizes
    n_trials                        : Number of HPO trials
    num_boost_round                 : Max boosting rounds per trial

    Returns
    -------
    dict : Best hyperparameter config (keys from TUNE_SEARCH_SPACE)
    """
    # Date-agnostic RankIC feval — avoids date_info dependency.
    # Structural XGB params (depth, regularisation) are insensitive to cross-date aggregation.
    def _feval_tune(predt, dmat):
        labels = dmat.get_label()
        if len(np.unique(predt)) < 2 or len(np.unique(labels)) < 2:
            return [("RankIC", 0.0)]
        c = spearmanr(predt, labels).correlation
        return [("RankIC", float(c) if not np.isnan(c) else 0.0)]

    def _trainable(config):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            **config,
            "objective": "reg:squarederror",
            "device": "cuda",
            "tree_method": "hist",
            "disable_default_eval_metric": 1,
        }
        evals_result = {}
        xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "valid")],
            feval=_feval_tune,
            evals_result=evals_result,
            verbose_eval=0,
        )
        valid_rankic = evals_result.get("valid", {}).get("RankIC", [0.0])
        tune.report({"RankIC": valid_rankic[-1] if valid_rankic else 0.0})

    scheduler = ASHAScheduler(
        metric="RankIC",
        mode="max",
        max_t=num_boost_round,
        grace_period=50,
    )

    analysis = tune.run(
        _trainable,
        config=TUNE_SEARCH_SPACE,
        num_samples=n_trials,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1, "cpu": 4},
        verbose=1,
    )

    return analysis.best_config

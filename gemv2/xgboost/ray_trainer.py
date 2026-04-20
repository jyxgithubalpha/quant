"""
Ray multi-GPU XGBoost training module.

Drop-in replacement for dask_trainer.DaskXGBRankModel.
Uses native xgb.train() with Ray for GPU resource management.
No dependency on xgboost_ray.

Key advantages:
  1. Standard xgb.callback.EarlyStopping works natively.
  2. Custom composite_metric with date_info fully supported.
  3. No NCCL env-var tuning required.
  4. Multiple train() calls are safe.

Interface is identical to XGBRankModel and DaskXGBRankModel for drop-in replacement.
"""

import numpy as np
import polars as pl
import xgboost as xgb
import ray

from model_core import composite_metric


# ============================================================
# Group computation for XGBoost ranking
# ============================================================
def _compute_groups(dates: np.ndarray) -> np.ndarray:
    """
    Convert sorted date array to group sizes for XGBoost ranking.

    Parameters
    ----------
    dates : 1-D array of date values (must be sorted)

    Returns
    -------
    groups : int32 array of group sizes (one element per unique date)
    """
    if len(dates) == 0:
        return np.array([], dtype=np.int32)
    unique_dates, counts = np.unique(dates, return_counts=True)
    return counts.astype(np.int32)


def _compute_qids(dates: np.ndarray) -> np.ndarray:
    """
    Convert dates to group IDs (one ID per row). Kept for backward compat.

    Parameters
    ----------
    dates : 1-D array of date values (datetime objects or timestamps)

    Returns
    -------
    qid : int32 array of group IDs, same length as input dates
    """
    _, qid = np.unique(dates, return_inverse=True)
    return qid.astype(np.int32)


# ============================================================
# Ray remote training / prediction functions
# ============================================================
@ray.remote(num_gpus=1)
def _ray_xgb_train(params, X_train, y_train, dates_train,
                    X_val, y_val, dates_val,
                    early_stopping_rounds, ndcg_weight, rankic_weight,
                    ndcg_k, num_boost_round, verbose):
    """Run xgb.train on a single GPU inside a Ray task."""
    import xgboost as xgb
    from model_core import composite_metric

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.date_info = dates_train

    evals = [(dtrain, "train")]
    dval = None
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.date_info = dates_val
        evals.append((dval, "valid"))

    def _feval(preds, dmat):
        return composite_metric(
            preds, dmat,
            ndcg_k=ndcg_k,
            ndcg_weight=ndcg_weight,
            rankic_weight=rankic_weight,
        )

    early_stop = xgb.callback.EarlyStopping(
        rounds=early_stopping_rounds,
        metric_name="Composite",
        data_name="valid" if dval is not None else "train",
        maximize=True,
    )

    evals_result = {}
    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        custom_metric=_feval,
        maximize=True,
        verbose_eval=10 if verbose else False,
        evals_result=evals_result,
        callbacks=[early_stop],
    )

    best_iteration = getattr(early_stop, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration", None)

    # Serialize model to bytes for transfer back
    model_bytes = model.save_raw("json")
    n_imp = len(model.get_score(importance_type="gain") or {})
    return model_bytes, best_iteration, n_imp


@ray.remote(num_gpus=1)
def _ray_xgb_predict(model_bytes, X):
    """Run xgb.Booster.predict on a single GPU inside a Ray task."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)


# ============================================================
# RayXGBRankModel — drop-in for XGBRankModel / DaskXGBRankModel
# ============================================================
class RayXGBRankModel:
    """
    Multi-GPU XGBoost model using Ray for GPU resource management.

    Training and prediction run as Ray remote tasks, each pinned to a
    single GPU.  The native xgb.train() API is used, so custom metrics
    (composite_metric with date_info) work without modification.
    """

    def __init__(self, params: dict, num_actors: int = 4,
                 cpus_per_actor: int = 4, gpus_per_actor: int = 1):
        """
        Parameters
        ----------
        params          : XGBoost parameter dict
        num_actors      : (kept for interface compat, not used in single-GPU mode)
        cpus_per_actor  : (kept for interface compat)
        gpus_per_actor  : (kept for interface compat)
        """
        self.params = {**params, "device": "cuda", "tree_method": "hist"}
        self.model = None
        self.best_iteration = None
        self._model_bytes = None

    def train(self, X_train, y_train, dates_train,
              X_val=None, y_val=None, dates_val=None,
              early_stopping_rounds: int = 50,
              ndcg_weight: float = 0.3, rankic_weight: float = 0.7,
              ndcg_k: int = 200, num_boost_round: int = 2000,
              verbose: bool = True):
        """
        Train XGBoost model on a Ray-managed GPU.

        Parameters
        ----------
        X_train, y_train, dates_train : Training arrays
        X_val, y_val, dates_val       : Validation arrays (required for early stopping)
        early_stopping_rounds         : Rounds without improvement before stopping
        ndcg_weight, rankic_weight    : Composite metric weights
        ndcg_k                        : Top-k for NDCG
        num_boost_round               : Maximum boosting rounds
        verbose                       : Print training progress
        """
        model_bytes, best_iter, n_imp = ray.get(
            _ray_xgb_train.remote(
                self.params, X_train, y_train, dates_train,
                X_val, y_val, dates_val,
                early_stopping_rounds, ndcg_weight, rankic_weight,
                ndcg_k, num_boost_round, verbose,
            )
        )

        # Reconstruct Booster from bytes
        self.model = xgb.Booster()
        self.model.load_model(bytearray(model_bytes))
        self._model_bytes = model_bytes
        self.best_iteration = best_iter

        if verbose:
            print(f"best_iteration: {self.best_iteration}")
            print(f"features with importance: {n_imp}")

    def predict(self, X, dates=None, codes=None):
        """
        Predict scores.

        Returns pl.DataFrame(date, Code, score) if dates+codes provided,
        else raw numpy array.
        """
        if self._model_bytes is not None:
            preds = ray.get(_ray_xgb_predict.remote(self._model_bytes, X))
        else:
            dtest = xgb.DMatrix(X)
            preds = self.model.predict(dtest)
        if dates is not None and codes is not None:
            return pl.DataFrame({"date": dates, "Code": codes, "score": preds})
        return preds

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model = xgb.Booster()
        self.model.load_model(path)
        self._model_bytes = self.model.save_raw("json")

    def get_feature_importance(self) -> dict:
        if self.model is None:
            return {}
        return self.model.get_score(importance_type="gain") or {}

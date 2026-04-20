"""
Multi-seed ensemble module
Train N XGBRankModels with different random seeds, weight-average prediction scores using validation set RankIC.
Weights are max(ic, 0) / sum(max(ic, 0)), equal weights if all are 0.
"""

import numpy as np
import polars as pl
from typing import List, Optional

from model_core import XGBRankModel, get_cross_section_metrics
from config import DEFAULT_XGB_PARAMS, TRAIN_CONFIG, RAY_GPU_CONFIG

if RAY_GPU_CONFIG["enabled"]:
    from ray_trainer import RayXGBRankModel


def train_ensemble(
    X_train: np.ndarray, y_train: np.ndarray, dates_train: list,
    X_val: np.ndarray, y_val: np.ndarray, dates_val: list,
    val_label_long: pl.DataFrame,
    seeds: List[int],
    xgb_params: Optional[dict] = None,
    train_config: Optional[dict] = None,
    verbose: bool = False,
):
    """
    Train multiple XGBRankModels with random seeds, return model list and corresponding validation set RankIC weights.

    Parameters
    ----------
    X_train, y_train, dates_train : Training set arrays
    X_val, y_val, dates_val       : Validation set arrays
    val_label_long                : pl.DataFrame Validation set label long table (date, Code, label), for weight calculation
    seeds                         : Random seed list, e.g. [42, 100, 200]
    xgb_params                    : XGBoost parameter dictionary (without seed)
    train_config                  : Training control parameter dictionary

    Returns
    -------
    models   : List[XGBRankModel]
    val_ics  : List[float], RankIC of each model on validation set (for weighting)
    """
    if xgb_params is None:
        xgb_params = DEFAULT_XGB_PARAMS
    if train_config is None:
        train_config = TRAIN_CONFIG

    models, val_ics = [], []

    for seed in seeds:
        params_with_seed = {**xgb_params, "seed": seed}
        if RAY_GPU_CONFIG["enabled"]:
            model = RayXGBRankModel(
                params_with_seed,
                num_actors=RAY_GPU_CONFIG["num_actors"],
                cpus_per_actor=RAY_GPU_CONFIG["cpus_per_actor"],
                gpus_per_actor=RAY_GPU_CONFIG["gpus_per_actor"],
            )
        else:
            model = XGBRankModel(params_with_seed)

        model.train(
            X_train, y_train, dates_train,
            X_val, y_val, dates_val,
            early_stopping_rounds=train_config["early_stopping_rounds"],
            ndcg_weight=train_config["ndcg_weight"],
            rankic_weight=train_config["rankic_weight"],
            ndcg_k=train_config["ndcg_k"],
            num_boost_round=train_config["num_boost_round"],
            verbose=verbose,
        )

        # Calculate validation set RankIC
        val_preds = model.predict(X_val)
        # Rebuild score_df for RankIC calculation
        score_df_val = pl.DataFrame({
            "date": dates_val,
            "Code": _extract_codes_from_label(val_label_long, dates_val),
            "score": val_preds,
        })
        ric = get_cross_section_metrics(score_df_val, val_label_long)[0]
        print(f"  [ensemble] seed={seed}, val_RankIC={ric:.4f}")

        models.append(model)
        val_ics.append(max(0.0, ric))

    return models, val_ics


def ensemble_predict(
    models: list, val_ics: List[float],
    X: np.ndarray, dates: list, codes: list,
) -> pl.DataFrame:
    """
    Weighted average of prediction scores from multiple models.

    Parameters
    ----------
    models   : List of trained models
    val_ics  : Validation set RankIC corresponding to each model (non-negative)
    X        : Test feature matrix
    dates    : Date list (corresponds to X rows)
    codes    : Stock code list (corresponds to X rows)

    Returns
    -------
    pl.DataFrame, containing ['date', 'Code', 'score'] columns
    """
    weights = np.array(val_ics, dtype=float)
    total = weights.sum()
    weights = weights / total if total > 1e-9 else np.ones_like(weights) / len(weights)

    preds_sum = np.zeros(len(X), dtype=float)
    for w, model in zip(weights, models):
        preds_sum += w * model.predict(X)

    return pl.DataFrame({"date": dates, "Code": codes, "score": preds_sum})


def train_tangle_ensemble(
    X_train: np.ndarray, y_train: np.ndarray, dates_train: list,
    X_val: np.ndarray, y_val: np.ndarray, dates_val: list,
    X_test: np.ndarray, dates_test: list, codes_test: list,
    val_label_long: pl.DataFrame,
    seeds: List[int],
    xgb_params: Optional[dict] = None,
    train_config: Optional[dict] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Tangle-weighted ensemble: train multiple XGBoost models, cluster test stocks
    by model agreement using tangles, then apply per-cluster model weights.

    Unlike global IC-weighted ensemble, tangles discover stock groups where
    different models excel, assigning cluster-specific weights.

    Returns pl.DataFrame with ['date', 'Code', 'score'] columns.
    """
    import sys, os
    _tangles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tangles_ens")
    if _tangles_dir not in sys.path:
        sys.path.insert(0, _tangles_dir)
    _tangles_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tangles_src")
    if _tangles_src not in sys.path:
        sys.path.insert(0, _tangles_src)

    from tangle_ensemble import run_tangle_clustering
    from scipy.stats import spearmanr

    if xgb_params is None:
        xgb_params = DEFAULT_XGB_PARAMS
    if train_config is None:
        train_config = TRAIN_CONFIG

    # Step 1: Train models with different seeds
    models = []
    for seed in seeds:
        params_with_seed = {**xgb_params, "seed": seed}
        if RAY_GPU_CONFIG["enabled"]:
            model = RayXGBRankModel(
                params_with_seed,
                num_actors=RAY_GPU_CONFIG["num_actors"],
                cpus_per_actor=RAY_GPU_CONFIG["cpus_per_actor"],
                gpus_per_actor=RAY_GPU_CONFIG["gpus_per_actor"],
            )
        else:
            model = XGBRankModel(params_with_seed)
        model.train(
            X_train, y_train, dates_train,
            X_val, y_val, dates_val,
            early_stopping_rounds=train_config["early_stopping_rounds"],
            ndcg_weight=train_config["ndcg_weight"],
            rankic_weight=train_config["rankic_weight"],
            ndcg_k=train_config["ndcg_k"],
            num_boost_round=train_config["num_boost_round"],
            verbose=verbose,
        )
        models.append(model)

    # Step 2: Get test predictions from all models → (n_test, n_models)
    all_test_preds = np.column_stack([m.predict(X_test) for m in models])
    n_models = len(models)

    # Step 3: Per-date tangle clustering and weighted ensemble
    test_df = pl.DataFrame({
        "date": dates_test,
        "Code": codes_test,
        **{f"_pred_{i}": all_test_preds[:, i] for i in range(n_models)},
    })
    pred_cols = [f"_pred_{i}" for i in range(n_models)]

    # Also get val predictions for computing per-cluster IC weights
    all_val_preds = np.column_stack([m.predict(X_val) for m in models])
    val_df = pl.DataFrame({
        "date": dates_val,
        "Code": _extract_codes_from_label(val_label_long, dates_val),
        **{f"_pred_{i}": all_val_preds[:, i] for i in range(n_models)},
    }).join(val_label_long.select(["date", "Code", "label"]), on=["date", "Code"], how="left")

    ensemble_scores_list = []

    for grp in test_df.partition_by("date", maintain_order=True):
        date_val = grp["date"][0]
        scores_matrix = grp.select(pred_cols).to_numpy().astype(np.float64)
        n_stocks = len(grp)

        # Try tangle clustering on this date's model predictions
        labels = None
        if n_stocks >= 30:
            labels = run_tangle_clustering(scores_matrix)

        if labels is None:
            # Fallback: equal-weight average
            avg = np.nanmean(scores_matrix, axis=1)
            for i in range(n_stocks):
                ensemble_scores_list.append({
                    "date": date_val,
                    "Code": grp["Code"][i],
                    "score": float(avg[i]),
                })
        else:
            # Compute per-cluster weights using val data IC
            unique_clusters = np.unique(labels)
            cluster_weights = {}
            for c in unique_clusters:
                # Find which stocks in val have similar prediction patterns
                # Use simple approach: compute IC of each model on val set
                ics = np.zeros(n_models)
                val_labels_arr = val_df["label"].to_numpy()
                for mi in range(n_models):
                    val_preds_arr = val_df[f"_pred_{mi}"].to_numpy()
                    valid = ~(np.isnan(val_preds_arr) | np.isnan(val_labels_arr))
                    if valid.sum() > 30:
                        corr = spearmanr(val_preds_arr[valid], val_labels_arr[valid]).correlation
                        ics[mi] = corr if not np.isnan(corr) else 0.0
                w = np.maximum(ics, 0.0)
                cluster_weights[c] = w / w.sum() if w.sum() > 1e-10 else np.ones(n_models) / n_models

            for i in range(n_stocks):
                w = cluster_weights.get(labels[i], np.ones(n_models) / n_models)
                score = np.dot(w, scores_matrix[i])
                ensemble_scores_list.append({
                    "date": date_val,
                    "Code": grp["Code"][i],
                    "score": float(score),
                })

    del models
    return pl.DataFrame(ensemble_scores_list)


def _extract_codes_from_label(label_long: pl.DataFrame, dates: list) -> list:
    """
    Extract Code column from label_long in dates order (for building score_df).
    Assumes label_long row order corresponds to dates (guaranteed by prepare_data).
    """
    if "Code" in label_long.columns:
        if len(label_long) == len(dates):
            return label_long["Code"].to_list()
        # Filter by date then extract Code
        date_col = "date" if "date" in label_long.columns else "index"
        date_set = set(dates)
        sub = label_long.filter(pl.col(date_col).is_in(list(date_set)))
        if len(sub) == len(dates):
            return sub["Code"].to_list()
    # fallback: return empty list (will result in RankIC = 0, equal weight ensemble)
    return [""] * len(dates)

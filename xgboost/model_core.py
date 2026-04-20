"""
Core model module
Reuses logic from hhx/dynamic3/model_XGBoost.py, supports seed parameter for ensemble use.
"""

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# I/O helper: read feather/IPC with type normalisation
# ============================================================
def _read_feather(path: str) -> pl.DataFrame:
    df = pl.read_ipc(path, memory_map=False)
    for col in ("date", "index"):
        if col in df.columns:
            if df[col].dtype == pl.String:
                df = df.with_columns(pl.col(col).str.strptime(pl.Datetime("us"), "%Y%m%d"))
            elif df[col].dtype != pl.Datetime("us"):
                df = df.with_columns(pl.col(col).cast(pl.Datetime("us")))
    if "Code" in df.columns and df["Code"].dtype == pl.String:
        df = df.with_columns(pl.col("Code").cast(pl.Categorical))
    return df


# ============================================================
# Polars acceleration: wide label → long + fac inner join
# ============================================================
def _pl_melt_join(fac_df: pl.DataFrame, label_wide: pl.DataFrame) -> pl.DataFrame:
    label_cols = [c for c in label_wide.columns if c != "index"]

    label_long = (
        label_wide.unpivot(
            on=label_cols,
            index="index",
            variable_name="Code",
            value_name="label",
        )
        .rename({"index": "date"})
    )

    fac = fac_df
    if fac["Code"].dtype != label_long["Code"].dtype:
        label_long = label_long.with_columns(pl.col("Code").cast(fac["Code"].dtype))

    return fac.join(label_long, on=["date", "Code"], how="inner")


# ============================================================
# Data preparation: cross-sectional MAD standardization (feature side)
# ============================================================
def prepare_features(
    fac_df: pl.DataFrame,
    label_df_raw: pl.DataFrame,
    feat_cols: list = None,
):
    merged = _pl_melt_join(fac_df, label_df_raw)

    # feature columns selection
    if feat_cols is None:
        candidate_cols = [c for c in merged.columns if c not in ("date", "Code", "label")]
        if candidate_cols:
            all_null_flags = merged.select(
                [pl.col(c).is_null().all().alias(c) for c in candidate_cols]
            ).row(0)
            feat_cols_out = [
                c for c, is_all_null in zip(candidate_cols, all_null_flags) if not is_all_null
            ]
        else:
            feat_cols_out = []
    else:
        feat_cols_out = [c for c in feat_cols if c in merged.columns]

    # early filters
    merged = merged.filter(pl.col("label").is_not_null())
    if len(merged) == 0:
        return np.array([]), [], [], feat_cols_out, np.array([])

    merged = merged.sort(["date", "Code"])

    # Filter out dates with fewer than 5 stocks (avoid window-in-agg; do via semi-join)
    ldf = merged.lazy()
    valid_dates = (
        ldf.group_by("date")
        .len()
        .filter(pl.col("len") >= 5)
        .select("date")
    )
    ldf = ldf.join(valid_dates, on="date", how="semi")

    # Robust normalization params
    EPS = 1e-6
    K = 1.4826
    keys = ["date"]

    if len(feat_cols_out) > 0:
        # 1) per-date median for each feature
        med_stats = ldf.group_by(keys).agg([
            pl.col(c).median().alias(f"__{c}_med") for c in feat_cols_out
        ])

        # 2) join medians, fill nulls using median (row-wise)
        ldf = (
            ldf.join(med_stats, on=keys, how="left")
            .with_columns([
                pl.col(c).fill_null(pl.col(f"__{c}_med")).alias(f"__{c}_filled")
                for c in feat_cols_out
            ])
        )

        # 3) per-date MAD and STD from filled + med columns
        mad_std_stats = ldf.group_by(keys).agg(
            [
                (pl.col(f"__{c}_filled") - pl.col(f"__{c}_med"))
                .abs()
                .median()
                .alias(f"__{c}_mad")
                for c in feat_cols_out
            ]
            + [
                pl.col(f"__{c}_filled").std(ddof=1).alias(f"__{c}_std")
                for c in feat_cols_out
            ]
        )

        # 4) join MAD/STD, normalize
        ldf = (
            ldf.join(mad_std_stats, on=keys, how="left")
            .with_columns([
                (
                    (pl.col(f"__{c}_filled") - pl.col(f"__{c}_med")) /
                    pl.when(K * pl.col(f"__{c}_mad") < EPS)
                      .then(pl.col(f"__{c}_std") + EPS)
                      .otherwise(K * pl.col(f"__{c}_mad"))
                ).alias(c)
                for c in feat_cols_out
            ])
        )

        # 5) drop temp cols
        tmp_cols = []
        for c in feat_cols_out:
            tmp_cols += [f"__{c}_med", f"__{c}_filled", f"__{c}_mad", f"__{c}_std"]
        ldf = ldf.drop(tmp_cols)

    merged = ldf.collect()

    if len(merged) == 0:
        return np.array([]), [], [], feat_cols_out, np.array([])

    X_out = merged.select(feat_cols_out).to_numpy().astype(np.float32) if feat_cols_out else np.empty((len(merged), 0), dtype=np.float32)
    dates = merged["date"].to_list()
    codes = merged["Code"].to_list()
    aligned_raw_labels = merged["label"].to_numpy()

    print(f"  prepare_features: X={X_out.shape}")
    return X_out, dates, codes, feat_cols_out, aligned_raw_labels


def prepare_labels_from_aligned(dates: list, codes: list,
                                 label_source: pl.DataFrame) -> np.ndarray:
    """
    Look up label values from pre-indexed label_source (date, Code, label),
    perform z-score normalization for each date cross-section.

    Parameters
    ----------
    dates        : Aligned date list (from prepare_features, already sorted)
    codes        : Aligned stock code list (from prepare_features)
    label_source : pl.DataFrame, containing ['date', 'Code', 'label'] columns

    Returns
    -------
    y : (n_samples,) float32 ndarray
    """
    EPS = 1e-6
    joined = (
        pl.DataFrame({"date": dates, "Code": codes})
        .join(label_source, on=["date", "Code"], how="left")
        .with_columns(pl.col("label").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("label").std().over("date") > EPS)
            .then(
                (pl.col("label") - pl.col("label").mean().over("date"))
                / pl.col("label").std().over("date")
            )
            .otherwise(0.0)
            .alias("label")
        )
    )
    return joined["label"].to_numpy().astype(np.float32)


# ============================================================
# XGBoost model wrapper (supports seed)
# ============================================================
class XGBRankModel:
    def __init__(self, params: dict):
        self.params = dict(params)
        self.model = None
        self.best_iteration = None

    def train(self, X_train, y_train, dates_train,
              X_val=None, y_val=None, dates_val=None,
              early_stopping_rounds=50,
              ndcg_weight=0.3, rankic_weight=0.7,
              ndcg_k=200, num_boost_round=2000, verbose=True,
              groups_train=None, groups_val=None,
              sample_weight=None):

        dtrain = xgb.DMatrix(X_train, label=y_train,
                             weight=sample_weight)
        dtrain.date_info = dates_train
        if groups_train is not None:
            dtrain.set_group(groups_train)

        evals = [(dtrain, "train")]
        dval = None
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.date_info = dates_val
            if groups_val is not None:
                dval.set_group(groups_val)
            evals.append((dval, "valid"))

        def _feval(preds, dmat):
            return composite_metric(
                preds, dmat, ndcg_k=ndcg_k,
                ndcg_weight=ndcg_weight, rankic_weight=rankic_weight
            )

        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="Composite",
            data_name="valid" if dval else "train",
            maximize=True,
        )

        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            custom_metric=_feval,
            maximize=True,
            verbose_eval=10 if verbose else False,
            evals_result=evals_result,
            callbacks=[early_stop],
        )

        self.best_iteration = getattr(early_stop, "best_iteration", None)
        if self.best_iteration is None:
            self.best_iteration = getattr(self.model, "best_iteration", None)

        if verbose:
            print(f"best_iteration: {self.best_iteration}")
            n_imp = len(self.model.get_score(importance_type="gain") or {})
            print(f"features with importance: {n_imp}")

    def predict(self, X, dates=None, codes=None):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        if dates is not None and codes is not None:
            return pl.DataFrame({"date": dates, "Code": codes, "score": preds})
        return preds

    def save(self, path):
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)

    def get_feature_importance(self):
        return self.model.get_score(importance_type="gain")


# ============================================================
# Evaluation metrics: RankIC / NDCG / composite
# ============================================================
def ndcg_k_metric(preds: np.ndarray, dtrain: xgb.DMatrix, k: int = 200):
    labels = dtrain.get_label()

    def _ndcg_nonneg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        """sklearn requires non-negative relevance scores.
        z-score labels span [-inf, inf], so we shift each group by its minimum
        value before computing NDCG. This preserves the relative ranking of
        labels within the group and makes all values >= 0.
        """
        y_min = y_true.min()
        if y_min < 0:
            y_true = y_true - y_min
        try:
            v = ndcg_score([y_true], [y_score], k=k)
            return float(v) if not np.isnan(v) else 0.0
        except Exception:
            return 0.0

    if not hasattr(dtrain, "date_info"):
        actual_k = min(k, len(preds))
        return f"NDCG@{k}", _ndcg_nonneg(labels, preds, actual_k)

    dates = dtrain.date_info
    tmp = pl.DataFrame({"date": dates, "pred": preds, "label": labels})
    ndcgs = []
    for grp in tmp.partition_by("date", maintain_order=True):
        if len(grp) < 2:
            continue
        ak = min(k, len(grp))
        v = _ndcg_nonneg(grp["label"].to_numpy(), grp["pred"].to_numpy(), ak)
        if v > 0.0:
            ndcgs.append(v)
    return f"NDCG@{k}", float(np.mean(ndcgs)) if ndcgs else 0.0


def rank_ic_metric(preds: np.ndarray, dtrain: xgb.DMatrix):
    labels = dtrain.get_label()
    if not hasattr(dtrain, "date_info"):
        if len(np.unique(preds)) < 2 or len(np.unique(labels)) < 2:
            return "RankIC", 0.0
        c = spearmanr(preds, labels).correlation
        return "RankIC", float(c) if not np.isnan(c) else 0.0

    dates = dtrain.date_info
    tmp = pl.DataFrame({"date": dates, "pred": preds, "label": labels})
    corrs = []
    for grp in tmp.partition_by("date", maintain_order=True):
        if len(grp) < 2:
            continue
        p, y = grp["pred"].to_numpy(), grp["label"].to_numpy()
        if len(np.unique(p)) < 2 or len(np.unique(y)) < 2:
            continue
        c = spearmanr(p, y).correlation
        if not np.isnan(c):
            corrs.append(c)
    return "RankIC", float(np.mean(corrs)) if corrs else 0.0


def composite_metric(preds: np.ndarray, dtrain: xgb.DMatrix,
                     ndcg_k: int = 250, ndcg_weight: float = 0.3,
                     rankic_weight: float = 0.7):
    _, ndcg_val = ndcg_k_metric(preds, dtrain, k=ndcg_k)
    _, ric_val = rank_ic_metric(preds, dtrain)
    composite = ndcg_weight * ndcg_val + rankic_weight * ric_val
    return [
        (f"NDCG@{ndcg_k}", float(ndcg_val)),
        ("RankIC", float(ric_val)),
        ("Composite", float(composite)),
    ]


# ============================================================
# Final evaluation: IC / ICIR / RankIC / RankICIR / top_return
# ============================================================
def get_cross_section_metrics(score_df: pl.DataFrame, ret_data: pl.DataFrame,
                               ndcg_k: int = 200):
    """
    Merge RankIC + NDCG groupby, single join + single scan.

    Returns
    -------
    (rankic_mean, rankicir, ndcg_mean)
    """
    merged = score_df.join(ret_data, on=["date", "Code"], how="inner")

    rics, ndcgs = [], []
    for grp in merged.partition_by("date", maintain_order=True):
        if len(grp) < 2:
            continue

        scores_np = grp["score"].to_numpy()
        labels_np = grp["label"].to_numpy()

        # RankIC
        if len(np.unique(scores_np)) >= 2 and len(grp) >= 5:
            c = spearmanr(scores_np, labels_np).correlation
            if not np.isnan(c):
                rics.append(c)

        # NDCG
        ak = min(ndcg_k, len(grp))
        try:
            v = ndcg_score([labels_np], [scores_np], k=ak)
            if not np.isnan(v):
                ndcgs.append(v)
        except Exception:
            pass

    ric_mean = float(np.mean(rics)) if rics else 0.0
    ric_std = float(np.std(rics)) if rics else 0.0
    rankicir = ric_mean / ric_std if ric_std > 1e-9 else 0.0
    ndcg_mean = float(np.mean(ndcgs)) if ndcgs else 0.0

    return ric_mean, rankicir, ndcg_mean


def get_ret_ic(score_df: pl.DataFrame, ret_data: pl.DataFrame,
               liquid_wide: pl.DataFrame,
               start=None, end=None, money: float = 1.5e9):
    """
    Simulate buying highest-scoring stocks, limited by liquidity, calculate daily returns and IC.

    Optimization: convert liquid_wide to long format once, then do a single 3-way join
    instead of 3 per-date filter scans (O(T) Polars filter calls → O(1) join + partition_by).
    """
    from datetime import datetime as _dt

    start_ts = _dt.strptime(str(start), "%Y%m%d") if start else None
    end_ts = _dt.strptime(str(end), "%Y%m%d") if end else None

    # Convert liquid_wide from wide to long format once (avoids per-date filter scans)
    liq = liquid_wide
    if "index" in liq.columns:
        liq = liq.rename({"index": "date"})
    liq_codes = [c for c in liq.columns if c != "date"]
    liq_long = liq.unpivot(on=liq_codes, index="date", variable_name="Code", value_name="liq")

    # Unify Code dtype with score_df
    score_code_dtype = score_df["Code"].dtype
    if liq_long["Code"].dtype != score_code_dtype:
        liq_long = liq_long.with_columns(pl.col("Code").cast(score_code_dtype))
    ret_renamed = ret_data.rename({"label": "ret"})
    if ret_renamed["Code"].dtype != score_code_dtype:
        ret_renamed = ret_renamed.with_columns(pl.col("Code").cast(score_code_dtype))

    # Filter date range, then single 3-way join → one combined long table
    combined = (
        score_df
        .filter((pl.col("date") >= start_ts) & (pl.col("date") <= end_ts))
        .join(ret_renamed, on=["date", "Code"], how="left")
        .with_columns(pl.col("ret").fill_null(0.0))
        .join(liq_long, on=["date", "Code"], how="left")
        .with_columns(pl.col("liq").fill_null(0.0))
        .sort(["date", "score"], descending=[False, True])
    )

    if combined.is_empty():
        empty = pl.DataFrame({"date": [], "ret": pl.Series([], dtype=pl.Float64),
                              "ic": pl.Series([], dtype=pl.Float64)})
        return empty["ret"], empty["ic"]

    rets_out, ics_out, dates_out = [], [], []

    # partition_by("date") once — already sorted by (date, score desc)
    for grp in combined.partition_by("date", maintain_order=True):
        dates_out.append(grp["date"][0])

        scores_np = grp["score"].to_numpy()
        ret_np = grp["ret"].to_numpy() * 100
        liq_np = np.clip(grp["liq"].to_numpy(), 0, None)

        # Vectorized portfolio simulation (grp already sorted by score desc)
        top_n = min(500, len(grp))
        liq_top = liq_np[:top_n]
        ret_top = ret_np[:top_n]
        cumliq = np.cumsum(liq_top)
        prev_cumliq = np.concatenate([[0.0], cumliq[:-1]])
        holds = np.minimum(liq_top, np.maximum(0.0, money - prev_cumliq))
        rets_out.append(float(np.dot(holds, ret_top)) / money)

        # IC: Pearson correlation between score and return (all stocks in group)
        if len(scores_np) >= 2 and np.std(scores_np) > 1e-9 and np.std(ret_np) > 1e-9:
            ic_val = float(np.corrcoef(scores_np, ret_np)[0, 1])
            ics_out.append(ic_val if not np.isnan(ic_val) else np.nan)
        else:
            ics_out.append(np.nan)

    result_df = pl.DataFrame({"date": dates_out, "ret": rets_out, "ic": ics_out})
    return result_df["ret"], result_df["ic"]


def get_metrics(score_df: pl.DataFrame, ret_data: pl.DataFrame,
                liquid_wide: pl.DataFrame,
                start=None, end=None, money: float = 1.5e9) -> dict:
    """
    Calculate IC / ICIR / RankIC / RankICIR / NDCG@200 / top_return / Stability.
    RankIC + NDCG merged into single groupby scan.
    """
    ret_series, ic_series = get_ret_ic(
        score_df, ret_data, liquid_wide, start=start, end=end, money=money
    )
    rankic, rankicir, ndcg200 = get_cross_section_metrics(score_df, ret_data, ndcg_k=200)

    # polars Series .mean() / .std() returns float | None
    ic_mean = float(ic_series.mean() or 0.0)
    ic_std = float(ic_series.std() or 0.0)
    icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0

    ret_mean = float(ret_series.mean() or 0.0)
    ret_std = float(ret_series.std() or 0.0)
    stability = ret_mean / ret_std if ret_std > 1e-9 else 0.0

    return {
        "IC": ic_mean,
        "ICIR": icir,
        "RankIC": float(rankic),
        "RankICIR": float(rankicir),
        "NDCG@200": float(ndcg200),
        "top_return": ret_mean,
        "Stability": stability,
    }

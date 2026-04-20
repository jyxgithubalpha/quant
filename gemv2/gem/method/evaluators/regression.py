"""
RegressionEvaluator -- generic regression evaluator with IC and portfolio metrics.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ...core.data import ProcessedBundle, SplitView
from ...core.training import EvalResult
from ..base.evaluator import BaseEvaluator
from .metrics import MetricRegistry
from .portfolio import PortfolioBacktest, PortfolioConfig


class RegressionEvaluator(BaseEvaluator):
    """
    Evaluator for regression / ranking models.

    Combines registry-based metrics (IC, ICIR, MSE, ...) with portfolio backtest
    metrics (top_ret, excess, stability, ...).
    """

    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        portfolio_top_k: int = 500,
        portfolio_money: float = 1.5e9,
        portfolio_min_trade_money: float = 1.0,
        portfolio_ret_scale: float = 1.0,
        portfolio_relative_improve_denom_floor: float = 0.01,
        portfolio_relative_improve_output_scale: float = 1.0,
        ret_col_candidates: Optional[List[str]] = None,
        liquidity_col_candidates: Optional[List[str]] = None,
        benchmark_col_candidates: Optional[List[str]] = None,
        benchmark_member_cols: Optional[List[str]] = None,
        use_benchmark_ensemble: bool = True,
    ) -> None:
        self.metric_names = metric_names or [
            "pearsonr_ic", "pearsonr_icir", "top_ret", "top_ret_std",
            "top_ret_stability", "model_benchmark_corr",
            "top_ret_excess", "top_ret_relative_improve_pct",
        ]
        self.portfolio = PortfolioBacktest(
            PortfolioConfig(
                top_k=portfolio_top_k,
                money=portfolio_money,
                min_trade_money=portfolio_min_trade_money,
                ret_scale=portfolio_ret_scale,
                relative_improve_denom_floor=portfolio_relative_improve_denom_floor,
                relative_improve_output_scale=portfolio_relative_improve_output_scale,
                ret_col_candidates=tuple(ret_col_candidates or ("ret__ret_value", "ret_value", "ret")),
                liquidity_col_candidates=tuple(liquidity_col_candidates or ("liquidity__liquidity_value", "liquidity_value", "liquidity")),
                benchmark_col_candidates=tuple(benchmark_col_candidates or ("score__score_value", "benchmark__benchmark_value", "score_value", "benchmark_value")),
                benchmark_member_cols=tuple(benchmark_member_cols or ("bench1__bench1_value", "bench2__bench2_value", "bench3__bench3_value", "bench4__bench4_value", "bench5__bench5_value", "bench6__bench6_value")),
                use_benchmark_ensemble=use_benchmark_ensemble,
            )
        )

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        views: ProcessedBundle,
    ) -> Dict[str, EvalResult]:
        backtest_names = PortfolioBacktest.METRIC_NAMES
        needs_backtest = any(name in backtest_names for name in self.metric_names)

        results: Dict[str, EvalResult] = {}
        for split, pred in predictions.items():
            view = views.get(split)

            # Portfolio backtest
            bt_metrics: Dict[str, float] = {}
            bt_series: Dict[str, pl.Series] = {}
            if needs_backtest:
                bt_metrics, bt_series = self.portfolio.compute(pred, view)

            # Registry metrics + backtest metrics
            metrics: Dict[str, float] = {}
            for name in self.metric_names:
                if MetricRegistry.has(name):
                    metrics[name] = MetricRegistry.get(name).compute(pred, view)
                elif name in backtest_names:
                    metrics[name] = float(bt_metrics.get(name, np.nan))

            # Daily IC series
            series = self._daily_ic_series(pred, view)
            series.update(bt_series)

            results[split] = EvalResult(
                metrics=metrics, series=series, predictions=pred, split=split,
            )

        return results

    @staticmethod
    def _daily_ic_series(pred: np.ndarray, view: SplitView) -> Dict[str, pl.Series]:
        if view.keys is None or "date" not in view.keys.columns:
            return {}

        pred = np.asarray(pred).ravel()
        y_true = np.asarray(view.y).ravel()
        dates = view.keys["date"].to_numpy()

        ic_values: List[float] = []
        ic_dates: List[int] = []

        for day in np.unique(dates):
            mask = dates == day
            if int(mask.sum()) < 2:
                continue
            pred_day, true_day = pred[mask], y_true[mask]
            if np.std(pred_day) < 1e-8 or np.std(true_day) < 1e-8:
                continue
            ic = float(np.corrcoef(pred_day, true_day)[0, 1])
            if np.isfinite(ic):
                ic_dates.append(int(day))
                ic_values.append(ic)

        return {
            "daily_ic": pl.Series("daily_ic", ic_values),
            "daily_ic_date": pl.Series("daily_ic_date", ic_dates),
        }

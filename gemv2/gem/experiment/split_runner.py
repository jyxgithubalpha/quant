"""
Split execution orchestration for one split task.
"""


import math
import traceback
from typing import Dict, Optional, Tuple

from ..core.data import GlobalDataset, SplitBundle
from ..core.training import TransformContext
from ..method.factory import PipelineFactory
from .context import RunContext
from .results import SplitResult, SplitTask
from .states import RollingState


class SplitRunner:
    """Run a single split end-to-end in a deterministic workflow."""

    def run(
        self,
        task: SplitTask,
        global_dataset: GlobalDataset,
        rolling_state: Optional[RollingState],
        ctx: RunContext,
    ) -> SplitResult:
        split_id = task.split_id

        try:
            views_bundle, skip_reason = self._build_split_bundle(task, global_dataset)
            if views_bundle is None:
                return SplitResult(
                    split_id=split_id,
                    metrics={},
                    skipped=True,
                    skip_reason=skip_reason,
                )

            pipeline, resolved_train_config = self._build_pipeline(ctx, split_id)

            # Translate RollingState -> TransformContext
            context: Optional[TransformContext] = (
                rolling_state.to_transform_context() if rolling_state else None
            )

            output = pipeline.run(
                views_bundle,
                resolved_train_config,
                should_tune=ctx.should_tune and (pipeline.tuner is not None),
                save_dir=ctx.split_dir(split_id),
                context=context,
            )

            metrics_flat = self._flatten_metrics(output.metrics)
            metric_series_rows = self._flatten_metric_series(output.metrics)
            test_predictions = None
            if "test" in output.metrics:
                test_predictions = output.metrics["test"].predictions

            state_delta = output.get_state_delta()
            return SplitResult(
                split_id=split_id,
                importance_vector=state_delta.importance,
                feature_names_hash=state_delta.feature_hash,
                metrics=metrics_flat,
                best_params=state_delta.best_params,
                best_objective=state_delta.best_objective,
                state_delta=state_delta,
                test_predictions=test_predictions,
                test_keys=views_bundle.test.keys,
                test_extra=views_bundle.test.extra,
                metric_series_rows=metric_series_rows,
                failed=False,
            )

        except Exception as exc:
            trace_text = traceback.format_exc()
            trace_path = ctx.split_dir(split_id) / "error_traceback.txt"
            trace_path.write_text(trace_text, encoding="utf-8")
            return SplitResult(
                split_id=split_id,
                metrics={"error": f"{type(exc).__name__}: {exc}"},
                failed=True,
                skipped=False,
                skip_reason=None,
                error_message=f"{type(exc).__name__}: {exc}",
                error_trace_path=str(trace_path),
            )

    def _build_split_bundle(
        self,
        task: SplitTask,
        global_dataset: GlobalDataset,
    ) -> Tuple[Optional[SplitBundle], Optional[str]]:
        spec = task.splitspec

        idx_train = global_dataset.get_indices_by_dates(spec.train_date_list)
        idx_val = global_dataset.get_indices_by_dates(spec.val_date_list)
        idx_test = global_dataset.get_indices_by_dates(spec.test_date_list)

        empty_sets = []
        if len(idx_train) == 0:
            empty_sets.append("train")
        if len(idx_val) == 0:
            empty_sets.append("val")
        if len(idx_test) == 0:
            empty_sets.append("test")

        if empty_sets:
            return None, f"Empty sets: {', '.join(empty_sets)}"

        views_bundle = SplitBundle(
            train=global_dataset.take(idx_train),
            val=global_dataset.take(idx_val),
            test=global_dataset.take(idx_test),
            split_spec=spec,
        )
        return views_bundle, None

    def _build_pipeline(self, ctx: RunContext, split_id: int):
        return PipelineFactory.build(
            method_config=ctx.method_config,
            train_config=ctx.train_config,
            n_trials=ctx.n_trials,
            parallel_trials=ctx.parallel_trials,
            base_seed=ctx.seed,
            split_id=split_id,
        )

    def _flatten_metrics(self, metrics: Dict[str, object]) -> Dict[str, float]:
        metrics_flat: Dict[str, float] = {}

        for split, eval_result in metrics.items():
            for name, value in eval_result.metrics.items():
                metrics_flat[f"{split}_{name}"] = value

        return metrics_flat

    @staticmethod
    def _coerce_date_int(raw_value: object) -> Optional[int]:
        if raw_value is None:
            return None
        try:
            date_int = int(raw_value)
        except (TypeError, ValueError):
            return None
        text = str(date_int)
        if len(text) != 8:
            return None
        return date_int

    def _flatten_metric_series(self, metrics: Dict[str, object]) -> list[dict]:
        rows: list[dict] = []

        for split, eval_result in metrics.items():
            series_map = getattr(eval_result, "series", None)
            if not series_map:
                continue

            for metric_name, value_series in series_map.items():
                if metric_name.endswith("_date"):
                    continue
                date_key = f"{metric_name}_date"
                date_series = series_map.get(date_key)
                if date_series is None:
                    continue

                value_list = value_series.to_list() if hasattr(value_series, "to_list") else list(value_series)
                date_list = date_series.to_list() if hasattr(date_series, "to_list") else list(date_series)
                n = min(len(value_list), len(date_list))

                for idx in range(n):
                    date_int = self._coerce_date_int(date_list[idx])
                    if date_int is None:
                        continue
                    try:
                        value = float(value_list[idx])
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value):
                        continue
                    rows.append(
                        {
                            "split": split,
                            "metric": metric_name,
                            "date": date_int,
                            "value": value,
                        }
                    )

        return rows

"""
EnsemblePipeline -- composite pipeline that trains multiple ModelPipelines
and combines their predictions via an ensemble strategy.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ...core.data import ProcessedBundle, SplitBundle, SplitView
from ...core.training import (
    EvalResult,
    RunOutput,
    StateDelta,
    TrainConfig,
    TransformContext,
)
from ..base.evaluator import BaseEvaluator
from ..base.pipeline import ModelPipeline, _hash_feature_names
from .strategy import BaseEnsembleStrategy

log = logging.getLogger(__name__)


class EnsemblePipeline:
    """
    Composite pipeline: manages multiple ModelPipelines, combines via strategy.

    ``run()`` has the same signature as ``ModelPipeline.run()`` so that
    SplitRunner can dispatch without isinstance checks.
    """

    def __init__(
        self,
        pipelines: Dict[str, ModelPipeline],
        configs: Dict[str, TrainConfig],
        strategy: BaseEnsembleStrategy,
        evaluator: BaseEvaluator,
    ) -> None:
        self.pipelines = pipelines
        self.configs = configs
        self.strategy = strategy
        self.evaluator = evaluator

    def run(
        self,
        views: Union[SplitBundle, ProcessedBundle],
        config: TrainConfig,
        should_tune: bool = True,
        save_dir: Optional[Path] = None,
        context: Optional[TransformContext] = None,
    ) -> RunOutput:
        # ① Train each sub-pipeline independently
        sub_outputs: Dict[str, RunOutput] = {}
        for name, pipeline in self.pipelines.items():
            sub_config = self.configs.get(name, config)
            sub_dir = save_dir / name if save_dir else None
            sub_outputs[name] = pipeline.run(views, sub_config, should_tune, sub_dir, context)

        # ② Collect predictions per split
        def _collect(split: str) -> Dict[str, np.ndarray]:
            return {
                name: out.metrics[split].predictions
                for name, out in sub_outputs.items()
                if out.metrics.get(split) is not None and out.metrics[split].predictions is not None
            }

        # ③ Fit strategy on val, combine all splits
        val_view = self._get_view(views, "val")
        self.strategy.fit(_collect("val"), val_view)

        combined: Dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            preds = _collect(split)
            if preds:
                combined[split] = self.strategy.combine(preds, self._get_view(views, split))

        # ④ Evaluate combined predictions
        eval_views = self._as_processed(views)
        eval_results = self.evaluator.evaluate(combined, eval_views)

        # ⑤ Aggregate importance
        importance = self._aggregate_importance(sub_outputs)
        feature_names = self._get_view(views, "train").feature_names
        fhash = _hash_feature_names(feature_names)

        return RunOutput(
            best_params={name: out.best_params for name, out in sub_outputs.items()},
            metrics=eval_results,
            importance=importance,
            feature_hash=fhash,
            state_delta=StateDelta(importance=importance, feature_hash=fhash),
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _get_view(views: Union[SplitBundle, ProcessedBundle], split: str) -> SplitView:
        if isinstance(views, ProcessedBundle):
            return views.get(split)
        return {"train": views.train, "val": views.val, "test": views.test}[split]

    @staticmethod
    def _as_processed(views: Union[SplitBundle, ProcessedBundle]) -> ProcessedBundle:
        if isinstance(views, ProcessedBundle):
            return views
        return ProcessedBundle(
            train=views.train, val=views.val, test=views.test,
            split_spec=views.split_spec,
        )

    @staticmethod
    def _aggregate_importance(sub_outputs: Dict[str, RunOutput]) -> np.ndarray:
        vecs = [o.importance for o in sub_outputs.values() if o.importance is not None and o.importance.size > 0]
        if not vecs:
            return np.array([])
        dims = {v.shape[0] for v in vecs}
        if len(dims) == 1:
            return np.mean(vecs, axis=0)
        # Different dimensions: take first sub-pipeline's importance
        return next(iter(sub_outputs.values())).importance

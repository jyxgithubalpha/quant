"""
ModelPipeline -- single-model training pipeline.

Orchestrates: transform -> tune -> train -> predict -> evaluate -> importance.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ...core.data import ProcessedBundle, SplitBundle
from ...core.training import (
    EvalResult,
    FitResult,
    RunOutput,
    StateDelta,
    TrainConfig,
    TransformContext,
    TuneResult,
)
from ...transforms.pipeline import TransformPipeline
from .evaluator import BaseEvaluator
from .importance import BaseImportanceExtractor
from .trainer import BaseTrainer
from .tuner import BaseTuner

log = logging.getLogger(__name__)


def _hash_feature_names(names) -> str:
    return hashlib.md5(json.dumps(sorted(names)).encode()).hexdigest()[:12]


class ModelPipeline:
    """
    Single-model training pipeline.

    Components are injected at construction time (by PipelineFactory from Hydra config).
    The ``run()`` method drives the full workflow and returns a ``RunOutput``.
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        evaluator: BaseEvaluator,
        importance_extractor: BaseImportanceExtractor,
        transform_pipeline: Optional[TransformPipeline] = None,
        tuner: Optional[BaseTuner] = None,
        adapter: Optional[Any] = None,
    ) -> None:
        self.trainer = trainer
        self.evaluator = evaluator
        self.importance_extractor = importance_extractor
        self.transform_pipeline = transform_pipeline
        self.tuner = tuner
        self.adapter = adapter

    # -- public API -------------------------------------------------------

    def run(
        self,
        views: Union[SplitBundle, ProcessedBundle],
        config: TrainConfig,
        should_tune: bool = True,
        save_dir: Optional[Path] = None,
        context: Optional[TransformContext] = None,
    ) -> RunOutput:
        """
        Execute the full pipeline.

        Args:
            views: raw SplitBundle or already-processed ProcessedBundle.
            config: training configuration.
            should_tune: whether to run hyper-parameter search.
            save_dir: directory for model / importance artifacts.
            context: transform context from RollingState (feature weights, etc.).
        """
        # ① Transform
        processed = self._ensure_processed(views, context)

        # ② Tune
        best_params, tune_result = self._run_tuning(processed, config, should_tune)

        # ③ Build final config
        final_config = TrainConfig(
            params=best_params,
            max_iterations=config.max_iterations,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metrics=list(config.monitor_metrics),
            seed=config.seed,
            log_interval=config.log_interval,
        )

        # ④ Train
        fit_result = self.trainer.fit(processed, final_config, phase="full")

        # ⑤ Predict
        predictions = self._predict_splits(fit_result.model, processed)

        # ⑥ Evaluate
        eval_results = self.evaluator.evaluate(predictions, processed)

        # ⑦ Importance
        imp_vec, imp_df = self.importance_extractor.extract(
            fit_result.model, processed.train.feature_names,
        )
        fit_result.feature_importance = imp_df

        # ⑧ Artifacts
        artifacts = self._save_artifacts(fit_result.model, imp_df, save_dir)

        # ⑨ Output
        fhash = _hash_feature_names(processed.train.feature_names)
        state_delta = StateDelta(
            importance=imp_vec,
            feature_hash=fhash,
            best_params=best_params,
            best_objective=tune_result.best_value if tune_result else None,
        )

        return RunOutput(
            best_params=best_params,
            metrics=eval_results,
            importance=imp_vec,
            feature_hash=fhash,
            tune_result=tune_result,
            fit_result=fit_result,
            state_delta=state_delta,
            artifacts=artifacts,
        )

    # -- internal helpers -------------------------------------------------

    def _ensure_processed(
        self, views: Union[SplitBundle, ProcessedBundle], context: Optional[TransformContext],
    ) -> ProcessedBundle:
        if isinstance(views, ProcessedBundle):
            return views
        if self.transform_pipeline is not None:
            return self.transform_pipeline.fit_transform_views(views, context)
        return ProcessedBundle(
            train=views.train, val=views.val, test=views.test,
            split_spec=views.split_spec,
        )

    def _run_tuning(
        self, views: ProcessedBundle, config: TrainConfig, should_tune: bool,
    ) -> Tuple[Dict[str, Any], Optional[TuneResult]]:
        best_params = dict(config.params)
        if not should_tune or self.tuner is None:
            return best_params, None
        tune_result = self.tuner.tune(views, self.trainer, config)
        if tune_result.best_params:
            best_params = dict(tune_result.best_params)
        return best_params, tune_result

    def _predict_splits(
        self, model: Any, views: ProcessedBundle,
    ) -> Dict[str, np.ndarray]:
        return {
            split: self.trainer.predict(model, views.get(split).X)
            for split in ("train", "val", "test")
        }

    @staticmethod
    def _save_artifacts(
        model: Any, imp_df: Any, save_dir: Optional[Path],
    ) -> Dict[str, Path]:
        if save_dir is None:
            return {}
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Path] = {}

        if hasattr(model, "save_model"):
            model_path = path / "model.txt"
            model.save_model(str(model_path))
            artifacts["model"] = model_path

        if imp_df is not None:
            imp_path = path / "feature_importance.csv"
            imp_df.write_csv(str(imp_path))
            artifacts["importance"] = imp_path

        return artifacts

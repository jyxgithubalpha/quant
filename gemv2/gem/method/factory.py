"""
PipelineFactory -- assembles ModelPipeline or EnsemblePipeline from Hydra config.
"""

import inspect
import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from ..core.training import TrainConfig
from .base.pipeline import ModelPipeline
from .ensemble.pipeline import EnsemblePipeline

log = logging.getLogger(__name__)


class PipelineFactory:
    """
    Build a ModelPipeline or EnsemblePipeline from Hydra DictConfig.

    Usage::

        pipeline, train_config = PipelineFactory.build(
            method_config=cfg.method,
            train_config=train_config,
            n_trials=50,
            ...
        )
        output = pipeline.run(views, train_config, ...)
    """

    @staticmethod
    def build(
        *,
        method_config: Optional[Mapping[str, Any]],
        train_config: TrainConfig,
        n_trials: int,
        parallel_trials: int = 1,
        base_seed: int = 42,
        split_id: int = 0,
    ) -> Tuple[Union[ModelPipeline, EnsemblePipeline], TrainConfig]:
        config = PipelineFactory._to_dict(method_config)

        if "ensemble" in config:
            return PipelineFactory._build_ensemble(
                config, train_config, n_trials, parallel_trials, base_seed, split_id,
            )
        return PipelineFactory._build_single(
            config, train_config, n_trials, parallel_trials, base_seed, split_id,
        )

    # -- single model pipeline -------------------------------------------

    @staticmethod
    def _build_single(
        config: dict,
        train_config: TrainConfig,
        n_trials: int,
        parallel_trials: int,
        base_seed: int,
        split_id: int,
    ) -> Tuple[ModelPipeline, TrainConfig]:

        adapter = None
        if config.get("adapter") is not None:
            adapter = instantiate(config["adapter"])

        trainer_overrides: Dict[str, Any] = {}
        if adapter is not None:
            trainer_overrides["adapter"] = adapter
        model_config = config.get("model")
        if model_config is not None:
            trainer_overrides["model_config"] = model_config
        trainer = instantiate(
            config["trainer"],
            **PipelineFactory._filter_overrides(config["trainer"], trainer_overrides),
        )

        evaluator = instantiate(config["evaluator"])
        importance_extractor = instantiate(config["importance_extractor"])

        transform_pipeline = None
        if config.get("transform_pipeline") is not None:
            transform_pipeline = instantiate(config["transform_pipeline"])

        tuner = None
        if n_trials > 0 and config.get("tuner") is not None:
            tuner = instantiate(config["tuner"])

        pipeline = ModelPipeline(
            trainer=trainer,
            evaluator=evaluator,
            importance_extractor=importance_extractor,
            transform_pipeline=transform_pipeline,
            tuner=tuner,
            adapter=adapter,
        )

        resolved_tc = PipelineFactory._resolve_train_config(train_config, model_config)
        return pipeline, resolved_tc

    # -- ensemble pipeline -----------------------------------------------

    @staticmethod
    def _build_ensemble(
        config: dict,
        train_config: TrainConfig,
        n_trials: int,
        parallel_trials: int,
        base_seed: int,
        split_id: int,
    ) -> Tuple[EnsemblePipeline, TrainConfig]:

        pipelines: Dict[str, ModelPipeline] = {}
        configs: Dict[str, TrainConfig] = {}

        for name, sub_cfg in config["ensemble"]["methods"].items():
            sub_dict = PipelineFactory._to_dict(sub_cfg)
            p, tc = PipelineFactory._build_single(
                sub_dict, train_config, n_trials, parallel_trials, base_seed, split_id,
            )
            pipelines[name] = p
            configs[name] = tc

        strategy = instantiate(config["ensemble"]["strategy"])
        evaluator = instantiate(config["evaluator"])

        ensemble = EnsemblePipeline(
            pipelines=pipelines,
            configs=configs,
            strategy=strategy,
            evaluator=evaluator,
        )
        return ensemble, train_config

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _to_dict(config: Optional[Mapping[str, Any]]) -> dict:
        if config is None:
            return {}
        if OmegaConf.is_config(config):
            resolved = OmegaConf.to_container(config, resolve=True)
            return dict(resolved) if isinstance(resolved, Mapping) else {}
        return dict(config) if isinstance(config, Mapping) else {}

    @staticmethod
    def _filter_overrides(component_config: Any, overrides: Mapping[str, Any]) -> Dict[str, Any]:
        target = None
        if isinstance(component_config, Mapping):
            target = component_config.get("_target_")
        elif hasattr(component_config, "get"):
            target = component_config.get("_target_")
        if not target:
            return dict(overrides)
        try:
            cls = get_class(str(target))
            sig = inspect.signature(cls.__init__)
        except Exception:
            return dict(overrides)
        return {k: v for k, v in overrides.items() if k in sig.parameters}

    @staticmethod
    def _resolve_train_config(
        train_config: TrainConfig,
        model_config: Optional[Any],
    ) -> TrainConfig:
        if model_config is None:
            return train_config
        model_defaults = PipelineFactory._to_dict(model_config)
        model_defaults.pop("_target_", None)
        merged = {**model_defaults, **dict(train_config.params or {})}
        return TrainConfig(
            params=merged,
            max_iterations=train_config.max_iterations,
            early_stopping_patience=train_config.early_stopping_patience,
            monitor_metrics=list(train_config.monitor_metrics),
            seed=train_config.seed,
            log_interval=train_config.log_interval,
        )

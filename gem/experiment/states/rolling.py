"""
RollingState - Rolling state container.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np

from .base import BaseState
from .concrete import FeatureImportanceState, SampleWeightState, TuningState


@dataclass
class RollingState:
    """
    Rolling state container - manages multiple pluggable State components

    Supports dynamic registration and retrieval of different types of State, and can convert all State
    into unified transform context to pass to Transform Pipeline.

    Attributes:
        states: State dictionary {state_key: BaseState}
        split_history: Historical split ID list
        metadata: Additional metadata
    """
    states: Dict[str, BaseState] = field(default_factory=dict)
    split_history: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def register_state(self, state: BaseState) -> None:
        """Register a state component"""
        self.states[state.state_key()] = state

    def get_state(self, state_class: Type[BaseState]) -> Optional[BaseState]:
        """Get state of specified type"""
        key = state_class.state_key()
        return self.states.get(key)

    def get_or_create_state(self, state_class: Type[BaseState]) -> BaseState:
        """Get or create state of specified type"""
        key = state_class.state_key()
        if key not in self.states:
            self.states[key] = state_class()
        return self.states[key]  # type: ignore

    def has_state(self, state_class: Type[BaseState]) -> bool:
        """Check if state of specified type exists"""
        return state_class.state_key() in self.states

    def to_transform_context(self) -> Dict[str, Any]:
        """
        Convert all states to transform context

        The returned dict can be passed to BaseTransformPipeline.process_views()
        """
        context = {}
        for state in self.states.values():
            context.update(state.to_transform_context())
        return context

    # =========================================================================
    # Convenience methods - shortcuts for common operations
    # =========================================================================

    def update_importance(
        self,
        new_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.3,
    ) -> None:
        """Update feature importance (convenience method)"""
        state = self.get_or_create_state(FeatureImportanceState)
        state.update(new_importance, feature_names, alpha)

    def update_tuning(
        self,
        best_params: Dict[str, Any],
        best_objective: float,
    ) -> None:
        """Update tuning state (convenience method)"""
        state = self.get_or_create_state(TuningState)
        state.update(best_params, best_objective)

    @property
    def feature_importance(self) -> Optional[FeatureImportanceState]:
        """Get feature importance state (convenience property)"""
        return self.get_state(FeatureImportanceState)

    @property
    def tuning(self) -> Optional[TuningState]:
        """Get tuning state (convenience property)"""
        return self.get_state(TuningState)

    @property
    def sample_weight(self) -> Optional[SampleWeightState]:
        """Get sample weight state (convenience property)"""
        return self.get_state(SampleWeightState)

    def save(self, path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "RollingState":
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a RollingState when loading '{path}', "
                f"got {type(obj).__name__}. Only load files saved by RollingState.save()."
            )
        return obj

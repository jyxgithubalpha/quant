"""
BaseState - Abstract base class for state components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseState(ABC):
    """
    Pluggable state base class

    All state components that can be stored in RollingState should inherit from this class.
    Each State class should define its own state_key as a unique identifier.
    """

    @classmethod
    @abstractmethod
    def state_key(cls) -> str:
        """Unique identifier for the state"""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update state"""
        pass

    def to_transform_context(self) -> Dict[str, Any]:
        """
        Convert to context usable by transform

        The returned dict will be passed to Transform.fit/transform methods
        """
        return {}

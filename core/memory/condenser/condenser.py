from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

# Replace placeholder with the real import
from core.config.condenser_config import CondenserConfig
from core.events.action import CondensationAction
from core.memory.view import View
from core.controller.state.state import State

# Placeholder class until the actual State file is created/located
# class State(BaseModel):
#     extra_data: dict = {}
#     view: 'View' = View(events=[])
#     def to_llm_metadata(self, key: str) -> dict:
#         return {}

CONDENSER_METADATA_KEY = 'condenser_meta'
"""Key identifying where metadata is stored in a `State` object's `extra_data` field."""

def get_condensation_metadata(state: State) -> list[dict[str, Any]]:
    """Utility function to retrieve a list of metadata batches from a `State`."""
    if CONDENSER_METADATA_KEY in state.extra_data:
        return state.extra_data[CONDENSER_METADATA_KEY]
    return []


CONDENSER_REGISTRY: dict[type[CondenserConfig], type[Condenser]] = {}
"""Registry of condenser configurations to their corresponding condenser classes."""


class Condensation(BaseModel):
    """Produced by a condenser to indicate the history has been condensed."""

    action: CondensationAction


class Condenser(ABC):
    """Abstract condenser interface."""

    def __init__(self):
        self._metadata_batch: dict[str, Any] = {}
        self._llm_metadata: dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any) -> None:
        self._metadata_batch[key] = value

    def write_metadata(self, state: State) -> None:
        if CONDENSER_METADATA_KEY not in state.extra_data:
            state.extra_data[CONDENSER_METADATA_KEY] = []
        if self._metadata_batch:
            state.extra_data[CONDENSER_METADATA_KEY].append(self._metadata_batch)
        self._metadata_batch = {}

    @contextmanager
    def metadata_batch(self, state: State):
        try:
            yield
        finally:
            self.write_metadata(state)

    @abstractmethod
    def condense(self, view: View) -> View | Condensation:
        """Condense a sequence of events into a potentially smaller list.

        New condenser strategies should override this method to implement their own condensation logic. Call `self.add_metadata` in the implementation to record any relevant per-condensation diagnostic information.

        Args:
            view: A view of the history containing all events that should be condensed.

        Returns:
            View | Condensation: A condensed view of the events or an event indicating the history has been condensed.
        """

    def condensed_history(self, state: State) -> View | Condensation:
        self._llm_metadata = state.to_llm_metadata('condenser')
        with self.metadata_batch(state):
            return self.condense(state.view)

    @classmethod
    def register_config(cls, configuration_type: type[CondenserConfig]) -> None:
        if configuration_type in CONDENSER_REGISTRY:
            raise ValueError(
                f'Condenser configuration {configuration_type} is already registered'
            )
        CONDENSER_REGISTRY[configuration_type] = cls

    @classmethod
    def from_config(cls, config: CondenserConfig) -> Condenser:
        try:
            condenser_class = CONDENSER_REGISTRY[type(config)]
            return condenser_class.from_config(config)
        except KeyError:
            raise ValueError(f'Unknown condenser config: {config}')


class RollingCondenser(Condenser, ABC):
    """Base class for a specialized condenser strategy that applies condensation to a rolling history."""

    @abstractmethod
    def should_condense(self, view: View) -> bool:
        """Determine if a view should be condensed."""

    @abstractmethod
    def get_condensation(self, view: View) -> Condensation:
        """Get the condensation from a view."""

    def condense(self, view: View) -> View | Condensation:
        if self.should_condense(view):
            return self.get_condensation(view)
        else:
            return view 

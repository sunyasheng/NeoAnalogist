from dataclasses import dataclass

from core.events.event import Action, ActionType


@dataclass
class NullAction(Action):
    """An action that does nothing."""

    action: str = ActionType.NULL

    @property
    def message(self) -> str:
        return "No action"

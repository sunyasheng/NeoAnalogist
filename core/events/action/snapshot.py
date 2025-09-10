from dataclasses import dataclass, field
from typing import ClassVar

from core.events.event import Action, ActionType


@dataclass
class SnapshotAction(Action):
    """Action for creating a repository snapshot."""
    repo_directory: str = field(default="", init=True)
    thought: str = ""
    action: str = ActionType.SNAPSHOT
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"I am creating a snapshot of the repository at: {self.repo_directory}"

    def __str__(self) -> str:
        ret = "**SnapshotAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_DIRECTORY: {self.repo_directory}"
        return ret 
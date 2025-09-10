from dataclasses import dataclass, field
from typing import ClassVar

from core.events.event import Action, ActionType


@dataclass
class RollbackAction(Action):
    """Action for rolling back a repository to a previous snapshot."""
    repo_directory: str = field(default="", init=True)
    tag: str = field(default="", init=True)
    thought: str = ""
    action: str = ActionType.ROLLBACK
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"I am rolling back the repository at {self.repo_directory} to snapshot: {self.tag}"

    def __str__(self) -> str:
        ret = "**RollbackAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_DIRECTORY: {self.repo_directory}\n"
        ret += f"TAG: {self.tag}"
        return ret

from dataclasses import dataclass, field
from typing import ClassVar

from core.events.event import Observation, ObservationType


@dataclass
class SnapshotObservation(Observation):
    """Observation that represents the result of a snapshot action.
    
    This observation contains the git commit tag of the created snapshot,
    which can be used to reference or restore the snapshot later.
    
    Attributes:
        tag (str): The git commit tag of the created snapshot
        observation (str): The observation type, namely ObservationType.SNAPSHOT
    """
    
    tag: str = field(default="", init=True)
    content: str = ""
    observation: str = ObservationType.SNAPSHOT
    
    @property
    def message(self) -> str:
        return f"Successfully created snapshot with tag: {self.tag}"

    def __str__(self) -> str:
        ret = "**SnapshotObservation**\n"
        ret += f"TAG: {self.tag}"
        return ret


@dataclass
class RollbackObservation(Observation):
    """Observation containing the result of a rollback operation."""
    tag: str = field(default="", init=True)
    content: str = ""
    observation: str = ObservationType.ROLLBACK

    @property
    def message(self) -> str:
        return f"Successfully rolled back to snapshot with tag: {self.tag}"

    def __str__(self) -> str:
        ret = "**RollbackObservation**\n"
        ret += f"TAG: {self.tag}"
        return ret
from dataclasses import dataclass

from core.events.event import Observation, ObservationType


@dataclass
class ErrorObservation(Observation):
    """This data class represents an error encountered by the agent.

    This is the type of error that LLM can recover from.
    E.g., Linter error after editing a file.
    """

    observation: str = ObservationType.ERROR
    error_id: str = ""

    @property
    def message(self) -> str:
        return self.content

    def __str__(self) -> str:
        return f"**ErrorObservation**\n{self.content}"

from dataclasses import dataclass
from core.events.event import Observation, ObservationType

@dataclass
class TaskGraphBuildObservation(Observation):
    """This data class represents the task graph built from a task description.
    
    The observation contains a string representation of the task graph,
    showing the execution topology of the task.
    """

    task_description: str
    task_graph_str: str
    observation: str = ObservationType.TASK_GRAPH_BUILD

    @property
    def message(self) -> str:
        """Get a human-readable message describing the task graph building operation."""
        return f"I built a task graph from the description: {self.task_description[:50]}..."

    def __str__(self) -> str:
        """Get a string representation of the task graph observation."""
        return f"[Task graph built successfully.]\n{self.task_graph_str}" 
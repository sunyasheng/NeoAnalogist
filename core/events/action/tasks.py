from dataclasses import dataclass
from typing import ClassVar
from core.events.event import Action, ActionType

@dataclass
class TaskGraphBuildAction(Action):
    """Action that builds a task graph from a task description.
    
    This action takes a task description as input and will produce a task graph
    as its observation, representing the execution topology of the task.
    
    Attributes:
        task_description (str): The description of the task to be decomposed
        thought (str): The reasoning behind the task decomposition
        action (str): The action type, namely ActionType.TASK_GRAPH_BUILD
    """
    
    task_description: str
    thought: str = ""
    action: str = ActionType.TASK_GRAPH_BUILD
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Building task graph from description: {self.task_description[:50]}..."

    def __repr__(self) -> str:
        ret = "**TaskGraphBuildAction**\n"
        ret += f"Task Description: {self.task_description}\n"
        ret += f"Thought: {self.thought}\n"
        return ret

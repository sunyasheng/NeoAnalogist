import abc
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

# from core.graph import DAG, Edge, Node


class Agent(abc.ABC):
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
    ):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"Agent:{agent_id}")
        self.state = None
        self.done = False

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def reset(self):
        self.state = None

    def policy(self, observation: Any, task: Optional[str] = None) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def step(self, observation: Any, task: Optional[str] = None) -> Dict[str, Any]:
        action = self.policy(observation, task)
        result = self.execute(action)

        response = {
            "policy": action,
            "execution_result": result,
            "done": self.done,
            "step_time": time.time(),
        }

        if self.episodic_memory.active_episode is None:
            self.episodic_memory.create_episode(
                name=f"Session_{int(time.time())}",
                description=f"Auto-created session for agent {self.agent_id}",
                tags=["auto"],
            )

        return response

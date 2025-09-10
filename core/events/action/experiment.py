from dataclasses import dataclass
from typing import ClassVar

from core.events.event import Action, ActionType


@dataclass
class ExperimentManagerAction(Action):
    """Action for managing and executing experiments.

    Supports modes like:
    - auto: build an experiment (optionally using exp_name) and/or run with a provided command
    - list: enumerate previously executed experiments and their results

    Attributes:
        command: Bash command to execute for the experiment (e.g., "python train.py"). Optional in list mode.
        output_dir: Directory to store experiment plans/results.
        mode: One of {"auto", "list"}.
        repo_path: Target repository path in the container.
        experiment_name: Optional experiment name; auto mode may default if not provided.
        runs: Optional run identifier(s) under the experiment.
        thought: Reasoning or notes.
        action: The action type, namely ActionType.EXPERIMENT_MANAGER
    """

    command: str = ""
    output_dir: str = ""
    mode: str = "auto"
    repo_path: str = ""
    experiment_name: str = ""
    runs: str = ""
    thought: str = ""
    action: str = ActionType.EXPERIMENT_MANAGER
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        mode_part = f"mode={self.mode}"
        name_part = f", exp={self.experiment_name}" if self.experiment_name else ""
        runs_part = f", runs={self.runs}" if self.runs else ""
        cmd_part = f", cmd={self.command}" if self.command else ""
        return f"ExperimentManager: {mode_part}{name_part}{runs_part}{cmd_part}"

    def __str__(self) -> str:
        ret = "**ExperimentManagerAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"MODE: {self.mode}\n"
        if self.experiment_name:
            ret += f"EXPERIMENT_NAME: {self.experiment_name}\n"
        if self.runs:
            ret += f"RUNS: {self.runs}\n"
        if self.command:
            ret += f"COMMAND: {self.command}\n"
        if self.repo_path:
            ret += f"REPO_PATH: {self.repo_path}\n"
        if self.output_dir:
            ret += f"OUTPUT_DIR: {self.output_dir}\n"
        return ret

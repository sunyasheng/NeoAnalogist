from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar

from litellm import ModelResponse
from pydantic import BaseModel, Field


class ToolCallMetadata(BaseModel):
    # See https://docs.litellm.ai/docs/completion/function_call#step-3---second-litellmcompletion-call
    function_name: str  # Name of the function that was called
    tool_call_id: str  # ID of the tool call

    model_response: ModelResponse
    total_calls_in_response: int


class EventSource(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENVIRONMENT = "environment"


@dataclass
class Event:
    INVALID_ID = -1

    @property
    def message(self) -> str | None:
        if hasattr(self, "_message"):
            return self._message  # type: ignore[attr-defined]
        return ""

    @property
    def id(self) -> int:
        if hasattr(self, "_id"):
            return self._id  # type: ignore[attr-defined]
        return Event.INVALID_ID

    @property
    def timestamp(self):
        if hasattr(self, "_timestamp") and isinstance(self._timestamp, str):
            return self._timestamp

    @timestamp.setter
    def timestamp(self, value: datetime) -> None:
        if isinstance(value, datetime):
            self._timestamp = value.isoformat()

    @property
    def source(self) -> EventSource | None:
        if hasattr(self, "_source"):
            return self._source  # type: ignore[attr-defined]
        return None

    @property
    def cause(self) -> int | None:
        if hasattr(self, "_cause"):
            return self._cause  # type: ignore[attr-defined]
        return None

    @property
    def timeout(self) -> int | None:
        if hasattr(self, "_timeout"):
            return self._timeout  # type: ignore[attr-defined]
        return None

    def set_hard_timeout(self, value: int | None, blocking: bool = True) -> None:
        """Set the timeout for the event.

        NOTE, this is a hard timeout, meaning that the event will be blocked
        until the timeout is reached.
        """
        self._timeout = value
        if value is not None and value > 600:
            from core.utils.logger import get_logger
            logger = get_logger(__name__)

            logger.warning(
                "Timeout greater than 600 seconds may not be supported by "
                "the runtime. Consider setting a lower timeout."
            )

        # Check if .blocking is an attribute of the event
        if hasattr(self, "blocking"):
            # .blocking needs to be set to True if .timeout is set
            self.blocking = blocking

    # optional field
    @property
    def tool_call_metadata(self) -> ToolCallMetadata | None:
        if hasattr(self, "_tool_call_metadata"):
            return self._tool_call_metadata  # type: ignore[attr-defined]
        return None

    @tool_call_metadata.setter
    def tool_call_metadata(self, value: ToolCallMetadata) -> None:
        self._tool_call_metadata = value


@dataclass
class Action(Event):
    runnable: ClassVar[bool] = False


@dataclass
class Observation(Event):
    content: str


class EventSource(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENVIRONMENT = "environment"


class FileEditSource(str, Enum):
    LLM_BASED_EDIT = "llm_based_edit"
    OH_ACI = "oh_aci"  # openhands-aci


class FileReadSource(str, Enum):
    OH_ACI = "oh_aci"  # openhands-aci
    DEFAULT = "default"


class ObservationTypeSchema(BaseModel):
    READ: str = Field(default="read")
    """The content of a file
    """

    WRITE: str = Field(default="write")

    EDIT: str = Field(default="edit")

    BROWSE: str = Field(default="browse")
    """The HTML content of a URL
    """

    RUN: str = Field(default="run")
    """The output of a command
    """

    RUN_IPYTHON: str = Field(default="run_ipython")
    """Runs a IPython cell.
    """

    CHAT: str = Field(default="chat")
    """A message from the user
    """

    DELEGATE: str = Field(default="delegate")
    """The result of a task delegated to another agent
    """

    MESSAGE: str = Field(default="message")

    ERROR: str = Field(default="error")

    SUCCESS: str = Field(default="success")

    NULL: str = Field(default="null")

    THINK: str = Field(default="think")

    AGENT_STATE_CHANGED: str = Field(default="agent_state_changed")

    USER_REJECTED: str = Field(default="user_rejected")

    CONDENSE: str = Field(default="condense")
    """Result of a condensation operation."""

    RECALL: str = Field(default="recall")
    """Result of a recall operation. This can be the workspace context, a microagent, or other types of information."""

    TASK_GRAPH_BUILD: str = Field(default="task_graph_build")
    """The task graph representation of a task description."""

    SNAPSHOT: str = Field(default="snapshot")
    """The snapshot of the repository."""

    ROLLBACK: str = Field(default="rollback")
    """Rollback to a snapshot."""

    REPO_PLAN: str = Field(default="repo_plan")
    """The planning results for repository implementation."""

    REPO_CREATE: str = Field(default="repo_create")
    """The results of repository creation with code generation."""

    REPO_ANALYZER: str = Field(default="repo_analyzer")
    """The results of repository analysis comparing paper with codebase."""

    REPO_UPDATE: str = Field(default="repo_update")
    """The results of repository code updates based on requirements."""

    REPO_VERIFY: str = Field(default="repo_verify")
    """Verifies repository implementation and functionality.
    """

    REPO_RUN: str = Field(default="repo_run")
    """Runs repository reproduce.sh script in isolated environment.
    """

    PAPER_REPRODUCTION_ANALYZER: str = Field(default="paper_reproduction_analyzer")
    """Analyzes paper reproduction requirements and implementation guidance.
    """

    REPO_DEBUG: str = Field(default="repo_debug")
    """Debug and fix code issues in repositories using refact agent.
    """

    REPO_EDIT: str = Field(default="repo_edit")
    """Edit repository code based on user instructions."""

    REPO_JUDGE: str = Field(default="repo_judge")
    """Judge repository code based on rubric questions."""

    PAPER_RUBRIC: str = Field(default="paper_rubric")
    """Extract rubrics from PDF papers for implementation requirements."""

    PDF_QUERY: str = Field(default="pdf_query")
    """Query PDF documents with semantic search and retrieval."""

    EXPERIMENT_MANAGER: str = Field(default="experiment_manager")
    """Manage and execute experiments (build/list/run)."""

    CONDENSATION: str = Field(default="condensation")
    """Result of a condensation operation."""


ObservationType = ObservationTypeSchema()


class ActionTypeSchema(BaseModel):
    SYSTEM: str = Field(default="system")
    """Represents a system message.
    """
    MESSAGE: str = Field(default="message")
    """Represents a message.
    """

    START: str = Field(default="start")
    """Starts a new development task OR send chat from the user. Only sent by the client.
    """

    READ: str = Field(default="read")
    """Reads the content of a file.
    """

    WRITE: str = Field(default="write")
    """Writes the content to a file.
    """

    EDIT: str = Field(default="edit")
    """Edits a file by providing a draft.
    """

    TASK_GRAPH_BUILD: str = Field(default="task_graph_build")
    """Builds a task graph from a task description.
    """

    REPO_PLAN: str = Field(default="repo_plan")
    """Plans repository implementation based on a paper.
    """

    REPO_CREATE: str = Field(default="repo_create")
    """Creates full repository implementation with code generation based on a paper.
    """

    REPO_ANALYZER: str = Field(default="repo_analyzer")
    """Analyzes repository implementation by comparing paper with existing codebase.
    """

    REPO_UPDATE: str = Field(default="repo_update")
    """Updates repository code based on user requirements.
    """

    REPO_VERIFY: str = Field(default="repo_verify")
    """Verifies repository implementation and functionality.
    """

    REPO_RUN: str = Field(default="repo_run")
    """Runs repository reproduce.sh script in isolated environment.
    """

    PAPER_REPRODUCTION_ANALYZER: str = Field(default="paper_reproduction_analyzer")
    """Analyzes paper reproduction requirements and implementation guidance.
    """

    REPO_DEBUG: str = Field(default="repo_debug")
    """Debug and fix code issues in repositories using refact agent.
    """

    REPO_EDIT: str = Field(default="repo_edit")
    """Edit repository code based on user instructions."""

    REPO_JUDGE: str = Field(default="repo_judge")
    """Judge repository code based on rubric questions."""

    PAPER_RUBRIC: str = Field(default="paper_rubric")
    """Extract rubrics from PDF papers for implementation requirements."""

    PDF_QUERY: str = Field(default="pdf_query")
    """Query PDF documents with semantic search and retrieval."""

    EXPERIMENT_MANAGER: str = Field(default="experiment_manager")
    """Manage and execute experiments (build/list/run)."""

    RUN: str = Field(default="run")
    """Runs a command.
    """

    RUN_IPYTHON: str = Field(default="run_ipython")
    """Runs a IPython cell.
    """

    BROWSE: str = Field(default="browse")
    """Opens a web page.
    """

    BROWSE_INTERACTIVE: str = Field(default="browse_interactive")
    """Interact with the browser instance.
    """

    DELEGATE: str = Field(default="delegate")
    """Delegates a task to another agent.
    """

    THINK: str = Field(default="think")
    """Logs a thought.
    """

    FINISH: str = Field(default="finish")
    """If you're absolutely certain that you've completed your task and have tested your work,
    use the finish action to stop working.
    """

    REJECT: str = Field(default="reject")
    """If you're absolutely certain that you cannot complete the task with given requirements,
    use the reject action to stop working.
    """

    NULL: str = Field(default="null")

    SUMMARIZE: str = Field(default="summarize")

    PAUSE: str = Field(default="pause")
    """Pauses the task.
    """

    RESUME: str = Field(default="resume")
    """Resumes the task.
    """

    STOP: str = Field(default="stop")
    """Stops the task. Must send a start action to restart a new task.
    """

    CHANGE_AGENT_STATE: str = Field(default="change_agent_state")

    PUSH: str = Field(default="push")
    """Push a branch to github."""

    SEND_PR: str = Field(default="send_pr")
    """Send a PR to github."""

    RECALL: str = Field(default="recall")
    """Retrieves content from a user workspace, microagent, or other source."""

    SNAPSHOT: str = Field(default="snapshot")
    """Creates a snapshot of the repository."""

    ROLLBACK: str = Field(default="rollback")
    """Rollback to a snapshot."""

    CONDENSATION: str = Field(default="condensation")
    """Condenses a list of events into a summary."""


ActionType = ActionTypeSchema()


def truncate_content(content: str, max_chars: int | None = None) -> str:
    """Truncate the middle of the observation content if it is too long."""
    if max_chars is None or len(content) <= max_chars or max_chars < 0:
        return content

    # truncate the middle and include a message to the LLM about it
    half = max_chars // 2
    return (
        content[:half]
        + "\n[... Observation truncated due to length ...]\n"
        + content[-half:]
    )


class ActionSecurityRisk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecallType(str, Enum):
    """The type of information that can be retrieved from microagents."""

    WORKSPACE_CONTEXT = 'workspace_context'
    """Workspace context (repo instructions, runtime, etc.)"""

    KNOWLEDGE = 'knowledge'
    """A knowledge microagent."""

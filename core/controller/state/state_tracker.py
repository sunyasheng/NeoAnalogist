from core.agent.base import Agent
from core.controller.state.state import State
from core.utils.logger import get_logger
logger = get_logger(__name__)
from core.events.action.empty import NullAction
from core.events.event import Event
from core.events.event_filter import EventFilter
from core.events.observation.empty import NullObservation
from core.events.serialization import event_to_trajectory
from core.events.stream import EventStream
from core.llm.metrics import Metrics
from core.storage.files import FileStore


class StateTracker:
    """Manages and synchronizes the state of an agent throughout its lifecycle.

    It is responsible for:
    1. Maintaining agent state persistence across sessions
    2. Managing agent history by filtering and tracking relevant events (previously done in the agent controller)
    3. Synchronizing metrics between the controller and LLM components
    4. Updating control flags for budget and iteration limits
    """

    def __init__(
        self, sid: str | None, file_store: FileStore | None, user_id: str | None
    ):
        self.sid = sid
        self.file_store = file_store
        self.user_id = user_id

        # filter out events that are not relevant to the agent
        # so they will not be included in the agent history
        self.agent_history_filter = EventFilter(
            exclude_types=(
                NullAction,
                NullObservation,
            ),
            exclude_hidden=True,
        )

    def set_initial_state(
        self,
        id: str,
        agent: Agent,
        state: State | None,
        max_iterations: int,
        max_budget_per_task: float | None,
        confirmation_mode: bool = False,
    ) -> None:
        """Sets the initial state for the agent, either from the previous session, or from a parent agent, or by creating a new one.

        Args:
            state: The state to initialize with, or None to create a new state.
            max_iterations: The maximum number of iterations allowed for the task.
            confirmation_mode: Whether to enable confirmation mode.
        """
        # state can come from:
        # - the previous session, in which case it has history
        # - from a parent agent, in which case it has no history
        # - None / a new state

        # NOTE: iteration_flag, budget_flag 已移除，如需恢复请补充
        if state is None:
            self.state = State(
                session_id=id.removesuffix('-delegate'),
                inputs={},
                confirmation_mode=confirmation_mode,
            )
            self.state.start_id = 0

            logger.info(
                f'AgentController {id} - created new state. start_id: {self.state.start_id}'
            )
        else:
            self.state = state
            if self.state.start_id <= -1:
                self.state.start_id = 0

            logger.info(
                f'AgentController {id} initializing history from event {self.state.start_id}',
            )

        # Share the state metrics with the agent's LLM metrics
        # This ensures that all accumulated metrics are always in sync between controller and llm
        agent.llm.metrics = self.state.metrics

    def _init_history(self, event_stream: EventStream) -> None:
        """Initializes the agent's history from the event stream.

        The history is a list of events that:
        - Excludes events of types listed in self.filter_out
        - Excludes events with hidden=True attribute
        """
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else event_stream.get_latest_event_id()
        )

        if start_id > end_id + 1:
            logger.warning(
                f'start_id {start_id} is greater than end_id + 1 ({end_id + 1}). History will be empty.',
            )
            self.state.history = []
            return

        self.state.history = list(
            event_stream.search_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter=self.agent_history_filter,
            )
        )

    def close(self, event_stream: EventStream):
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else event_stream.get_latest_event_id()
        )

        self.state.history = list(
            event_stream.search_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter=self.agent_history_filter,
            )
        )

    def add_history(self, event: Event):
        if self.agent_history_filter.include(event):
            self.state.history.append(event)

    def get_trajectory(self, include_screenshots: bool = False) -> list[dict]:
        return [
            event_to_trajectory(event, include_screenshots)
            for event in self.state.history
        ]

    def get_metrics_snapshot(self):
        return self.state.metrics.copy()

    def save_state(self):
        if self.sid and self.file_store:
            self.state.save_to_session(self.sid, self.file_store, self.user_id)

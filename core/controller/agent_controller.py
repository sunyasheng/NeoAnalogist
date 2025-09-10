from __future__ import annotations

import logging
from core.controller.state.state import State
from core.controller.state.state_tracker import StateTracker
from core.events.stream import EventStream
from core.storage.files import FileStore
from core.memory.view import View
from core.events.action import CondensationAction, SystemMessageAction, RecallAction, MessageAction
from core.events.event import Event, EventSource
from core.events.observation import Observation
from core.events.action.repo import RepoCreateAction
from core.events.observation.repo import RepoCreateObservation

class AgentController:
    def __init__(
        self,
        agent,  # Add agent parameter
        event_stream: EventStream,
        iteration_delta: int,
        budget_per_task_delta: float | None = None,
        sid: str | None = None,
        file_store: FileStore | None = None,
        user_id: str | None = None,
        confirmation_mode: bool = False,
        initial_state: State | None = None,
    ):
        self.id = sid or event_stream.sid
        self.user_id = user_id
        self.file_store = file_store
        self.agent = agent
        self.event_stream = event_stream
        self.state_tracker = StateTracker(self.id, self.file_store, self.user_id)
        self.set_initial_state(
            state=initial_state,
            max_iterations=iteration_delta,
            max_budget_per_task=budget_per_task_delta,
            confirmation_mode=confirmation_mode,
        )
        self.state = self.state_tracker.state
        # Subscribe to event stream for real-time state.history updates
    #     self.event_stream.subscribe(
    #         subscriber_id="agent_controller",
    #         callback=self.on_event,
    #         callback_id="agent_controller_history"
    #     )

    # def on_event(self, event):
    #     self.state_tracker.add_history(event)

    def set_initial_state(
        self,
        state: State | None,
        max_iterations: int,
        max_budget_per_task: float | None,
        confirmation_mode: bool = False,
    ):
        self.state_tracker.set_initial_state(
            self.id,
            self.agent,
            state,
            max_iterations,
            max_budget_per_task,
            confirmation_mode,
        )
        self.state_tracker._init_history(self.event_stream)

    def restore_state(self, sid: str, user_id: str | None = None) -> State:
        """Restore the agent state from persistent storage."""
        return State.restore_from_session(sid, self.file_store, user_id)

    def save_state(self):
        """Save the current agent state to persistent storage."""
        self.state_tracker.save_state()

    def log(self, level: str, message: str, extra: dict | None = None) -> None:
        """Logs a message to the agent controller's logger."""
        message = f'[Agent Controller {self.id}] {message}'
        if extra is None:
            extra = {}
        extra_merged = {'session_id': self.id, **extra}
        getattr(logging, level)(message, extra=extra_merged, stacklevel=2)

    def _apply_conversation_window(self, history: list[Event]) -> list[Event]:
        """Cuts history roughly in half when context window is exceeded.

        It preserves action-observation pairs and ensures that the system message,
        the first user message, and its associated recall observation are always included
        at the beginning of the context window.
        """
        if not history:
            return []
        system_message = next((e for e in history if isinstance(e, SystemMessageAction)), None)
        assert (
            system_message is None
            or isinstance(system_message, SystemMessageAction)
            and system_message.id == history[0].id
        )
        first_user_msg = self._first_user_message(history)
        if first_user_msg is None:
            first_user_msg = self._first_user_message()
            if first_user_msg is None:
                raise RuntimeError('No first user message found in the event stream.')
            self.log('warning', 'First user message not found in history. Using cached version from event stream.')
        first_user_msg_index = -1
        for i, event in enumerate(history):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                first_user_msg_index = i
                break
        recall_action = None
        recall_observation = None
        for i in range(first_user_msg_index + 1, len(history)):
            event = history[i]
            if isinstance(event, RecallAction) and getattr(event, 'query', None) == getattr(first_user_msg, 'content', None):
                recall_action = event
                for j in range(i + 1, len(history)):
                    obs_event = history[j]
                    if isinstance(obs_event, Observation) and getattr(obs_event, 'cause', None) == recall_action.id:
                        recall_observation = obs_event
                        break
                break
        essential_events = []
        if system_message:
            essential_events.append(system_message)
        if history:
            essential_events.append(first_user_msg)
            if recall_action and recall_observation:
                essential_events.append(recall_action)
                essential_events.append(recall_observation)
            elif recall_action:
                essential_events.append(recall_action)
        # Always keep all RepoCreateAction as essential
        for e in history:
            if isinstance(e, RepoCreateAction) and e not in essential_events:
                essential_events.append(e)
        # Always keep all RepoCreateObservation as essential
        for e in history:
            if isinstance(e, RepoCreateObservation) and e not in essential_events:
                essential_events.append(e)
        # Always keep all PaperRubricAction as essential
        from core.events.action.repo import PaperRubricAction
        for e in history:
            if isinstance(e, PaperRubricAction) and e not in essential_events:
                essential_events.append(e)
        # Always keep all PaperRubricObservation as essential
        from core.events.observation.repo import PaperRubricObservation
        for e in history:
            if isinstance(e, PaperRubricObservation) and e not in essential_events:
                essential_events.append(e)
        num_non_essential_events = len(history) - len(essential_events)
        num_recent_to_keep = max(1, num_non_essential_events // 2)
        slice_start_index = len(history) - num_recent_to_keep
        recent_events = history[slice_start_index:]
        # Remove any essential events from recent_events to avoid duplication
        recent_events = [e for e in recent_events if e not in essential_events]
        return essential_events + recent_events

    def _first_user_message(self, events: list[Event] | None = None) -> MessageAction | None:
        """Get the first user message for this agent."""
        if events is not None:
            return next((e for e in events if isinstance(e, MessageAction) and e.source == EventSource.USER), None)
        if hasattr(self, '_cached_first_user_message') and self._cached_first_user_message is not None:
            return self._cached_first_user_message
        self._cached_first_user_message = next(
            (e for e in self.event_stream.search_events(start_id=self.state.start_id)
             if isinstance(e, MessageAction) and e.source == EventSource.USER),
            None,
        )
        return self._cached_first_user_message

    def _handle_long_context_error(self) -> None:
        # When context window is exceeded, keep roughly half of agent interactions
        current_view = View.from_events(self.state.history)
        kept_events = self._apply_conversation_window(current_view.events)
        kept_event_ids = {e.id for e in kept_events}

        self.log(
            'info',
            f'Context window exceeded. Keeping events with IDs: {kept_event_ids}',
        )

        # The events to forget are those that are not in the kept set
        forgotten_event_ids = {e.id for e in self.state.history} - kept_event_ids

        # Debug print: check if forgotten_event_ids are continuous
        sorted_ids = sorted(forgotten_event_ids)
        is_continuous = sorted_ids == list(range(sorted_ids[0], sorted_ids[-1] + 1)) if sorted_ids else True
        print(f"[DEBUG] forgotten_event_ids: {sorted_ids}")
        print(f"[DEBUG] is_continuous: {is_continuous}")

        # Use list if not continuous, else use range
        if sorted_ids and not is_continuous:
            condensation_action = CondensationAction(
                forgotten_event_ids=sorted_ids
            )
        else:
            condensation_action = CondensationAction(
                forgotten_events_start_id=min(forgotten_event_ids) if forgotten_event_ids else 0,
                forgotten_events_end_id=max(forgotten_event_ids) if forgotten_event_ids else 0,
            )
        
        # self.event_stream.add_event(
        #     condensation_action,
        #     EventSource.AGENT,
        # )
        # self.state_tracker.add_history(condensation_action)
        ## NOTE: state-tracker state is not equal to the state
        ## print(f"[DEBUG] state_tracker.state ID: {id(self.controller.state_tracker.state)}") state_tracker.state ID: 140003677650000
        ## print(f"[DEBUG] controller.state ID: {id(self.controller.state)}") controller.state ID: 140003287732192
        return condensation_action
from __future__ import annotations

from core.events.action import CondensationAction
from core.events.observation import AgentCondensationObservation
from core.events.event import truncate_content
from core.llm.interface import LLMInterface as LLM
from core.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)
from core.utils.types.message import Message, TextContent


class LLMSummarizingCondenser(RollingCondenser):
    """A condenser that summarizes forgotten events."""

    def __init__(
        self,
        llm: LLM,
        max_size: int = 100,
        keep_first: int = 1,
        max_event_length: int = 10_000,
    ):
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if max_size < 1:
            raise ValueError(f'max_size ({max_size}) cannot be non-positive')

        self.max_size = max_size
        self.keep_first = keep_first
        self.max_event_length = max_event_length
        self.llm = llm

        super().__init__()

    def _truncate(self, content: str) -> str:
        """Truncate the content to fit within the specified maximum event length."""
        return truncate_content(content, max_chars=self.max_event_length)

    def get_condensation(self, view: View) -> Condensation:
        head = view[: self.keep_first]
        target_size = self.max_size // 2
        events_from_tail = target_size - len(head) - 1

        summary_event = (
            view[self.keep_first]
            if isinstance(view[self.keep_first], AgentCondensationObservation)
            else AgentCondensationObservation(content='No events summarized')
        )

        forgotten_events = []
        # Correctly calculate the end of the slice for forgotten events
        end_slice = len(view) - events_from_tail if events_from_tail > 0 else len(view)
        for event in view[self.keep_first : end_slice]:
            if not isinstance(event, AgentCondensationObservation):
                forgotten_events.append(event)
        
        if not forgotten_events:
            # Nothing to condense, just return the view
            return Condensation(action=CondensationAction(summary=""))


        prompt = """You are maintaining a context-aware state summary for an interactive agent. You will be given a list of events..."""  # Truncated for brevity

        summary_event_content = self._truncate(
            summary_event.content if summary_event.content else ''
        )
        prompt += f'<PREVIOUS SUMMARY>\n{summary_event_content}\n</PREVIOUS SUMMARY>\n\n'

        for forgotten_event in forgotten_events:
            event_content = self._truncate(str(forgotten_event))
            prompt += f'<EVENT id={forgotten_event.id}>\n{event_content}\n</EVENT>\n'

        prompt += 'Now summarize the events using the rules above.'

        messages = [Message(role='user', content=[TextContent(text=prompt)])]

        # Use the chat method from the existing LLMInterface
        response_dict = self.llm.chat(
            messages=self.llm.format_messages_for_llm(messages),
        )
        
        # The 'response_dict' is the processed response from LLMInterface.
        # The actual raw response from the LLM provider is in the 'raw_response' key.
        raw_response = response_dict.get("raw_response")
        
        summary = ""
        if raw_response and hasattr(raw_response, 'choices') and raw_response.choices:
            message = raw_response.choices[0].message
            if hasattr(message, 'content'):
                summary = message.content or ""
        
        if not summary:
            # Fallback if the structure is not as expected or content is empty
            summary = response_dict.get("text", "Could not generate summary.")
        
        # import pdb; pdb.set_trace()
        # self.add_metadata('response', response_dict)
        # self.add_metadata('metrics', self.llm.get_usage_stats())

        return Condensation(
            action=CondensationAction(
                forgotten_events_start_id=min(event.id for event in forgotten_events),
                forgotten_events_end_id=max(event.id for event in forgotten_events),
                summary=summary,
                summary_offset=self.keep_first,
            )
        )

    def should_condense(self, view: View) -> bool:
        return len(view) > self.max_size 
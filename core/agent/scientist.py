import importlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.agent.base import Agent
from core.agent.function_calling import get_tools, response_to_actions
from core.config.condenser_config import LLMSummarizingCondenserConfig
from core.events.action import AgentFinishAction, MessageAction, CondensationAction
from core.events.event import Action, Event, EventSource, Observation
from core.events.observation import NullObservation, AgentCondensationObservation
from core.llm.interface import LLMInterface, ToolCall
from core.memory.conversation_memory import ConversationMemory
from core.memory.condenser.condenser import Condensation, RollingCondenser
from core.memory.condenser.impl.browser_output_condenser import BrowserOutputCondenser
from core.memory.condenser.impl.pipeline import CondenserPipeline
from core.memory.condenser.impl.llm_summarizing_condenser import LLMSummarizingCondenser
from core.memory.view import View
from core.prompt.prompt_manager import PromptManager
from core.runtime.impl.docker.docker_runtime import DockerRuntime
from core.utils.types.message import ImageContent, Message, TextContent
from litellm.exceptions import ContextWindowExceededError, BadRequestError, OpenAIError

import litellm


class Scientist(Agent):
    def __init__(
        self, agent_id: str, config: Dict[str, Any], working_dir: Optional[str] = None, event_stream=None
    ):
        super().__init__(agent_id, config)
        self.agent_name = config.get("agent_name", agent_id)
        self.agent_description = config.get("agent_description", "")

        if not working_dir:
            raise ValueError("working_dir must be provided for Scientist agent")

        self.working_dir = working_dir
        self.context_window = self.config.get("context_window", 5)
        self.task = None
        self.done = False
        self.current_intent = None
        self.intent_recognized = False
        self.event_history: list[Event] = []
        self.condenser: RollingCondenser | None = None
        # Add event ID counter for proper event ID assignment
        self._next_event_id = 0
        self.event_stream = event_stream
        self.initialize()
        # 移除 event_stream.subscribe 相关代码

    def initialize(self):
        self.logger.info(f"Initializing Scientist agent: {self.agent_name}")

        # Create a SimpleConfig object for DockerRuntime
        class SimpleConfig:
            def __init__(self, working_dir: str):
                self.sandbox = type("Sandbox", (), {"timeout": 30})()
                self.workspace_base = working_dir
                self.workspace_mount_path = None
                self.workspace_mount_path_in_sandbox = None
                self.debug = True

        self.tools = get_tools(codeact_enable_browsing=True)

        self.action_executor = DockerRuntime(
            config=SimpleConfig(working_dir=self.working_dir),
            sid=self.agent_id,
            env_vars=self.config.get("env_vars", {}),
            status_callback=None,
            attach_to_existing=False,
            headless_mode=True,
        )

        # Start the container
        self.logger.info("Starting Docker container...")
        try:
            success = self.action_executor.start_container(image="mc-scientist:latest")
            if not success:
                # Get container logs if available
                if self.action_executor.container:
                    logs = self.action_executor.get_container_logs()
                    self.logger.error(f"Container logs:\n{logs}")
                raise RuntimeError("Failed to start Docker container")
            self.logger.info(
                f"Container started successfully at {self.action_executor.api_url}"
            )
        except Exception as e:
            # Get container logs if available
            if self.action_executor.container:
                logs = self.action_executor.get_container_logs()
                self.logger.error(f"Container logs:\n{logs}")
            raise RuntimeError(f"Failed to start Docker container: {str(e)}")

        self.prompt_manager = PromptManager(self.config)
        self.enable_som_visual_browsing = self.config.get(
            "enable_som_visual_browsing", False
        )
        self.llm = LLMInterface(self.config, working_dir=self.working_dir)

        self.conversation_memory = ConversationMemory(
            enable_som_visual_browsing=self.enable_som_visual_browsing,
            prompt_manager=self.prompt_manager,
        )

        # Initialize the condenser from agent configuration
        condenser_config_dict = self.config.get("condenser", {})
        # If the config specifies a pipeline, combine multiple condensers in order
        if condenser_config_dict.get("type") == "pipeline":
            condensers = []
            for c_cfg in condenser_config_dict.get("condensers", []):
                # Add a BrowserOutputCondenser if specified in the config
                if c_cfg.get("type") == "browser_output":
                    condensers.append(BrowserOutputCondenser(attention_window=c_cfg.get("attention_window", 1)))
                # Add an LLM-based summarizing condenser if specified
                elif c_cfg.get("type") == "llm_summarizing":
                    from core.config.condenser_config import LLMSummarizingCondenserConfig
                    from core.memory.condenser.impl.llm_summarizing_condenser import LLMSummarizingCondenser
                    llm_cfg = LLMSummarizingCondenserConfig(**c_cfg)
                    condensers.append(LLMSummarizingCondenser(
                        llm=self.llm,
                        max_size=llm_cfg.max_size,
                        keep_first=llm_cfg.keep_first,
                        max_event_length=llm_cfg.max_event_length,
                    ))
            # Combine all specified condensers into a pipeline
            self.condenser = CondenserPipeline(*condensers)
            self.logger.info("CondenserPipeline initialized with: %s", [type(c).__name__ for c in condensers])
        # Fallback: use a single LLM summarizing condenser if specified
        elif condenser_config_dict.get("type") == "llm_summarizing":
            condenser_config = LLMSummarizingCondenserConfig(**condenser_config_dict)
            self.condenser = LLMSummarizingCondenser(
                llm=self.llm,
                max_size=condenser_config.max_size,
                keep_first=condenser_config.keep_first,
                max_event_length=condenser_config.max_event_length,
            )
            self.logger.info("LLM Summarizing Condenser initialized with validated config.")
        # If no condenser is configured, memory will not be condensed
        else:
            self.logger.info("No condenser configured. Memory will not be condensed.")

        self.logger.info(f"Scientist agent {self.agent_name} initialized successfully")

    def reset(self):
        pass
    #     """Reset the agent's state for a new task."""
    #     self.done = False
    #     self.task = None
    #     self.event_history = []
    #     # Reset event ID counter
    #     self._next_event_id = 0
    #     self.logger.info(f"Agent reset for new task")

    def _enhance_messages(
        self, messages: list[Message], task: str, paper_path: str
    ) -> list[Message]:
        """Enhances the user message with additional context based on keywords matched.

        Args:
            messages (list[Message]): The list of messages to enhance

        Returns:
            list[Message]: The enhanced list of messages
        """
        assert self.prompt_manager, "Prompt Manager not instantiated."

        results: list[Message] = []
        is_first_message_handled = False
        prev_role = None

        for msg in messages:
            if msg.role == "user" and not is_first_message_handled:
                is_first_message_handled = True
                # compose the first user message with examples
                if self.config["agent"].get("task_type", "") == "devai":
                    self.prompt_manager.add_query_to_initial_message(
                        msg, task, self.config["devai_path"]
                    )
                else:
                    # For ImageBrush mode, do not inject paper reproduction examples or env setup block
                    pass
                    # import pdb; pdb.set_trace()

            elif msg.role == "user":
                # Add double newline between consecutive user messages
                if prev_role == "user" and len(msg.content) > 0:
                    # Find the first TextContent in the message to add newlines
                    for content_item in msg.content:
                        if isinstance(content_item, TextContent):
                            # If the previous message was also from a user, prepend two newlines to ensure separation
                            content_item.text = "\n\n" + content_item.text
                            break

            results.append(msg)
            prev_role = msg.role

        return results

    def _get_messages(self) -> List[Message]:

        print("[DEBUG] event_history length before prompt:", len(self.event_history))
        if len(self.event_history) > 0:
            print("[DEBUG] first event:", self.event_history[0])
            print("[DEBUG] last event:", self.event_history[-1])

        if self.config["agent"].get("task_type", "") == "devai":
            system_prompt_name = "devai_system_prompt.j2"
            suffix_prompt = "\n\nYour workspace root should be at ``/app_sci/workspace{}`` . Please implement the task in this workspace.\n\n⚠️ CRITICAL INSTRUCTION: You MUST generate only ONE tool call at a time. Never generate multiple tool calls in a single response. Wait for the result of each tool call before proceeding to the next one.".format(
                self.working_dir.split("workspace")[1]
            )
            # import pdb; pdb.set_trace()
        else:
            system_prompt_name = "image_manipulation_system_prompt.j2"
            suffix_prompt = ""
        # initial system prompt
        messages = self.conversation_memory.process_initial_messages(
            with_caching=False,
            system_prompt_name=system_prompt_name,
            suffix_prompt=suffix_prompt,
        )

        events = [
            event
            for event in self.event_history
            if not isinstance(event, NullObservation)
        ]

        processed_events = events

        messages = self.conversation_memory.process_events(
            condensed_history=processed_events,
            initial_messages=messages,
            max_message_chars=self.llm.config["max_message_chars"],
            vision_is_active=False,
        )

        paper_path = self.config.get("paper_path", "")
        workspace_paper_path = os.path.join(self.working_dir, paper_path)
        # import pdb; pdb.set_trace()
        messages = self._enhance_messages(messages, self.task, workspace_paper_path)

        return messages

    def policy(self, observation: Any, task: Optional[str] = None) -> Dict[str, Any]:
        if task and not any(isinstance(e, MessageAction) and e.content == f"TASK: {task}" for e in self.event_history):
            self.task = task
            # Add task as a MessageAction to event history
            message_action = MessageAction(
                content=f"TASK: {task}", wait_for_response=False
            )
            message_action._source = EventSource.USER
            self.event_stream.add_event(message_action, EventSource.USER)
            self.event_history.append(message_action)

        # prepare what we want to send to the LLM
        messages = self._get_messages()

        # print(self.llm.format_messages_for_llm(messages))
        # import pdb; pdb.set_trace()
        response = self.llm.chat(
            messages=self.llm.format_messages_for_llm(messages),
            temperature=self.config.get("llm", {}).get("temperature", 0.7),
            tools=self.tools,
            tool_choice="auto",
        )

        actions = response_to_actions(response["raw_response"])

        return actions

    def execute(self, actions: List[Action]) -> List[Observation]:
        """Execute code"""
        observations = []

        for action_i in actions:
            observation_i = self.action_executor.run_action(action_i)
            observation_i.tool_call_metadata = action_i.tool_call_metadata

            observations.append(observation_i)

        return observations

    def get_llm_usage_stats(self) -> Dict[str, Any]:
        return self.llm.get_usage_stats()

    def save_llm_usage_log(self, filename: Optional[str] = None) -> str:
        return self.llm.save_usage_log(filename)

    def maybe_condense_event_history(self):
        """Automatically check and insert CondensationAction to event_history if necessary"""
        # effective_event_count = len(view)

        # Use should_condense if available, to support CondenserPipeline
        if self.condenser:# and hasattr(self.condenser, "should_condense"):
            # if self.condenser.should_condense(view):
            #     self.logger.info(f"Effective event count ({effective_event_count}) exceeds condensation threshold.")
            #     self.logger.info(f"Raw event history length: {len(self.event_history)}")
            view = View.from_events(self.event_history)
            condensation_result = self.condenser.condense(view)
            
            if isinstance(condensation_result, Condensation):
                condensation_result.action._source = EventSource.AGENT
                self.event_stream.add_event(condensation_result.action, EventSource.AGENT)
                self.event_history.append(condensation_result.action)
                self.logger.info(f"CondensationAction inserted. New event history length: {len(self.event_history)}")
                self.logger.info(f"Condensation summary length: {len(condensation_result.action.summary)} characters")
            elif isinstance(condensation_result, View):
                # If condense returns a View, it means no condensation was needed
                # This can happen if should_condense returns False
                self.logger.info(f"Condensation returned View instead of Condensation, no action inserted")
                self.logger.info(f"View length: {len(condensation_result)}")
            else:
                self.logger.warning(f"Unexpected condensation result type: {type(condensation_result)}")

    def run(self, task: str, max_steps: int = 10, controller=None) -> Dict[str, Any]:
        """Run the agent on a task"""
        self.task = task
        self.controller = controller
        step = 0
        while not self.done and step < max_steps:
            try:
                actions = self.policy(None, task if step == 0 else None)
            except (litellm.ContextWindowExceededError, litellm.BadRequestError, ContextWindowExceededError, OpenAIError) as e:
            # except Exception as e:
                error_str = str(e).lower()
                if (
                    'contextwindowexceedederror' in error_str
                    or 'prompt is too long' in error_str
                    or 'input length and `max_tokens` exceed context limit' in error_str
                    or 'please reduce the length of either one' in error_str
                ):
                # if True:
                    if hasattr(self, 'controller') and hasattr(self.controller, '_handle_long_context_error'):
                        self.controller.state.history = self.event_history.copy()
                        condensation_action = self.controller._handle_long_context_error()
                        self.event_stream.add_event(condensation_action, EventSource.AGENT)
                        self.event_history.append(condensation_action)
                        print("[DEBUG] event_history after condensation:", len(self.event_history))
                        #### the state_tracker.state.history is updated to add the condensation action
                        # print("[DEBUG] controller.state.history:", len(self.controller.state_tracker.state.history))
                        # self.event_history = self.controller.state_tracker.state.history.copy()
                        # import pdb; pdb.set_trace()
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Condensation triggered due to context window error: {e}")
                        continue  # Retry this step after condensation
                raise

            # Check for AgentFinishAction
            for action in actions:
                if isinstance(action, AgentFinishAction):
                    if action.task_completed == "true" or action.task_completed == "partial":
                        self.done = True
                        return {"status": "success", "steps": step + 1}
                    # elif action.task_completed == "partial":
                        # continue_prompt = MessageAction(
                        #     content=self.prompt_manager.get_continue_prompt(),
                        #     wait_for_response=False,
                        # )
                        # continue_prompt._source = EventSource.USER
                        # self.event_stream.add_event(continue_prompt, EventSource.USER)
                        # self.event_history.append(continue_prompt)
                        # break
                if isinstance(action, MessageAction):
                    import pdb; pdb.set_trace()
                    if action.wait_for_response:
                        continue_prompt = MessageAction(
                            content=self.prompt_manager.get_continue_prompt(),
                            wait_for_response=False,
                        )
                        continue_prompt._source = EventSource.USER
                        self.event_stream.add_event(continue_prompt, EventSource.USER)
                        self.event_history.append(continue_prompt)
                        break

                self.event_stream.add_event(action, getattr(action, '_source', EventSource.AGENT))
                self.event_history.append(action)

            observations = self.execute(actions)
            for obs in observations:
                self.event_stream.add_event(obs, getattr(obs, '_source', EventSource.ENVIRONMENT))
                self.event_history.append(obs)
            # Auto condensation check
            self.maybe_condense_event_history()
            step += 1

        return {"status": "success" if self.done else "error", "steps": step}

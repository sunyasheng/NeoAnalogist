import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from core.llm import LiteLLMProvider, OpenRouterProvider
from core.utils.logger import (LLMFileHandler, get_logger, log_llm_request,
                               log_llm_response)
from core.utils.types.message import Message

# current_log_level = logging.INFO
current_log_level = logging.DEBUG

MESSAGE_SEPARATOR = "\n\n----------\n\n"


class ToolCall:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.name: str = "unknown"
        self.arguments: Dict[str, Any] = {}

        if data is None:
            return

        if isinstance(data, dict):
            function_data = data.get("function", {})
            self.name = function_data.get("name", "unknown")
            try:
                self.arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                self.arguments = {}
        else:
            self.name = getattr(data.function, "name", "unknown")
            try:
                self.arguments = json.loads(getattr(data.function, "arguments", "{}"))
            except (json.JSONDecodeError, AttributeError):
                self.arguments = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)}
        }


class LLMInterface:
    def __init__(self, config: Dict[str, Any], working_dir: Optional[str] = None):
        self.config = config
        self.working_dir = working_dir
        self.logger = logging.getLogger("AgentLLM")

        # 初始化日志处理器
        if self.working_dir:
            log_dir = os.path.join(self.working_dir, "data", "logs")
            os.makedirs(log_dir, exist_ok=True)

            # 创建自定义的LLMFileHandler
            llm_file_handler = LLMFileHandler(log_dir, prefix="prompt")
            llm_file_handler.setLevel(current_log_level)
            llm_file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )

            # 创建logger并添加文件处理器
            self.llm_prompt_logger = get_logger("prompt", current_log_level)
            self.llm_prompt_logger.addHandler(llm_file_handler)
        else:
            self.llm_prompt_logger = get_logger("prompt", current_log_level)

        self.llm_response_logger = get_logger("response", current_log_level)

        self.initialize_llm()

    def initialize_llm(self):
        llm_config = self.config.get("llm", {})
        providers_config = llm_config.get("providers", {})

        if providers_config.get("litellm", {}).get("use", False):
            provider_config = providers_config.get("litellm", {})
            provider_config["temperature"] = llm_config.get("temperature", 0.7)
            self.llm = LiteLLMProvider(provider_config, working_dir=self.working_dir)

        elif providers_config.get("openrouter", {}).get("use", False):
            provider_config = providers_config.get("openrouter", {})
            provider_config["temperature"] = llm_config.get("temperature", 0.7)
            self.llm = OpenRouterProvider(provider_config, working_dir=self.working_dir)

        else:
            print("llm initialization failed")

    def _format_message_content(self, message: dict[str, Any]) -> str:
        parts = []
        content = message.get("content")
        if content:
            if isinstance(content, list):
                parts.append("\n".join(
                    self._format_content_element(element) for element in content
                ))
            else:
                parts.append(str(content))

        # Special handling for tool calls
        if message.get("tool_calls"):
            tool_calls_str_list = []
            for tc in message.get("tool_calls", []):
                try:
                    name = tc.get("function", {}).get("name", "unknown_function")
                    args = tc.get("function", {}).get("arguments", "{}")
                    tool_calls_str_list.append(f"Tool Call: {name}({args})")
                except Exception:
                    # Fallback for any unexpected structure
                    tool_calls_str_list.append(f"Tool Call: (Could not parse: {str(tc)})")
            
            if tool_calls_str_list:
                parts.append("\n--- TOOL CALLS ---\n" + "\n".join(tool_calls_str_list))

        return "\n".join(parts)

    def _format_content_element(self, element: dict[str, Any] | Any) -> str:
        if isinstance(element, dict):
            if "text" in element:
                return str(element["text"])
        return str(element)

    def log_prompt(self, messages: list[dict[str, Any]] | dict[str, Any]) -> None:
        if not messages:
            return

        debug_messages = messages if isinstance(messages, list) else [messages]
        # Ensure we log messages that have content OR tool_calls
        debug_message = MESSAGE_SEPARATOR.join(
            self._format_message_content(msg)
            for msg in debug_messages
            if msg.get("content") is not None or msg.get("tool_calls")
        )

        if debug_message:
            self.llm_prompt_logger.debug(debug_message)

    # def log_response(self, message_back: str) -> None:
    #     if message_back:
    #         llm_response_logger.debug(message_back)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # log_llm_request(messages=messages, model=kwargs.get("model", self.llm.model), tools=tools)

        self.log_prompt(messages)

        response = self.llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        log_llm_response(response)

        if "text" in response:
            response_content = response["text"]
            input_tokens = response.get("input_tokens", 0)
            output_tokens = response.get("output_tokens", 0)
        elif "choices" in response and len(response["choices"]) > 0:
            response_content = response["choices"][0]["message"]["content"]
            input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        else:
            response_content = str(response)
            input_tokens = 0
            output_tokens = 0

        return response

    def get_usage_stats(self) -> Dict[str, Any]:
        return self.llm.get_usage_stats()

    def save_usage_log(self, filename: Optional[str] = None) -> str:
        return self.llm.save_usage_log(filename)

    def parse_tool_calls(
        self, response: Dict[str, Any], content: str
    ) -> Optional[List[Dict[str, Any]]]:
        tool_calls = []
        response = response["raw_response"]

        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            message = choice.get("message", {})

            if message.tool_calls:
                for call in message.tool_calls:
                    try:
                        tool_call = {
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        tool_calls.append(tool_call)
                    except (KeyError, TypeError) as e:
                        self.logger.warning(f"Failed to parse tool call: {e}")
                        continue

        return tool_calls if tool_calls else None

    def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not tool_calls:
            return {"error": "No tool calls found", "status": "error"}

        actions = []
        for tool_call_data in tool_calls:
            tool_call = ToolCall(tool_call_data)

            action = {
                "type": tool_call.name,
                "params": tool_call.arguments,
                "status": "pending",
            }
            actions.append(action)

        return {"actions": actions, "status": "success"}

    def _construct_tool_call_message(self, tool_call: ToolCall) -> str:
        message = f"TOOL CALL: {tool_call.name.upper()}"

        if tool_call.arguments:
            args_str = str(tool_call.arguments)
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            message += f"\nPARAMS: {args_str}"

        return message

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            messages = [messages]

        # set flags to know how to serialize the messages
        for message in messages:
            # message.cache_enabled = self.is_caching_prompt_active()
            # message.vision_enabled = self.vision_is_active()
            # message.function_calling_enabled = self.is_function_calling_active()
            message.cache_enabled = False
            message.vision_enabled = False
            message.function_calling_enabled = True
            # if 'deepseek' in self.config.model:
            #     message.force_string_serializer = True

        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]

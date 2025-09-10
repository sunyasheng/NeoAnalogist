"""Base LLM class for agent language model integration."""

import abc
from typing import Any, Dict, List, Optional, Union

from litellm import ChatCompletionToolParam


class LLM(abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize()

    def _initialize(self):
        pass

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abc.abstractmethod
    def get_cost(
        self, input_tokens: int, output_tokens: int, model: Optional[str] = None
    ) -> float:
        pass

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI

from core.llm.base import LLM
from core.llm.token_counter import TokenCounter


class OpenRouterProvider(LLM):
    def __init__(self, config: Dict[str, Any], working_dir: Optional[str] = None):
        self.logger = logging.getLogger("OpenRouterProvider")
        self.working_dir = working_dir
        super().__init__(config)

    def _initialize(self):
        self.model = self.config.get("model", "deepseek/deepseek-chat-v3-0324:free")
        self.fallback_models = self.config.get("fallback_models", [])

        self.api_key = self.config.get("api_key")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            if self.api_key is None:
                self.logger.warning(
                    "No API key provided. Set OPENROUTER_API_KEY in environment variables."
                )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        self.extra_headers = {}
        if "http_referer" in self.config:
            self.extra_headers["HTTP-Referer"] = self.config["http_referer"]
        if "site_name" in self.config:
            self.extra_headers["X-Title"] = self.config["site_name"]

        self.timeout = self.config.get("timeout", 60)
        self.max_tokens = self.config.get("max_tokens", 8192)

        token_counter_config = self.config.get("token_counter_config", {})
        self.token_counter = TokenCounter(
            token_counter_config, working_dir=self.working_dir
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model = kwargs.pop("model", self.model)
        max_tokens = max_tokens or self.max_tokens
        start_time = time.time()

        try:
            input_token_count = self.count_tokens(prompt, model)
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {e}. Using approximation.")
            input_token_count = len(prompt) // 4

        try:
            messages = [{"role": "user", "content": prompt}]

            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_headers": self.extra_headers,
            }

            if stop:
                params["stop"] = stop

            extra_body = {}
            for key, value in kwargs.items():
                if key not in params:
                    extra_body[key] = value

            if extra_body:
                params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**params)

            output_text = response.choices[0].message.content

            try:
                output_token_count = self.count_tokens(output_text, model)
            except Exception as e:
                self.logger.warning(
                    f"Error counting output tokens: {e}. Using approximation."
                )
                output_token_count = len(output_text) // 4

            try:
                usage_record = self.token_counter.add_usage(
                    model=model,
                    input_tokens=input_token_count,
                    output_tokens=output_token_count,
                    metadata={
                        "type": "generation",
                        "duration": time.time() - start_time,
                        "prompt_length": len(prompt),
                        "response_length": len(output_text),
                    },
                )
                cost = usage_record.get("cost", 0.0)
            except Exception as e:
                self.logger.error(f"Error tracking token usage: {e}")
                cost = 0.0

            return {
                "text": output_text,
                "model": model,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "cost": cost,
                "finish_reason": response.choices[0].finish_reason,
                "raw_response": response,
            }

        except Exception as e:
            self.logger.error(f"Error with {model}: {str(e)}")

            if self.fallback_models:
                fallback_model = self.fallback_models[0]
                self.logger.info(f"Trying fallback model: {fallback_model}")
                kwargs["model"] = fallback_model
                try:
                    return self.generate(
                        prompt, max_tokens, temperature, stop, **kwargs
                    )
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback {fallback_model} also failed: {fallback_error}"
                    )

            error_response = {
                "text": f"Sorry, I encountered an error processing your request.",
                "model": model,
                "input_tokens": input_token_count,
                "output_tokens": 20,
                "cost": 0.0,
                "finish_reason": "error",
                "error": str(e),
                "raw_response": None,
            }
            return error_response

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        model = kwargs.pop("model", self.model)
        max_tokens = max_tokens or self.max_tokens
        start_time = time.time()

        input_token_count = 0
        for message in messages:
            try:
                input_token_count += self.count_tokens(
                    message.get("content", ""), model
                )
            except Exception as e:
                self.logger.warning(f"Error counting tokens: {e}. Using approximation.")
                input_token_count += len(message.get("content", "")) // 4

        try:
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_headers": self.extra_headers,
            }

            if stop:
                params["stop"] = stop

            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice

            extra_body = {}
            for key, value in kwargs.items():
                if key not in params:
                    extra_body[key] = value

            if extra_body:
                params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**params)

            if (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                message = response.choices[0].message
                content = message.content or ""
                tool_calls = message.tool_calls

                try:
                    output_token_count = self.count_tokens(
                        json.dumps(tool_calls), model
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Error counting tool call tokens: {e}. Using approximation."
                    )
                    output_token_count = len(json.dumps(tool_calls)) // 4

                try:
                    usage_record = self.token_counter.add_usage(
                        model=model,
                        input_tokens=input_token_count,
                        output_tokens=output_token_count,
                        metadata={
                            "type": "chat_with_tools",
                            "duration": time.time() - start_time,
                        },
                    )
                    cost = usage_record.get("cost", 0.0)
                except Exception as e:
                    self.logger.error(f"Error tracking token usage: {e}")
                    cost = 0.0

                return {
                    "text": content,
                    "model": model,
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                    "cost": cost,
                    "finish_reason": response.choices[0].finish_reason,
                    "raw_response": response,
                    "tool_calls": tool_calls,
                }
            else:
                output_text = response.choices[0].message.content

                try:
                    output_token_count = self.count_tokens(output_text, model)
                except Exception as e:
                    self.logger.warning(
                        f"Error counting output tokens: {e}. Using approximation."
                    )
                    output_token_count = len(output_text) // 4

                try:
                    usage_record = self.token_counter.add_usage(
                        model=model,
                        input_tokens=input_token_count,
                        output_tokens=output_token_count,
                        metadata={
                            "type": "chat",
                            "duration": time.time() - start_time,
                            "messages_count": len(messages),
                            "response_length": len(output_text),
                        },
                    )
                    cost = usage_record.get("cost", 0.0)
                except Exception as e:
                    self.logger.error(f"Error tracking token usage: {e}")
                    cost = 0.0

                return {
                    "text": output_text,
                    "model": model,
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                    "cost": cost,
                    "finish_reason": response.choices[0].finish_reason,
                    "raw_response": response,
                }

        except Exception as e:
            self.logger.error(f"Error in chat with {model}: {str(e)}")

            if self.fallback_models:
                fallback_model = self.fallback_models[0]
                self.logger.info(f"Trying fallback model: {fallback_model}")
                kwargs["model"] = fallback_model
                try:
                    return self.chat(
                        messages,
                        max_tokens,
                        temperature,
                        stop,
                        tools,
                        tool_choice,
                        **kwargs,
                    )
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback {fallback_model} also failed: {fallback_error}"
                    )

            error_response = {
                "text": f"Sorry, I encountered an error processing your request.",
                "model": model,
                "input_tokens": input_token_count,
                "output_tokens": 20,
                "cost": 0.0,
                "finish_reason": "error",
                "error": str(e),
                "raw_response": None,
            }
            return error_response

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4

    def get_cost(
        self, input_tokens: int, output_tokens: int, model: Optional[str] = None
    ) -> float:
        return (input_tokens * 0.000001) + (output_tokens * 0.000002)

    def get_usage_stats(self) -> Dict[str, Any]:
        return self.token_counter.get_usage_stats()

    def save_usage_log(self, filename: Optional[str] = None) -> str:
        return self.token_counter.save_usage_log(filename)

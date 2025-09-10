import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

try:
    import litellm
    from litellm import ChatCompletionToolParam, completion, embedding
except ImportError:
    raise ImportError(
        "LiteLLM package is required. Install it with: pip install litellm"
    )

litellm.drop_params = True

from core.llm.base import LLM
from core.llm.token_counter import TokenCounter

try:
    from litellm import Cache
except ImportError:
    Cache = None

try:
    from litellm.types.chat_completion import ChatCompletionToolParam
except ImportError:
    ChatCompletionToolParam = Any


def serialize_tool_call(tc):
    try:
        return {
            "name": tc.function.name if hasattr(tc, "function") else "unknown",
            "arguments": tc.function.arguments if hasattr(tc, "function") else {},
            "id": tc.id if hasattr(tc, "id") else "",
        }
    except Exception:
        return {"error": "Failed to serialize tool call"}


class LiteLLMProvider(LLM):
    def __init__(self, config: Dict[str, Any], working_dir: Optional[str] = None):
        self.logger = logging.getLogger("LiteLLMProvider")
        self.working_dir = working_dir
        super().__init__(config)

    def _initialize(self):
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.fallback_models = self.config.get("fallback_models", [])

        litellm.set_verbose = self.config.get("verbose", False)

        if self.config.get("cache_responses", False):
            try:
                litellm.cache = Cache()
                self.logger.info("LiteLLM cache enabled")
            except ImportError:
                self.logger.warning(
                    "Failed to initialize LiteLLM cache. Caching disabled."
                )
                litellm.cache = None
        else:
            litellm.cache = None

        litellm_config = self.config.get("litellm_config", None)
        if litellm_config:
            for key, value in litellm_config.items():
                setattr(litellm, key, value)

        token_counter_config = self.config.get("token_counter_config", {})
        self.token_counter = TokenCounter(
            token_counter_config, working_dir=self.working_dir
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model = kwargs.pop("model", self.model)
        start_time = time.time()

        try:
            input_token_count = self.count_tokens(prompt, model)
        except Exception as e:
            self.logger.warning(
                f"Error counting tokens: {e}. Using length as approximation."
            )
            input_token_count = len(prompt) // 4

        try:
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            response = litellm.completion(
                model=model,
                base_url=base_url,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )

            output_text = response.choices[0].text
            try:
                output_token_count = self.count_tokens(output_text, model)
            except Exception as e:
                self.logger.warning(
                    f"Error counting output tokens: {e}. Using length as approximation."
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
                "finish_reason": response.choices[0].finish_reason
                if hasattr(response.choices[0], "finish_reason")
                else "unknown",
                "raw_response": response,
            }

        except Exception as e:
            self.logger.error(f"Error generating completion with {model}: {str(e)}")

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
                        f"Fallback model {fallback_model} also failed: {fallback_error}"
                    )

            error_response = {
                "text": f"I'm sorry, I encountered an error processing your request. Please try again later.",
                "model": model,
                "input_tokens": input_token_count,
                "output_tokens": 20,
                "cost": 0.0,
                "finish_reason": "error",
                "error": str(e),
                "raw_response": None,
            }

            self.logger.error(
                f"Failed text generation for prompt: '{prompt[:50]}...'. Error: {str(e)}"
            )
            return error_response

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        model = kwargs.pop("model", self.model)
        start_time = time.time()

        try:
            input_token_count = 0
            for message in messages:
                try:
                    input_token_count += self.count_tokens(
                        message.get("content", ""), model
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Error counting tokens for message: {e}. Using length as approximation."
                    )
                    input_token_count += len(message.get("content", "")) // 4

            base_url = os.environ.get("OPENAI_BASE_URL", None)
            response = litellm.completion(
                model=model,
                base_url=base_url,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

            if hasattr(response.choices[0], "message") and hasattr(
                response.choices[0].message, "tool_calls"
            ):
                message = response.choices[0].message
                content = message.content or ""
                tool_calls = message.tool_calls
                if tool_calls:
                    try:
                        json_tool_calls = json.dumps(
                            [serialize_tool_call(tc) for tc in tool_calls]
                        )
                        output_token_count = self.count_tokens(
                            json_tool_calls, model
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Error counting output tokens: {e}. Using approximation for tool calls."
                        )
                        output_token_count = len(json.dumps(tool_calls)) // 4
                else:
                    output_token_count = 0

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
                    "finish_reason": response.choices[0].finish_reason
                    if hasattr(response.choices[0], "finish_reason")
                    else "tool_calls",
                    "raw_response": response,
                    "tool_calls": tool_calls,
                }
            else:
                output_text = response.choices[0].message.content
                try:
                    output_token_count = self.count_tokens(output_text, model)
                except Exception as e:
                    self.logger.warning(
                        f"Error counting output tokens: {e}. Using length as approximation."
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
                    "finish_reason": response.choices[0].finish_reason
                    if hasattr(response.choices[0], "finish_reason")
                    else "unknown",
                    "raw_response": response,
                }


        except Exception as e:

            error_str = str(e).lower()
            if (
                'contextwindowexceedederror' in error_str
                or 'context window' in error_str
                or 'prompt is too long' in error_str
                or 'input length and `max_tokens` exceed context limit' in error_str
                or 'please reduce the length of either one' in error_str
            ):
                raise
            self.logger.error(f"Error in chat completion with {model}: {str(e)}")

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
                        f"Fallback model {fallback_model} also failed: {fallback_error}"
                    )

            error_response = {
                "text": f"I'm sorry, I encountered an error processing your request. Please try again later.",
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
        model_name = model or self.model
        return self.token_counter.count_tokens(text, model_name)

    def get_cost(
        self, input_tokens: int, output_tokens: int, model: Optional[str] = None
    ) -> float:
        model_name = model or self.model
        return self.token_counter.calculate_cost(
            input_tokens, output_tokens, model_name
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        return self.token_counter.get_usage_stats()

    def save_usage_log(self, filename: Optional[str] = None) -> str:
        return self.token_counter.save_usage_log(filename)

    def embedding(self, text: str, model: Optional[str] = None) -> Dict[str, Any]:
        embedding_model = model or "text-embedding-ada-002"
        start_time = time.time()
        input_token_count = self.count_tokens(text, embedding_model)

        response = litellm.embedding(model=embedding_model, input=text)

        usage_record = self.token_counter.add_usage(
            model=embedding_model,
            input_tokens=input_token_count,
            output_tokens=0,
            metadata={
                "type": "embedding",
                "duration": time.time() - start_time,
                "text_length": len(text),
            },
        )

        return {
            "embedding": response.data[0].embedding,
            "model": embedding_model,
            "input_tokens": input_token_count,
            "cost": usage_record["cost"],
            "raw_response": response,
        }

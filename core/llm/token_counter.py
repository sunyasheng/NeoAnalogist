"""Token counting and cost tracking module."""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import tiktoken


class TokenCounter:
    # Default cost per 1K tokens for various models (in USD)
    DEFAULT_COSTS = {
        # OpenAI models
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-vision": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini": {"input": 0.000015, "output": 0.00006},
        "gpt-4o-2024-08-06": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-2024-08-06-mini": {"input": 0.000015, "output": 0.00006},
        "gpt-4o-2024-05-13": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-2024-05-13-mini": {"input": 0.000015, "output": 0.00006},
        "gpt-4o-2024-03-14": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-2024-03-14-mini": {"input": 0.000015, "output": 0.00006},
        "gpt-4o-2024-02-15": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-2024-02-15-mini": {"input": 0.000015, "output": 0.00006},
        "gpt-4o-2024-01-18": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-2024-01-18-mini": {"input": 0.000015, "output": 0.00006},
        # Anthropic models
        "claude-2": {"input": 0.008, "output": 0.024},
        "claude-instant-1": {"input": 0.0016, "output": 0.0054},
        "claude-3-5-sonnet": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet-20240620": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet-20240307": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet-20240229": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet-20240215": {"input": 0.00015, "output": 0.0006},
        # Default fallback
        "default": {"input": 0.01, "output": 0.02},
    }

    def __init__(
        self, config: Dict[str, Any] = None, working_dir: Optional[str] = None
    ):
        self.config = config or {}
        self.model_costs = self.config.get("model_costs", self.DEFAULT_COSTS)
        self.working_dir = working_dir

        # Initialize counters
        self.token_counts = defaultdict(lambda: {"input": 0, "output": 0})
        self.request_counts = defaultdict(int)
        self.total_cost = 0.0
        self.start_time = time.time()

        # Maintain a log of all requests
        self.usage_log = []

        # Directory for saving usage data
        if self.working_dir:
            # Use the agent's working directory if provided
            self.log_dir = os.path.join(self.working_dir, "data", "token_usage")
            os.makedirs(self.log_dir, exist_ok=True)

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        try:
            # For OpenAI models, use tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except (KeyError, ValueError):
            # Fallback for non-OpenAI models: rough approximation
            # GPT models use ~4 characters per token on average
            return len(text) // 4

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str = "gpt-3.5-turbo"
    ) -> float:
        # Get cost rates, falling back to default if model not found
        model_rates = self.model_costs.get(model, self.model_costs["default"])

        # Calculate cost (rates are per 1K tokens)
        input_cost = (input_tokens / 1000) * model_rates["input"]
        output_cost = (output_tokens / 1000) * model_rates["output"]

        return input_cost + output_cost

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        # Update counters
        self.token_counts[model]["input"] += input_tokens
        self.token_counts[model]["output"] += output_tokens
        self.request_counts[model] += 1
        self.total_cost += cost

        # Create usage record
        usage_record = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "metadata": metadata or {},
        }

        # Add to log
        self.usage_log.append(usage_record)

        return usage_record

    def get_usage_stats(self) -> Dict[str, Any]:
        runtime_seconds = time.time() - self.start_time
        hours = runtime_seconds / 3600

        total_tokens = sum(
            counts["input"] + counts["output"] for counts in self.token_counts.values()
        )

        return {
            "start_time": self.start_time,
            "runtime_seconds": runtime_seconds,
            "total_requests": sum(self.request_counts.values()),
            "total_tokens": total_tokens,
            "total_cost_usd": self.total_cost,
            "cost_per_hour": self.total_cost / hours if hours > 0 else 0,
            "tokens_per_second": total_tokens / runtime_seconds
            if runtime_seconds > 0
            else 0,
            "model_usage": {
                model: {
                    "requests": self.request_counts[model],
                    "input_tokens": counts["input"],
                    "output_tokens": counts["output"],
                    "total_tokens": counts["input"] + counts["output"],
                    "estimated_cost": self.calculate_cost(
                        counts["input"], counts["output"], model
                    ),
                }
                for model, counts in self.token_counts.items()
            },
        }

    def save_usage_log(self, filename: Optional[str] = None) -> str:
        if not self.working_dir:
            return ""

        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"token_usage_{timestamp}.json"

        filepath = os.path.join(self.log_dir, filename)

        # Combine usage log and overall stats
        data = {"stats": self.get_usage_stats(), "log": self.usage_log}

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def reset(self) -> None:
        self.token_counts = defaultdict(lambda: {"input": 0, "output": 0})
        self.request_counts = defaultdict(int)
        self.total_cost = 0.0
        self.start_time = time.time()
        self.usage_log = []

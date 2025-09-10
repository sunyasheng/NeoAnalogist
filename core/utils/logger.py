import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from rich.box import SIMPLE
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import install

# Install rich traceback handler
install()

# Define custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "debug": "dim cyan",
        "agent": "bold green",
        "action": "bold blue",
        "success": "green",
        "failure": "red",
        "thinking": "dim white",
        "execution": "magenta",
        "separator": "dim cyan",
        "highlight": "bold yellow",
    }
)

# Create console
console = Console(theme=custom_theme)


class LLMFileHandler(logging.FileHandler):
    def __init__(self, log_dir: str, prefix: str = "log"):
        self.log_dir = log_dir
        self.prefix = prefix
        self.message_counter = 1
        self.session = datetime.now().strftime("%y-%m-%d_%H-%M")

        # 创建日志目录
        self.log_directory = os.path.join(log_dir, self.session)
        os.makedirs(self.log_directory, exist_ok=True)

        # 生成初始文件名
        filename = f"{self.prefix}_{self.message_counter:03d}.log"
        self.baseFilename = os.path.join(self.log_directory, filename)

        # 初始化父类
        super().__init__(self.baseFilename, mode="w", encoding="utf-8")

    def emit(self, record):
        """重写emit方法，每次写入都创建新文件"""
        try:
            # 生成新的文件名
            filename = f"{self.prefix}_{self.message_counter:03d}.log"
            self.baseFilename = os.path.join(self.log_directory, filename)

            # 打开新文件
            self.stream = self._open()

            # 写入日志
            super().emit(record)

            # 关闭文件
            self.stream.close()

            # 增加计数器
            self.message_counter += 1

        except Exception:
            self.handleError(record)

    def _open(self):
        """打开新的日志文件"""
        return open(self.baseFilename, self.mode, encoding=self.encoding)


# Configure default logging format
DEFAULT_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO

_loggers: Dict[str, logging.Logger] = {}


def setup_logging(log_file="agent.log", level="INFO"):
    """Setup both rich console logging and file logging"""

    # Create file handler for the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Create rich handler for console output
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
        log_time_format=DEFAULT_DATE_FORMAT,
    )
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(rich_handler)

    return root_logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get or create a logger with the given name"""

    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level if level is not None else DEFAULT_LEVEL)

    # Store logger reference
    _loggers[name] = logger

    return logger


def configure_root_logger(level: int = DEFAULT_LEVEL):
    """Configure the root logger with Rich handler"""

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Create rich handler
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
        log_time_format=DEFAULT_DATE_FORMAT,
    )
    handler.setLevel(level)

    # Add handler to root logger
    root_logger.addHandler(handler)


def add_file_handler(logger_name: str, file_path: str, level: int = DEFAULT_LEVEL):
    """Add a file handler to the logger"""

    # Get logger
    logger = get_logger(logger_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create file handler
    handler = logging.FileHandler(file_path)
    handler.setLevel(level)

    # Set formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)


def set_log_level(name: str, level: int):
    """Set the log level for a specific logger"""

    logger = get_logger(name)
    logger.setLevel(level)


def log_agent_selection(agent_id: str, reasoning: str):
    """Log agent selection with Rich formatting"""

    console.print(
        Rule(f"[agent]Agent Selection: {agent_id}[/agent]", style="separator")
    )

    if reasoning:
        # Create a panel with the reasoning
        panel = Panel(
            Text(reasoning, style="thinking"),
            title="Selection Reasoning",
            border_style="dim",
            expand=False,
        )
        console.print(panel)


def log_agent_action(agent_id: str, action_type: str, params: Dict[str, Any] = None):
    """Log an agent action with Rich formatting"""

    # Build the action information
    action_text = f"[agent]{agent_id}[/agent] → [action]{action_type}[/action]"

    console.print(Rule(action_text, style="separator"))

    # Display parameters if provided
    if params:
        # Filter out large content or sensitive info
        filtered_params = {
            k: v for k, v in params.items() if k not in ["content", "file_content"]
        }

        if filtered_params:
            param_table = Table(
                show_header=False, box=None, expand=False, padding=(0, 1)
            )
            param_table.add_column("Parameter")
            param_table.add_column("Value")

            for key, value in filtered_params.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 80:
                    value = value[:77] + "..."
                param_table.add_row(f"[dim]{key}[/dim]", str(value))

            console.print(param_table)


def log_execution_result(
    success: bool, output: Optional[str] = None, error: Optional[str] = None
):
    """Log execution result with Rich formatting"""
    status = "[success]Success[/success]" if success else "[failure]Failure[/failure]"
    console.print(f"Result: {status}")

    if output:
        # Truncate long outputs
        if len(output) > 500:
            console.print("[dim]Output (truncated):[/dim]")
            console.print(Text(output[:200] + "...", style="dim"))
            console.print(Text("..." + output[-200:], style="dim"))
        else:
            console.print("[dim]Output:[/dim]")
            console.print(Text(output, style="dim"))

    if error:
        console.print("[error]Error:[/error]", Text(error, style="error"))


def log_step(step_number: int, step_result: Dict[str, Any], verbose: bool = False):
    """Log a step with Rich formatting"""

    # Create a header for the step
    step_header = f"Step {step_number}"
    console.print("\n")
    console.print(Rule(f"[highlight]{step_header}[/highlight]", style="separator"))

    action = step_result.get("action", {})
    execution_result = step_result.get("execution_result", {})

    # Log the thinking process
    thought = action.get("thought")
    if thought and verbose:
        panel = Panel(
            Markdown(thought),
            title="Thinking Process",
            border_style="dim",
            expand=False,
        )
        console.print(panel)

    # Log the action
    action_type = action.get("type")
    action_params = action.get("params", {})
    tool_calls = action.get("tool_calls", [])

    if tool_calls:
        for call in tool_calls:
            console.print(f"[action]Tool:[/action] {call.function.name}")

            # Format arguments as JSON
            try:
                args = json.loads(call.function.arguments)
                args_syntax = Syntax(
                    json.dumps(args, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                )
                console.print(args_syntax)
            except:
                console.print(f"Arguments: {call.function.arguments}")
    elif action_type:
        log_agent_action(
            step_result.get("agent_id", "agent"),
            action_type,
            action_params if verbose else None,
        )

    # Log the execution result
    if execution_result:
        if isinstance(execution_result, dict):
            success = execution_result.get("status", "error") == "success"
            output = execution_result["results"][0]["content"]
            error = execution_result.get("error")

            log_execution_result(success, output, error)
        else:
            console.print(f"Result: {execution_result}")

    console.print(Rule(style="separator"))


def log_task_completion(completed: bool, stats: Dict[str, Any] = None):
    """Log task completion status with statistics"""

    status = (
        "[success]Completed[/success]" if completed else "[failure]Terminated[/failure]"
    )
    console.print("\n")
    console.print(Rule(f"Task {status}", style="separator"))

    if stats:
        stats_table = Table(title="Usage Statistics")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")

        for key, value in stats.items():
            if key.endswith("tokens"):
                stats_table.add_row(key.replace("_", " ").title(), f"{value:,}")
            elif key.endswith("cost"):
                stats_table.add_row(key.replace("_", " ").title(), f"${value:.5f}")
            else:
                stats_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(stats_table)


def log_llm_request(messages=None, prompt=None, model=None, tools=None):
    """Log an LLM request with Rich formatting"""

    console.print(Rule("[highlight]LLM Request[/highlight]", style="separator"))

    if model:
        console.print(f"[dim]Model:[/dim] [agent]{model}[/agent]")

    if prompt:
        # Create a panel with the prompt
        panel = Panel(
            Markdown(prompt), title="Prompt", border_style="cyan", expand=False
        )
        console.print(panel)

    if messages:
        msg_table = Table(show_header=True, box=SIMPLE, expand=False, title="Messages")
        msg_table.add_column("Role", style="bold")
        msg_table.add_column("Content")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format based on role
            role_style = "cyan"
            if role == "system":
                role_style = "yellow"
            elif role == "assistant":
                role_style = "green"
            elif role == "user":
                role_style = "blue"

            # Truncate very long content for display
            if len(content) > 1000:
                displayed_content = content[:500] + "\n...\n" + content[-500:]
            else:
                displayed_content = content

            msg_table.add_row(f"[{role_style}]{role}[/{role_style}]", displayed_content)

        console.print(msg_table)

    if tools:
        tool_table = Table(
            show_header=True, box=SIMPLE, expand=False, title="Available Tools"
        )
        tool_table.add_column("Name", style="bold")
        tool_table.add_column("Description")

        for tool in tools:
            name = tool.get("function", {}).get("name", "unknown")
            description = tool.get("function", {}).get("description", "")
            tool_table.add_row(name, description)

        console.print(tool_table)


def log_llm_response(response):
    """Log an LLM response with Rich formatting"""

    console.print(Rule("[highlight]LLM Response[/highlight]", style="separator"))

    # Extract basic information
    if isinstance(response, dict):
        model = response.get("model", "unknown")
        text = response.get("text", "")
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)
        cost = response.get("cost", 0.0)
        finish_reason = response.get("finish_reason", "unknown")
        tool_calls = []

        # Try to extract tool calls if available
        raw_response = response.get("raw_response", {})
        if raw_response:
            try:
                # Check for litellm/openai format
                if hasattr(raw_response, "choices") and raw_response.choices:
                    choice = raw_response.choices[0]
                    if hasattr(choice, "message") and choice.message:
                        message = choice.message
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            tool_calls = message.tool_calls
                # Handle dictionary format
                elif isinstance(raw_response, dict) and "choices" in raw_response:
                    choice = raw_response["choices"][0]
                    if "message" in choice and "tool_calls" in choice["message"]:
                        tool_calls = choice["message"]["tool_calls"]
            except Exception as e:
                console.print(f"[dim]Warning: Error extracting tool calls: {e}[/dim]")
    else:
        # Fallback for unexpected response format
        text = str(response)
        model = "unknown"
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        finish_reason = "unknown"
        tool_calls = []

    # Display model and token info
    info_table = Table(show_header=False, box=None, expand=False)
    info_table.add_column("Property")
    info_table.add_column("Value")

    info_table.add_row("[dim]Model[/dim]", f"[agent]{model}[/agent]")
    info_table.add_row("[dim]Tokens[/dim]", f"{input_tokens} in / {output_tokens} out")
    info_table.add_row("[dim]Cost[/dim]", f"${cost:.6f}")
    info_table.add_row("[dim]Finish Reason[/dim]", finish_reason)

    console.print(info_table)

    # Display the response content
    if text:
        content_panel = Panel(
            Markdown(text), title="Content", border_style="green", expand=False
        )
        console.print(content_panel)

    # Display tool calls if available
    if tool_calls:
        console.print("[bold]Tool Calls:[/bold]")

        for call in tool_calls:
            try:
                name = call.function.name
                args = call.function.arguments

                # Try to parse and pretty print the arguments
                try:
                    if isinstance(args, str):
                        args = json.loads(args)
                    args_str = json.dumps(args, indent=2)
                except:
                    args_str = str(args)

                call_panel = Panel(
                    Syntax(args_str, "json", theme="monokai", line_numbers=False),
                    title=f"[bold blue]{name}[/bold blue]",
                    border_style="blue",
                    expand=False,
                )
                console.print(call_panel)
            except Exception as e:
                console.print(f"[error]Error displaying tool call: {e}[/error]")
                console.print(call)


# Initialize console logging by default
configure_root_logger()

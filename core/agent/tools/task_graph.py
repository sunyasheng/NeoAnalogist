from typing import Any, Dict
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk


_TASK_GRAPH_DESCRIPTION = """Build a task graph (DAG) from a task description.
The tool will analyze the task and create a directed acyclic graph showing the execution flow
and dependencies between different steps of the task.

### Task Analysis
* Task decomposition: The tool breaks down complex tasks into smaller, manageable steps
* Dependency identification: Identifies dependencies and relationships between tasks
* Execution flow: Creates a clear visualization of task execution order

### Graph Structure
* DAG format: The graph is a Directed Acyclic Graph (DAG)
* Node representation: Each node represents a distinct step or subtask
* Edge representation: Edges show dependencies between tasks
* No cycles: The graph ensures no circular dependencies

### Output Format
* Tree-like structure: The output is formatted as a tree-like string
* Node descriptions: Each node includes a brief description of its purpose
* Hierarchical layout: Shows clear parent-child relationships between tasks
"""

def create_task_graph_tool() -> ChatCompletionToolParam:
    """Create a tool for building task graphs from task descriptions."""
    return ChatCompletionToolParam(
        type="function",
        function=ChatCompletionToolParamFunctionChunk(
            name="build_task_graph",
            description=_TASK_GRAPH_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "A detailed description of the task to be analyzed and converted into a task graph.",
                    }
                },
                "required": ["task_description"],
            },
        ),
    ) 
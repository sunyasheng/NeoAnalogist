"""Repo Edit Tool for Agent

This tool provides agent access to the repo edit functionality,
allowing the agent to edit repository code based on a natural language instruction.
"""

from litellm import ChatCompletionToolParam

RepoEditTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "repo_edit",
        "description": (
            "Edit repository code or fix bugs according to a user-provided natural language instruction. "
            "This tool directly applies code edits or bug fixes as described by the user. "
            "It is strongly recommended to provide the full Python error message and traceback when making a bug fix request, as this will help accurately locate and resolve the issue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to be edited"
                },
                "edit_description": {
                    "type": "string",
                    "description": "A natural language description of the edit to perform."
                },
                "traceback": {
                    "type": "string",
                    "description": "Optional: The error message and traceback if this is a bug fix request."
                }
            },
            "required": ["repo_path", "edit_description"]
        }
    }
} 
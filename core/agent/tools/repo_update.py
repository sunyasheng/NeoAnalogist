"""Repo Update Tool for Agent

This tool provides agent access to the repo update functionality,
allowing the agent to edit and modify repository code based on requirements.
"""

from litellm import ChatCompletionToolParam

RepoUpdateTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "repo_update",
        "description": (
            "Update and modify repository code based on user requirements. "
            "This tool can analyze the repository structure, identify relevant files, "
            "and generate code modifications to implement new features, improve code quality, "
            "and add functionality. It provides intelligent multi-file modifications."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to be updated"
                },
                "requirements": {
                    "type": "string", 
                    "description": "Detailed requirements describing what changes need to be made to the repository, such as 'Add logging to all functions', 'Implement user authentication', or 'Add data validation'"
                },
                "target_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific files to modify. If not provided, relevant files will be auto-detected"
                },
                "context": {
                    "type": "string", 
                    "description": "Optional additional context about the changes needed"
                },
                "apply_changes": {
                    "type": "boolean",
                    "description": "Whether to apply the changes to actual files",
                    "default": False
                }
            },
            "required": ["repo_path", "requirements"]
        }
    }
} 
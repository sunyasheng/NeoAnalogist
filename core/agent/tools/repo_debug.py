"""Repo Debug Tool for Agent

This tool provides agent access to the repo debug functionality,
allowing the agent to automatically fix code issues and perform code editing using refact agent.
"""

from litellm import ChatCompletionToolParam

RepoDebugTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "repo_debug",
        "description": (
            "Debug, fix, and edit code using refact agent. "
            ""
            "For ERROR FIXING: paste the exact error message and traceback "
            "- 'Fix error: NameError: name \"epochs\" is not defined in trainer.py line 81' "
            ""
            "For CODE EDITING: describe what to add/modify "
            "- 'Add validate_data() function to utils.py' "
            ""
            "Avoid vague descriptions like 'Fix the code'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository containing the code to debug or edit"
                },
                "action_description": {
                    "type": "string", 
                    "description": "For errors: paste exact error message. For editing: describe what to add/modify"
                }
            },
            "required": ["repo_path", "action_description"]
        }
    }
} 
"""Repo Verify Tool for Agent

This tool provides agent access to the repo verify functionality,
allowing the agent to verify repository implementation and functionality.
"""

from litellm import ChatCompletionToolParam

RepoVerifyTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "repo_verify",
        "description": (
            "Verify repository implementation and functionality with static analysis (no execution). "
            "This tool performs repository analysis including structure understanding, "
            "dependency analysis, and code review. It focuses on reproduce.sh as the main entry point "
            "and analyzes code structure without running it. Use this tool for initial assessment "
            "and to identify potential issues, but remember that it cannot solve problems - it only "
            "identifies them. For active problem-solving, use other tools like web_read, browser, "
            "or repo_debug. This tool provides a summary report but does not fix issues automatically. "
            "IMPORTANT: If this tool identifies dataset issues (placeholder URLs, missing files), "
            "use browser and web_read tools to search for real dataset sources before running repo_run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to be verified"
                },
                "requirement": {
                    "type": "string", 
                    "description": "Optional description of what the repository should accomplish (e.g., 'Implement a calculator', 'Reproduce research results'). If not provided, performs a general completeness and feasibility check of the entire codebase."
                },
                "verification_level": {
                    "type": "string",
                    "enum": ["basic", "functional", "comprehensive"],
                    "description": "Level of verification to perform",
                    "default": "comprehensive"
                }
            },
            "required": ["repo_path"]
        }
    }
} 
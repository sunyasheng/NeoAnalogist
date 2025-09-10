"""Repo Run Tool for Agent

This tool provides agent access to the repo run functionality,
allowing the agent to execute repository reproduce.sh scripts in isolated Docker containers.
"""

from litellm import ChatCompletionToolParam

RepoRunTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "repo_run",
        "description": (
            "Execute a repository's reproduce.sh script in an isolated Docker container. "
            "This tool mounts the repository into a container, installs dependencies, "
            "and runs the reproduce.sh script with full isolation and resource control. "
            "It provides real-time output streaming and comprehensive execution results. "
            "IMPORTANT: This tool will fail if reproduce.sh is missing or empty. "
            "Check the error message carefully - if it says 'reproduce.sh not found' or "
            "'reproduce.sh exists but is empty', you need to create or fix the script first. "
            "Useful for reproducing research results, testing implementations, and "
            "validating repository functionality in a controlled environment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository containing reproduce.sh script"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout for script execution in seconds",
                    "default": 3600
                },
                "docker_image": {
                    "type": "string",
                    "description": "Docker image to use for isolation",
                    "default": "pb-reproducer:latest"
                },
                "memory_limit": {
                    "type": "string",
                    "description": "Memory limit for container (e.g., '4g', '8g')",
                    "default": "4g"
                },
                "network_enabled": {
                    "type": "boolean",
                    "description": "Whether to enable network access in container",
                    "default": True
                },
                "gpu_enabled": {
                    "type": "boolean",
                    "description": "Whether to enable GPU access in container",
                    "default": True
                },
                "use_persistent_containers": {
                    "type": "boolean",
                    "description": "Whether to use persistent containers to avoid reinstalling dependencies. Set to false for complete isolation.",
                    "default": True
                }
            },
            "required": ["repo_path"]
        }
    }
} 
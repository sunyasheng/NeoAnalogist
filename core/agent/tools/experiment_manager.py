"""Experiment Manager Tool for Agent

This tool lets the agent manage experiments:
- wrap: generate an MLflow wrapper script for a given command in a repo, so the run is tracked.
- query: list existing experiments and basic status.

Typical flow:
1) Use wrap to create the wrapper file near the repo (mlflow_scripts directory next to submission)
2) Then run the produced Python file via cmd tool to record the run
3) Later use query to see what experiments exist
"""

from litellm import ChatCompletionToolParam

ExperimentManagerTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "experiment_manager",
        "description": (
            "Experiment manager. Use 'wrap' to generate an MLflow tracking wrapper for a bash command in a repo; "
            "then run the produced wrapper with the cmd tool to record the experiment. Use 'query' to list experiments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["wrap", "query"],
                    "description": "Mode: wrap (create MLflow wrapper) or query (list experiments)."
                },
                "command": {
                    "type": "string",
                    "description": "Bash command to wrap (e.g., 'python main.py --config config.yaml'). Required in wrap mode.",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Repository path used for wrapping (used to place wrapper under mlflow_scripts)."
                },
                "experiment_name": {
                    "type": "string",
                    "description": "Optional experiment name (wrapper filename will derive from this)."
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional explicit output dir for wrappers; normally auto-set near the repo."
                }
            },
            "required": ["mode"]
        }
    }
}

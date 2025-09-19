"""Image Edit Judge Tool for Agent

This tool lets the agent evaluate the quality of image editing by comparing
original and edited images with the given instruction.
"""

from litellm import ChatCompletionToolParam


ImageEditJudgeTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "image_edit_judge",
        "description": (
            "Evaluate the quality of image editing by comparing original and edited images. "
            "Provides correctness assessment, score, and detailed feedback about what went wrong or right."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "original_path": {
                    "type": "string",
                    "description": "Absolute path to the original image file.",
                },
                "edited_path": {
                    "type": "string", 
                    "description": "Absolute path to the edited image file.",
                },
                "instruction": {
                    "type": "string",
                    "description": "The edit instruction that was given to the agent.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600).",
                    "default": 600,
                },
            },
            "required": ["original_path", "edited_path", "instruction"],
        },
    },
}

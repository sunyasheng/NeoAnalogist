"""Image Edit Judge Tool for Agent

This tool lets the agent evaluate the quality of image editing by comparing
original and edited images using AnyBench metrics and optional Qwen analysis.
"""

from litellm import ChatCompletionToolParam


ImageEditJudgeTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "image_edit_judge",
        "description": (
            "Evaluate the quality of image editing by comparing original and edited images. "
            "Uses AnyBench metrics (FID, CLIP-Score, DINO-Score) and optional Qwen analysis. "
            "Provides detailed evaluation results including success rate and analysis."
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
                "input_caption": {
                    "type": "string",
                    "description": "Description of what the original image contains.",
                },
                "output_caption": {
                    "type": "string",
                    "description": "Description of what the edited image should contain.",
                },
                "use_qwen_analysis": {
                    "type": "boolean",
                    "description": "Whether to use Qwen API for intelligent analysis (default: True).",
                    "default": True,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600).",
                    "default": 600,
                },
            },
            "required": ["original_path", "edited_path", "input_caption", "output_caption"],
        },
    },
}

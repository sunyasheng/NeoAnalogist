"""Qwen2.5-VL API Tool for Agent

This tool lets the agent invoke the Qwen2.5-VL API for image analysis and text generation.
"""

from litellm import ChatCompletionToolParam


QwenAPITool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "qwen_api",
        "description": (
            "Analyze images using Qwen2.5-VL vision-language model. "
            "This tool requires an image_path and is designed for image analysis tasks. "
            "Use 'generate' mode for single image analysis requests, "
            "or 'chat' mode for structured conversation about images."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt for the model. Required for 'generate' mode.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Required path to image file for analysis.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["generate", "chat"],
                    "description": "Operation mode: 'generate' for single request, 'chat' for structured conversation",
                    "default": "generate",
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": "Maximum number of new tokens to generate",
                    "default": 128,
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (0.0 to 1.0)",
                    "default": 0.7,
                },
                "top_p": {
                    "type": "number",
                    "description": "Nucleus sampling parameter (0.0 to 1.0)",
                    "default": 0.9,
                },
                "messages": {
                    "type": "string",
                    "description": "JSON string of structured messages for 'chat' mode. Format: [{'role': 'user', 'content': [{'type': 'text', 'text': 'message'}]}]",
                },
            },
            "required": ["prompt", "image_path"],
        },
    },
}

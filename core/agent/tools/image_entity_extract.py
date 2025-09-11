"""Image Entity Extraction Tool for Agent

This tool lets the agent extract a caption and entities (with bounding boxes and descriptions)
from an image using the runtime's `image_entity_extract` action.
"""

from litellm import ChatCompletionToolParam

ImageEntityExtractTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "image_entity_extract",
        "description": (
            "Extract a global caption and entities from an image. "
            "Each entity includes label, score, bbox [x,y,h,w] in pixels and a short description. "
            "Path must be readable inside the sandbox container (e.g., /app_sci/... or workspace/...)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image inside the container (e.g., /app_sci/test.png)",
                },
                "model": {
                    "type": "string",
                    "description": "Vision model to use (default gpt-4o)",
                    "default": "gpt-4o",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 180)",
                    "default": 180,
                },
            },
            "required": ["image_path"],
        },
    },
}



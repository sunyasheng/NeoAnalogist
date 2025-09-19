"""GroundingSAM Tool for Agent

Text-prompted instance segmentation via external GroundingSAM FastAPI, routed through
the runtime server using `GroundingSAMAction`.
"""

from litellm import ChatCompletionToolParam


GroundingSAMTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "grounding_sam_segment",
            "description": (
            "Text-prompted instance segmentation using GroundingSAM (Florence2+SAM2). "
            "Supports both simple labels (e.g., 'cat, dog') and complex descriptions for precise targeting. "
            "Streams PNG and saves to output_path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Container-absolute path to the input image (e.g., /app_sci/workspace/img.png).",
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Text prompt for segmentation. Can be simple labels (e.g., 'cat, dog') or complex descriptions for precise targeting.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Explicit file path to save streamed mask PNG.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600)",
                    "default": 600,
                },
            },
            "required": [
                "image_path",
                "text_prompt",
            ],
        },
    },
}




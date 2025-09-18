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
            "Provide container-absolute image path and comma-separated text prompt labels. "
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
                    "description": "Comma-separated labels, e.g., 'person,car'.",
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




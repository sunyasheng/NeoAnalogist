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
            "When return_type is json and output_dir is provided, masks are saved to that directory."
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
                "return_type": {
                    "type": "string",
                    "description": "Return type from external API: 'image' (PNG stream) or 'json' (paths).",
                    "enum": ["image", "json"],
                    "default": "json",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Host-absolute directory to save masks when return_type=json.",
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




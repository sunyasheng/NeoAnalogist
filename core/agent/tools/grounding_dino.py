"""GroundingDINO Tool for Agent

Text-prompted object detection via external GroundingDINO FastAPI, routed through
the runtime server using `GroundingDINOAction`.
"""

from litellm import ChatCompletionToolParam


GroundingDINOTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "grounding_dino_detect",
        "description": (
            "Text-prompted object detection using GroundedSAM (includes GroundingDINO functionality). "
            "Detects objects in images based on text descriptions. "
            "Returns bounding boxes and labels for detected objects."
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
                    "description": "Text prompt for detection. Can be simple labels (e.g., 'cat, dog') or complex descriptions (e.g., 'cat in mirror').",
                },
                "box_threshold": {
                    "type": "number",
                    "description": "Box confidence threshold (default 0.3)",
                    "default": 0.3,
                },
                "text_threshold": {
                    "type": "number", 
                    "description": "Text confidence threshold (default 0.25)",
                    "default": 0.25,
                },
                "return_type": {
                    "type": "string",
                    "description": "Return format: 'json' for detection data, 'image' for annotated image",
                    "enum": ["json", "image"],
                    "default": "json",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save annotated image (only used when return_type='image')",
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

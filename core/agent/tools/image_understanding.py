"""
Image Understanding Tool

Comprehensive image analysis tool that can analyze images with masks, bounding boxes, and labels.
"""

from typing import Dict, Any
from litellm.types.utils import ChatCompletionToolParam

ImageUnderstandingTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "image_understanding",
        "description": (
            "Comprehensive image understanding and analysis. "
            "Analyzes images with optional masks, bounding boxes, and labels to provide detailed understanding including "
            "object descriptions, spatial relationships, scene context, and visual elements. "
            "Can work with just an image or with additional detection/segmentation data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Container-absolute path to the input image (e.g., /app_sci/workspace/img.png).",
                },
                "boxes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Bounding box coordinates [x1, y1, x2, y2]"
                    },
                    "description": "Optional list of bounding boxes for detected objects. Each box is [x1, y1, x2, y2].",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of object labels corresponding to the bounding boxes.",
                },
                "masks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of paths to mask images for segmented objects.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600).",
                    "default": 600,
                },
            },
            "required": ["image_path"],
        },
    },
}

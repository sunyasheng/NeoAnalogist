"""
Inpaint-Anything Remove Tool for LLM Function Calling

Provides LLM with the ability to remove objects from images using Inpaint-Anything.
"""

from litellm import ChatCompletionToolParam

InpaintRemoveTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "inpaint_remove",
        "description": (
            "Remove objects from images using Inpaint-Anything (LaMa). "
            "Provide one input image and one binary mask; an optional dilate_kernel_size controls edge expansion. "
            "Returns JSON with output image path(s)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Container-absolute path to the input image.",
                },
                "mask_path": {
                    "type": "string",
                    "description": "Container-absolute path to the binary mask image (white=remove).",
                },
                "dilate_kernel_size": {
                    "type": "integer",
                    "description": "Dilate kernel size for mask expansion to avoid edge artifacts (default: 0)",
                    "default": 0,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600)",
                    "default": 600,
                },
            },
            "required": ["image_path", "mask_path"],
        },
    },
}

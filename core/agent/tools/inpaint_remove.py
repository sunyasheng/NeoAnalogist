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
            "Remove objects from images using Inpaint-Anything (SAM + LaMa). "
            "Provide either point coordinates to click on an object, or a mask image. "
            "Returns inpainted image paths if output_dir is provided, otherwise streams the first result as PNG."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Container-absolute path to the input image.",
                },
                "point_coords": {
                    "type": "string",
                    "description": "Point coordinates as 'x,y' to click on object for removal (e.g., '200,300'). Alternative to mask_path.",
                },
                "mask_path": {
                    "type": "string",
                    "description": "Container-absolute path to mask image for object removal. Alternative to point_coords.",
                },
                "dilate_kernel_size": {
                    "type": "integer",
                    "description": "Dilate kernel size for mask expansion to avoid edge effects (default: 10)",
                    "default": 10,
                },
                "return_type": {
                    "type": "string",
                    "enum": ["image", "json"],
                    "default": "image",
                    "description": "Return type: 'image' (PNG stream of first result) or 'json' (list of result paths).",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Host-absolute path to save results when return_type is 'json'. If not provided, results are saved to a temporary cache.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600)",
                    "default": 600,
                },
            },
            "required": ["image_path"],
        },
    },
}

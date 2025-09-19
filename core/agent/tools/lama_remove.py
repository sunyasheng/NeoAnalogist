"""
LAMA Object Removal Tool for LLM Function Calling

Provides LLM with the ability to remove objects from images using LAMA (LaMa) model.
Specialized for object removal with automatic background inpainting.
"""

from litellm import ChatCompletionToolParam

LAMARemoveTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "lama_remove",
        "description": (
            "Remove objects from images using LAMA (LaMa) model for automatic background inpainting. "
            "Provide input image and binary mask; the tool will automatically fill the masked region "
            "with appropriate background content. Streams PNG and saves to output_path."
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
                    "description": "Container-absolute path to the binary mask image (white=remove region).",
                },
                "dilate_kernel_size": {
                    "type": "integer",
                    "description": "Dilate kernel size for mask expansion to avoid edge artifacts (default: 0)",
                    "default": 0,
                },
                "output_path": {
                    "type": "string",
                    "description": "Explicit file path to save streamed output PNG.",
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

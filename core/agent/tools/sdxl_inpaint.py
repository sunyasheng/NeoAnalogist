"""
SDXL Inpainting Tool for LLM Function Calling

Provides LLM with the ability to perform high-quality text-guided image inpainting using Stable Diffusion XL.
Uses smart cropping for better results.
"""

from litellm import ChatCompletionToolParam

SDXLInpaintTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "sdxl_inpaint",
        "description": (
            "High-quality text-guided image inpainting using Stable Diffusion XL. "
            "Fill masked areas with content described by text prompt. "
            "Uses smart cropping for better results. Streams PNG and saves to output_path."
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
                    "description": "Container-absolute path to the binary mask image (white=inpaint).",
                },
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing what to fill in the masked area.",
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Guidance scale (1.0-20.0). Higher values follow prompt more closely.",
                    "default": 8.0,
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of inference steps (15-50). More steps = better quality but slower.",
                    "default": 20,
                },
                "strength": {
                    "type": "number",
                    "description": "Strength (0.0-1.0). Use < 1.0 for better results.",
                    "default": 0.99,
                },
                "use_smart_crop": {
                    "type": "boolean",
                    "description": "Use smart cropping to focus on masked region (recommended for better results).",
                    "default": True,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible results (optional).",
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
            "required": ["image_path", "mask_path", "prompt"],
        },
    },
}

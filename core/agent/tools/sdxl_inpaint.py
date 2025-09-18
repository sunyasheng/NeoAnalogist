"""SDXL Inpainting Tool for Agent

High-quality image inpainting using Stable Diffusion XL inpainting model.
Provides LLM with the ability to inpaint images with text prompts.
"""

from litellm import ChatCompletionToolParam

SDXLInpaintTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "sdxl_inpaint",
        "description": (
            "High-quality image inpainting using Stable Diffusion XL. "
            "Provide input image, mask image, and text prompt to generate inpainted content. "
            "Supports various quality and style parameters."
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
                    "description": "Container-absolute path to the binary mask image (white=inpaint region).",
                },
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing what to generate in the masked region.",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Negative prompt to avoid certain elements (optional).",
                    "default": "",
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Guidance scale for prompt adherence (default: 8.0)",
                    "default": 8.0,
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of denoising steps (default: 20, range: 15-30)",
                    "default": 20,
                },
                "strength": {
                    "type": "number",
                    "description": "Inpainting strength (default: 0.99, must be < 1.0)",
                    "default": 0.99,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible results (optional).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Container-absolute path to save the inpainted result PNG.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default: 600)",
                    "default": 600,
                },
            },
            "required": [
                "image_path",
                "mask_path", 
                "prompt",
                "output_path",
            ],
        },
    },
}

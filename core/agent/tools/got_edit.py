"""GoT Image Edit/Generation Tool for Agent

This tool lets the agent invoke the GoT API via the runtime's `got_edit` action
for text-to-image (t2i) or edit modes.
"""

from litellm import ChatCompletionToolParam


GoTEditTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "got_edit",
        "description": (
            "Generate or edit an image using the GoT model. "
            "Use mode 't2i' for text-to-image, or 'edit' with an input image. "
            "When in 'edit' mode, provide image_path accessible on the host."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "Absolute path on host to the input image for edit mode. "
                        "Leave empty for t2i."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "Prompt instructing the generation or edit.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["t2i", "edit"],
                    "description": "Operation mode: text-to-image or edit",
                    "default": "t2i",
                },
                "height": {
                    "type": "integer",
                    "description": "Output image height (multiple of 8, typically 1024)",
                    "default": 1024,
                },
                "width": {
                    "type": "integer",
                    "description": "Output image width (multiple of 8, typically 1024)",
                    "default": 1024,
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": "Max new tokens for MLLM chain-of-thought",
                    "default": 1024,
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Diffusion steps",
                    "default": 50,
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "CFG guidance scale",
                    "default": 7.5,
                },
                "image_guidance_scale": {
                    "type": "number",
                    "description": "Edit image guidance scale",
                    "default": 1.0,
                },
                "cond_image_guidance_scale": {
                    "type": "number",
                    "description": "Conditional image guidance scale",
                    "default": 4.0,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600)",
                    "default": 600,
                },
            },
            "required": ["prompt", "mode"],
        },
    },
}



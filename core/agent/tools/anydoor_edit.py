"""AnyDoor Edit Tool for Agent

This tool lets the agent invoke the AnyDoor API via the runtime's `anydoor_edit` action.
Reference object transfer: ref (with mask/alpha) -> target region (mask).
"""

from litellm import ChatCompletionToolParam


AnyDoorEditTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "anydoor_edit",
        "description": (
            "Reference-guided object transfer using AnyDoor. "
            "Provide ref_image (PNG with alpha preferred or ref_mask), target_image, and target_mask."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref_image_path": {
                    "type": "string",
                    "description": "Container-absolute path to reference image (PNG with alpha preferred).",
                },
                "ref_mask_path": {
                    "type": "string",
                    "description": "Optional container-absolute path to reference mask (required if ref has no alpha).",
                },
                "target_image_path": {
                    "type": "string",
                    "description": "Container-absolute path to target image.",
                },
                "target_mask_path": {
                    "type": "string",
                    "description": "Container-absolute path to target mask (binary).",
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Guidance scale for AnyDoor.",
                    "default": 5.0,
                },
                "output_path": {
                    "type": "string",
                    "description": "Container-absolute path to save output PNG.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout seconds (default 600)",
                    "default": 600,
                },
            },
            "required": [
                "ref_image_path",
                "target_image_path",
                "target_mask_path",
                "output_path",
            ],
        },
    },
}



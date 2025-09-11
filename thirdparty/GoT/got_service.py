import os
from typing import Optional, Dict, Any
from PIL import Image

import torch

# Minimal lazy loader to avoid heavy imports at module import time
_GOT_MODEL = None
_GOT_PROCESSOR = None


def _load_got_once():
    global _GOT_MODEL, _GOT_PROCESSOR
    if _GOT_MODEL is not None:
        return _GOT_MODEL, _GOT_PROCESSOR

    # The upstream repo in this workspace does not provide builder helpers.
    # Expose a clear error with setup guidance instead of failing on import.
    from got.models.got_model import GenCot  # noqa: F401 (verify module exists)
    raise RuntimeError(
        "GoT setup incomplete: missing model builders. Please ensure: "
        "1) Pretrained weights under thirdparty/GoT/pretrained as README specifies; "
        "2) Implement model/processor builders or install the official GoT package providing them; "
        "3) Then update got_service._load_got_once() to construct (mllm, output_projector, output_projector_add, vae, unet, scheduler, processor)."
    )


def got_generate(
    prompt: str,
    mode: str = "t2i",
    image_path: Optional[str] = None,
    height: int = 1024,
    width: int = 1024,
    max_new_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.0,
    cond_image_guidance_scale: float = 4.0,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model, _ = _load_got_once()

    pil_img = None
    prompt_type = "t2i"
    if mode == "edit":
        prompt_type = "edit"
        if not image_path:
            raise ValueError("image_path is required when mode == 'edit'")
        pil_img = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        out = model.generate(
            text_input=prompt,
            image=pil_img,
            max_new_tokens=max_new_tokens,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            cond_image_guidance_scale=cond_image_guidance_scale,
            height=height,
            width=width,
            prompt_type=prompt_type,
        )

    return out



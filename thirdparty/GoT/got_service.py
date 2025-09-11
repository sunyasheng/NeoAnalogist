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

    # Minimal hydra-free builder using provided Hydra-like config values
    from got.models.got_model import GenCot
    from got.models.projector import LinearProjector
    from got.processer.qwen25_vl_processor import get_processor
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import AutoModelForCausalLM, AutoConfig

    root_dir = os.path.dirname(__file__)
    pretrained_dir = os.path.join(root_dir, "pretrained")

    # Processor
    processor = get_processor(os.path.join(pretrained_dir, "Qwen2.5-VL-3B-Instruct"), add_gen_token_num=64)

    # MLLM backbone (Qwen2.5-VL-3B-Instruct). If vision variant requires a specific class, users must adjust here.
    mllm = AutoModelForCausalLM.from_pretrained(
        os.path.join(pretrained_dir, "Qwen2.5-VL-3B-Instruct"),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Output projectors per config
    output_projector = LinearProjector(in_hidden_size=2048, out_hidden_size=2048)
    output_projector_add = LinearProjector(in_hidden_size=2048, out_hidden_size=1280)

    # Diffusion parts per config
    sdxl_root = os.path.join(pretrained_dir, "stable-diffusion-xl-base-1.0")
    vae = AutoencoderKL.from_pretrained(sdxl_root, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sdxl_root, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(sdxl_root, subfolder="scheduler")

    model = GenCot.from_pretrained(
        mllm=mllm,
        output_projector=output_projector,
        output_projector_add=output_projector_add,
        scheduler=scheduler,
        vae=vae,
        unet=unet,
        processor=processor,
        pretrained_model_path=os.path.join(pretrained_dir, "GoT-6B", "pytorch_model.bin"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _GOT_MODEL = model
    _GOT_PROCESSOR = processor
    return _GOT_MODEL, _GOT_PROCESSOR


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



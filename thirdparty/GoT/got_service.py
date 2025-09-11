import os
from typing import Optional, Dict, Any
from PIL import Image

import torch

# Minimal lazy loader to avoid heavy imports at module import time
_GOT_MODEL = None
_GOT_PROCESSOR = None


def _load_sharded_state_dict(pretrained_dir: str) -> dict:
    """Load sharded state dict using HF-style index json."""
    import json
    index_path = os.path.join(pretrained_dir, "GoT-6B", "pytorch_model.bin.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    shard_map = index.get("weight_map", {})
    shard_files = sorted({v for v in shard_map.values()})
    combined = {}
    for shard in shard_files:
        shard_path = os.path.join(pretrained_dir, "GoT-6B", shard)
        sd = torch.load(shard_path, map_location="cpu")
        combined.update(sd)
    return combined


def _load_got_once():
    global _GOT_MODEL, _GOT_PROCESSOR
    if _GOT_MODEL is not None:
        return _GOT_MODEL, _GOT_PROCESSOR

    # Minimal hydra-free builder using provided Hydra-like config values
    from got.models.got_model import GenCot
    from got.models.projector import LinearProjector
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    root_dir = os.path.dirname(__file__)
    pretrained_dir = os.path.join(root_dir, "pretrained")

    # Processor (trust remote code for Qwen2.5-VL) and add special image tokens
    qwen_path = os.path.join(pretrained_dir, "Qwen2.5-VL-3B-Instruct")
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
    add_token_list = ['<|im_gen_start|>', '<|im_gen_end|>']
    for i in range(64):
        add_token_list.append(f"<|im_gen_{i:04d}|>")
    processor.tokenizer.add_tokens(add_token_list, special_tokens=True)

    # MLLM backbone (trust remote code for VL architecture)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # Ensure embeddings cover added tokens
    mllm.resize_token_embeddings(len(processor.tokenizer))

    # Output projectors per config
    output_projector = LinearProjector(in_hidden_size=2048, out_hidden_size=2048)
    output_projector_add = LinearProjector(in_hidden_size=2048, out_hidden_size=1280)

    # Diffusion parts per config
    sdxl_root = os.path.join(pretrained_dir, "stable-diffusion-xl-base-1.0")
    vae = AutoencoderKL.from_pretrained(sdxl_root, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sdxl_root, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(sdxl_root, subfolder="scheduler")

    # Build GenCot and load sharded weights (no single-file ckpt present)
    model = GenCot(
        mllm=mllm,
        output_projector=output_projector,
        output_projector_add=output_projector_add,
        scheduler=scheduler,
        vae=vae,
        unet=unet,
        processor=processor,
        num_img_out_tokens=64,
        img_gen_start_id=151667,
        box_start_id=151648,
        box_end_id=151649,
    )
    try:
        state_dict = _load_sharded_state_dict(pretrained_dir)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) or len(unexpected):
            print(f"[GoT] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    except FileNotFoundError:
        # Fall back: try single-file ckpt if user later places it
        single_path = os.path.join(pretrained_dir, "GoT-6B", "pytorch_model.bin")
        if os.path.exists(single_path):
            sd = torch.load(single_path, map_location="cpu")
            model.load_state_dict(sd, strict=False)
        else:
            raise
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



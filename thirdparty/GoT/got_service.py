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


def _load_sharded_state_dict_from_folder(folder_path: str) -> dict:
    """Load sharded state dict from a folder containing index json + bin shards."""
    import json
    index_path = os.path.join(folder_path, "pytorch_model.bin.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    shard_map = index.get("weight_map", {})
    shard_files = sorted({v for v in shard_map.values()})
    combined = {}
    for shard in shard_files:
        shard_path = os.path.join(folder_path, shard)
        sd = torch.load(shard_path, map_location="cpu")
        combined.update(sd)
    return combined


def _load_got_once():
    global _GOT_MODEL, _GOT_PROCESSOR
    if _GOT_MODEL is not None:
        return _GOT_MODEL, _GOT_PROCESSOR

    # Hydra-based builder (mirror notebook)
    import hydra
    from omegaconf import OmegaConf

    root_dir = os.path.dirname(__file__)
    cfg_mllm = OmegaConf.load(os.path.join(root_dir, 'configs/clm_models/llm_qwen25_vl_3b_lora.yaml'))
    cfg_got = OmegaConf.load(os.path.join(root_dir, 'configs/clm_models/agent_got.yaml'))
    weight_dtype = torch.bfloat16

    mllm_model = hydra.utils.instantiate(cfg_mllm, torch_dtype='bf16')
    if hasattr(mllm_model, 'config'):
        try:
            mllm_model.config.use_cache = True
        except Exception:
            pass
    mllm_model = mllm_model.eval()

    got_model = hydra.utils.instantiate(cfg_got, mllm=mllm_model)
    got_model = got_model.to(weight_dtype)
    if torch.cuda.is_available():
        got_model = got_model.cuda()
    got_model = got_model.eval()

    # Load sharded GoT weights from pretrained/GoT-6B
    pretrained_dir = os.path.join(root_dir, 'pretrained')
    try:
        ckpt = _load_sharded_state_dict_from_folder(os.path.join(pretrained_dir, 'GoT-6B'))
        logs = got_model.load_state_dict(ckpt, strict=False)
        print(logs)
    except FileNotFoundError:
        single_path = os.path.join(pretrained_dir, 'GoT-6B', 'pytorch_model.bin')
        if os.path.exists(single_path):
            sd = torch.load(single_path, map_location='cpu')
            logs = got_model.load_state_dict(sd, strict=False)
            print(logs)
        else:
            raise

    _GOT_MODEL = got_model
    _GOT_PROCESSOR = getattr(got_model, 'processor', None)
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



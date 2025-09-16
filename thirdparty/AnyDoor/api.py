from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
import os
import cv2
import numpy as np

import einops
import torch
import random
from omegaconf import OmegaConf
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
from run_inference import process_pairs, crop_back


app = FastAPI(title="AnyDoor API", description="Reference-guided object transfer/editing", version="1.0.0")

# Global model cache
_MODEL = None
_SAMPLER = None


def _load_anydoor_once():
    global _MODEL, _SAMPLER
    if _MODEL is not None:
        return _MODEL, _SAMPLER
    disable_verbosity()
    # Load config for pretrained path
    cfg = OmegaConf.load('./AnyDoor/configs/inference.yaml')
    model_ckpt = cfg.pretrained_model
    model_config = cfg.config_file
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location='cuda' if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    sampler = DDIMSampler(model)
    _MODEL, _SAMPLER = model, sampler
    return _MODEL, _SAMPLER


def _infer_with_cached_model(ref_image, ref_mask, tar_image, tar_mask, guidance_scale: float) -> np.ndarray:
    model, sampler = _load_anydoor_once()
    save_memory = False
    if save_memory:
        enable_sliced_attention()
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    num_samples = 1
    control = torch.from_numpy(item['hint'].copy()).float().cuda() if torch.cuda.is_available() else torch.from_numpy(item['hint'].copy()).float()
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(item['ref'].copy()).float().cuda() if torch.cuda.is_available() else torch.from_numpy(item['ref'].copy()).float()
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H, W = 512, 512
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    ddim_steps = 50
    eta = 0.0
    scale = guidance_scale
    model.control_scales = ([1.0] * 13)  # default
    samples, _ = sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
    return gen_image


@app.on_event("startup")
async def _warmup_model():
    try:
        _load_anydoor_once()
        print("[AnyDoor] Model loaded on startup.")
    except Exception as e:
        print(f"[AnyDoor] Startup load failed: {e}")


def _read_image_rgb(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    img_arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Failed to decode image")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)
    return rgb


def _get_ref_image_and_mask(ref_file: UploadFile, ref_mask_file: Optional[UploadFile]) -> tuple[np.ndarray, np.ndarray]:
    data = ref_file.file.read()
    img_arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Failed to decode ref image")
    if bgr.shape[-1] == 4 and ref_mask_file is None:
        # Use alpha as mask
        alpha = (bgr[:, :, 3] > 128).astype(np.uint8)
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        return rgb, alpha
    # No alpha or explicit mask provided
    rgb = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)
    if ref_mask_file is None:
        raise ValueError("ref_mask is required when ref image has no alpha channel")
    mask_rgb = _read_image_rgb(ref_mask_file)
    if mask_rgb.ndim == 3:
        mask = (cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY) > 128).astype(np.uint8)
    else:
        mask = (mask_rgb > 128).astype(np.uint8)
    return rgb, mask


@app.post("/anydoor/edit")
async def anydoor_edit(
    ref_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    ref_mask: Optional[UploadFile] = File(None),
    target_mask: UploadFile = File(...),
    guidance_scale: float = Form(5.0),
):
    """Reference object transfer: ref (with mask) -> target region (mask). Returns PNG image."""
    # Read inputs
    # Note: UploadFile streams; read() exhausts, so don't reuse file handles
    ref_img, ref_m = _get_ref_image_and_mask(ref_image, ref_mask)

    tgt_rgb = _read_image_rgb(target_image)
    tgt_mask_rgb = _read_image_rgb(target_mask)
    tgt_m = (cv2.cvtColor(tgt_mask_rgb, cv2.COLOR_RGB2GRAY) > 128).astype(np.uint8)

    # Run AnyDoor (cached model)
    gen_rgb = _infer_with_cached_model(ref_img, ref_m, tgt_rgb, tgt_m, guidance_scale=guidance_scale)

    # Encode to PNG and stream
    bgr = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError("Failed to encode output image")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ANYDOOR_API_PORT", 8401))
    uvicorn.run(app, host="0.0.0.0", port=port)



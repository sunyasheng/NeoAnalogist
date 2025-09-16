from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
import os
import cv2
import numpy as np

from run_inference import inference_single_image


app = FastAPI(title="AnyDoor API", description="Reference-guided object transfer/editing", version="1.0.0")


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

    # Run AnyDoor
    gen_rgb = inference_single_image(ref_img, ref_m, tgt_rgb, tgt_m, guidance_scale=guidance_scale)

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



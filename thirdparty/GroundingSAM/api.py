"""
GroundingSAM API: Text-prompted segmentation via GroundingDINO + SAM.

POST /grounding-sam/segment
  - image: UploadFile (required)
  - text_prompt: str (required, e.g., "cat, dog, person")
  - box_threshold: float (optional, default 0.3)
  - text_threshold: float (optional, default 0.25)
  - output_dir: str (optional; if set, saves masks as PNGs)

Returns: application/json with fields:
  - success: bool
  - error: str (if failed)
  - num_instances: int
  - mask_paths: list[str] (when saved)

Note:
  - This service attempts to import GroundingDINO and SAM from your thirdparty tree.
  - If dependencies/weights are missing, it returns 501 with a setup message.
"""

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional, List
import io
import os
import sys
import numpy as np
from PIL import Image

app = FastAPI(title="GroundingSAM API", description="Text-prompted segmentation (GroundingDINO + SAM)", version="1.0.0")


def _ensure_thirdparty_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    gdino_dir = os.path.join(repo_root, "thirdparty", "GroundingDINO")
    if os.path.isdir(gdino_dir) and gdino_dir not in sys.path:
        sys.path.append(gdino_dir)


def _try_load_models():
    """Best-effort import GroundingDINO + SAM. Returns (pipeline, error_str)."""
    _ensure_thirdparty_on_path()
    try:
        # Prefer huggingface pipeline if available; else try local modules
        # We keep imports lazy to fail fast with a clean message
        from groundingdino.util.inference import Model as Grounder
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, (
            "GroundingDINO/SAM not available in environment. Install and ensure weights are present. "
            "Hint: pip install groundingdino segment-anything; or add your local GroundingDINO to PYTHONPATH. "
            f"Original import error: {e}"
        )
    return (Grounder, sam_model_registry, SamPredictor), None


def _load_image_to_numpy(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


@app.post("/grounding-sam/segment")
async def grounding_sam_segment(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    output_dir: Optional[str] = Form(None),
):
    # Load models (lazy)
    models, err = _try_load_models()
    if err is not None:
        return JSONResponse(status_code=501, content={"success": False, "error": err})

    Grounder, sam_model_registry, SamPredictor = models  # type: ignore[misc]

    try:
        # NOTE: Here we avoid hardcoding weight paths; your GroundingDINO module should be configured
        # the same way as your existing thirdparty/GroundingDINO. If not, provide env vars or defaults.
        image_np = _load_image_to_numpy(image)

        # Run GroundingDINO to get boxes per text prompt
        # Minimal inference util; expects Grounder to offer a high-level API.
        # If your local GroundingDINO differs, adapt here.
        grounder = Grounder()
        boxes, labels, scores = grounder.predict(image_np, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)

        # If no boxes, return empty result
        if boxes is None or len(boxes) == 0:
            return {"success": True, "num_instances": 0, "mask_paths": []}

        # Load SAM (choose a default variant via env or fall back to vit_h)
        sam_variant = os.environ.get("SAM_MODEL_TYPE", "vit_h")
        sam_ckpt = os.environ.get("SAM_CHECKPOINT", "")
        if not sam_ckpt:
            return JSONResponse(status_code=501, content={
                "success": False,
                "error": "SAM checkpoint not configured. Set SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth and SAM_MODEL_TYPE (e.g., vit_h)."
            })

        sam = sam_model_registry[sam_variant](checkpoint=sam_ckpt)
        predictor = SamPredictor(sam)
        predictor.set_image(image_np)

        mask_paths: List[str] = []
        h, w = image_np.shape[:2]
        for i, box in enumerate(boxes):
            # SAM expects box in XYXY, image space
            box_xyxy = np.array(box, dtype=np.float32)
            masks, _, _ = predictor.predict(
                box=box_xyxy,
                multimask_output=False,
            )
            mask = (masks[0].astype(np.uint8) * 255)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"mask_{i}.png")
                Image.fromarray(mask).save(out_path)
                mask_paths.append(out_path)

        return {"success": True, "num_instances": len(mask_paths) if output_dir else len(boxes), "mask_paths": mask_paths}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GROUNDING_SAM_PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)



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
import traceback
import numpy as np
from PIL import Image
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

app = FastAPI(title="GroundingSAM API", description="Text-prompted segmentation (GroundingDINO + SAM)", version="1.0.0")

# Provide sensible defaults for env vars (can be overridden by user env)
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_PRETRAINED_DIR = "/home/suny0a/Proj/ImageBrush/NeoAnalogist/pretrained/GroundingSAM"
_DEFAULT_SAM = os.path.join(_DEFAULT_PRETRAINED_DIR, "sam_vit_h_4b8939.pth")
_DEFAULT_GDINO_CKPT = os.path.join(_DEFAULT_PRETRAINED_DIR, "groundingdino_swint_ogc.pth")
_DEFAULT_GDINO_CFG = os.path.join(
    _REPO_ROOT,
    "thirdparty",
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinT_OGC.py",
)

# Defaults; can be overridden by environment.
os.environ.setdefault("SAM_MODEL_TYPE", "vit_h")
os.environ.setdefault("SAM_CHECKPOINT", _DEFAULT_SAM)
os.environ.setdefault("GROUNDING_DINO_CHECKPOINT", _DEFAULT_GDINO_CKPT)
os.environ.setdefault("GROUNDING_DINO_CONFIG", _DEFAULT_GDINO_CFG)

def _validate_env() -> Optional[str]:
    """Validate required weights/config paths; return error message if invalid."""
    missing: list[str] = []
    sam_ckpt_env = os.environ.get("SAM_CHECKPOINT", "")
    gd_ckpt_env = os.environ.get("GROUNDING_DINO_CHECKPOINT", "")
    gd_cfg_env = os.environ.get("GROUNDING_DINO_CONFIG", "")
    if not sam_ckpt_env or not os.path.exists(sam_ckpt_env):
        missing.append(f"SAM_CHECKPOINT={sam_ckpt_env or '<unset>'}")
    if not gd_ckpt_env or not os.path.exists(gd_ckpt_env):
        missing.append(f"GROUNDING_DINO_CHECKPOINT={gd_ckpt_env or '<unset>'}")
    if not gd_cfg_env or not os.path.exists(gd_cfg_env):
        missing.append(f"GROUNDING_DINO_CONFIG={gd_cfg_env or '<unset>'}")
    if missing:
        return (
            "Missing or invalid paths for required weights/config:\n  - "
            + "\n  - ".join(missing)
            + "\nPlease set these environment variables to existing files."
        )
    return None


def _load_image_to_numpy(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


@app.on_event("startup")
async def _warmup_models():
    # No eager model load; just validate env once and log.
    err = _validate_env()
    if err:
        print(f"[GroundingSAM] Startup error: {err}")


@app.post("/grounding-sam/segment")
async def grounding_sam_segment(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    output_dir: Optional[str] = Form(None),
):
    # Ensure environment ok
    err = _validate_env()
    if err is not None:
        return JSONResponse(status_code=501, content={"success": False, "error": err})

    try:
        # Build ontology from prompt (comma/space split)
        labels = [t.strip() for t in text_prompt.split(",") if t.strip()]
        if not labels:
            return JSONResponse(status_code=400, content={"success": False, "error": "text_prompt is empty"})
        ontology = CaptionOntology({label: label for label in labels})

        # Instantiate model per request (simple & robust)
        # autodistill-grounded-sam expects weights via environment variables;
        # do not pass grounded_dino_checkpoint/sam_checkpoint or device as kwargs.
        model = GroundedSAM(
            ontology=ontology,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Save upload to temp path for predict
        import tempfile
        image_np = _load_image_to_numpy(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            Image.fromarray(image_np).save(tmp.name)
            results = model.predict(tmp.name)

        # If SAM failed to produce masks, optionally fall back to box->rect masks
        masks_from_boxes = False
        if not getattr(results, "masks", None):
            xyxy = getattr(results, "xyxy", None)
            if xyxy is not None and len(xyxy) > 0 and os.environ.get("GSAM_BOX_AS_MASK", "1") == "1":
                H, W = int(image_np.shape[0]), int(image_np.shape[1])
                rect_masks: list[np.ndarray] = []
                for box in xyxy:
                    try:
                        x1, y1, x2, y2 = map(int, box)
                        x1 = max(0, min(W - 1, x1))
                        x2 = max(0, min(W - 1, x2))
                        y1 = max(0, min(H - 1, y1))
                        y2 = max(0, min(H - 1, y2))
                        if x2 > x1 and y2 > y1:
                            m = np.zeros((H, W), dtype=np.uint8)
                            m[y1:y2, x1:x2] = 1
                            rect_masks.append(m)
                    except Exception:
                        continue
                if rect_masks:
                    results.masks = rect_masks  # type: ignore[attr-defined]
                    masks_from_boxes = True

        # Save masks
        mask_paths: List[str] = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # autodistill results.masks -> list of numpy bool/uint8 (H,W)
            for i, m in enumerate(getattr(results, "masks", []) or []):
                arr = (m.astype("uint8") * 255)
                out_path = os.path.join(output_dir, f"mask_{i}.png")
                Image.fromarray(arr).save(out_path)
                mask_paths.append(out_path)

        num = len(getattr(results, "masks", []) or [])

        # Optional debug payload to understand zero-detection cases
        payload = {"success": True, "num_instances": num, "mask_paths": mask_paths}
        if os.environ.get("GSAM_DEBUG") == "1":
            try:
                debug: dict = {}
                debug["available_attrs"] = sorted([a for a in dir(results) if not a.startswith("_")])
                # Common autodistill fields
                for key in ("boxes", "xyxy", "class_id", "confidence", "scores", "labels", "text"):  # best-effort
                    val = getattr(results, key, None)
                    if val is not None:
                        # convert to lightweight lists if possible
                        try:
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            elif isinstance(val, (list, tuple)):
                                val = list(val)
                        except Exception:  # noqa: BLE001
                            pass
                        debug[key] = val
                debug["masks_from_boxes"] = masks_from_boxes
                payload["debug"] = debug
            except Exception as _:
                pass

        return payload
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        content = {"success": False, "error": str(e)}
        if os.environ.get("GSAM_DEBUG") == "1":
            content["traceback"] = tb
        return JSONResponse(status_code=500, content=content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GROUNDING_SAM_PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)



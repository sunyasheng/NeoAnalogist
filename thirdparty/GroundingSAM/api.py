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

# Global caches
_GROUNDER = None
_SAM_PREDICTOR = None
_STARTUP_ERROR: Optional[str] = None

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

os.environ.setdefault("SAM_MODEL_TYPE", "vit_h")
os.environ.setdefault("SAM_CHECKPOINT", _DEFAULT_SAM)
os.environ.setdefault("GROUNDING_DINO_CHECKPOINT", _DEFAULT_GDINO_CKPT)
os.environ.setdefault("GROUNDING_DINO_CONFIG", _DEFAULT_GDINO_CFG)


def _ensure_thirdparty_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    gdino_dir = os.path.join(repo_root, "thirdparty", "GroundingDINO")
    if os.path.isdir(gdino_dir) and gdino_dir not in sys.path:
        sys.path.append(gdino_dir)


def _try_import_modules():
    """Import GroundingDINO + SAM modules. Returns (Grounder, sam_model_registry, SamPredictor) or (None, err)."""
    _ensure_thirdparty_on_path()
    try:
        from groundingdino.util.inference import Model as Grounder  # type: ignore
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        return (Grounder, sam_model_registry, SamPredictor), None
    except Exception as e:  # noqa: BLE001
        return None, (
            "GroundingDINO/SAM not available. Install and ensure weights are present. "
            "Hint: pip install groundingdino segment-anything; or add your GroundingDINO to PYTHONPATH. "
            f"Import error: {e}"
        )


def _load_once():
    global _GROUNDER, _SAM_PREDICTOR, _STARTUP_ERROR
    if _GROUNDER is not None and _SAM_PREDICTOR is not None:
        return
    # Validate required files first for clearer errors
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
        _STARTUP_ERROR = (
            "Missing or invalid paths for required weights/config:\n  - "
            + "\n  - ".join(missing)
            + "\nPlease set these environment variables to existing files."
        )
        print(f"[GroundingSAM] Startup error: {_STARTUP_ERROR}")
        return

    modules, err = _try_import_modules()
    if err is not None:
        _STARTUP_ERROR = err
        print(f"[GroundingSAM] Import error: {_STARTUP_ERROR}")
        return
    Grounder, sam_model_registry, SamPredictor = modules  # type: ignore[misc]

    try:
        # Instantiate GroundingDINO with explicit cfg/ckpt from env
        device = "cuda" if (os.environ.get("FORCE_CUDA", "1") != "0" and \
                            os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1") else "cpu"
        grounder = Grounder(
            model_config_path=os.environ.get("GROUNDING_DINO_CONFIG"),
            model_checkpoint_path=os.environ.get("GROUNDING_DINO_CHECKPOINT"),
            device=device,
        )
    except Exception as e:  # noqa: BLE001
        _STARTUP_ERROR = f"Failed to initialize GroundingDINO: {e}"
        print(f"[GroundingSAM] {_STARTUP_ERROR}")
        return

    # SAM settings via env
    sam_variant = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    sam_ckpt = os.environ.get("SAM_CHECKPOINT", "")
    if not sam_ckpt:
        _STARTUP_ERROR = (
            "SAM checkpoint not configured. Set SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth "
            "and optionally SAM_MODEL_TYPE (e.g., vit_h)."
        )
        print(f"[GroundingSAM] {_STARTUP_ERROR}")
        return

    try:
        sam = sam_model_registry[sam_variant](checkpoint=sam_ckpt)  # type: ignore[index]
        predictor = SamPredictor(sam)
    except Exception as e:  # noqa: BLE001
        _STARTUP_ERROR = f"Failed to initialize SAM ({sam_variant}): {e}"
        print(f"[GroundingSAM] {_STARTUP_ERROR}")
        return

    _GROUNDER = grounder
    _SAM_PREDICTOR = predictor
    print("[GroundingSAM] Models loaded successfully.")


def _load_image_to_numpy(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


@app.on_event("startup")
async def _warmup_models():
    _load_once()


@app.post("/grounding-sam/segment")
async def grounding_sam_segment(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    output_dir: Optional[str] = Form(None),
):
    # Ensure models are loaded once
    if _STARTUP_ERROR is not None:
        return JSONResponse(status_code=501, content={"success": False, "error": _STARTUP_ERROR})
    if _GROUNDER is None or _SAM_PREDICTOR is None:
        _load_once()
    if _STARTUP_ERROR is not None:
        return JSONResponse(status_code=501, content={"success": False, "error": _STARTUP_ERROR})

    try:
        # Inference
        image_np = _load_image_to_numpy(image)
        boxes, labels, scores = _GROUNDER.predict(  # type: ignore[assignment]
            image_np,
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # If no boxes, return empty result
        if boxes is None or len(boxes) == 0:
            return {"success": True, "num_instances": 0, "mask_paths": []}

        predictor = _SAM_PREDICTOR
        predictor.set_image(image_np)  # type: ignore[union-attr]

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



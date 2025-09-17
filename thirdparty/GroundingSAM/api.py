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
import torch
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

app = FastAPI(title="GroundingSAM API", description="Text-prompted segmentation (GroundingDINO + SAM)", version="1.0.0")

# Global caches
_STARTUP_ERROR: Optional[str] = None
_MODEL: Optional[GroundedSAM] = None
_MODEL_LABELS: set[str] = set()

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
    """Validate imports needed for autodistill-grounded-sam."""
    try:
        _ = GroundedSAM  # noqa: F841
        _ = CaptionOntology  # noqa: F841
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"autodistill-grounded-sam not available: {e}"


def _load_once():
    global _STARTUP_ERROR, _MODEL, _MODEL_LABELS
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

    ok, err = _try_import_modules()
    if not ok:
        _STARTUP_ERROR = err
        print(f"[GroundingSAM] Import error: {_STARTUP_ERROR}")
        return

    # Build a base model at startup to load checkpoints into memory
    default_labels_env = os.environ.get(
        "GSAM_DEFAULT_LABELS",
        "person,dog,cat,car,bus,backpack,bottle,chair,table,tv,monitor,phone,laptop"
    )
    labels = [t.strip() for t in default_labels_env.split(",") if t.strip()]
    if not labels:
        labels = ["object"]
    ontology = CaptionOntology({label: label for label in labels})

    try:
        _MODEL = GroundedSAM(
            ontology=ontology,
            box_threshold=float(os.environ.get("GSAM_BOX_THRESHOLD", "0.3")),
            text_threshold=float(os.environ.get("GSAM_TEXT_THRESHOLD", "0.25")),
        )
        _MODEL_LABELS = set(labels)
        print(f"[GroundingSAM] Base model loaded on startup. labels={sorted(_MODEL_LABELS)}")
    except Exception as e:  # noqa: BLE001
        _STARTUP_ERROR = f"Failed to initialize GroundedSAM at startup: {e}"
        print(f"[GroundingSAM] {_STARTUP_ERROR}")
        return


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
    # Ensure environment ok
    if _STARTUP_ERROR is not None:
        return JSONResponse(status_code=501, content={"success": False, "error": _STARTUP_ERROR})
    _load_once()
    if _STARTUP_ERROR is not None:
        return JSONResponse(status_code=501, content={"success": False, "error": _STARTUP_ERROR})

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

        # Save masks
        mask_paths: List[str] = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, "vis.jpg")
            results.save(vis_path)
            # autodistill results.masks -> list of numpy bool/uint8 (H,W)
            for i, m in enumerate(getattr(results, "masks", []) or []):
                arr = (m.astype("uint8") * 255)
                out_path = os.path.join(output_dir, f"mask_{i}.png")
                Image.fromarray(arr).save(out_path)
                mask_paths.append(out_path)

        num = len(getattr(results, "masks", []) or [])
        return {"success": True, "num_instances": num, "mask_paths": mask_paths}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GROUNDING_SAM_PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)



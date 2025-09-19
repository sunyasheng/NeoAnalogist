"""
GroundingSAM API: Text-prompted segmentation via GroundingDINO + SAM.

POST /grounding-sam/segment
  - image: UploadFile (required)
  - text_prompt: str (required, e.g., "cat, dog, person")
  - output_dir: str (optional; if set, saves masks as PNGs)
  - return_type: str (optional, default "image"; "image" | "json")

Returns:
  - return_type="image" (default): image/png stream of first mask
  - return_type="json": application/json with fields:
    - success: bool
    - error: str (if failed)
    - num_instances: int
    - mask_paths: list[str] (saved to temp cache)

Note:
  - This service uses GroundedSAM2 (Florence2 + SAM2). Ensure dependencies are installed in the environment.
  - Matches GoT API pattern: streams first result by default, JSON with paths when requested.
"""

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List
import io
import os
import sys
import traceback
import uuid
import numpy as np
from PIL import Image
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

app = FastAPI(title="GroundingSAM API", description="Text-prompted segmentation (GSAM2: Florence2 + SAM2)", version="1.0.0")

_HERE = os.path.abspath(os.path.dirname(__file__))


def _load_image_to_numpy(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


@app.on_event("startup")
async def _warmup_models():
    # No eager model load.
    pass


@app.post("/grounding-sam/segment")
async def grounding_sam_segment(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    output_dir: Optional[str] = Form(None),
    return_type: str = Form("image"),  # image (default) | json
):
    # Create temp cache dir for this request
    temp_cache_dir = os.path.join("./tmp/gsam_cache", f"task_{uuid.uuid4()}")
    os.makedirs(temp_cache_dir, exist_ok=True)
    
    try:
        # Build ontology from prompt - support both simple labels and complex descriptions
        # First try comma separation, then fall back to treating as single complex description
        if "," in text_prompt:
            # Multiple simple labels: "cat, dog, person"
            labels = [t.strip() for t in text_prompt.split(",") if t.strip()]
        else:
            # Single complex description: "reflection of cat in mirror"
            # Ensure the text ends with a period for better GroundingDINO performance
            clean_prompt = text_prompt.strip()
            if not clean_prompt.endswith("."):
                clean_prompt += "."
            labels = [clean_prompt]
        
        if not labels or not labels[0]:
            return JSONResponse(status_code=400, content={"success": False, "error": "text_prompt is empty"})
        
        # Create ontology mapping each label to itself
        ontology = CaptionOntology({label: label for label in labels})

        # Instantiate GSAM2 model per request
        model = GroundedSAM2(
            ontology=ontology,
        )

        # Save upload to temp path for predict
        import tempfile
        image_np = _load_image_to_numpy(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            Image.fromarray(image_np).save(tmp.name)
            results = model.predict(tmp.name)

        # Normalize masks from results
        masks: List[np.ndarray] = []
        if hasattr(results, "masks") and results.masks is not None:
            # autodistill style: list/array of (H, W) boolean/uint8 masks
            rm = results.masks
            if isinstance(rm, list):
                masks = rm
            else:
                # numpy array: (N, H, W) or (H, W)
                arr = np.asarray(rm)
                if arr.ndim == 3:
                    masks = [arr[i] for i in range(arr.shape[0])]
                elif arr.ndim == 2:
                    masks = [arr]
        elif hasattr(results, "mask") and results.mask is not None:
            # some versions expose singular 'mask'
            arr = np.asarray(results.mask)
            if arr.ndim == 3:
                masks = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 2:
                masks = [arr]
        num = len(masks)

        # Default: stream first mask
        if return_type == "image":
            if not masks:
                return JSONResponse(status_code=404, content={"success": False, "error": "No masks found"})
            buf = io.BytesIO()
            first_mask = (masks[0].astype("uint8") * 255)
            Image.fromarray(first_mask).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        # JSON: save all masks to temp cache and return paths
        cache_mask_paths: List[str] = []
        for i, m in enumerate(masks):
            arr = (m.astype("uint8") * 255)
            out_path = os.path.join(temp_cache_dir, f"mask_{i}.png")
            Image.fromarray(arr).save(out_path)
            cache_mask_paths.append(out_path)

        # Also save to user-provided output_dir if specified
        output_mask_paths: List[str] = []
        if output_dir:
            # Sanitize and prefer a local outputs root for safety
            safe_dir = os.path.normpath(os.path.expanduser(output_dir))
            if os.path.isabs(safe_dir):
                # Place absolute requests under a local outputs root instead
                safe_dir = os.path.join("./outputs/grounding_sam", safe_dir.lstrip(os.sep))
            os.makedirs(safe_dir, exist_ok=True)
            for i, m in enumerate(masks):
                arr = (m.astype("uint8") * 255)
                out_path = os.path.join(safe_dir, f"mask_{i}.png")
                Image.fromarray(arr).save(out_path)
                output_mask_paths.append(out_path)

        # Prefer returning user-specified output paths if provided; otherwise return temp cache paths
        response_paths = output_mask_paths if output_mask_paths else cache_mask_paths
        return {"success": True, "num_instances": num, "mask_paths": response_paths}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GROUNDING_SAM_PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)



#!/usr/bin/env python3
"""
Inpaint-Anything FastAPI Service

Provides REST API endpoints for image inpainting operations:
- Remove objects using point coordinates or masks
- Fill objects with text prompts
- Replace object backgrounds

Based on Inpaint-Anything: https://github.com/geekyutao/Inpaint-Anything
"""

import os
import sys
import io
import uuid
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import Inpaint-Anything modules
from sam_segment import predict_masks_with_sam, build_sam_model
from lama_inpaint import inpaint_img_with_lama, build_lama_model
from utils import load_img_to_array, save_array_to_img, dilate_mask

app = FastAPI(title="Inpaint-Anything API", version="1.0.0")

# Environment variables with defaults
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "/home/suny0a/Proj/ImageBrush/NeoAnalogist/pretrained/InpaintAnything/sam_vit_h_4b8939.pth")
LAMA_CONFIG = os.environ.get("LAMA_CONFIG", "./lama/configs/prediction/default.yaml")
LAMA_CHECKPOINT = os.environ.get("LAMA_CHECKPOINT", "/home/suny0a/Proj/ImageBrush/NeoAnalogist/pretrained/InpaintAnything/big-lama")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Debug mode
INPAINT_DEBUG = os.environ.get("INPAINT_DEBUG", "0") == "1"

print(f"[Inpaint-Anything] Starting with device: {DEVICE}")
print(f"[Inpaint-Anything] SAM model: {SAM_MODEL_TYPE} at {SAM_CHECKPOINT}")
print(f"[Inpaint-Anything] LaMa config: {LAMA_CONFIG}")
print(f"[Inpaint-Anything] LaMa checkpoint: {LAMA_CHECKPOINT}")

# Global model cache
sam_model = None
lama_model = None

def load_sam_model():
    """Load SAM model once at startup."""
    global sam_model
    if sam_model is None:
        print(f"[Inpaint-Anything] Loading SAM model...")
        try:
            sam_model = build_sam_model(
                model_type=SAM_MODEL_TYPE,
                ckpt_p=SAM_CHECKPOINT,
                device=DEVICE,
            )
            print(f"[Inpaint-Anything] SAM model loaded successfully")
        except Exception as e:
            print(f"[Inpaint-Anything] Failed to load SAM model: {e}")
            sam_model = None
    return sam_model

def load_lama_model():
    """Load LaMa model once at startup."""
    global lama_model
    if lama_model is None:
        print(f"[Inpaint-Anything] Loading LaMa model...")
        try:
            lama_model = build_lama_model(LAMA_CONFIG, LAMA_CHECKPOINT, device=DEVICE)
            print(f"[Inpaint-Anything] LaMa model loaded successfully")
        except Exception as e:
            print(f"[Inpaint-Anything] Failed to load LaMa model: {e}")
            lama_model = None
    return lama_model

@app.on_event("startup")
async def startup_event():
    """Load models at startup."""
    print("[Inpaint-Anything] Loading models at startup...")
    load_sam_model()
    load_lama_model()
    print("[Inpaint-Anything] Startup complete!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Inpaint-Anything API is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/inpaint/remove")
async def remove_object(
    image: UploadFile = File(..., description="Input image"),
    point_coords: Optional[str] = Form(None, description="Point coordinates as 'x,y' (e.g., '200,300')"),
    point_labels: str = Form("1", description="Point labels (1 for foreground, 0 for background)"),
    mask: Optional[UploadFile] = File(None, description="Optional mask image (if not provided, will use SAM to generate from point)"),
    dilate_kernel_size: int = Form(10, description="Dilate kernel size for mask expansion"),
    output_dir: Optional[str] = Form(None, description="Output directory for saving results"),
    return_type: str = Form("image", description="Return type: 'image' (stream first result) or 'json' (return paths)")
):
    """
    Remove objects from image using point coordinates or mask.
    
    Either point_coords or mask must be provided.
    """
    try:
        # Create temp cache dir for this request
        temp_cache_dir = os.path.join("./tmp/inpaint_cache", f"task_{uuid.uuid4()}")
        os.makedirs(temp_cache_dir, exist_ok=True)
        
        # Save uploaded image
        input_image_path = os.path.join(temp_cache_dir, "input.png")
        with open(input_image_path, "wb") as f:
            f.write(await image.read())
        
        # Load image
        img = load_img_to_array(input_image_path)
        if INPAINT_DEBUG:
            print(f"[DEBUG] Loaded image shape: {img.shape}")
        
        # Process mask
        masks = None
        if mask is not None:
            # Use provided mask
            mask_path = os.path.join(temp_cache_dir, "input_mask.png")
            with open(mask_path, "wb") as f:
                f.write(await mask.read())
            mask_array = load_img_to_array(mask_path)
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]  # Convert to grayscale
            masks = [mask_array.astype(np.uint8)]
            if INPAINT_DEBUG:
                print(f"[DEBUG] Using provided mask, shape: {masks[0].shape}")
        elif point_coords is not None:
            # Generate mask from point coordinates
            try:
                coords = [float(x.strip()) for x in point_coords.split(',')]
                if len(coords) != 2:
                    raise ValueError("point_coords must be in format 'x,y'")
                point_coords_list = [coords]
                point_labels_list = [int(point_labels)]
                
                if INPAINT_DEBUG:
                    print(f"[DEBUG] Using point coordinates: {point_coords_list}, labels: {point_labels_list}")
                
                # Use preloaded SAM model
                sam_predictor = load_sam_model()
                if sam_predictor is None:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": "SAM model not loaded"}
                    )
                
                # Use the preloaded predictor
                sam_predictor.set_image(img)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=np.array(point_coords_list),
                    point_labels=np.array(point_labels_list),
                    multimask_output=True,
                )
                if INPAINT_DEBUG:
                    print(f"[DEBUG] Generated {len(masks)} masks from SAM")
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": f"Invalid point_coords format: {str(e)}"}
                )
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Either point_coords or mask must be provided"}
            )
        
        if masks is None or len(masks) == 0:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "No masks found"}
            )
        
        # Dilate masks to avoid edge effects
        if dilate_kernel_size > 0:
            if INPAINT_DEBUG:
                print(f"[DEBUG] Dilating {len(masks)} masks with kernel size {dilate_kernel_size}")
            masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
        
        # Inpaint each mask
        inpainted_images = []
        mask_paths = []
        result_paths = []
        
        target_output_dir = output_dir if output_dir else temp_cache_dir
        os.makedirs(target_output_dir, exist_ok=True)
        
        for i, mask in enumerate(masks):
            # Save mask
            mask_path = os.path.join(target_output_dir, f"mask_{i}.png")
            save_array_to_img(mask, mask_path)
            mask_paths.append(mask_path)
            
            # Inpaint using preloaded LaMa model
            try:
                lama_model = load_lama_model()
                if lama_model is None:
                    print(f"[ERROR] LaMa model not loaded, skipping mask {i}")
                    continue
                
                if INPAINT_DEBUG:
                    print(f"[DEBUG] Inpainting mask {i}, shape: {mask.shape}, dtype: {mask.dtype}")
                    print(f"[DEBUG] Image shape: {img.shape}, dtype: {img.dtype}")
                
                # Use the preloaded model
                from lama_inpaint import inpaint_img_with_builded_lama
                inpainted_img = inpaint_img_with_builded_lama(
                    lama_model, img, mask, device=DEVICE
                )
                inpainted_images.append(inpainted_img)
                
                # Save result
                result_path = os.path.join(target_output_dir, f"inpainted_{i}.png")
                save_array_to_img(inpainted_img, result_path)
                result_paths.append(result_path)
                
                if INPAINT_DEBUG:
                    print(f"[DEBUG] Inpainted mask {i}, saved to {result_path}")
            except Exception as e:
                print(f"[ERROR] Failed to inpaint mask {i}: {str(e)}")
                continue
        
        if len(inpainted_images) == 0:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to inpaint any masks"}
            )
        
        # Return response
        if return_type == "image":
            # Stream first inpainted image
            buf = io.BytesIO()
            Image.fromarray(inpainted_images[0]).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        else:
            # Return JSON with paths
            return {
                "success": True,
                "num_masks": len(masks),
                "mask_paths": mask_paths,
                "result_paths": result_paths,
                "message": f"Successfully inpainted {len(inpainted_images)} masks"
            }
            
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Remove object failed: {str(e)}")
        if INPAINT_DEBUG:
            print(f"[DEBUG] Traceback: {tb}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/inpaint/fill")
async def fill_object(
    image: UploadFile = File(..., description="Input image"),
    text_prompt: str = Form(..., description="Text prompt for filling"),
    point_coords: Optional[str] = Form(None, description="Point coordinates as 'x,y'"),
    point_labels: str = Form("1", description="Point labels"),
    mask: Optional[UploadFile] = File(None, description="Optional mask image"),
    dilate_kernel_size: int = Form(10, description="Dilate kernel size"),
    output_dir: Optional[str] = Form(None, description="Output directory"),
    return_type: str = Form("image", description="Return type: 'image' or 'json'")
):
    """
    Fill objects with text-guided content using Stable Diffusion.
    """
    # TODO: Implement fill functionality with Stable Diffusion
    return JSONResponse(
        status_code=501,
        content={"success": False, "error": "Fill functionality not yet implemented"}
    )


@app.post("/inpaint/replace")
async def replace_object(
    image: UploadFile = File(..., description="Input image"),
    text_prompt: str = Form(..., description="Text prompt for replacement"),
    point_coords: Optional[str] = Form(None, description="Point coordinates as 'x,y'"),
    point_labels: str = Form("1", description="Point labels"),
    mask: Optional[UploadFile] = File(None, description="Optional mask image"),
    dilate_kernel_size: int = Form(10, description="Dilate kernel size"),
    output_dir: Optional[str] = Form(None, description="Output directory"),
    return_type: str = Form("image", description="Return type: 'image' or 'json'")
):
    """
    Replace object backgrounds with text-guided content.
    """
    # TODO: Implement replace functionality with Stable Diffusion
    return JSONResponse(
        status_code=501,
        content={"success": False, "error": "Replace functionality not yet implemented"}
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inpaint-Anything FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8601, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"[Inpaint-Anything] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


#### 1st Mode
#### curl -X POST http://localhost:8601/inpaint/remove   -F "image=@anydoor_result.png"   -F "point_coords=260,300"   -F "point_labels=1"   -F "dilate_kernel_size=5"   -F "return_type=json"   -F "output_dir=/home/suny0a/Proj/ImageBrush/out_dir"

#### 2nd Mode, the kernel can be set to different values
#### curl -X POST http://localhost:8601/inpaint/remove   -F "image=@anydoor_result.png"   -F "mask=@/home/suny0a/Proj/ImageBrush/out_dir/mask_2.png"   -F "dilate_kernel_size=0"   -F "return_type=image"   --output /home/suny0a/Proj/ImageBrush/inpainted_final.png

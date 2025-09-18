#!/usr/bin/env python3
"""
LAMA Inpainting API

A simplified API that uses LAMA (LaMa) for image inpainting.
This API provides a clean interface for LAMA-based inpainting operations.
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

# Add Inpaint-Anything directory to path
inpaint_anything_dir = Path(__file__).parent.parent / "Inpaint-Anything"
sys.path.insert(0, str(inpaint_anything_dir))

# Import LAMA functions from Inpaint-Anything
from lama_inpaint import inpaint_img_with_lama, build_lama_model
from utils import load_img_to_array, save_array_to_img, dilate_mask

app = FastAPI(title="LAMA Inpainting API", version="1.0.0")

# Environment variables with defaults
LAMA_CONFIG = os.environ.get("LAMA_CONFIG", str(inpaint_anything_dir / "lama/configs/prediction/default.yaml"))
LAMA_CHECKPOINT = os.environ.get("LAMA_CHECKPOINT", "/home/suny0a/Proj/ImageBrush/NeoAnalogist/pretrained/InpaintAnything/big-lama")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Debug mode
LAMA_DEBUG = os.environ.get("LAMA_DEBUG", "0") == "1"

print(f"[LAMA] Starting with device: {DEVICE}")
print(f"[LAMA] LaMa config: {LAMA_CONFIG}")
print(f"[LAMA] LaMa checkpoint: {LAMA_CHECKPOINT}")

# Global model cache
lama_model = None

def load_lama_model():
    """Load LAMA model once at startup."""
    global lama_model
    if lama_model is None:
        print(f"[LAMA] Loading LaMa model...")
        try:
            lama_model = build_lama_model(LAMA_CONFIG, LAMA_CHECKPOINT, device=DEVICE)
            print(f"[LAMA] LaMa model loaded successfully")
        except Exception as e:
            print(f"[LAMA] Failed to load LaMa model: {e}")
            lama_model = None
    return lama_model

@app.on_event("startup")
async def startup_event():
    """Load models at startup."""
    print("[LAMA] Loading model at startup...")
    load_lama_model()
    print("[LAMA] Startup complete!")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "LAMA Inpainting API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": lama_model is not None,
        "device": DEVICE,
        "model_name": "LaMa (Large Mask Inpainting)"
    }

@app.post("/inpaint")
async def inpaint_with_lama(
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(..., description="Mask image (white=inpaint region)"),
    dilate_kernel_size: int = Form(0, description="Dilate kernel size for mask expansion"),
    output_dir: Optional[str] = Form(None, description="Output directory for saving results"),
    return_type: str = Form("image", description="Return type: 'image' (stream result) or 'json' (return paths)")
):
    """
    Inpaint image using LAMA (LaMa) model.
    
    This endpoint performs high-quality inpainting using the LaMa model.
    The mask should be a binary image where white areas indicate regions to inpaint.
    """
    try:
        # Create temp cache dir for this request
        temp_cache_dir = os.path.join("./tmp/lama_cache", f"task_{uuid.uuid4()}")
        os.makedirs(temp_cache_dir, exist_ok=True)
        
        # Save uploaded files
        input_image_path = os.path.join(temp_cache_dir, "input.png")
        mask_path = os.path.join(temp_cache_dir, "mask.png")
        
        with open(input_image_path, "wb") as f:
            f.write(await image.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())
        
        # Load image and mask
        img = load_img_to_array(input_image_path)
        mask_array = load_img_to_array(mask_path)
        
        # Convert mask to grayscale if needed
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        if LAMA_DEBUG:
            print(f"[DEBUG] Loaded image shape: {img.shape}")
            print(f"[DEBUG] Loaded mask shape: {mask_array.shape}")
        
        # Dilate mask if requested
        if dilate_kernel_size > 0:
            if LAMA_DEBUG:
                print(f"[DEBUG] Dilating mask with kernel size {dilate_kernel_size}")
            mask_array = dilate_mask(mask_array, dilate_kernel_size)
        
        # Load LAMA model
        model = load_lama_model()
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "LAMA model not loaded"}
            )
        
        # Perform inpainting
        if LAMA_DEBUG:
            print(f"[DEBUG] Starting inpainting...")
        
        try:
            # Use the preloaded model
            from lama_inpaint import inpaint_img_with_builded_lama
            inpainted_img = inpaint_img_with_builded_lama(
                model, img, mask_array, device=DEVICE
            )
        except Exception as e:
            # Fallback to original function if preloaded version fails
            if LAMA_DEBUG:
                print(f"[DEBUG] Preloaded model failed, using original function: {e}")
            inpainted_img = inpaint_img_with_lama(
                img, mask_array, LAMA_CONFIG, LAMA_CHECKPOINT, device=DEVICE
            )
        
        if LAMA_DEBUG:
            print(f"[DEBUG] Inpainting completed, result shape: {inpainted_img.shape}")
        
        # Save results
        target_output_dir = output_dir if output_dir else temp_cache_dir
        os.makedirs(target_output_dir, exist_ok=True)
        
        result_path = os.path.join(target_output_dir, "inpainted.png")
        save_array_to_img(inpainted_img, result_path)
        
        # Return response
        if return_type == "image":
            # Stream inpainted image
            buf = io.BytesIO()
            Image.fromarray(inpainted_img).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        else:
            # Return JSON with paths
            return {
                "success": True,
                "result_path": result_path,
                "mask_path": mask_path,
                "message": "Successfully inpainted image with LAMA"
            }
            
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] LAMA inpainting failed: {str(e)}")
        if LAMA_DEBUG:
            print(f"[DEBUG] Traceback: {tb}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/inpaint/simple")
async def simple_inpaint(
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(..., description="Mask image (white=inpaint region)")
):
    """
    Simple inpainting endpoint - just inpaint and return the result image.
    
    This is a simplified version that always returns the inpainted image directly.
    """
    try:
        # Create temp files
        temp_dir = os.path.join("./tmp/lama_simple", f"task_{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)
        
        input_path = os.path.join(temp_dir, "input.png")
        mask_path = os.path.join(temp_dir, "mask.png")
        
        with open(input_path, "wb") as f:
            f.write(await image.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())
        
        # Load and process
        img = load_img_to_array(input_path)
        mask_array = load_img_to_array(mask_path)
        
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # Load model and inpaint
        model = load_lama_model()
        if model is None:
            raise HTTPException(status_code=500, detail="LAMA model not loaded")
        
        try:
            from lama_inpaint import inpaint_img_with_builded_lama
            result = inpaint_img_with_builded_lama(model, img, mask_array, device=DEVICE)
        except:
            result = inpaint_img_with_lama(img, mask_array, LAMA_CONFIG, LAMA_CHECKPOINT, device=DEVICE)
        
        # Return image
        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LAMA Inpainting FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8602, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"[LAMA] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

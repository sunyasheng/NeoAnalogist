#!/usr/bin/env python3
"""
SDXL Inpainting FastAPI Service

Provides REST API endpoints for high-quality image inpainting using Stable Diffusion XL.
Based on diffusers AutoPipelineForInpainting with SDXL 1.0 inpainting model.
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

# Import diffusers
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# Import preprocessing utilities
import cv2

app = FastAPI(title="SDXL Inpainting API", version="1.0.0")

# Environment variables with defaults
MODEL_NAME = os.environ.get("SDXL_MODEL_NAME", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "float16" if DEVICE == "cuda" else "float32")
VARIANT = os.environ.get("VARIANT", "fp16" if TORCH_DTYPE == "float16" else None)

# Default generation parameters
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "8.0"))
DEFAULT_NUM_INFERENCE_STEPS = int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "20"))
DEFAULT_STRENGTH = float(os.environ.get("DEFAULT_STRENGTH", "0.99"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "1024"))

# Debug mode
SDXL_DEBUG = os.environ.get("SDXL_DEBUG", "0") == "1"

print(f"[SDXL Inpainting] Starting with device: {DEVICE}")
print(f"[SDXL Inpainting] Model: {MODEL_NAME}")
print(f"[SDXL Inpainting] Torch dtype: {TORCH_DTYPE}")
print(f"[SDXL Inpainting] Variant: {VARIANT}")

# Global model cache
pipe = None

def crop_for_filling_pre(image: np.ndarray, mask: np.ndarray, crop_size: int = 1024):
    """
    Preprocessing for inpainting: crop image and mask to focus on the masked region.
    Optimized for SDXL (1024x1024) instead of SD2 (512x512).
    """
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than crop_size, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Update the height and width of the resized image
    height, width = image.shape[:2]

    # If the crop_size square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

    # Crop the image
    cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    cropped_mask = mask[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return cropped_image, cropped_mask

def crop_for_filling_post(
    image: np.ndarray,
    mask: np.ndarray,
    filled_image: np.ndarray, 
    crop_size: int = 1024,
):
    """
    Postprocessing for inpainting: merge the inpainted result back to original image.
    Optimized for SDXL (1024x1024) instead of SD2 (512x512).
    """
    image_copy = image.copy()
    mask_copy = mask.copy()
    # Calculate the aspect ratio of the image
    height, width = image.shape[:2]
    height_ori, width_ori = height, width
    aspect_ratio = float(width) / float(height)

    # If the shorter side is less than crop_size, resize the image proportionally
    if min(height, width) < crop_size:
        if height < width:
            new_height = crop_size
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = crop_size
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))

    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Update the height and width of the resized image
    height, width = image.shape[:2]

    # If the crop_size square cannot cover the entire mask, resize the image accordingly
    if w > crop_size or h > crop_size:
        flag_padding = True
        # padding to square at first
        if height < width:
            padding = width - height
            image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
            mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
            padding_side = 'h'
        else:
            padding = height - width
            image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')
            padding_side = 'w'

        resize_factor = crop_size / max(w, h)
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
        x, y, w, h = cv2.boundingRect(mask)
    else:
        flag_padding = False

    # Calculate the crop coordinates
    crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
    crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

    # Fill the image
    image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = filled_image
    if flag_padding:
        image = cv2.resize(image, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
        if padding_side == 'h':
            image = image[padding // 2:padding // 2 + height_ori, :]
        else:
            image = image[:, padding // 2:padding // 2 + width_ori]

    image = cv2.resize(image, (width_ori, height_ori))

    image_copy[mask_copy==255] = image[mask_copy==255]
    return image_copy

def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 1024):
    """
    Resizes an image and its corresponding mask to have the longer side equal to target_size 
    and pads them to make them both have the same size. Optimized for SDXL.
    """
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    mask_padded = np.pad(mask_resized, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    return image_padded, mask_padded, (top_pad, bottom_pad, left_pad, right_pad)

def recover_size(image_padded: np.ndarray, mask_padded: np.ndarray, orig_size, padding_factors):
    """
    Resizes a padded and resized image and mask to the original size.
    """
    h, w, c = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors
    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    mask = mask_padded[top_pad:h-bottom_pad, left_pad:w-right_pad]
    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    return image_resized, mask_resized

def load_sdxl_model():
    """Load SDXL inpainting model once at startup."""
    global pipe
    if pipe is None:
        print(f"[SDXL Inpainting] Loading SDXL inpainting model...")
        try:
            torch_dtype = torch.float16 if TORCH_DTYPE == "float16" else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch_dtype, 
                variant=VARIANT
            ).to(DEVICE)
            print(f"[SDXL Inpainting] SDXL model loaded successfully")
        except Exception as e:
            print(f"[SDXL Inpainting] Failed to load SDXL model: {e}")
            pipe = None
    return pipe

@app.on_event("startup")
async def startup_event():
    """Load models at startup."""
    print("[SDXL Inpainting] Loading models at startup...")
    load_sdxl_model()
    print("[SDXL Inpainting] Startup complete!")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "SDXL Inpainting API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/inpaint")
async def inpaint_image(
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(..., description="Mask image (white areas will be inpainted)"),
    prompt: str = Form(..., description="Text prompt for inpainting"),
    negative_prompt: Optional[str] = Form("", description="Negative prompt"),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE, description="Guidance scale (1.0-20.0)"),
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS, description="Number of inference steps (15-50)"),
    strength: float = Form(DEFAULT_STRENGTH, description="Strength (0.0-1.0, use < 1.0)"),
    height: int = Form(DEFAULT_HEIGHT, description="Output height"),
    width: int = Form(DEFAULT_WIDTH, description="Output width"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
    output_path: Optional[str] = Form(None, description="[Ignored] Output path is not used; response is streamed"),
    return_type: str = Form("image", description="[Ignored] Always streams PNG image")
):
    """
    Inpaint image using SDXL with text prompt.
    
    The mask should have white areas indicating regions to be inpainted.
    """
    try:
        # Read uploaded files directly from memory (no disk I/O)
        input_bytes = await image.read()
        mask_bytes = await mask.read()
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB").resize((width, height))
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((width, height))
        
        if SDXL_DEBUG:
            print(f"[DEBUG] Loaded image size: {input_image.size}")
            print(f"[DEBUG] Loaded mask size: {mask_image.size}")
            print(f"[DEBUG] Prompt: {prompt}")
            print(f"[DEBUG] Parameters: guidance_scale={guidance_scale}, steps={num_inference_steps}, strength={strength}")
        
        # Load model
        sdxl_pipe = load_sdxl_model()
        if sdxl_pipe is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "SDXL model not loaded"}
            )
        
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # Generate inpainted image
        if SDXL_DEBUG:
            print(f"[DEBUG] Starting inpainting generation...")
        
        result = sdxl_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=input_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        )
        
        inpainted_image = result.images[0]
        
        if SDXL_DEBUG:
            print(f"[DEBUG] Inpainting completed, result size: {inpainted_image.size}")
        
        # Always stream PNG image in response (no file writes)
        buf = io.BytesIO()
        inpainted_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
            
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] SDXL inpainting failed: {str(e)}")
        if SDXL_DEBUG:
            print(f"[DEBUG] Traceback: {tb}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/inpaint/remove")
async def remove_object(
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(..., description="Mask image (white areas will be removed/inpainted)"),
    prompt: str = Form("", description="Optional text prompt for background (empty for automatic background inpainting)"),
    guidance_scale: float = Form(7.5, description="Guidance scale for removal"),
    num_inference_steps: int = Form(25, description="Number of inference steps"),
    strength: float = Form(0.95, description="Strength for removal"),
    height: int = Form(DEFAULT_HEIGHT, description="Output height"),
    width: int = Form(DEFAULT_WIDTH, description="Output width"),
    seed: Optional[int] = Form(None, description="Random seed"),
    output_path: Optional[str] = Form(None, description="Output path"),
    return_type: str = Form("image", description="Return type: 'image' or 'json'")
):
    """
    Remove objects from image using SDXL inpainting.
    
    If no prompt is provided, will attempt to inpaint with background context.
    """
    # Use empty or generic prompt for object removal
    if not prompt.strip():
        prompt = "background, seamless, natural"
    
    return await inpaint_image(
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt="object, person, animal, text, watermark, artifacts",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        height=height,
        width=width,
        seed=seed,
        output_path=output_path,
        return_type=return_type
    )

@app.post("/inpaint/fill")
async def fill_object(
    image: UploadFile = File(..., description="Input image"),
    mask: UploadFile = File(..., description="Mask image (white areas will be filled)"),
    prompt: str = Form(..., description="Text prompt for filling"),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE, description="Guidance scale"),
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS, description="Number of inference steps"),
    strength: float = Form(DEFAULT_STRENGTH, description="Strength"),
    seed: Optional[int] = Form(None, description="Random seed"),
    output_path: Optional[str] = Form(None, description="[Ignored] Output path is not used; response is streamed"),
    return_type: str = Form("image", description="[Ignored] Always streams PNG image"),
    use_smart_crop: bool = Form(False, description="[Internal] Use smart cropping for better results")
):
    """
    Fill masked areas with text-guided content using smart cropping.
    
    This endpoint uses intelligent cropping to focus on the masked region,
    which often produces better results than full-image inpainting.
    """
    try:
        # Read uploaded files directly from memory (no disk I/O)
        input_bytes = await image.read()
        mask_bytes = await mask.read()
        input_image = np.array(Image.open(io.BytesIO(input_bytes)).convert("RGB"))
        mask_image = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
        
        # Convert mask to grayscale if needed
        if len(mask_image.shape) == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        
        if SDXL_DEBUG:
            print(f"[DEBUG] Loaded image shape: {input_image.shape}")
            print(f"[DEBUG] Loaded mask shape: {mask_image.shape}")
            print(f"[DEBUG] Prompt: {prompt}")
            print(f"[DEBUG] Use smart crop: {use_smart_crop}")
        
        # Load model
        sdxl_pipe = load_sdxl_model()
        if sdxl_pipe is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "SDXL model not loaded"}
            )
        
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        if use_smart_crop:
            # Use smart cropping for better results
            if SDXL_DEBUG:
                print(f"[DEBUG] Using smart cropping...")
            
            # Preprocess: crop to focus on masked region
            cropped_image, cropped_mask = crop_for_filling_pre(input_image, mask_image, crop_size=1024)
            
            if SDXL_DEBUG:
                print(f"[DEBUG] Cropped image shape: {cropped_image.shape}")
                print(f"[DEBUG] Cropped mask shape: {cropped_mask.shape}")
            
            # Convert to PIL for SDXL
            cropped_pil_image = Image.fromarray(cropped_image)
            cropped_pil_mask = Image.fromarray(cropped_mask)
            
            # Generate inpainted image on cropped region
            result = sdxl_pipe(
                prompt=prompt,
                image=cropped_pil_image,
                mask_image=cropped_pil_mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                generator=generator,
            )
            
            inpainted_cropped = np.array(result.images[0])
            
            # Postprocess: merge back to original image
            inpainted_image = crop_for_filling_post(input_image, mask_image, inpainted_cropped, crop_size=1024)
            inpainted_pil = Image.fromarray(inpainted_image)
        else:
            # Use full image inpainting
            if SDXL_DEBUG:
                print(f"[DEBUG] Using full image inpainting...")
            
            # Resize to SDXL resolution
            input_pil = Image.fromarray(input_image).resize((1024, 1024))
            mask_pil = Image.fromarray(mask_image).resize((1024, 1024))
            
            result = sdxl_pipe(
                prompt=prompt,
                image=input_pil,
                mask_image=mask_pil,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                generator=generator,
            )
            
            inpainted_pil = result.images[0]
            # Resize back to original size
            inpainted_pil = inpainted_pil.resize((input_image.shape[1], input_image.shape[0]))
        
        if SDXL_DEBUG:
            print(f"[DEBUG] Inpainting completed, result size: {inpainted_pil.size}")
        
        # Always stream PNG image in response (no file writes)
        buf = io.BytesIO()
        inpainted_pil.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
            
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] SDXL fill inpainting failed: {str(e)}")
        if SDXL_DEBUG:
            print(f"[DEBUG] Traceback: {tb}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SDXL Inpainting FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8603, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"[SDXL Inpainting] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

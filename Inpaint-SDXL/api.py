#!/usr/bin/env python3
"""
SDXL Inpainting API Server

A FastAPI server that provides SDXL inpainting functionality.
This server can be used with the SDXLInpaintClient in the NeoAnalogist system.

Usage:
    python api.py

The server will start on http://localhost:8402 by default.
"""

import os
import time
from typing import Optional
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting

app = FastAPI(title="SDXL Inpainting API", version="1.0.0")

# Global pipeline variable
pipe = None

def load_model():
    """Load the SDXL inpainting model."""
    global pipe
    if pipe is None:
        print("Loading SDXL inpainting model...")
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
                torch_dtype=torch.float16, 
                variant="fp16"
            ).to("cuda")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Fallback to CPU if CUDA fails
            pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
                torch_dtype=torch.float32
            ).to("cpu")
            print("Model loaded on CPU as fallback!")

@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "SDXL Inpainting API is running", "status": "healthy"}

@app.post("/sdxl/inpaint")
async def inpaint(
    image: bytes = File(..., description="Input image file"),
    mask: bytes = File(..., description="Binary mask image (white=inpaint region)"),
    prompt: str = Form(..., description="Text prompt for inpainting"),
    negative_prompt: str = Form("", description="Negative prompt"),
    guidance_scale: float = Form(8.0, description="Guidance scale"),
    num_inference_steps: int = Form(20, description="Number of inference steps"),
    strength: float = Form(0.99, description="Inpainting strength"),
    seed: Optional[int] = Form(None, description="Random seed"),
):
    """
    Perform SDXL inpainting on the provided image and mask.
    
    Args:
        image: Input image bytes
        mask: Binary mask image bytes (white areas will be inpainted)
        prompt: Text description of what to generate
        negative_prompt: What to avoid generating
        guidance_scale: How closely to follow the prompt (higher = more adherence)
        num_inference_steps: Number of denoising steps (15-30 recommended)
        strength: Inpainting strength (must be < 1.0)
        seed: Random seed for reproducible results
    
    Returns:
        PNG image bytes of the inpainted result
    """
    try:
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Load and process input image
        input_image = Image.open(BytesIO(image)).convert("RGB")
        mask_image = Image.open(BytesIO(mask)).convert("RGB")
        
        # Resize to 1024x1024 (SDXL standard)
        input_image = input_image.resize((1024, 1024))
        mask_image = mask_image.resize((1024, 1024))
        
        # Set up generator with seed if provided
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() and pipe.device.type == "cuda" else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Perform inpainting
        start_time = time.time()
        result = pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]
        
        execution_time = time.time() - start_time
        print(f"Inpainting completed in {execution_time:.2f} seconds")
        
        # Convert result to bytes
        output_buffer = BytesIO()
        result.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        return StreamingResponse(
            BytesIO(output_buffer.getvalue()),
            media_type="image/png",
            headers={
                "X-Execution-Time": str(execution_time),
                "X-Model": "stable-diffusion-xl-1.0-inpainting-0.1"
            }
        )
        
    except Exception as e:
        print(f"Error during inpainting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "model_name": "stable-diffusion-xl-1.0-inpainting-0.1",
        "device": "cuda" if torch.cuda.is_available() and pipe and pipe.device.type == "cuda" else "cpu",
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/sdxl/load_model")
async def load_model_endpoint():
    """Manually trigger model loading."""
    try:
        load_model()
        return {"message": "Model loaded successfully", "model_loaded": pipe is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SDXL Inpainting API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8402, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting SDXL Inpainting API server on {args.host}:{args.port}")
    print("Make sure you have CUDA available and the model will be downloaded on first run.")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

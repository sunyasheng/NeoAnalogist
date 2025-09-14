from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import shutil
import traceback
from PIL import Image
import io

from got_service import got_generate

app = FastAPI(
    title="GoT API",
    description="API for GoT generation and editing following videomme_videollm_llava pattern",
    version="1.0.0",
)

@app.on_event("startup")
async def _warmup_model():
    """Preload GoT model at startup to avoid first-request latency."""
    try:
        # Lazy import to avoid circular
        from got_service import _load_got_once
        _load_got_once()
        print("[GoT] Model preloaded at startup.")
    except Exception as e:
        # Do not crash server; just log
        print(f"[GoT] Preload failed: {e}")


class GoTRequest(BaseModel):
    prompt: str
    mode: str = "t2i"  # "t2i" or "edit"
    height: int = 1024
    width: int = 1024
    max_new_tokens: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.0
    cond_image_guidance_scale: float = 4.0


class GoTResponse(BaseModel):
    got_text: str
    images: List[str]


@app.post("/got/generate", response_model=GoTResponse)
async def got_generate_api(
    prompt: str = Form(...),
    mode: str = Form("t2i"),
    height: int = Form(1024),
    width: int = Form(1024),
    max_new_tokens: int = Form(1024),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    image_guidance_scale: float = Form(1.0),
    cond_image_guidance_scale: float = Form(4.0),
    image: Optional[UploadFile] = File(None),
    return_type: str = Form("image"),  # image (default) | json
    output_path: Optional[str] = Form(None)  # Custom output path
):
    # Use custom output path if provided, otherwise use default temp directory
    if output_path:
        # Extract directory from output_path
        temp_cache_dir = os.path.dirname(output_path)
        os.makedirs(temp_cache_dir, exist_ok=True)
    else:
        temp_cache_dir = os.path.join("./tmp/got_cache", f"task_{uuid.uuid4()}")
        os.makedirs(temp_cache_dir, exist_ok=True)
    
    try:
        # Handle image upload for edit mode
        pil_img = None
        if mode == "edit":
            if not image:
                raise ValueError("image file is required when mode == 'edit'")
            
            # Read image from upload
            image_data = await image.read()
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Save uploaded image to temp cache for got_service
            temp_image_path = os.path.join(temp_cache_dir, "input_image.jpg")
            pil_img.save(temp_image_path)
        else:
            temp_image_path = None
        
        result = got_generate(
            prompt=prompt,
            mode=mode,
            image_path=temp_image_path,
            height=height,
            width=width,
            max_new_tokens=max_new_tokens,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            cond_image_guidance_scale=cond_image_guidance_scale,
            cache_dir=temp_cache_dir,
        )

        # Stream or JSON based on return_type
        images = result.get("images", [])
        
        # If output_path is specified, copy the generated image to that location
        if output_path and images:
            import shutil
            source_image = images[0]
            if os.path.exists(source_image):
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Copy the generated image to the specified output path
                shutil.copy2(source_image, output_path)
                # Update the images list to point to the new location
                images = [output_path]

        # Default: stream first image
        if return_type == "image":
            if not images:
                return GoTResponse(got_text=result.get("got_text", ""), images=[])
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        # JSON: save to disk and return paths
        saved_paths: List[str] = []
        for idx, img in enumerate(images):
            out_path = os.path.join(temp_cache_dir, f"output_{idx}.png")
            img.save(out_path)
            saved_paths.append(out_path)

        return GoTResponse(got_text=result.get("got_text", ""), images=saved_paths)
    except Exception as e:
        traceback.print_exc()
        raise e
    finally:
        # Do not auto-delete to allow caller to fetch images; caller can clean later
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GOT_API_PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)


# python /Users/suny0a/Proj/ImageBrush/NeoAnalogist/thirdparty/GoT/api.py
# 
# Text-to-Image (t2i) mode:
# curl -X POST http://localhost:8100/got/generate \
#   -F "prompt=a red car on the beach at sunset" \
#   -F "mode=t2i"
#
# Image editing mode:
# curl -X POST http://localhost:8100/got/generate \
#   -F "prompt=turn the hat to blue" \
#   -F "mode=edit" \
#   -F "image=@/path/to/your/image.jpg"

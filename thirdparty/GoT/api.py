from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import shutil
import traceback

from got_service import got_generate

app = FastAPI(
    title="GoT API",
    description="API for GoT generation and editing following videomme_videollm_llava pattern",
    version="1.0.0",
)


class GoTRequest(BaseModel):
    prompt: str
    mode: str = "t2i"  # "t2i" or "edit"
    image_path: Optional[str] = None  # required when mode == "edit"
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
def got_generate_api(request: GoTRequest):
    temp_cache_dir = os.path.join("./tmp/got_cache", f"task_{uuid.uuid4()}")
    os.makedirs(temp_cache_dir, exist_ok=True)
    try:
        result = got_generate(
            prompt=request.prompt,
            mode=request.mode,
            image_path=request.image_path,
            height=request.height,
            width=request.width,
            max_new_tokens=request.max_new_tokens,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            image_guidance_scale=request.image_guidance_scale,
            cond_image_guidance_scale=request.cond_image_guidance_scale,
            cache_dir=temp_cache_dir,
        )

        # Persist images to disk and return paths
        saved_paths: List[str] = []
        for idx, img in enumerate(result.get("images", [])):
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
# curl -X POST http://localhost:8100/got/generate -H "Content-Type: application/json" -d '{"prompt":"a red car on the beach at sunset","mode":"t2i"}'
# curl -X POST http://localhost:8100/got/generate -H "Content-Type: application/json" -d '{"prompt":"turn the hat to blue","mode":"edit","image_path":"/Users/suny0a/Proj/ImageBrush/NeoAnalogist/thirdparty/GoT/examples/hat.jpg"}'

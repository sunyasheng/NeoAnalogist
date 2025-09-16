from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import shutil
import traceback
from PIL import Image
import io
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from .eval.utils import run_evaluation

app = FastAPI(
    title="Thyme API",
    description="API for Thyme vision-language model with code execution capabilities",
    version="1.0.0",
)

# Global model and processor
_MODEL = None
_PROCESSOR = None

class ThymeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class ThymeResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

def _load_model_once():
    """Load Thyme model and processor once at startup."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    
    print("[Thyme] Loading model...")
    MODEL_PATH = "yifanzhang114/Thyme"  # or your local model path
    
    config = AutoConfig.from_pretrained(MODEL_PATH)
    _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Use eager to avoid PyTorch version issues
        config=config
    )
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
    print("[Thyme] Model loaded successfully.")
    return _MODEL, _PROCESSOR

@app.on_event("startup")
async def _warmup_model():
    """Preload Thyme model at startup to avoid first-request latency."""
    try:
        _load_model_once()
        print("[Thyme] Model preloaded at startup.")
    except Exception as e:
        print(f"[Thyme] Preload failed: {e}")

@app.post("/thyme/generate", response_model=ThymeResponse)
async def thyme_generate_api(
    prompt: str = Form(...),
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None)
):
    """Generate text response from Thyme model with optional image input."""
    try:
        model, processor = _load_model_once()
        
        # Prepare messages
        messages = [{"role": "user", "content": []}]
        
        # Add image if provided (either uploaded file or file path)
        temp_image_path = None
        if image:
            # Read and process uploaded image
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            temp_image_path = f"/tmp/thyme_temp_{uuid.uuid4()}.jpg"
        elif image_path and os.path.exists(image_path):
            # Load image from file path
            pil_image = Image.open(image_path).convert("RGB")
            temp_image_path = f"/tmp/thyme_temp_{uuid.uuid4()}.jpg"
        
        if temp_image_path:
            # Resize image to reduce memory usage
            max_size = 512  # Maximum width or height
            if pil_image.width > max_size or pil_image.height > max_size:
                # Calculate new size maintaining aspect ratio
                ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                new_width = int(pil_image.width * ratio)
                new_height = int(pil_image.height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"[Thyme] Image resized to {new_width}x{new_height}")
            
            # Save to temp file for processing with compression
            pil_image.save(temp_image_path, "JPEG", quality=95, optimize=True)
            
            messages[0]["content"].append({
                "type": "image",
                "image": temp_image_path,
            })
        
        # Add text prompt
        messages[0]["content"].append({
            "type": "text", 
            "text": prompt
        })
        
        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Clean up temp file
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return ThymeResponse(
            response=output_text[0] if output_text else "",
            success=True
        )
        
    except Exception as e:
        # Clean up temp file in case of error
        if 'temp_image_path' in locals() and temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        traceback.print_exc()
        return ThymeResponse(
            response="",
            success=False,
            error=str(e)
        )

@app.post("/thyme/chat", response_model=ThymeResponse)
async def thyme_chat_api(
    messages: str = Form(...),  # JSON string of messages
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9)
):
    """Chat with Thyme model using structured messages."""
    try:
        import json
        model, processor = _load_model_once()
        
        # Parse messages
        message_list = json.loads(messages)
        
        # Prepare for inference
        text = processor.apply_chat_template(
            message_list, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message_list)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return ThymeResponse(
            response=output_text[0] if output_text else "",
            success=True
        )
        
    except Exception as e:
        traceback.print_exc()
        return ThymeResponse(
            response="",
            success=False,
            error=str(e)
        )

@app.post("/thyme/chat_upload", response_model=ThymeResponse)
async def thyme_chat_upload_api(
    prompt: str = Form(...),
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None)
):
    """Chat with Thyme model using direct image uploads (supports 1-2 images)."""
    temp_image_paths = []
    try:
        model, processor = _load_model_once()
        
        # Prepare messages
        messages = [{"role": "user", "content": []}]
        
        # Process uploaded images
        for i, uploaded_file in enumerate([image1, image2], 1):
            if uploaded_file:
                # Read and process uploaded image
                image_data = await uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                # Resize image to reduce memory usage
                max_size = 512  # Maximum width or height
                if pil_image.width > max_size or pil_image.height > max_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                    new_width = int(pil_image.width * ratio)
                    new_height = int(pil_image.height * ratio)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"[Thyme] Image {i} resized to {new_width}x{new_height}")
                
                # Save to temp file
                temp_image_path = f"/tmp/thyme_upload_{uuid.uuid4()}.jpg"
                pil_image.save(temp_image_path, "JPEG", quality=95, optimize=True)
                temp_image_paths.append(temp_image_path)
                
                messages[0]["content"].append({
                    "type": "image",
                    "image": temp_image_path,
                })
        
        # Add text prompt
        messages[0]["content"].append({
            "type": "text", 
            "text": prompt
        })
        
        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return ThymeResponse(
            response=output_text[0] if output_text else "",
            success=True
        )
        
    except Exception as e:
        traceback.print_exc()
        return ThymeResponse(
            response="",
            success=False,
            error=str(e)
        )

@app.post("/thyme/evaluate", response_model=ThymeResponse)
async def thyme_evaluate_api(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None)
):
    """Use Thyme's full evaluation pipeline with code execution capabilities."""
    temp_image_path = None
    try:
        model, processor = _load_model_once()
        
        # Handle image input
        if image:
            # Read and process uploaded image
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            temp_image_path = f"/tmp/thyme_eval_{uuid.uuid4()}.jpg"
            pil_image.save(temp_image_path, "JPEG", quality=95, optimize=True)
        elif image_path and os.path.exists(image_path):
            temp_image_path = image_path
        else:
            return ThymeResponse(
                response="",
                success=False,
                error="No valid image provided"
            )
        
        # Use Thyme's evaluation pipeline
        final_assistant_response, final_answer = run_evaluation(
            question, temp_image_path, model, processor
        )
        
        # Clean up temp file if we created it
        if image and temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return ThymeResponse(
            response=final_assistant_response,
            success=True
        )
        
    except Exception as e:
        # Clean up temp file in case of error
        if image and temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        traceback.print_exc()
        return ThymeResponse(
            response="",
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("THYME_API_PORT", 8201))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Usage examples:
# 
# Text-only generation:
# curl -X POST http://localhost:8201/thyme/generate \
#   -F "prompt=Describe what you see in this image"
#
# Image + text generation:
# curl -X POST http://localhost:8201/thyme/generate \
#   -F "prompt=What is in this image?" \
#   -F "image=@/path/to/image.jpg"
#
# Chat with structured messages:
# curl -X POST http://localhost:8201/thyme/chat \
#   -F 'messages=[{"role":"user","content":[{"type":"text","text":"Hello"}]}]'
#
# Full evaluation with code execution:
# curl -X POST http://localhost:8201/thyme/evaluate \
#   -F "question=What is the plate number of the blue car?" \
#   -F "image=@/path/to/image.jpg"

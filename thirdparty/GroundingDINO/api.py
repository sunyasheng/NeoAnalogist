"""
GroundingDINO API: Text-prompted object detection.

POST /grounding-dino/detect
  - image: UploadFile (required)
  - text_prompt: str (required, e.g., "cat, dog, person")
  - box_threshold: float (optional, default 0.3)
  - text_threshold: float (optional, default 0.25)
  - return_type: str (optional, default "json"; "json" | "image")

Returns:
  - return_type="json" (default): application/json with fields:
    - success: bool
    - error: str (if failed)
    - num_detections: int
    - detections: list of dict with fields:
      - box: [x1, y1, x2, y2] in pixels
      - label: str
      - confidence: float
  - return_type="image": image/jpeg with bounding boxes drawn
"""

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import io
import os
import sys
import traceback
import uuid
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add GroundingDINO to path
sys.path.append(os.path.dirname(__file__))

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

app = FastAPI(title="GroundingDINO API", description="Text-prompted object detection", version="1.0.0")

_HERE = os.path.abspath(os.path.dirname(__file__))

# Global model variables
_model = None
_device = None

def _load_image_to_numpy(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)

def load_image(image_array):
    """Load and preprocess image array."""
    image_pil = Image.fromarray(image_array)
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    """Load GroundingDINO model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded: {load_res}")
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    """Get detection results from GroundingDINO."""
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    
    # get phrase
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
    
    return boxes_filt, pred_phrases

def plot_boxes_to_image(image_pil, boxes, labels):
    """Draw bounding boxes on image."""
    H, W = image_pil.size[1], image_pil.size[0]  # PIL size is (W, H)
    draw = ImageDraw.Draw(image_pil)
    
    # draw boxes and labels
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
    
    return image_pil

@app.on_event("startup")
async def _warmup_models():
    """Load model on startup."""
    global _model, _device
    
    # Model paths - using existing pretrained model
    model_config_path = os.path.join(_HERE, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    model_checkpoint_path = os.path.join(_HERE, "..", "..", "pretrained", "GroundingSAM", "groundingdino_swint_ogc.pth")
    
    # Check if model files exist
    if not os.path.exists(model_config_path):
        print(f"Warning: Model config not found at {model_config_path}")
        return
    if not os.path.exists(model_checkpoint_path):
        print(f"Warning: Model checkpoint not found at {model_checkpoint_path}")
        return
    
    try:
        _model = load_model(model_config_path, model_checkpoint_path, cpu_only=False)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GroundingDINO model loaded successfully on {_device}")
    except Exception as e:
        print(f"Failed to load GroundingDINO model: {e}")
        _model = None

@app.post("/grounding-dino/detect")
async def grounding_dino_detect(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    return_type: str = Form("json"),  # json | image
):
    """Detect objects using GroundingDINO."""
    if _model is None:
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": "Model not loaded"}
        )
    
    try:
        # Load and preprocess image
        image_np = _load_image_to_numpy(image)
        image_pil, image_tensor = load_image(image_np)
        
        # Run detection
        boxes_filt, pred_phrases = get_grounding_output(
            _model, image_tensor, text_prompt, box_threshold, text_threshold, 
            with_logits=False, cpu_only=(_device == "cpu")
        )
        
        # Convert boxes to pixel coordinates
        H, W = image_pil.size[1], image_pil.size[0]  # PIL size is (W, H)
        detections = []
        
        for box, label in zip(boxes_filt, pred_phrases):
            # Convert from normalized coordinates to pixel coordinates
            box = box * torch.Tensor([W, H, W, H])
            # Convert from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            
            detections.append({
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],  # [x1, y1, x2, y2]
                "label": label,
                "confidence": float(boxes_filt.max(dim=1)[0][len(detections)-1]) if len(boxes_filt) > 0 else 0.0
            })
        
        if return_type == "image":
            # Draw boxes and return image
            image_with_boxes = plot_boxes_to_image(image_pil, boxes_filt, pred_phrases)
            buf = io.BytesIO()
            image_with_boxes.save(buf, format="JPEG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")
        else:
            # Return JSON
            return {
                "success": True,
                "num_detections": len(detections),
                "detections": detections
            }
            
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GROUNDING_DINO_PORT", "8502"))
    uvicorn.run(app, host="0.0.0.0", port=port)

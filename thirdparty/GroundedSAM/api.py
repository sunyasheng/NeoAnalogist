"""
GroundedSAM API for object segmentation with bounding boxes and labels.

This API accepts an image, bounding boxes, and object labels to perform segmentation.
It uses GroundingDINO for detection and SAM for segmentation.
"""

import os
import io
import json
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# GroundingDINO imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI(title="GroundedSAM API", description="Object segmentation with bounding boxes and labels", version="1.0.0")

_HERE = os.path.abspath(os.path.dirname(__file__))
_grounding_dino_model = None
_sam_predictor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_pil):
    """Load and transform image for GroundingDINO."""
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    """Load GroundingDINO model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    """Get GroundingDINO output."""
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    print(f"DEBUG: get_grounding_output called with caption: '{caption}'")
    print(f"DEBUG: box_threshold: {box_threshold}, text_threshold: {text_threshold}")
    
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    
    print(f"DEBUG: Raw logits shape: {logits.shape}, boxes shape: {boxes.shape}")
    print(f"DEBUG: Raw boxes (first 5): {boxes[:5]}")
    print(f"DEBUG: Raw logits max values (first 5): {logits.max(dim=1)[0][:5]}")
    
    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    
    print(f"DEBUG: After filtering - boxes_filt shape: {boxes_filt.shape}")
    print(f"DEBUG: After filtering - boxes_filt: {boxes_filt}")
    
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())
    
    print(f"DEBUG: Final pred_phrases: {pred_phrases}")
    print(f"DEBUG: Final boxes_filt: {boxes_filt}")
    
    return boxes_filt, torch.Tensor(scores), pred_phrases

def segment_with_boxes(sam_predictor, image, boxes, labels):
    """Segment objects using SAM with provided bounding boxes."""
    sam_predictor.set_image(image)
    result_masks = []
    
    print(f"DEBUG: segment_with_boxes called with {len(boxes)} boxes")
    print(f"DEBUG: Image shape: {image.shape}")
    
    for i, box in enumerate(boxes):
        # Convert box format if needed
        if isinstance(box, list):
            box = np.array(box)
        elif hasattr(box, 'cpu'):  # PyTorch tensor
            box = box.cpu().numpy()
        
        print(f"DEBUG: Processing box {i}: {box} (type: {type(box)})")
        
        # Ensure box is in correct format [x1, y1, x2, y2]
        if len(box) == 4:
            # Convert to float32 numpy array
            box = box.astype(np.float32)
            print(f"DEBUG: Box {i} after conversion: {box}")
            
            # Check if box is valid (not all zeros)
            if np.all(box == 0):
                print(f"WARNING: Box {i} is all zeros, creating empty mask")
                result_masks.append(np.zeros(image.shape[:2], dtype=bool))
                continue
            
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            print(f"DEBUG: Box {i} - got {len(masks)} masks, scores: {scores}")
            
            # Select the best mask
            index = np.argmax(scores)
            best_mask = masks[index]
            print(f"DEBUG: Box {i} - selected mask {index}, shape: {best_mask.shape}, unique values: {np.unique(best_mask)}")
            result_masks.append(best_mask)
        else:
            print(f"Warning: Invalid box format for box {i}: {box}")
            # Create empty mask
            result_masks.append(np.zeros(image.shape[:2], dtype=bool))
    
    print(f"DEBUG: Returning {len(result_masks)} masks")
    return np.array(result_masks)

@app.on_event("startup")
async def _warmup_models():
    """Load models on startup."""
    global _grounding_dino_model, _sam_predictor, _device
    
    # GroundingDINO paths
    model_config_path = os.path.join(_HERE, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    model_checkpoint_path = os.path.join(_HERE, "..", "..", "pretrained", "GroundingSAM", "groundingdino_swint_ogc.pth")
    
    # SAM paths
    sam_checkpoint_path = os.path.join(_HERE, "..", "..", "pretrained", "GroundingSAM", "sam_vit_h_4b8939.pth")
    
    if not os.path.exists(model_config_path):
        print(f"Warning: GroundingDINO config not found at {model_config_path}")
        return
    if not os.path.exists(model_checkpoint_path):
        print(f"Warning: GroundingDINO checkpoint not found at {model_checkpoint_path}")
        return
    if not os.path.exists(sam_checkpoint_path):
        print(f"Warning: SAM checkpoint not found at {sam_checkpoint_path}")
        return
    
    try:
        # Load GroundingDINO
        _grounding_dino_model = load_model(model_config_path, model_checkpoint_path, _device)
        print(f"GroundingDINO model loaded successfully on {_device}")
        
        # Load SAM
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
        sam.to(device=_device)
        _sam_predictor = SamPredictor(sam)
        print(f"SAM model loaded successfully on {_device}")
        
    except Exception as e:
        print(f"Failed to load models: {e}")
        _grounding_dino_model = None
        _sam_predictor = None

@app.post("/grounded-sam/segment")
async def grounded_sam_segment(
    image: UploadFile = File(...),
    boxes: str = Form(...),  # JSON string of bounding boxes
    labels: str = Form(...),  # JSON string of labels
    return_type: str = Form("json"),  # json | image
    output_dir: Optional[str] = Form(None),
):
    """Segment objects using provided bounding boxes and labels."""
    global _grounding_dino_model, _sam_predictor, _device
    
    if _grounding_dino_model is None or _sam_predictor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Models not loaded"})
    
    try:
        # Parse inputs
        boxes_data = json.loads(boxes)
        labels_data = json.loads(labels)
        
        if len(boxes_data) != len(labels_data):
            return JSONResponse(status_code=400, content={"success": False, "error": "Number of boxes and labels must match"})
        
        # Load image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Convert boxes to numpy array
        boxes_array = np.array(boxes_data)
        
        # Perform segmentation
        masks = segment_with_boxes(_sam_predictor, image_rgb, boxes_array, labels_data)
        
        if return_type == "json":
            # Return segmentation results as JSON
            results = []
            for i, (box, label, mask) in enumerate(zip(boxes_data, labels_data, masks)):
                # Convert mask to list of coordinates or save as file
                mask_path = None
                if output_dir:
                    mask_filename = f"mask_{i}.png"
                    mask_path = os.path.join(output_dir, mask_filename)
                    os.makedirs(output_dir, exist_ok=True)
                    mask_image = (mask * 255).astype(np.uint8)
                    cv2.imwrite(mask_path, mask_image)
                
                results.append({
                    "box": box,
                    "label": label,
                    "mask_path": mask_path,
                    "mask_area": int(np.sum(mask))
                })
            
            return {
                "success": True,
                "num_instances": len(results),
                "results": results
            }
        
        elif return_type == "image":
            # Return annotated image
            annotated_image = image_cv.copy()
            
            # Draw masks
            for i, (mask, label) in enumerate(zip(masks, labels_data)):
                # Create colored mask
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask = np.zeros_like(image_cv)
                colored_mask[mask] = color
                
                # Blend with original image
                annotated_image = cv2.addWeighted(annotated_image, 0.7, colored_mask, 0.3, 0)
                
                # Draw bounding box
                box = boxes_data[i]
                cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                
                # Draw label
                cv2.putText(annotated_image, label, (int(box[0]), int(box[1]) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.png', annotated_image)
            img_bytes = io.BytesIO(buffer).getvalue()
            
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid return_type"})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/grounded-sam/segment-masks")
async def grounded_sam_segment_masks(
    image: UploadFile = File(...),
    boxes: str = Form(...),  # JSON string of bounding boxes
    labels: str = Form(...),  # JSON string of labels
    mask_index: int = Form(0),  # Which mask to return (0-based index)
):
    """Segment objects and return individual mask image by index."""
    global _grounding_dino_model, _sam_predictor, _device
    
    if _grounding_dino_model is None or _sam_predictor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Models not loaded"})
    
    try:
        # Parse inputs
        boxes_data = json.loads(boxes)
        labels_data = json.loads(labels)
        
        if len(boxes_data) != len(labels_data):
            return JSONResponse(status_code=400, content={"success": False, "error": "Number of boxes and labels must match"})
        
        if mask_index < 0 or mask_index >= len(boxes_data):
            return JSONResponse(status_code=400, content={"success": False, "error": f"mask_index {mask_index} out of range [0, {len(boxes_data)-1}]"})
        
        # Load image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Convert boxes to numpy array
        boxes_array = np.array(boxes_data)
        
        # Perform segmentation
        masks = segment_with_boxes(_sam_predictor, image_rgb, boxes_array, labels_data)
        
        # Get the specific mask
        mask = masks[mask_index]
        label = labels_data[mask_index]
        
        # Convert mask to image
        mask_image = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_image, mode='L')
        
        # Convert to bytes
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(mask_buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=mask_{mask_index}_{label.replace(' ', '_')}.png"}
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/grounded-sam/segment-all-masks")
async def grounded_sam_segment_all_masks(
    image: UploadFile = File(...),
    boxes: str = Form(...),  # JSON string of bounding boxes
    labels: str = Form(...),  # JSON string of labels
    output_dir: str = Form(...),  # Directory to save all masks
):
    """Segment objects and save all individual mask images to specified directory."""
    global _grounding_dino_model, _sam_predictor, _device
    
    if _grounding_dino_model is None or _sam_predictor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Models not loaded"})
    
    try:
        # Parse inputs
        boxes_data = json.loads(boxes)
        labels_data = json.loads(labels)
        
        if len(boxes_data) != len(labels_data):
            return JSONResponse(status_code=400, content={"success": False, "error": "Number of boxes and labels must match"})
        
        # Load image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Convert boxes to numpy array
        boxes_array = np.array(boxes_data)
        
        # Perform segmentation
        masks = segment_with_boxes(_sam_predictor, image_rgb, boxes_array, labels_data)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all masks
        mask_paths = []
        for i, (box, label, mask) in enumerate(zip(boxes_data, labels_data, masks)):
            # Convert mask to image
            mask_image = (mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_image, mode='L')
            
            # Save mask
            mask_filename = f"mask_{i}_{label.replace(' ', '_')}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            mask_pil.save(mask_path)
            mask_paths.append(mask_path)
        
        return {
            "success": True,
            "num_masks": len(mask_paths),
            "mask_paths": mask_paths,
            "output_dir": output_dir
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/grounded-sam/detect")
async def grounded_sam_detect(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    return_type: str = Form("json"),  # json | image
):
    """Detect objects using GroundingDINO (same as GroundingDINO API)."""
    global _grounding_dino_model, _device
    
    if _grounding_dino_model is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "GroundingDINO model not loaded"})
    
    try:
        # Load image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        _, image_tensor = load_image(image_pil)
        
        # Get GroundingDINO output
        boxes_filt, scores, pred_phrases = get_grounding_output(
            _grounding_dino_model, image_tensor, text_prompt, box_threshold, text_threshold, device=_device
        )
        
        detections = []
        for box, label in zip(boxes_filt, pred_phrases):
            x0, y0, x1, y1 = box.tolist()
            # Convert from normalized coordinates [0,1] to pixel coordinates
            size = image_pil.size
            W, H = size[0], size[1]
            x0_pixel = int(x0 * W)
            y0_pixel = int(y0 * H)
            x1_pixel = int(x1 * W)
            y1_pixel = int(y1 * H)
            detections.append({
                "box": [x0_pixel, y0_pixel, x1_pixel, y1_pixel],
                "label": label,
                "confidence": float(label.split('(')[-1][:-1]) if '(' in label and ')' in label else 0.0
            })
        
        if return_type == "json":
            return {
                "success": True,
                "num_detections": len(detections),
                "detections": detections
            }
        elif return_type == "image":
            # Create annotated image
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            for detection in detections:
                box = detection["box"]
                label = detection["label"]
                confidence = detection["confidence"]
                
                # Draw bounding box
                cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(image_cv, label_text, (box[0], box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.png', image_cv)
            img_bytes = io.BytesIO(buffer).getvalue()
            
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid return_type"})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/grounded-sam/detect-and-segment")
async def grounded_sam_detect_and_segment(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    return_type: str = Form("json"),  # json | image
    output_dir: Optional[str] = Form(None),
):
    """Complete pipeline: detect objects with GroundingDINO and segment with SAM."""
    global _grounding_dino_model, _sam_predictor, _device
    
    if _grounding_dino_model is None or _sam_predictor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Models not loaded"})
    
    try:
        # Load image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        _, image_tensor = load_image(image_pil)
        
        # Step 1: Detect objects with GroundingDINO
        boxes_filt, scores, pred_phrases = get_grounding_output(
            _grounding_dino_model, image_tensor, text_prompt, box_threshold, text_threshold, device=_device
        )
        
        if len(boxes_filt) == 0:
            return {
                "success": True,
                "num_detections": 0,
                "num_masks": 0,
                "detections": [],
                "masks": []
            }
        
        # Convert boxes to proper format for SAM
        # GroundingDINO returns normalized coordinates [0,1], convert to pixel coordinates
        size = image_pil.size
        H, W = size[1], size[0]
        boxes_for_sam = []
        for i in range(boxes_filt.size(0)):
            # Convert from normalized [0,1] to pixel coordinates [x1, y1, x2, y2]
            box = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_for_sam.append(box.cpu().numpy())
        
        # Step 2: Segment objects with SAM
        masks = segment_with_boxes(_sam_predictor, image_rgb, boxes_for_sam, pred_phrases)
        
        if return_type == "json":
            # Return both detections and masks
            detections = []
            mask_paths = []
            
            for i, (box, label, mask, score) in enumerate(zip(boxes_for_sam, pred_phrases, masks, scores)):
                x0, y0, x1, y1 = box
                detections.append({
                    "box": [int(x0), int(y0), int(x1), int(y1)],
                    "label": label,
                    "confidence": float(score)
                })
                
                # Save mask if output_dir provided
                mask_path = None
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    mask_image = (mask * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_image, mode='L')
                    mask_filename = f"mask_{i}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                    mask_path = os.path.join(output_dir, mask_filename)
                    mask_pil.save(mask_path)
                    mask_paths.append(mask_path)
            
            return {
                "success": True,
                "num_detections": len(detections),
                "num_masks": len(masks),
                "detections": detections,
                "mask_paths": mask_paths if output_dir else []
            }
        
        elif return_type == "image":
            # Return annotated image with masks
            annotated_image = image_cv.copy()
            
            for i, (box, label, mask, score) in enumerate(zip(boxes_for_sam, pred_phrases, masks, scores)):
                # Create colored mask
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask = np.zeros_like(image_cv)
                colored_mask[mask] = color
                
                # Blend with original image
                annotated_image = cv2.addWeighted(annotated_image, 0.7, colored_mask, 0.3, 0)
                
                # Draw bounding box
                x0, y0, x1, y1 = box
                cv2.rectangle(annotated_image, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
                
                # Draw label
                label_text = f"{label}: {float(score):.2f}"
                cv2.putText(annotated_image, label_text, (int(x0), int(y0) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.png', annotated_image)
            img_bytes = io.BytesIO(buffer).getvalue()
            
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid return_type"})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/alive")
async def alive():
    """Health check endpoint."""
    return {"status": "alive", "models_loaded": _grounding_dino_model is not None and _sam_predictor is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GROUNDED_SAM_PORT", "8503"))
    uvicorn.run(app, host="0.0.0.0", port=port)

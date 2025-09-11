from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
import json
from litellm import completion

from core.events.action.image import ImageEntityExtractAction
from core.events.observation.image import ImageEntityExtractObservation


class ImageEntityExtractTask:
    """Server-side task that runs object/entity detection on an image.

    Minimal reference implementation using OpenCV DNN (no heavy deps). If Ultralytics
    YOLO is available in your image, you can swap the backend to a YOLO runner.
    """

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def _load_image(self, action: ImageEntityExtractAction) -> Tuple[np.ndarray, Tuple[int, int]]:
        if action.image_path:
            img = Image.open(action.image_path).convert("RGB")
        elif action.image_bytes:
            data = base64.b64decode(action.image_bytes)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            raise ValueError("ImageEntityExtractAction requires image_path or image_bytes")
        arr = np.array(img)
        h, w = arr.shape[:2]
        return arr, (w, h)

    def run(self, action: ImageEntityExtractAction) -> ImageEntityExtractObservation:
        t0 = time.time()
        img, (w, h) = self._load_image(action)

        # Build data URL for vision input
        buffered = io.BytesIO()
        Image.fromarray(img).save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        system_prompt = (
            "You are a strict vision parser. Return ONLY valid JSON with this schema:"\
            " {\n"\
            "  \"caption\": string,\n"\
            "  \"entities\": [ { \"label\": string, \"score\": number, \"bbox_norm_xyxy\": [x1, y1, x2, y2], \"desc\": string } ]\n"\
            " }\n"\
            "Rules: (1) bbox_norm_xyxy are floats normalized to [0,1] relative to image width/height;"\
            " (2) x1<x2, y1<y2; tightly enclose the visible pixels;"\
            " (3) Output JSON only, no extra text." 
        )

        user_prompt = (
            f"Image size is width={w}, height={h}. Provide a single overall caption and a list of entities with bbox [x,y,h,w] and a short desc."
        )

        try:
            resp = completion(
                model=action.model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message["content"]  # type: ignore[index]
            # Be robust to any accidental non-JSON tokens around the JSON
            def _extract_first_json(s: str) -> str:
                start = s.find('{')
                end = s.rfind('}')
                if start != -1 and end != -1 and end > start:
                    return s[start:end+1]
                return s
            safe_text = _extract_first_json(text)
            parsed: Dict[str, Any] = json.loads(safe_text)
            raw_entities: List[Dict] = parsed.get("entities", []) if isinstance(parsed, dict) else []
            entities: List[Dict] = []
            for ent in raw_entities or []:
                bbox_px = None
                if isinstance(ent, dict):
                    # Prefer normalized [x1,y1,x2,y2]
                    norm = ent.get("bbox_norm_xyxy") or ent.get("bbox_norm")
                    if isinstance(norm, (list, tuple)) and len(norm) == 4:
                        x1n, y1n, x2n, y2n = [float(max(0.0, min(1.0, v))) for v in norm]
                        if x2n < x1n:
                            x1n, x2n = x2n, x1n
                        if y2n < y1n:
                            y1n, y2n = y2n, y1n
                        x1 = int(round(x1n * w)); y1 = int(round(y1n * h))
                        x2 = int(round(x2n * w)); y2 = int(round(y2n * h))
                        x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
                        x2 = max(0, min(w, x2)); y2 = max(0, min(h, y2))
                        bw = max(1, x2 - x1); bh = max(1, y2 - y1)
                        bbox_px = [x1, y1, bh, bw]
                    else:
                        # Fallback: accept pixel [x,y,h,w] or [x,y,w,h]
                        bbox = ent.get("bbox")
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x, y, a, b = bbox
                            # Heuristic: treat third as h if within height, else swap
                            ah, aw = int(a), int(b)
                            if y + ah <= h and x + aw <= w:
                                bh, bw = ah, aw
                            else:
                                bw, bh = ah, aw
                            x = int(max(0, min(w - 1, int(x))))
                            y = int(max(0, min(h - 1, int(y))))
                            bw = int(max(1, min(w - x, bw)))
                            bh = int(max(1, min(h - y, bh)))
                            bbox_px = [x, y, bh, bw]
                if bbox_px is not None:
                    entities.append({
                        "label": ent.get("label", ""),
                        "score": ent.get("score", 0.0),
                        "bbox": bbox_px,
                        "desc": ent.get("desc", ""),
                    })
            # For debugging, return the raw model output in content
            content_str = text
        except Exception as e:
            entities = []
            content_str = f"vision_parse_error: {str(e)}"

        dt = int((time.time() - t0) * 1000)
        return ImageEntityExtractObservation(
            content=content_str,
            entities=entities,
            image_size=(w, h),
            model=action.model,
            time_ms=dt,
        )



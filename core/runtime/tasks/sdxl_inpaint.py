import os
import time
from typing import Optional

import requests


class SDXLInpaintClient:
    """Client for SDXL Inpainting API.

    Endpoint: POST /sdxl/inpaint
    - image: file (input image)
    - mask: file (binary mask image, white=inpaint region)
    - prompt: string (text prompt for inpainting)
    - negative_prompt: string (optional)
    - guidance_scale: float (default 8.0)
    - num_inference_steps: int (default 20)
    - strength: float (default 0.99)
    - seed: int (optional)
    Returns: image/png bytes. Caller should save to a file.
    """

    def __init__(self, base_url: str = "http://localhost:8402"):
        self.base_url = base_url.rstrip("/")

    def inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 8.0,
        num_inference_steps: int = 20,
        strength: float = 0.99,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        timeout: int = 600,
    ) -> str:
        """Run SDXL inpainting and save output PNG to output_path.

        Returns the absolute path to the saved output image.
        """
        url = f"{self.base_url}/sdxl/inpaint"

        # Prepare files
        files = {
            "image": open(image_path, "rb"),
            "mask": open(mask_path, "rb"),
        }

        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": str(guidance_scale),
            "num_inference_steps": str(num_inference_steps),
            "strength": str(strength),
        }
        
        if seed is not None:
            data["seed"] = str(seed)

        # Choose output path
        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(os.getcwd(), "workspace", "sdxl_outputs")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"sdxl_inpaint_{ts}.png")
        else:
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.getcwd(), output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with requests.post(url, files=files, data=data, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        finally:
            # Close file handles
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass

        return output_path

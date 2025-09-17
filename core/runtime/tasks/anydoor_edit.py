import os
import time
from typing import Optional

import requests


class AnyDoorClient:
    """Client for AnyDoor API (reference object transfer).

    Endpoint: POST /anydoor/edit
    - ref_image: file (PNG with alpha preferred; else provide ref_mask)
    - target_image: file
    - target_mask: file (binary mask; required)
    - ref_mask: file (optional; required if ref_image has no alpha)
    - guidance_scale: float (default 5.0)
    Returns: image/png bytes. Caller should save to a file.
    """

    def __init__(self, base_url: str = "http://localhost:8401"):
        self.base_url = base_url.rstrip("/")

    def edit(
        self,
        ref_image: str,
        target_image: str,
        target_mask: str,
        ref_mask: Optional[str] = None,
        guidance_scale: float = 5.0,
        output_path: Optional[str] = None,
        timeout: int = 600,
    ) -> str:
        """Run AnyDoor edit and save output PNG to output_path.

        Returns the absolute path to the saved output image.
        """
        url = f"{self.base_url}/anydoor/edit"

        # Prepare files
        files = {
            "ref_image": open(ref_image, "rb"),
            "target_image": open(target_image, "rb"),
            "target_mask": open(target_mask, "rb"),
        }
        if ref_mask:
            files["ref_mask"] = open(ref_mask, "rb")

        data = {
            "guidance_scale": str(guidance_scale),
        }

        # Choose output path
        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(os.getcwd(), "workspace", "anydoor_outputs")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"anydoor_{ts}.png")
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



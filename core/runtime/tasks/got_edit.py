import os
import requests
from typing import Optional, Dict, Any


class GoTEditClient:
    """Client for GoT API image editing and generation."""

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url.rstrip("/")

    def edit(self, image_path: str, prompt: str, height: int = 1024, width: int = 1024) -> Dict[str, Any]:
        url = f"{self.base_url}/got/generate"
        payload = {
            "prompt": prompt,
            "mode": "edit",
            "image_path": image_path,
            "height": height,
            "width": width,
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()

    def generate(self, prompt: str, height: int = 1024, width: int = 1024) -> Dict[str, Any]:
        url = f"{self.base_url}/got/generate"
        payload = {
            "prompt": prompt,
            "mode": "t2i",
            "height": height,
            "width": width,
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()



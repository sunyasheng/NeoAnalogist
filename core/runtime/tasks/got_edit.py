import os
import requests
from typing import Optional, Dict, Any


class GoTEditClient:
    """Client for GoT API image editing and generation."""

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url.rstrip("/")

    def edit(self, image_path: str, prompt: str, height: int = 1024, width: int = 1024) -> Dict[str, Any]:
        url = f"{self.base_url}/got/generate"
        
        # Prepare multipart form data
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'prompt': prompt,
                'mode': 'edit',
                'height': str(height),
                'width': str(width),
                'return_type': 'json'  # Return JSON with file paths
            }
            resp = requests.post(url, files=files, data=data, timeout=600)
        
        resp.raise_for_status()
        return resp.json()

    def generate(self, prompt: str, height: int = 1024, width: int = 1024) -> Dict[str, Any]:
        url = f"{self.base_url}/got/generate"
        
        # Prepare multipart form data for t2i
        data = {
            'prompt': prompt,
            'mode': 't2i',
            'height': str(height),
            'width': str(width),
            'return_type': 'json'  # Return JSON with file paths
        }
        resp = requests.post(url, data=data, timeout=600)
        resp.raise_for_status()
        return resp.json()



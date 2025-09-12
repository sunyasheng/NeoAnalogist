import os
import requests
from typing import Optional, Dict, Any, List
import uuid
import tempfile


class QwenAPIClient:
    """Client for Qwen2.5-VL API image analysis and text generation."""

    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 128, 
                 temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
        """Generate text response from Qwen2.5-VL model with optional image input."""
        url = f"{self.base_url}/qwen/generate"
        
        # Prepare form data
        data = {
            'prompt': prompt,
            'max_new_tokens': str(max_new_tokens),
            'temperature': str(temperature),
            'top_p': str(top_p),
        }
        
        files = {}
        if image_path and os.path.exists(image_path):
            files['image'] = open(image_path, 'rb')
        
        try:
            resp = requests.post(url, data=data, files=files, timeout=600)  # Increased timeout to 10 minutes
            resp.raise_for_status()
            return resp.json()
        finally:
            if 'image' in files:
                files['image'].close()

    def chat(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128,
             temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
        """Chat with Qwen2.5-VL model using structured messages."""
        url = f"{self.base_url}/qwen/chat"
        
        import json
        data = {
            'messages': json.dumps(messages),
            'max_new_tokens': str(max_new_tokens),
            'temperature': str(temperature),
            'top_p': str(top_p),
        }
        
        resp = requests.post(url, data=data, timeout=600)  # Increased timeout to 10 minutes
        resp.raise_for_status()
        return resp.json()

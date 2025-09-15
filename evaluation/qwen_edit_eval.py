import argparse
import io
import os
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image
import requests
import json


def concatenate_images_horizontally(left_path: str, right_path: str, max_side: int = 1024) -> Image.Image:
    """Concatenate two images horizontally with optional resizing to fit within max_side height.

    Args:
        left_path: Path to the left (source) image.
        right_path: Path to the right (edited) image.
        max_side: Max height of the output image.

    Returns:
        PIL.Image.Image: Concatenated image.
    """
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # Resize both to have the same height (min of their heights, capped by max_side)
    target_height = min(max(left.height, right.height), max_side)

    def resize_keep_aspect(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        ratio = target_h / img.height
        new_w = max(1, int(img.width * ratio))
        return img.resize((new_w, target_h), Image.Resampling.LANCZOS)

    left_resized = resize_keep_aspect(left, target_height)
    right_resized = resize_keep_aspect(right, target_height)

    out_w = left_resized.width + right_resized.width
    out = Image.new("RGB", (out_w, target_height), color=(0, 0, 0))
    out.paste(left_resized, (0, 0))
    out.paste(right_resized, (left_resized.width, 0))
    return out


def build_eval_prompt(instruction: str) -> str:
    """Simplified prompt to debug message format with Qwen chat (two images)."""
    return (
        "Describe the two images provided in this message.\n"
        "The first image is the Source image, the second image is the Edited image.\n"
        f"Editing instruction (for reference only): {instruction}\n"
    )


def call_qwen_generate(api_url: str, prompt: str, image: Image.Image, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Call Qwen generate endpoint with one concatenated image and prompt."""
    url = api_url.rstrip("/") + "/qwen/generate"

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, optimize=True)
    buf.seek(0)

    files = {"image": ("concat.jpg", buf, "image/jpeg")}
    data = {
        "prompt": prompt,
        "max_new_tokens": str(max_new_tokens),
        "temperature": str(temperature),
        "top_p": str(top_p),
    }
    resp = requests.post(url, files=files, data=data, timeout=180)
    resp.raise_for_status()
    js = resp.json()
    if not js.get("success", False):
        raise RuntimeError(js.get("error", "Qwen API returned failure"))
    return js.get("response", "")


def call_qwen_chat(api_url: str, prompt: str, image_paths: list[str], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Call Qwen chat endpoint with two image paths and prompt (multi-image without concatenation)."""
    url = api_url.rstrip("/") + "/qwen/chat"

    # Build messages using official Qwen2.5 format: file:// URLs for local images
    content = []
    for img_path in image_paths:
        # Use file:// protocol for local paths as per Qwen2.5 documentation
        content.append({"type": "image", "image": f"file://{os.path.abspath(img_path)}"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    data = {
        "messages": json.dumps(messages, ensure_ascii=False),
        "max_new_tokens": str(max_new_tokens),
        "temperature": str(temperature),
        "top_p": str(top_p),
    }
    resp = requests.post(url, data=data, timeout=180)
    resp.raise_for_status()
    js = resp.json()
    if not js.get("success", False):
        raise RuntimeError(js.get("error", "Qwen chat API returned failure"))
    return js.get("response", "")


def main():
    parser = argparse.ArgumentParser(description="Evaluate edit success via Qwen (chat, two images)")
    parser.add_argument("--original", required=True, help="Path to source/original image")
    parser.add_argument("--edited", required=True, help="Path to edited image")
    parser.add_argument("--instruction", required=True, help="Editing instruction text")
    parser.add_argument("--api-url", default="http://localhost:8200", help="Qwen API base URL")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if not os.path.exists(args.original):
        print(f"Original image not found: {args.original}")
        sys.exit(1)
    if not os.path.exists(args.edited):
        print(f"Edited image not found: {args.edited}")
        sys.exit(1)

    prompt = build_eval_prompt(args.instruction)

    # Always use chat with two images
    response = call_qwen_chat(
        api_url=args.api_url,
        prompt=prompt,
        image_paths=[args.original, args.edited],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Print the model's response as-is; the prompt instructs the required JSON-like format
    print(response)


if __name__ == "__main__":
    main()



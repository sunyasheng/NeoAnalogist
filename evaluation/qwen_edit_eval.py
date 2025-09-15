import argparse
import io
import os
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image
import requests


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
    """Build the evaluation prompt for Qwen using the provided template."""
    return (
        "Human:\n"
        "You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on\n"
        "the given rules. You will have to give your output in this way (Keep your reasoning concise and short.): ”score” : [...],\n"
        "”reasoning” : ”...” and don’t output anything else.\n"
        "Two images will be provided:\n"
        "The first being the original AI-generated image and the second being an edited version of the first. The objective is to\n"
        "evaluate how successfully the editing instruction has been executed in the second image. Note that sometimes the two\n"
        "images might look identical due to the failure of image edit.\n"
        "From a scale 0 to 10:\n"
        "A score from 0 to 10 will be given based on the success of the editing.\n"
        "- 0 indicates that the scene in the edited image does not follow the editing instruction at all.\n"
        "- 10 indicates that the scene in the edited image follow the editing instruction text perfectly.\n"
        "- If the object in the instruction is not present in the original image at all, the score will be 0.\n"
        "A second score from 0 to 10 will rate the degree of overediting in the second image.\n"
        "- 0 indicates that the scene in the edited image is completely different from the original.\n"
        "- 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.\n"
        "Put the score in a list such that output score = [score1, score2], where ’score1’ evaluates the editing success and ’score2’\n"
        "evaluates the degree of overediting.\n"
        f"Editing instruction: {instruction}\n"
        "<Image> Source Image (left) + Edited Image (right) </Image>\n"
        "Assistant:"
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate edit success via Qwen with a GPT-4o-style prompt")
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

    concat = concatenate_images_horizontally(args.original, args.edited, max_side=1024)
    prompt = build_eval_prompt(args.instruction)
    response = call_qwen_generate(
        api_url=args.api_url,
        prompt=prompt,
        image=concat,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Print the model's response as-is; the prompt instructs the required JSON-like format
    print(response)


if __name__ == "__main__":
    main()



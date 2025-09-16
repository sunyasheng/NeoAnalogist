import argparse
import io
import os
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image
import requests
import json


def build_eval_prompt(instruction: str) -> str:
    """Build the evaluation prompt for Thyme using the provided template."""
    return (
        "你是一位专业的数字艺术家。你需要根据给定的规则评估由 AI 生成的图像效果。\n"
        "必须严格输出：{\"score\":[score1,score2],\"reasoning\":\"一句简短原因\",\"edited_description\":\"对编辑后图片的一两句客观描述\",\"suggestion\":\"若编辑欠佳给出改进建议与原因；若良好则写：保持当前方案\"}，且不要输出其他任何内容。\n"
        "- 必须包含 reasoning（简洁一句话）。\n"
        "- edited_description：用一两句话客观描述编辑后的图像内容与显著变化。\n"
        "- suggestion：如 score1<7，给出具体可执行的改进建议与原因；否则写"保持当前方案"。\n"
        "将提供两张图像：\n"
        "第一张是原始的 AI 生成图像，第二张是编辑后的版本。你的目标是评估第二张图是否成功执行了编辑指令。注意，有时由于编辑失败，两张图可能看起来相同。\n"
        "评分尺度为 0 到 10：\n"
        "根据编辑是否成功给出 0 到 10 的分数。\n"
        "- 0 表示编辑后的场景完全没有遵循编辑指令。\n"
        "- 10 表示编辑后的场景完美地遵循了编辑指令文本。\n"
        "- 如果指令中的目标对象在原始图像中不存在，分数为 0。\n"
        "第二个 0 到 10 的分数用于评估过度编辑的程度：\n"
        "- 0 表示编辑后的场景与原始场景完全不同。\n"
        "- 10 表示在尽可能少变动的前提下完成了有效编辑。\n"
        "将两个分数放入列表，输出为 score = [score1, score2]，其中 score1 评估编辑成功度，score2 评估过度编辑程度。\n"
        f"编辑指令：{instruction}\n"
    )


def call_thyme_chat(api_url: str, prompt: str, image_paths: list[str], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Call Thyme chat endpoint with two image paths and prompt (multi-image without concatenation)."""
    url = api_url.rstrip("/") + "/thyme/chat"

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
        raise RuntimeError(js.get("error", "Thyme chat API returned failure"))
    return js.get("response", "")


def call_thyme_upload(api_url: str, prompt: str, image_paths: list[str], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Call Thyme chat_upload endpoint with direct image uploads (supports 1-2 images)."""
    url = api_url.rstrip("/") + "/thyme/chat_upload"

    # Prepare files for upload
    files = {}
    data = {
        "prompt": prompt,
        "max_new_tokens": str(max_new_tokens),
        "temperature": str(temperature),
        "top_p": str(top_p),
    }
    
    # Upload up to 2 images
    for i, img_path in enumerate(image_paths[:2], 1):
        if os.path.exists(img_path):
            files[f"image{i}"] = (os.path.basename(img_path), open(img_path, 'rb'), 'image/jpeg')
        else:
            print(f"Warning: Image {i} not found: {img_path}")
    
    try:
        resp = requests.post(url, files=files, data=data, timeout=180)
        resp.raise_for_status()
        js = resp.json()
        if not js.get("success", False):
            raise RuntimeError(js.get("error", "Thyme upload API returned failure"))
        return js.get("response", "")
    finally:
        # Close file handles
        for file_handle in files.values():
            if hasattr(file_handle[1], 'close'):
                file_handle[1].close()


def call_thyme_evaluate(api_url: str, question: str, image_path: str) -> str:
    """Call Thyme's full evaluation pipeline with code execution capabilities."""
    url = api_url.rstrip("/") + "/thyme/evaluate"

    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    files = {"image": (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')}
    data = {"question": question}
    
    try:
        resp = requests.post(url, files=files, data=data, timeout=300)  # Longer timeout for code execution
        resp.raise_for_status()
        js = resp.json()
        if not js.get("success", False):
            raise RuntimeError(js.get("error", "Thyme evaluate API returned failure"))
        return js.get("response", "")
    finally:
        # Close file handle
        if hasattr(files["image"][1], 'close'):
            files["image"][1].close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate edit success via Thyme (supports chat, upload, and full evaluation modes)")
    parser.add_argument("--original", required=True, help="Path to source/original image")
    parser.add_argument("--edited", required=True, help="Path to edited image")
    parser.add_argument("--instruction", required=True, help="Editing instruction text")
    parser.add_argument("--api-url", default="http://localhost:8201", help="Thyme API base URL")
    parser.add_argument("--mode", choices=["chat", "upload", "evaluate"], default="upload", 
                       help="API mode: chat (file paths), upload (direct upload), evaluate (full pipeline)")
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

    # Choose API mode
    if args.mode == "evaluate":
        print("Using Thyme full evaluation pipeline...")
        # For evaluate mode, we need to create a combined question
        combined_question = f"{prompt}\n\nPlease compare these two images and provide your evaluation."
        response = call_thyme_evaluate(
            api_url=args.api_url,
            question=combined_question,
            image_path=args.original  # Use original image, Thyme will handle the comparison
        )
    elif args.mode == "upload":
        print("Using direct image upload mode...")
        response = call_thyme_upload(
            api_url=args.api_url,
            prompt=prompt,
            image_paths=[args.original, args.edited],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:  # chat mode
        print("Using chat mode with file paths...")
        response = call_thyme_chat(
            api_url=args.api_url,
            prompt=prompt,
            image_paths=[args.original, args.edited],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # Print the model's response as-is
    print(response)


if __name__ == "__main__":
    main()

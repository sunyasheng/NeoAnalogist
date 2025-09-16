import argparse
import base64
import os
from typing import List, Dict, Any
import json

from dotenv import load_dotenv
from openai import OpenAI


def build_eval_prompt(instruction: str) -> str:
    return (
        "You are a professional digital artist. Evaluate the effectiveness of the AI-edited image following these rules.\n"
        "Output strictly in JSON and nothing else: {\"score\":[score1,score2],\"reasoning\":\"one short sentence\"}.\n"
        "Two images are provided: the first is the original, the second is the edited version.\n"
        "Goal: judge how well the edit instruction was executed in the second image. The two images may be identical if editing failed.\n"
        "Scoring (0–10):\n"
        "- score1 (editing success): 0 = not following the instruction at all; 10 = perfectly follows the instruction.\n"
        "- If the target object in the instruction does not appear in the original image, score1 = 0.\n"
        "- score2 (degree of over-editing): 0 = scene completely different from original; 10 = minimal yet effective change.\n"
        "Return score as [score1, score2].\n"
        f"Editing instruction: {instruction}\n"
    )


def encode_image_to_data_url(path: str) -> str:
    mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_gpt4o(prompt: str, image_paths: List[str], model: str, max_tokens: int, temperature: float) -> str:
    # Load API key from environment (.env supported)
    load_dotenv()
    # OpenAI() will read OPENAI_API_KEY from environment
    client = OpenAI()

    # Build multimodal message with two images
    content = [{"type": "text", "text": prompt}]
    for p in image_paths[:2]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        data_url = encode_image_to_data_url(p)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate edit success via GPT-4o (no local VRAM needed)")
    # Single-pair inputs
    parser.add_argument("--original", help="Path to source/original image")
    parser.add_argument("--edited", help="Path to edited image")
    parser.add_argument("--instruction", help="Editing instruction text")
    # Batch mode from JSON (EMU format)
    parser.add_argument("--json", help="Path to generation_results.json to evaluate sequentially")
    parser.add_argument("--use-gt", action="store_true", help="Use ground_truth_image as 'edited' for evaluation")
    parser.add_argument("--save", help="Optional path to save JSONL outputs or JSON (auto-detected by extension)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model, e.g., gpt-4o or gpt-4o-mini")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    def eval_one(original_path: str, edited_path: str, instruction_text: str) -> str:
        prompt = build_eval_prompt(instruction_text)
        return call_gpt4o(
            prompt=prompt,
            image_paths=[original_path, edited_path],
            model=args.model,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Batch mode
    if args.json:
        json_path = os.path.abspath(args.json)
        base_dir = os.path.dirname(json_path)              # .../evaluation/emu_eval_results
        project_eval_dir = os.path.dirname(base_dir)       # .../evaluation
        with open(json_path, "r", encoding="utf-8") as f:
            items: List[Dict[str, Any]] = json.load(f)

        outputs = []
        for it in items:
            original = it.get("original_image")
            edited = None
            if args.use_gt:
                edited = it.get("ground_truth_image")
            if not edited:
                edited = it.get("generated_image") or it.get("edited_image")
            instruction = it.get("instruction", "")
            if not original or not edited:
                continue
            # Resolve relative paths robustly: try json dir then its parent (evaluation root)
            def _resolve(p: str) -> str:
                if os.path.isabs(p):
                    return p
                cand1 = os.path.join(base_dir, p)
                if os.path.exists(cand1):
                    return cand1
                cand2 = os.path.join(project_eval_dir, p)
                return cand2
            original = _resolve(original)
            edited = _resolve(edited)
            result = eval_one(original, edited, instruction)
            print(result)
            outputs.append({
                "idx": it.get("idx"),
                "instruction": instruction,
                "original_image": it.get("original_image"),
                "generated_image": it.get("generated_image") or it.get("edited_image"),
                "model": args.model,
                "response": result,
            })

        if args.save:
            save_path = os.path.abspath(args.save)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if save_path.lower().endswith(".jsonl"):
                with open(save_path, "w", encoding="utf-8") as f:
                    for row in outputs:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=2)
        return

    # Single pair mode
    if not (args.original and args.edited and args.instruction):
        raise SystemExit("Provide --original --edited --instruction for single evaluation, or --json for batch mode.")

    out_text = eval_one(args.original, args.edited, args.instruction)
    print(out_text)


if __name__ == "__main__":
    main()



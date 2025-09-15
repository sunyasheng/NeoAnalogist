#!/usr/bin/env python3
"""
Run Emu-Edit Dataset Evaluation with GoT API

This script:
1. Loads Emu-Edit dataset
2. Calls GoT API to generate edited images
3. Evaluates the results using evaluation metrics
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from PIL import Image
import io

# Add evaluation directory to path
sys.path.append(str(Path(__file__).parent))

from emu_edit_evaluator import EmuEditEvaluator

class GoTAPIClient:
    """GoT API Client for image editing"""
    
    def __init__(self, base_url="http://localhost:8100"):
        self.base_url = base_url
        
    def generate_image_edit(self, prompt, image_path, **kwargs):
        """Edit image using GoT API"""
        url = f"{self.base_url}/got/generate"
        
        data = {
            "prompt": prompt,
            "mode": "edit",
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "image_guidance_scale": kwargs.get("image_guidance_scale", 1.0),
            "cond_image_guidance_scale": kwargs.get("cond_image_guidance_scale", 4.0),
            "return_type": "json"
        }
        
        try:
            with open(image_path, 'rb') as f:
                files = {"image": f}
                response = requests.post(url, data=data, files=files, timeout=300)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error calling GoT API: {e}")
            return None

    def generate_image_edit_stream(self, prompt: str, image_path: str, output_path: str, **kwargs) -> bool:
        """Edit image using GoT API and stream the image back directly, saving to output_path.

        Returns True on success, False otherwise.
        """
        url = f"{self.base_url}/got/generate"

        data = {
            "prompt": prompt,
            "mode": "edit",
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "image_guidance_scale": kwargs.get("image_guidance_scale", 1.0),
            "cond_image_guidance_scale": kwargs.get("cond_image_guidance_scale", 4.0),
            "return_type": "image"
        }

        try:
            with open(image_path, 'rb') as f:
                files = {"image": f}
                # stream binary image back
                response = requests.post(url, data=data, files=files, timeout=600)
                response.raise_for_status()
                # Save bytes to output_path
                with open(output_path, 'wb') as out_f:
                    out_f.write(response.content)
                return True
        except Exception as e:
            print(f"Error calling GoT API (stream): {e}")
            return False

def download_image_from_url(url: str, save_path: str) -> bool:
    """Download image from URL and save to local path"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

def run_emu_evaluation(
    dataset_name: str = "validation",
    max_samples: int = 10,  # Limit for testing
    output_dir: str = "./emu_eval_results",
    got_api_url: str = "http://localhost:8100"
):
    """Run Emu-Edit dataset evaluation with GoT API"""
    
    print("="*80)
    print("Emu-Edit Dataset Evaluation with GoT API")
    print("="*80)
    
    # Initialize GoT API client
    got_client = GoTAPIClient(base_url=got_api_url)
    
    # Initialize evaluator
    evaluator = EmuEditEvaluator(device='cuda')
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create organized directories
    src_images_dir = output_path / "src-images"
    edited_images_dir = output_path / "edited-images"
    src_images_dir.mkdir(parents=True, exist_ok=True)
    edited_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    from datasets import load_dataset
    
    try:
        ds = load_dataset("facebook/emu_edit_test_set_generations")
        dataset = ds[dataset_name]
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Process samples
    results = []
    successful_generations = 0
    
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
            
        print(f"\nProcessing sample {i+1}/{max_samples}")
        print(f"Sample ID: {sample['idx']}")
        print(f"Instruction: {sample['instruction']}")
        
        # Save original image (it's already a PIL Image object)
        original_image_path = src_images_dir / f"src_{sample['idx']}.jpg"
        try:
            sample['image'].save(str(original_image_path))
            print(f"Saved original image: {original_image_path}")
        except Exception as e:
            print(f"Failed to save original image for sample {sample['idx']}: {e}")
            continue
        
        # Generate edited image using GoT API (prefer streaming to avoid tmp paths)
        print("Calling GoT API for image editing (stream mode)...")

        # Determine output path first
        generated_image_path = edited_images_dir / f"edited_{sample['idx']}.png"

        ok = got_client.generate_image_edit_stream(
            prompt=sample['instruction'],
            image_path=str(original_image_path),
            output_path=str(generated_image_path)
        )

        if not ok:
            print("Stream mode failed, falling back to JSON mode...")
            got_result = got_client.generate_image_edit(
                prompt=sample['instruction'],
                image_path=str(original_image_path)
            )

            print(f"GoT API response: {got_result}")
            
            if not got_result or not got_result.get("images"):
                print(f"GoT API returned no images for sample {sample['idx']}")
                continue
            
            # Fallback: try to resolve relative paths
            source_image_path = got_result["images"][0]
            possible_paths = [
                source_image_path,
                Path("/Users/suny0a/Proj/ImageBrush/NeoAnalogist/thirdparty/GoT") / source_image_path,
                Path.cwd() / source_image_path,
                Path("/home/suny0a/Proj/ImageBrush/NeoAnalogist/thirdparty/GoT") / source_image_path,
            ]
            resolved = None
            for path in possible_paths:
                if os.path.exists(path):
                    resolved = path
                    break
            if not resolved:
                print("Could not locate generated image from JSON response.")
                continue
            with open(resolved, 'rb') as src, open(generated_image_path, 'wb') as dst:
                dst.write(src.read())
            print(f"Generated image saved: {generated_image_path}")
        
        # Store result info
        result_info = {
            'idx': sample['idx'],
            'instruction': sample['instruction'],
            'original_image': str(original_image_path),
            'generated_image': str(generated_image_path),
            'ground_truth_image': sample['edited_image'],
            'got_reasoning': got_result.get("got_text", ""),
            'success': True
        }
        results.append(result_info)
        successful_generations += 1
        
        print(f"Successfully processed sample {sample['idx']}")
        
        # Add delay to avoid overwhelming the API
        time.sleep(1)
    
    print(f"\nGeneration completed: {successful_generations}/{max_samples} successful")
    
    if successful_generations == 0:
        print("No successful generations. Cannot proceed with evaluation.")
        return
    
    # Save generation results
    results_file = output_path / "generation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Generation results saved to: {results_file}")
    
    # Run evaluation
    print("\n" + "="*80)
    print("Running Evaluation Metrics")
    print("="*80)
    
    try:
        # Create image pairs for evaluation
        image_pairs = []
        for result in results:
            if result['success']:
                image_pairs.append((
                    result['generated_image'],
                    result['ground_truth_image']
                ))
        
        if not image_pairs:
            print("No valid image pairs for evaluation")
            return
        
        print(f"Evaluating {len(image_pairs)} image pairs...")
        
        # Run evaluation using the evaluator
        eval_results = evaluator.evaluate_image_pairs(
            image_pairs=image_pairs,
            captions=[r['instruction'] for r in results if r['success']]
        )
        
        # Save evaluation results
        eval_file = output_path / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation results saved to: {eval_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total samples processed: {max_samples}")
        print(f"Successful generations: {successful_generations}")
        print(f"Evaluation samples: {len(image_pairs)}")
        
        if eval_results:
            metrics = eval_results.get('metrics', {})
            print("\nMetrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Emu-Edit evaluation with GoT API')
    parser.add_argument('--dataset', default='validation', 
                       help='Dataset split to use (validation, test)')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to process')
    parser.add_argument('--output_dir', default='./emu_eval_results',
                       help='Output directory for results')
    parser.add_argument('--got_api_url', default='http://localhost:8100',
                       help='GoT API URL')
    
    args = parser.parse_args()
    
    # Check if GoT API is available
    try:
        response = requests.get(f"{args.got_api_url}/docs", timeout=5)
        print("✅ GoT API is available")
    except Exception as e:
        print(f"❌ GoT API is not available at {args.got_api_url}")
        print("Please start the GoT API service first:")
        print("python /Users/suny0a/Proj/ImageBrush/NeoAnalogist/thirdparty/GoT/api.py")
        return
    
    # Run evaluation
    run_emu_evaluation(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        got_api_url=args.got_api_url
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for image editing evaluation tools.

This script demonstrates how to use the evaluation tools with sample data.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.simple_evaluator import SimpleImageEditEvaluator


def test_simple_evaluation():
    """Test the simple evaluator with sample images."""
    print("Testing Simple Image Edit Evaluator...")
    
    # Initialize evaluator
    evaluator = SimpleImageEditEvaluator(
        device='cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    )
    
    # Test with sample data (you'll need to provide actual image paths)
    test_cases = [
        {
            'original': 'workspace/imgs/test.png',  # Replace with actual paths
            'edited': 'workspace/imgs/edited_pirate_dog.png',
            'caption': 'Change the major object within the image to a pirate dog.'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        
        # Check if files exist
        if not os.path.exists(test_case['original']):
            print(f"Warning: Original image not found: {test_case['original']}")
            continue
        if not os.path.exists(test_case['edited']):
            print(f"Warning: Edited image not found: {test_case['edited']}")
            continue
            
        # Run evaluation
        try:
            results = evaluator.evaluate_with_analysis(
                original_path=test_case['original'],
                edited_path=test_case['edited'],
                caption=test_case['caption']
            )
            
            # Print results
            evaluator.print_results(results)
            
            # Save results
            output_file = f'test_results_{i+1}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error in evaluation: {e}")


def test_emu_dataset_loading():
    """Test loading the Emu-Edit dataset."""
    print("\nTesting Emu-Edit Dataset Loading...")
    
    try:
        from datasets import load_dataset
        
        # Load a small sample
        ds = load_dataset("facebook/emu_edit_test_set_generations")
        val_dataset = ds['validation']
        
        print(f"Dataset loaded successfully!")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Sample keys: {val_dataset[0].keys()}")
        print(f"Sample idx: {val_dataset[0]['idx']}")
        print(f"Sample caption: {val_dataset[0]['output_caption'][:100]}...")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")


def test_clip_model():
    """Test CLIP model loading."""
    print("\nTesting CLIP Model...")
    
    try:
        import clip
        
        device = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
        model, transform = clip.load("ViT-B/32", device=device)
        
        print(f"CLIP model loaded successfully on {device}")
        
        # Test with a dummy image
        from PIL import Image
        import torch
        
        dummy_img = Image.new('RGB', (224, 224), color='red')
        img_tensor = transform(dummy_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encode_image(img_tensor)
            print(f"Image features shape: {features.shape}")
            
        print("CLIP model test passed!")
        
    except Exception as e:
        print(f"Error testing CLIP model: {e}")


def main():
    """Run all tests."""
    print("="*60)
    print("IMAGE EDITING EVALUATION TOOLS - TEST SUITE")
    print("="*60)
    
    # Test CLIP model
    test_clip_model()
    
    # Test dataset loading
    test_emu_dataset_loading()
    
    # Test simple evaluation (requires actual images)
    test_simple_evaluation()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()

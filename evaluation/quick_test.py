#!/usr/bin/env python3
"""
Quick test script for the evaluation tools.

This script tests the evaluation tools without requiring actual image files.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from evaluation.simple_evaluator import SimpleImageEditEvaluator
        print("✓ SimpleImageEditEvaluator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import SimpleImageEditEvaluator: {e}")
        return False
        
    try:
        from evaluation.emu_edit_evaluator import EmuEditEvaluator
        print("✓ EmuEditEvaluator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import EmuEditEvaluator: {e}")
        return False
        
    return True

def test_clip_loading():
    """Test if CLIP model can be loaded."""
    print("\nTesting CLIP model loading...")
    
    try:
        import clip
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model, transform = clip.load("ViT-B/32", device=device)
        print("✓ CLIP model loaded successfully")
        
        # Test with a dummy image
        from PIL import Image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        img_tensor = transform(dummy_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encode_image(img_tensor)
            print(f"✓ Image encoding test passed, features shape: {features.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ CLIP model test failed: {e}")
        return False

def test_dataset_loading():
    """Test if Emu-Edit dataset can be loaded."""
    print("\nTesting Emu-Edit dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Load a small sample
        ds = load_dataset("facebook/emu_edit_test_set_generations")
        val_dataset = ds['validation']
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Validation set size: {len(val_dataset)}")
        print(f"  - Sample keys: {list(val_dataset[0].keys())}")
        
        # Test a sample
        sample = val_dataset[0]
        print(f"  - Sample idx: {sample['idx']}")
        print(f"  - Sample caption: {sample['output_caption'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        return False

def test_evaluator_initialization():
    """Test if evaluators can be initialized."""
    print("\nTesting evaluator initialization...")
    
    try:
        from evaluation.simple_evaluator import SimpleImageEditEvaluator
        
        evaluator = SimpleImageEditEvaluator(device='cpu')  # Use CPU to avoid GPU issues
        print("✓ SimpleImageEditEvaluator initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluator initialization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("EVALUATION TOOLS - QUICK TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_clip_loading,
        test_dataset_loading,
        test_evaluator_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Evaluation tools are ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    print("="*60)

if __name__ == "__main__":
    main()

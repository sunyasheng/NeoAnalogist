#!/usr/bin/env python3
"""
Test script for SDXL Inpainting API

This script demonstrates how to use the SDXL inpainting API.
"""

import os
import time
from PIL import Image
import requests

def create_test_mask(width=1024, height=1024):
    """Create a simple test mask with a rectangular region."""
    mask = Image.new('RGB', (width, height), (0, 0, 0))  # Black background
    # Create a white rectangle in the center
    left = width // 4
    top = height // 4
    right = 3 * width // 4
    bottom = 3 * height // 4
    mask.paste((255, 255, 255), (left, top, right, bottom))
    return mask

def test_api():
    """Test the SDXL inpainting API."""
    print("Testing SDXL Inpainting API...")
    
    # Create test images
    test_image = Image.new('RGB', (1024, 1024), (128, 128, 128))  # Gray background
    test_mask = create_test_mask()
    
    # Save test images
    test_image.save("test_input.png")
    test_mask.save("test_mask.png")
    
    # Test API endpoint
    url = "http://localhost:8402/sdxl/inpaint"
    
    with open("test_input.png", "rb") as img_file, open("test_mask.png", "rb") as mask_file:
        files = {
            "image": img_file,
            "mask": mask_file
        }
        data = {
            "prompt": "a beautiful sunset over mountains",
            "negative_prompt": "blurry, low quality",
            "guidance_scale": "8.0",
            "num_inference_steps": "20",
            "strength": "0.99",
            "seed": "42"
        }
        
        try:
            print("Sending request to API...")
            response = requests.post(url, files=files, data=data, timeout=300)
            response.raise_for_status()
            
            # Save result
            with open("test_result.png", "wb") as f:
                f.write(response.content)
            
            print("✓ API test successful! Result saved as test_result.png")
            print(f"Execution time: {response.headers.get('X-Execution-Time', 'unknown')} seconds")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API test failed: {e}")
            return False
    
    return True

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    
    try:
        response = requests.get("http://localhost:8402/health", timeout=10)
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✓ Health check successful!")
        print(f"  Status: {health_data['status']}")
        print(f"  Model loaded: {health_data['model_loaded']}")
        print(f"  Device: {health_data['device']}")
        print(f"  CUDA available: {health_data['cuda_available']}")
        
        return health_data['model_loaded']
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("SDXL Inpainting API Test")
    print("=" * 30)
    
    # Check if API server is running
    if not test_health_check():
        print("\n❌ API server is not running or not healthy.")
        print("Please start the server with: python api.py")
        return
    
    # Run API test
    api_success = test_api()
    
    # Summary
    print("\n" + "=" * 30)
    print("Test Summary:")
    print(f"  Health check: ✓")
    print(f"  API test: {'✓' if api_success else '✗'}")
    
    if api_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ API test failed.")
    
    # Cleanup
    try:
        os.remove("test_input.png")
        os.remove("test_mask.png")
        print("\nCleaned up test files.")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()

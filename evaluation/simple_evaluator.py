#!/usr/bin/env python3
"""
Simple Image Edit Evaluator

A simplified version for quick testing and evaluation of image editing results.
This tool can evaluate single image pairs or batches of images.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
import clip
import numpy as np
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision import transforms

# Import AnyBench utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'thirdparty' / 'AnySD' / 'anybench'))
from eval.utils import eval_distance, eval_clip_i, eval_clip_t, is_all_black

# Qwen API client removed - evaluation works independently


class SimpleImageEditEvaluator:
    """Simple evaluator for image editing tasks."""
    
    def __init__(self, device: str = None):
        """
        Initialize the simple evaluator.
        
        Args:
            device: Device to use for evaluation (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.clip_model = None
        self.clip_transform = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load CLIP model for evaluation."""
        self.logger.info("Loading evaluation models...")
        
        # Load CLIP model
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=self.device)
        self.logger.info("CLIP model loaded successfully")
            
    def evaluate_single_pair(self, original_path: str, edited_path: str, caption: str) -> Dict[str, float]:
        """
        Evaluate a single image pair.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image
            caption: Caption describing the edit
            
        Returns:
            Dictionary of metric scores
        """
        if not self.clip_model:
            self.load_models()
            
        # Load images
        try:
            original_img = Image.open(original_path).convert('RGB')
            edited_img = Image.open(edited_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Error loading images: {e}")
            return {}
            
        # Create image pair
        image_pairs = [(edited_img, original_img, caption)]
        
        # Evaluate metrics
        results = {}
        
        try:
            # CLIP-I Score (image similarity)
            clip_i = eval_clip_i(
                image_pairs=image_pairs, 
                model=self.clip_model, 
                transform=self.clip_transform, 
                url_flag=False
            )
            results['clip_i'] = clip_i
            
            # CLIP-T Score (text-image alignment)
            clip_t, _ = eval_clip_t(
                image_pairs=image_pairs, 
                model=self.clip_model, 
                transform=self.clip_transform, 
                url_flag=False
            )
            results['clip_t'] = clip_t
            
            # L1 Distance
            l1 = eval_distance(
                image_pairs=image_pairs, 
                metric='l1', 
                url_flag=False
            )
            results['l1'] = l1
            
            # L2 Distance
            l2 = eval_distance(
                image_pairs=image_pairs, 
                metric='l2', 
                url_flag=False
            )
            results['l2'] = l2
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            
        return results
        
            
    def evaluate_with_analysis(self, original_path: str, edited_path: str, caption: str) -> Dict:
        """
        Evaluate image pair with metrics.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image
            caption: Caption describing the edit
            
        Returns:
            Complete evaluation results
        """
        self.logger.info("Starting evaluation...")
        
        # Load models
        self.load_models()
        
        # Evaluate metrics
        metrics = self.evaluate_single_pair(original_path, edited_path, caption)
        
        # Compile results
        results = {
            'original_path': original_path,
            'edited_path': edited_path,
            'caption': caption,
            'metrics': metrics
        }
        
        return results
        
    def print_results(self, results: Dict):
        """Print evaluation results in a formatted way."""
        print("="*60)
        print("IMAGE EDITING EVALUATION RESULTS")
        print("="*60)
        print(f"Original: {results['original_path']}")
        print(f"Edited: {results['edited_path']}")
        print(f"Caption: {results['caption']}")
        print("-"*60)
        
        if results['metrics']:
            print("METRICS:")
            print(f"  CLIP-I (Image Similarity): {results['metrics'].get('clip_i', 'N/A'):.3f}")
            print(f"  CLIP-T (Text Alignment):   {results['metrics'].get('clip_t', 'N/A'):.3f}")
            print(f"  L1 Distance:               {results['metrics'].get('l1', 'N/A'):.3f}")
            print(f"  L2 Distance:               {results['metrics'].get('l2', 'N/A'):.3f}")
        else:
            print("METRICS: Failed to compute")
            
        print("-"*60)
        print("="*60)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Simple Image Edit Evaluator')
    parser.add_argument('--original', required=True,
                       help='Path to original image')
    parser.add_argument('--edited', required=True,
                       help='Path to edited image')
    parser.add_argument('--caption', required=True,
                       help='Caption describing the edit')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', default=None,
                       help='Path to save results JSON file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SimpleImageEditEvaluator(device=args.device)
    
    # Run evaluation
    results = evaluator.evaluate_with_analysis(
        original_path=args.original,
        edited_path=args.edited,
        caption=args.caption
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

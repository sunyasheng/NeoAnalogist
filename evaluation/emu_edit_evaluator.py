#!/usr/bin/env python3
"""
Emu-Edit Dataset Evaluator based on AnyBench

This module provides evaluation functionality for image editing tasks using the Emu-Edit test set.
It implements metrics like CLIP-I, CLIP-T, L1/L2 distance, and DINO score as used in AnyBench.

Usage:
    python emu_edit_evaluator.py --dataset emu_val --generated_path ./results --output_path ./eval_results
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
from datasets import load_dataset

# Import AnyBench utilities
import sys
sys.path.append('/Users/suny0a/Proj/ImageBrush/NeoAnalogist/thirdparty/AnySD/anybench')
from eval.utils import eval_distance, eval_clip_i, eval_clip_t, is_all_black

# Qwen API client removed - evaluation works independently


class EmuEditEvaluator:
    """Evaluator for Emu-Edit dataset using AnyBench metrics."""
    
    def __init__(self, device: str = None):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use for evaluation (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.clip_model = None
        self.clip_transform = None
        self.dino_model = None
        self.dino_transform = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler('emu_edit_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load CLIP and DINO models for evaluation."""
        self.logger.info("Loading evaluation models...")
        
        # Load CLIP model
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=self.device)
        self.logger.info("CLIP model loaded successfully")
        
        # Load DINO model
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.dino_model.eval()
        self.dino_model.to(self.device)
        self.dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.logger.info("DINO model loaded successfully")
        
            
    def load_emu_dataset(self, bench_name: str = 'emu_val'):
        """
        Load Emu-Edit dataset.
        
        Args:
            bench_name: Dataset split ('emu_test' or 'emu_val')
            
        Returns:
            Dataset object
        """
        self.logger.info(f"Loading {bench_name} dataset...")
        
        if bench_name == 'emu_test':
            # test: 2022 samples
            ds = load_dataset("facebook/emu_edit_test_set_generations")
            return ds['test']
        elif bench_name == 'emu_val':
            # val: 3589 samples
            ds = load_dataset("facebook/emu_edit_test_set_generations")
            return ds['validation']
        else:
            raise ValueError(f"Unknown benchmark named {bench_name}")
            
    def prepare_image_pairs(self, dataset, generated_path: str) -> List[Tuple[Image.Image, Image.Image, str]]:
        """
        Prepare image pairs for evaluation.
        
        Args:
            dataset: Emu-Edit dataset
            generated_path: Path to generated images
            
        Returns:
            List of (generated_image, ground_truth_image, caption) tuples
        """
        self.logger.info("Preparing image pairs for evaluation...")
        
        image_pairs = []
        skipped_count = 0
        
        for item in tqdm(dataset, desc="Loading image pairs"):
            generated_img_path = f"{generated_path}/{item['idx']}.png"
            
            # Skip if generated image doesn't exist or is all black
            if not os.path.exists(generated_img_path):
                self.logger.warning(f"Generated image not found: {generated_img_path}")
                skipped_count += 1
                continue
                
            if is_all_black(generated_img_path):
                self.logger.warning(f"Skipping black image: {generated_img_path}")
                skipped_count += 1
                continue
                
            try:
                generated_img = Image.open(generated_img_path).convert('RGB')
                gt_img = item['edited_image']
                caption = item['output_caption']
                
                image_pairs.append((generated_img, gt_img, caption))
                
            except Exception as e:
                self.logger.error(f"Error loading image pair {item['idx']}: {e}")
                skipped_count += 1
                
        self.logger.info(f"Prepared {len(image_pairs)} image pairs, skipped {skipped_count}")
        return image_pairs
        
    def evaluate_metrics(self, image_pairs: List[Tuple[Image.Image, Image.Image, str]]) -> Dict[str, float]:
        """
        Evaluate all metrics on image pairs.
        
        Args:
            image_pairs: List of (generated_image, ground_truth_image, caption) tuples
            
        Returns:
            Dictionary of metric scores
        """
        self.logger.info(f"Evaluating {len(image_pairs)} image pairs...")
        
        results = {}
        
        # CLIP-I Score (image similarity)
        self.logger.info("Computing CLIP-I score...")
        clip_i = eval_clip_i(
            image_pairs=image_pairs, 
            model=self.clip_model, 
            transform=self.clip_transform, 
            url_flag=False
        )
        results['clip_i'] = clip_i
        
        # CLIP-T Score (text-image alignment)
        self.logger.info("Computing CLIP-T score...")
        clip_t, _ = eval_clip_t(
            image_pairs=image_pairs, 
            model=self.clip_model, 
            transform=self.clip_transform, 
            url_flag=False
        )
        results['clip_t'] = clip_t
        
        # L1 Distance
        self.logger.info("Computing L1 distance...")
        l1 = eval_distance(
            image_pairs=image_pairs, 
            metric='l1', 
            url_flag=False
        )
        results['l1'] = l1
        
        # L2 Distance
        self.logger.info("Computing L2 distance...")
        l2 = eval_distance(
            image_pairs=image_pairs, 
            metric='l2', 
            url_flag=False
        )
        results['l2'] = l2
        
        # DINO Score
        self.logger.info("Computing DINO score...")
        dino_score = eval_clip_i(
            image_pairs=image_pairs, 
            model=self.dino_model, 
            transform=self.dino_transform, 
            url_flag=False, 
            metric='dino'
        )
        results['dino'] = dino_score
        
        return results
        
        
    def run_evaluation(self, dataset_name: str, generated_path: str, output_path: str) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            dataset_name: Name of dataset ('emu_test' or 'emu_val')
            generated_path: Path to generated images
            output_path: Path to save evaluation results
            
        Returns:
            Complete evaluation results
        """
        self.logger.info(f"Starting evaluation for {dataset_name}")
        
        # Load models
        self.load_models()
        
        # Load dataset
        dataset = self.load_emu_dataset(dataset_name)
        
        # Prepare image pairs
        image_pairs = self.prepare_image_pairs(dataset, generated_path)
        
        if not image_pairs:
            raise ValueError("No valid image pairs found for evaluation")
        
        # Evaluate metrics
        metrics = self.evaluate_metrics(image_pairs)
        
        # Compile results
        results = {
            'dataset': dataset_name,
            'total_samples': len(dataset),
            'valid_samples': len(image_pairs),
            'metrics': metrics,
            'timestamp': str(torch.datetime.now()) if hasattr(torch, 'datetime') else None
        }
        
        # Save results
        os.makedirs(output_path, exist_ok=True)
        results_file = os.path.join(output_path, f'evaluation_results_{dataset_name}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.logger.info(f"Evaluation results saved to {results_file}")
        
        # Print summary
        self.logger.info("="*50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Valid samples: {len(image_pairs)}/{len(dataset)}")
        self.logger.info(f"CLIP-I: {metrics['clip_i']:.3f}")
        self.logger.info(f"CLIP-T: {metrics['clip_t']:.3f}")
        self.logger.info(f"DINO: {metrics['dino']:.3f}")
        self.logger.info(f"L1: {metrics['l1']:.3f}")
        self.logger.info(f"L2: {metrics['l2']:.3f}")
        self.logger.info("="*50)
        
        return results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Emu-Edit Dataset Evaluator')
    parser.add_argument('--dataset', choices=['emu_test', 'emu_val'], default='emu_val',
                       help='Dataset to evaluate')
    parser.add_argument('--generated_path', required=True,
                       help='Path to generated images')
    parser.add_argument('--output_path', required=True,
                       help='Path to save evaluation results')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = EmuEditEvaluator(device=args.device)
    
    # Run evaluation
    results = evaluator.run_evaluation(
        dataset_name=args.dataset,
        generated_path=args.generated_path,
        output_path=args.output_path
    )
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()

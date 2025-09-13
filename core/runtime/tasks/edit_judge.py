"""
Image Edit Judge Task

Image editing quality evaluation task using AnyBench metrics.
Input: original image path, edited image path, edit instruction
Output: quality metrics + reasoning and suggestions
"""

import logging
import time
import os
import torch
import clip
from PIL import Image
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import evaluation functions from anybench
from thirdparty.AnySD.anybench.eval.utils import eval_clip_i, eval_clip_t, eval_distance, is_all_black

logger = logging.getLogger(__name__)

@dataclass
class ImageJudgeResult:
    """Image judge result."""
    clip_i: float
    clip_t: float
    l1_distance: float
    l2_distance: float
    overall_score: float
    suggestions: List[str]
    status: str = "success"

class ImageEditJudgeTask:
    """Task for judging image editing quality using AnyBench metrics."""
    
    def __init__(self, device: Optional[str] = None, clip_model_type: str = "ViT-B/32"):
        """
        Initialize image edit judge task.
        
        Args:
            device: Device to run on (auto-detect if None)
            clip_model_type: CLIP model type to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_type = clip_model_type
        self.start_time = time.time()
        
        # Load CLIP model
        self.clip_model = None
        self.clip_transform = None
        self._load_clip_model()
    
    def _load_clip_model(self):
        """Load CLIP model for evaluation."""
        try:
            self.clip_model, self.clip_transform = clip.load(
                self.clip_model_type, 
                device=self.device
            )
            logger.info(f"Loaded CLIP model: {self.clip_model_type} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def _check_inputs(self, original_path: str, edited_path: str) -> tuple[bool, str]:
        """Check if input files exist and are valid."""
        if not os.path.exists(original_path):
            return False, f"Original image not found: {original_path}"
        
        if not os.path.exists(edited_path):
            return False, f"Edited image not found: {edited_path}"
        
        if is_all_black(edited_path):
            return False, "Edited image is all black (generation failed)"
        
        return True, ""
    
    def _calculate_metrics(self, original_path: str, edited_path: str, input_caption: str, output_caption: str) -> Dict[str, float]:
        """Calculate evaluation metrics using AnyBench functions."""
        # Create image pairs for evaluation functions (AnySD format: [ground_truth_image, generated_image, caption])
        # Load images as PIL objects to match AnySD's url_flag=False usage
        original_img = Image.open(original_path).convert('RGB')
        edited_img = Image.open(edited_path).convert('RGB')
        image_pairs = [[original_img, edited_img, output_caption]]
        
        # Calculate CLIP-I (image similarity)
        clip_i_score = eval_clip_i(
            args=None, 
            image_pairs=image_pairs, 
            model=self.clip_model, 
            transform=self.clip_transform,
            url_flag=False, 
            skip_flag=False
        )
        
        # Calculate CLIP-T (text-image alignment) using AnySD's method
        clip_t_score, _ = eval_clip_t(
            args=None, 
            image_pairs=image_pairs, 
            model=self.clip_model, 
            transform=self.clip_transform,
            url_flag=False, 
            caption_dict=None,  # Not needed when url_flag=False
            skip_flag=False
        )
        
        # Calculate L1 and L2 distances
        l1_distance = eval_distance(image_pairs, metric='l1', url_flag=False, skip_flag=False)
        l2_distance = eval_distance(image_pairs, metric='l2', url_flag=False, skip_flag=False)
        
        return {
            "clip_i": clip_i_score,
            "clip_t": clip_t_score,
            "l1_distance": l1_distance,
            "l2_distance": l2_distance
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        # Weighted combination: CLIP-I(40%) + CLIP-T(40%) + L1(20%)
        # Normalize L1 distance (lower is better)
        normalized_l1 = max(0, 1 - metrics["l1_distance"])
        
        overall = (
            metrics["clip_i"] * 0.4 +
            metrics["clip_t"] * 0.4 +
            normalized_l1 * 0.2
        )
        
        return overall
    
    def _generate_suggestions(self, metrics: Dict[str, float], overall_score: float) -> List[str]:
        """Generate improvement suggestions based on metrics."""
        suggestions = []
        
        # CLIP-I suggestions
        if metrics["clip_i"] < 0.7:
            suggestions.append("Low image similarity, suggest preserving more original structure")
        elif metrics["clip_i"] > 0.9:
            suggestions.append("Very high image similarity, edit might be too conservative")
        
        # CLIP-T suggestions
        if metrics["clip_t"] < 0.2:
            suggestions.append("Poor text alignment, suggest adjusting edit instruction or regenerating")
        elif metrics["clip_t"] > 0.4:
            suggestions.append("Good text alignment, edit effect meets expectations")
        
        # L1 distance suggestions
        if metrics["l1_distance"] > 0.4:
            suggestions.append("Large pixel differences, suggest more conservative editing")
        elif metrics["l1_distance"] < 0.1:
            suggestions.append("Small pixel differences, edit effect is very natural")
        
        # Overall score suggestions
        if overall_score >= 0.7:
            suggestions.append("Excellent overall quality!")
        elif overall_score >= 0.5:
            suggestions.append("Good overall quality, can be further optimized")
        elif overall_score >= 0.3:
            suggestions.append("Medium overall quality, suggest adjusting edit strategy")
        else:
            suggestions.append("Poor overall quality, suggest re-editing")
        
        if not suggestions:
            suggestions.append("All metrics normal, edit quality is good")
        
        return suggestions
    
    async def _analyze_with_qwen(self, original_path: str, edited_path: str, input_caption: str, output_caption: str) -> Optional[str]:
        """Analyze image editing quality using Qwen API."""
        try:
            # Import Qwen API client
            from core.runtime.tasks.qwen_api import QwenAPIClient
            
            # Create analysis prompt
            analysis_prompt = f"""
Please analyze the quality of this image editing task:

Original image description: "{input_caption}"
Expected edited image description: "{output_caption}"

Compare the original and edited images. Please evaluate:
1. Does the edited image successfully implement the requested changes?
2. Are there any artifacts or quality issues in the edited image?
3. How well does the edited image match the expected description?
4. Any specific suggestions for improvement?

Please provide a concise analysis (2-3 sentences) focusing on the most important aspects.
"""

            # Use edited image for analysis (the result we want to evaluate)
            qwen_client = QwenAPIClient(base_url="http://10.64.74.69:8200")
            result = qwen_client.generate(
                prompt=analysis_prompt,
                image_path=edited_path,
                max_new_tokens=150,
                temperature=0.3
            )
            
            if result and result.get('content'):
                return result['content'].strip()
            else:
                logger.warning("Qwen API analysis failed or returned empty result")
                return None
                
        except Exception as e:
            logger.error(f"Error in Qwen API analysis: {str(e)}")
            return None
    
    async def run(self, original_path: str, edited_path: str, input_caption: str, output_caption: str, use_qwen_analysis: bool = True) -> ImageJudgeResult:
        """
        Run image editing quality evaluation.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image
            input_caption: Description of original image
            output_caption: Description of edited image
            use_qwen_analysis: Whether to use Qwen API for intelligent analysis
            
        Returns:
            ImageJudgeResult with evaluation results
        """
        logger.info(f"Starting image edit evaluation: {original_path} -> {edited_path}")
        
        try:
            # Check inputs
            is_valid, error_msg = self._check_inputs(original_path, edited_path)
            if not is_valid:
                return ImageJudgeResult(
                    clip_i=0.0,
                    clip_t=0.0,
                    l1_distance=1.0,
                    l2_distance=1.0,
                    overall_score=0.0,
                    suggestions=[f"Input error: {error_msg}"],
                    status="error"
                )
            
            # Calculate metrics
            metrics = self._calculate_metrics(original_path, edited_path, input_caption, output_caption)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(metrics, overall_score)
            
            # Add Qwen API analysis if enabled
            if use_qwen_analysis:
                qwen_suggestion = await self._analyze_with_qwen(original_path, edited_path, input_caption, output_caption)
                if qwen_suggestion:
                    suggestions.append(f"AI Analysis: {qwen_suggestion}")
            
            execution_time = time.time() - self.start_time
            logger.info(f"Image edit evaluation completed in {execution_time:.2f}s")
            
            return ImageJudgeResult(
                clip_i=round(metrics["clip_i"], 3),
                clip_t=round(metrics["clip_t"], 3),
                l1_distance=round(metrics["l1_distance"], 3),
                l2_distance=round(metrics["l2_distance"], 3),
                overall_score=round(overall_score, 3),
                suggestions=suggestions,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error in image edit evaluation: {str(e)}")
            return ImageJudgeResult(
                clip_i=0.0,
                clip_t=0.0,
                l1_distance=1.0,
                l2_distance=1.0,
                overall_score=0.0,
                suggestions=[f"Evaluation error: {str(e)}"],
                status="error"
            )
    
    def get_feedback_summary(self, result: ImageJudgeResult) -> str:
        """Generate feedback summary for agent framework."""
        if result.status == "error":
            return f"❌ Image evaluation failed: {result.suggestions[0]}"
        
        feedback_lines = [
            "=== IMAGE EDIT QUALITY ASSESSMENT ===",
            f"CLIP-I (Image Similarity): {result.clip_i:.3f}",
            f"CLIP-T (Text Alignment): {result.clip_t:.3f}",
            f"L1 Distance: {result.l1_distance:.3f}",
            f"L2 Distance: {result.l2_distance:.3f}",
            f"Overall Score: {result.overall_score:.3f}",
            "",
            "RECOMMENDATIONS:"
        ]
        
        for i, suggestion in enumerate(result.suggestions, 1):
            feedback_lines.append(f"{i}. {suggestion}")
        
        feedback_lines.extend([
            "",
            f"Status: {'PASS' if result.overall_score >= 0.5 else 'FAIL'}",
            "================================"
        ])
        
        return "\n".join(feedback_lines)


# Convenience function
async def judge_image_edit(original_path: str, edited_path: str, prompt: str) -> ImageJudgeResult:
    """
    Convenience function for image edit evaluation.
    
    Args:
        original_path: Path to original image
        edited_path: Path to edited image
        prompt: Text prompt used for editing
        
    Returns:
        ImageJudgeResult with evaluation results
    """
    task = ImageEditJudgeTask()
    return await task.run(original_path, edited_path, prompt)


def print_results(result: ImageJudgeResult):
    """Print evaluation results."""
    print(result.get_feedback_summary(result))


if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python image_edit_judge.py <original_path> <edited_path> <prompt>")
        print("Example: python image_edit_judge.py original.jpg edited.jpg 'make the sky more dramatic'")
        sys.exit(1)
    
    original_path = sys.argv[1]
    edited_path = sys.argv[2]
    prompt = sys.argv[3]
    
    async def main():
        result = await judge_image_edit(original_path, edited_path, prompt)
        print_results(result)
    
    asyncio.run(main())

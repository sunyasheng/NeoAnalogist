"""
Image Edit Judge Task - Simplified Version

Simple image editing evaluation task.
Input: original image path, edited image path, edit instruction
Output: correctness assessment and feedback
"""

import logging
import base64
import json
from PIL import Image
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class EditJudgeResult:
    """Result of image edit evaluation."""
    is_correct: bool
    score: float  # 0-10 scale
    feedback: str  # What's wrong or what's good
    reasoning: str  # Detailed explanation

def encode_image_to_data_url(path: str) -> str:
    """Encode image to base64 data URL."""
    mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_eval_prompt(instruction: str) -> str:
    """Build evaluation prompt for GPT-4o."""
    return f"""You are an expert image editor and quality assessor. Your task is to evaluate whether an image edit was executed correctly.

**Task**: Compare the original image and the edited image to determine if the edit instruction was followed correctly.

**Edit Instruction**: {instruction}

**Analysis Steps**:
1. **Analyze Original Image**: Identify all objects, their properties, and relationships
2. **Analyze Edited Image**: Identify what changed and what remained the same
3. **Compare with Instruction**: Check if the edit matches the instruction requirements
4. **Identify Errors**: If incorrect, specify exactly what went wrong

**Evaluation Criteria**:
1. **Correctness**: Does the edited image show the requested changes as specified?
2. **Quality**: Is the edit well-executed (realistic, properly integrated)?
3. **Completeness**: Are all parts of the instruction addressed?
4. **Accuracy**: Was the edit applied correctly according to the instruction?

**Output Format**: Return a JSON object with exactly these fields:
{{
    "is_correct": true/false,
    "score": 0-10,
    "feedback": "Brief description of what's wrong or what's good",
    "reasoning": "Detailed explanation including: 1) What was in the original image, 2) What changed in the edited image, 3) How the edit compares to the instruction, 4) Specific errors if any"
}}

**Scoring Guide**:
- 9-10: Perfect execution, exactly matches instruction
- 7-8: Good execution with minor issues
- 5-6: Partial success, some issues
- 3-4: Poor execution, major problems
- 1-2: Very poor, barely follows instruction
- 0: Complete failure, doesn't follow instruction at all

**Important Analysis Points**:
- Pay attention to all aspects of the instruction: objects, locations, colors, directions, relationships, etc.
- If the instruction specifies certain details, check if they were followed correctly
- Be specific about what was wrong: identify the exact discrepancy between instruction and result
- Consider both what was changed and what should have been changed
- **Critical**: If the instruction targets a specific object/location, check if the edit was applied to the RIGHT target
- **Critical**: Identify what should have been left unchanged but was modified instead

**Example of Good Reasoning**:
"Original image shows [describe original with all objects and their locations]. The instruction was to [restate instruction with specific target]. However, the edited image shows [describe what actually happened]. This is incorrect because [specific reason why it doesn't match the instruction]. The edit was applied to [wrong target] instead of [correct target], and [what should have been left unchanged] was modified when it should have remained the same."
"""

def call_gpt4o_eval(prompt: str, image_paths: list[str], model: str = "gpt-4o-mini", 
                   max_tokens: int = 1000, temperature: float = 0.1) -> Dict[str, Any]:
    """Call GPT-4o for image editing evaluation."""
    load_dotenv()
    client = OpenAI()
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_to_data_url(image_paths[0])}
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": encode_image_to_data_url(image_paths[1])}
                }
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # If not JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Fallback: create a structured response
                return {
                    "is_correct": False,
                    "score": 0,
                    "feedback": "Failed to parse GPT response",
                    "reasoning": content
                }
                
    except Exception as e:
        logger.error(f"GPT-4o evaluation failed: {str(e)}")
        return {
            "is_correct": False,
            "score": 0,
            "feedback": f"Evaluation failed: {str(e)}",
            "reasoning": "Error occurred during evaluation"
        }

class ImageEditJudgeTask:
    """Simplified task for judging image editing quality."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize image edit judge task.
        
        Args:
            model: GPT model to use for evaluation
        """
        self.model = model
        logger.info(f"Initialized ImageEditJudgeTask with model: {model}")
    
    def evaluate(self, original_path: str, edited_path: str, instruction: str) -> EditJudgeResult:
        """
        Evaluate if the image edit was executed correctly.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image  
            instruction: The edit instruction that was given
            
        Returns:
            EditJudgeResult with evaluation details
        """
        try:
            # Build evaluation prompt
            prompt = build_eval_prompt(instruction)
            
            # Call GPT-4o for evaluation
            result = call_gpt4o_eval(
                prompt=prompt,
                image_paths=[original_path, edited_path],
                model=self.model
            )
            
            # Create result object
            return EditJudgeResult(
                is_correct=result.get("is_correct", False),
                score=float(result.get("score", 0)),
                feedback=result.get("feedback", "No feedback provided"),
                reasoning=result.get("reasoning", "No reasoning provided")
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return EditJudgeResult(
                is_correct=False,
                score=0.0,
                feedback=f"Evaluation error: {str(e)}",
                reasoning="An error occurred during evaluation"
            )

# Convenience function for direct usage
def judge_edit(original_path: str, edited_path: str, instruction: str, 
               model: str = "gpt-4o-mini") -> EditJudgeResult:
    """
    Convenience function to judge an image edit.
    
    Args:
        original_path: Path to original image
        edited_path: Path to edited image
        instruction: The edit instruction
        model: GPT model to use
        
    Returns:
        EditJudgeResult with evaluation
    """
    judge = ImageEditJudgeTask(model=model)
    return judge.evaluate(original_path, edited_path, instruction)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python edit_judge.py <original_image> <edited_image> <instruction>")
        sys.exit(1)
    
    original_path = sys.argv[1]
    edited_path = sys.argv[2] 
    instruction = sys.argv[3]
    
    result = judge_edit(original_path, edited_path, instruction)
    
    print(f"Correct: {result.is_correct}")
    print(f"Score: {result.score}/10")
    print(f"Feedback: {result.feedback}")
    print(f"Reasoning: {result.reasoning}")
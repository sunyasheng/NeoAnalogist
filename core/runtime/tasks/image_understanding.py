"""
Image Understanding Task

Analyze images with masks, bounding boxes, and labels to provide comprehensive image understanding.
Input: image path, masks (optional), bounding boxes (optional), labels (optional)
Output: detailed image analysis and understanding
"""

import logging
import base64
import json
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ImageUnderstandingResult:
    """Result of image understanding analysis."""
    success: bool
    image_description: str  # Overall description of the image
    object_analysis: List[Dict[str, Any]]  # Analysis of each detected object
    spatial_relationships: List[str]  # Spatial relationships between objects
    scene_context: str  # Scene context and setting
    visual_elements: Dict[str, Any]  # Colors, lighting, composition, etc.
    error_message: str = ""

def encode_image_to_data_url(path: str) -> str:
    """Encode image to base64 data URL."""
    mime = "image/jpeg"
    if path.lower().endswith(('.png', '.PNG')):
        mime = "image/png"
    elif path.lower().endswith(('.gif', '.GIF')):
        mime = "image/gif"
    elif path.lower().endswith(('.webp', '.WEBP')):
        mime = "image/webp"
    
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime};base64,{encoded_string}"

def load_mask_as_image(mask_path: str) -> Optional[Image.Image]:
    """Load mask image and return as PIL Image."""
    try:
        if os.path.exists(mask_path):
            return Image.open(mask_path).convert("L")  # Convert to grayscale
        return None
    except Exception as e:
        logger.warning(f"Failed to load mask {mask_path}: {e}")
        return None

def create_visualization(image_path: str, boxes: List[List[float]], labels: List[str], masks: List[str] = None) -> str:
    """Create a visualization of the image with boxes, labels, and masks."""
    try:
        # Load original image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Draw label
                draw.text((x1, y1-20), f"{label}", fill="red", font=font)
        
        # Overlay masks if provided
        if masks:
            for i, mask_path in enumerate(masks):
                mask_img = load_mask_as_image(mask_path)
                if mask_img:
                    # Create colored overlay for mask
                    mask_array = np.array(mask_img)
                    if mask_array.max() > 0:  # If mask is not empty
                        # Create colored mask
                        colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
                        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][i % 6]
                        colored_mask[mask_array > 0] = color
                        
                        # Convert back to PIL and blend
                        colored_mask_img = Image.fromarray(colored_mask)
                        img = Image.blend(img, colored_mask_img, 0.3)
        
        # Save visualization
        viz_path = image_path.replace('.png', '_understanding_viz.png').replace('.jpg', '_understanding_viz.jpg')
        img.save(viz_path)
        return viz_path
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        return ""

class ImageUnderstandingTask:
    """Task for comprehensive image understanding analysis."""
    
    def __init__(self, runtime: Any):
        self.runtime = runtime
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        logger.info("Initialized ImageUnderstandingTask with OpenAI client")

    def run(self, action) -> ImageUnderstandingResult:
        """Run image understanding analysis."""
        try:
            # Extract parameters from action
            image_path = action.image_path
            boxes = getattr(action, 'boxes', [])
            labels = getattr(action, 'labels', [])
            masks = getattr(action, 'masks', [])
            
            if not image_path or not os.path.exists(image_path):
                return ImageUnderstandingResult(
                    success=False,
                    image_description="",
                    object_analysis=[],
                    spatial_relationships=[],
                    scene_context="",
                    visual_elements={},
                    error_message="Image path not provided or file does not exist"
                )
            
            # Create visualization if we have boxes/labels/masks
            viz_path = ""
            if boxes and labels:
                viz_path = create_visualization(image_path, boxes, labels, masks)
            
            # Encode images for analysis
            image_data_url = encode_image_to_data_url(image_path)
            
            # Prepare analysis prompt
            analysis_prompt = self._build_analysis_prompt(boxes, labels, masks)
            
            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert image analyst. Analyze the provided image comprehensively, considering:
1. Overall scene description
2. Individual object analysis (if bounding boxes/labels provided)
3. Spatial relationships between objects
4. Scene context and setting
5. Visual elements (colors, lighting, composition, etc.)

Provide detailed, accurate analysis in JSON format."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to text parsing
            try:
                analysis_data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: parse the text response
                analysis_data = self._parse_text_response(content)
            
            # Create result
            result = ImageUnderstandingResult(
                success=True,
                image_description=analysis_data.get("image_description", ""),
                object_analysis=analysis_data.get("object_analysis", []),
                spatial_relationships=analysis_data.get("spatial_relationships", []),
                scene_context=analysis_data.get("scene_context", ""),
                visual_elements=analysis_data.get("visual_elements", {}),
                error_message=""
            )
            
            # Add visualization path if created
            if viz_path:
                result.visual_elements["visualization_path"] = viz_path
            
            logger.info(f"Image understanding analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in image understanding analysis: {str(e)}")
            return ImageUnderstandingResult(
                success=False,
                image_description="",
                object_analysis=[],
                spatial_relationships=[],
                scene_context="",
                visual_elements={},
                error_message=f"Analysis failed: {str(e)}"
            )

    def _build_analysis_prompt(self, boxes: List[List[float]], labels: List[str], masks: List[str]) -> str:
        """Build the analysis prompt based on available data."""
        prompt = "Please analyze this image comprehensively. "
        
        if boxes and labels:
            prompt += f"\n\nDetected objects with bounding boxes and labels:\n"
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    prompt += f"- Object {i+1}: '{label}' at position ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})\n"
        
        if masks:
            prompt += f"\n\nSegmentation masks are available for {len(masks)} objects.\n"
        
        prompt += """
Please provide your analysis in the following JSON format:
{
    "image_description": "Overall description of what's in the image",
    "object_analysis": [
        {
            "object_id": 1,
            "label": "object name",
            "description": "detailed description of this object",
            "attributes": ["attribute1", "attribute2"],
            "position": "spatial position description",
            "interactions": "how this object interacts with others"
        }
    ],
    "spatial_relationships": [
        "relationship description 1",
        "relationship description 2"
    ],
    "scene_context": "overall scene setting and context",
    "visual_elements": {
        "colors": "dominant colors and color scheme",
        "lighting": "lighting conditions and mood",
        "composition": "compositional elements and layout",
        "style": "artistic style or photographic style"
    }
}"""
        
        return prompt

    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        return {
            "image_description": content[:500] + "..." if len(content) > 500 else content,
            "object_analysis": [],
            "spatial_relationships": [],
            "scene_context": "",
            "visual_elements": {
                "raw_analysis": content
            }
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from core.events.event import Observation


@dataclass
class ImageEntityExtractObservation(Observation):
    """Entities extracted from an image.

    entities: list of {label:str, score:float, bbox:[x,y,w,h], props:dict}
    image_size: (width, height)
    """

    content: str = ""
    entities: List[Dict] = None  # type: ignore[assignment]
    image_size: Optional[Tuple[int, int]] = None
    model: str = ""
    time_ms: int = 0
    observation: str = "image_entity_extract"


@dataclass
class ImageEditJudgeObservation(Observation):
    """Image editing quality evaluation results.
    
    Contains correctness assessment, score, feedback and reasoning.
    """

    content: str = ""
    is_correct: bool = False
    score: float = 0.0
    feedback: str = ""
    reasoning: str = ""
    status: str = "success"
    execution_time: float = 0.0
    error_message: str = ""
    observation: str = "image_edit_judge"



@dataclass
class GroundingSAMObservation(Observation):
    """Observation for GroundingSAM segmentation operation."""

    content: str = ""
    num_instances: int = 0
    mask_paths: List[str] = None
    boxes: List[List[float]] = None  # Bounding boxes [[x1, y1, x2, y2], ...]
    labels: List[str] = None  # Object labels
    success: bool = False
    error_message: str = ""
    observation: str = "GROUNDING_SAM"
    # When using streaming response (PNG), we return base64 data URL of the first mask
    image_b64: str = ""

    def __post_init__(self):
        if self.mask_paths is None:
            self.mask_paths = []
        if self.boxes is None:
            self.boxes = []
        if self.labels is None:
            self.labels = []


@dataclass
class InpaintRemoveObservation(Observation):
    """Observation for Inpaint-Anything remove operation."""
    
    content: str = ""
    num_masks: int = 0
    mask_paths: List[str] = None
    result_paths: List[str] = None
    success: bool = False
    error_message: str = ""
    observation: str = "inpaint_remove"
    
    def __post_init__(self):
        if self.mask_paths is None:
            self.mask_paths = []
        if self.result_paths is None:
            self.result_paths = []


@dataclass
class SDXLInpaintObservation(Observation):
    """Observation for SDXL inpainting results."""
    
    content: str = ""
    prompt: str = ""
    output_path: str = ""
    parameters: Optional[dict] = None
    success: bool = True
    error_message: str = ""
    observation: str = "SDXL_INPAINT"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class LAMARemoveObservation(Observation):
    """Observation for LAMA object removal results."""
    
    content: str = ""
    output_path: str = ""
    dilate_kernel_size: int = 0
    success: bool = True
    error_message: str = ""
    observation: str = "LAMA_REMOVE"


@dataclass
class GroundingDINOObservation(Observation):
    """Observation for GroundingDINO detection results."""
    
    content: str = ""
    num_detections: int = 0
    detections: list = None  # List of detection dicts
    output_path: str = ""  # Path to annotated image if return_type=image
    success: bool = True
    error_message: str = ""
    observation: str = "GROUNDING_DINO_DETECT"
    
    def __post_init__(self):
        if self.detections is None:
            self.detections = []


@dataclass
class ImageUnderstandingObservation(Observation):
    """Observation for image understanding analysis results."""
    
    content: str = ""
    image_description: str = ""
    object_analysis: list = None  # List of object analysis dicts
    spatial_relationships: list = None  # List of spatial relationship strings
    scene_context: str = ""
    visual_elements: dict = None  # Dict of visual elements
    success: bool = True
    error_message: str = ""
    observation: str = "IMAGE_UNDERSTANDING"
    
    def __post_init__(self):
        if self.object_analysis is None:
            self.object_analysis = []
        if self.spatial_relationships is None:
            self.spatial_relationships = []
        if self.visual_elements is None:
            self.visual_elements = {}



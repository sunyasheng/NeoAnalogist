from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from core.events.event import Action


@dataclass
class ImageEntityExtractAction(Action):
    """Extract entities/objects from an image.

    Provide either image_path (preferred, path inside container) or image_bytes (base64 string).
    """

    image_path: Optional[str] = None
    image_bytes: Optional[str] = None
    model: str = "gpt-4o"
    thought: str = ""

    # action routing key used by server/client
    action: str = "image_entity_extract"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        src = self.image_path or ("<bytes>" if self.image_bytes else "<none>")
        return f"Extract entities from image: {src} (model={self.model})"



@dataclass
class GoTEditAction(Action):
    """Edit an image using GoT API.

    image_path should be an absolute path inside container or accessible path on host.
    """

    image_path: Optional[str] = None
    prompt: str = ""
    mode: str = "t2i"  # "t2i" for text-to-image, "edit" for image editing
    height: int = 1024
    width: int = 1024
    max_new_tokens: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.0
    cond_image_guidance_scale: float = 4.0
    output_path: str = ""
    thought: str = ""

    action: str = "got_edit"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        mode_desc = "text-to-image" if self.mode == "t2i" else "image editing"
        if self.mode == "edit" and self.image_path:
            return f"GoT {mode_desc}: {self.prompt} on {self.image_path} ({self.width}x{self.height})"
        else:
            return f"GoT {mode_desc}: {self.prompt} ({self.width}x{self.height})"


@dataclass
class QwenAPIAction(Action):
    """Analyze images using Qwen2.5-VL API."""

    prompt: str = ""
    image_path: str = ""
    mode: str = "generate"  # "generate" for single request, "chat" for structured chat
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    messages: Optional[str] = None  # JSON string for chat mode
    thought: str = ""

    action: str = "qwen_api"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        if self.mode == "chat":
            return f"Qwen chat API: {len(self.messages or '')} characters"
        elif self.image_path:
            return f"Qwen API: {self.prompt} with image {self.image_path}"
        else:
            return f"Qwen API: {self.prompt}"


@dataclass
class AnyDoorEditAction(Action):
    """Edit image using AnyDoor API (reference object transfer).

    All paths should be absolute container paths when called inside DockerRuntime server.
    """

    ref_image_path: str = ""
    target_image_path: str = ""
    target_mask_path: str = ""
    ref_mask_path: Optional[str] = None
    guidance_scale: float = 5.0
    output_path: str = ""
    thought: str = ""

    action: str = "anydoor_edit"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return (
            f"AnyDoor edit: ref={self.ref_image_path} -> target={self.target_image_path}"
        )

@dataclass
class GroundingSAMAction(Action):
    """Text-prompted segmentation using GroundingSAM service.
    
    All paths should be absolute container paths when called inside DockerRuntime server.
    """

    image_path: str = ""
    text_prompt: str = ""
    # Optional knobs (currently handled on server side / API defaults)
    return_type: str = "image"  # image | json (prefer streaming PNG; server defaults to image)
    output_dir: Optional[str] = None
    thought: str = ""

    action: str = "grounding_sam"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"GroundingSAM segment: {self.text_prompt} on {self.image_path}"

@dataclass
class ImageEditJudgeAction(Action):
    """Judge image editing quality using AnyBench metrics.
    
    Evaluates the quality of image editing by comparing original and edited images
    using CLIP-I, CLIP-T, L1/L2 distances and providing suggestions.
    """

    original_path: str = ""
    edited_path: str = ""
    input_caption: str = ""
    output_caption: str = ""
    use_qwen_analysis: bool = True  # Whether to use Qwen API for intelligent analysis
    thought: str = ""

    action: str = "image_edit_judge"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Judge image edit quality: {self.original_path} -> {self.edited_path} (input: {self.input_caption[:30]}... -> output: {self.output_caption[:30]}...)"


@dataclass
class InpaintRemoveAction(Action):
    """Remove objects from image using Inpaint-Anything.

    All paths should be absolute container paths when called inside DockerRuntime server.
    """
    image_path: str = ""
    point_coords: Optional[str] = None  # "x,y" format
    point_labels: str = "1"  # "1" for foreground, "0" for background
    mask_path: Optional[str] = None  # Optional mask image path
    dilate_kernel_size: int = 10
    return_type: str = "json"  # "image" (stream first result) | "json" (return paths)
    output_dir: Optional[str] = None  # Container path to save results if return_type is json
    thought: str = ""

    action: str = "inpaint_remove"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        if self.mask_path:
            return f"Inpaint remove with mask: {self.mask_path} on {self.image_path}"
        elif self.point_coords:
            return f"Inpaint remove at point: {self.point_coords} on {self.image_path}"
        else:
            return f"Inpaint remove on {self.image_path}"


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
class ImageEditJudgeAction(Action):
    """Judge image editing quality using AnyBench metrics.
    
    Evaluates the quality of image editing by comparing original and edited images
    using CLIP-I, CLIP-T, L1/L2 distances and providing suggestions.
    """

    original_path: str = ""
    edited_path: str = ""
    input_caption: str = ""
    output_caption: str = ""
    thought: str = ""

    action: str = "image_edit_judge"
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Judge image edit quality: {self.original_path} -> {self.edited_path} (input: {self.input_caption[:30]}... -> output: {self.output_caption[:30]}...)"


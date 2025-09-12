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


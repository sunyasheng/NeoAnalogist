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



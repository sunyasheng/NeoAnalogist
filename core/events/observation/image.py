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



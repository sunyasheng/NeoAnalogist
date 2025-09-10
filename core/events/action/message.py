from dataclasses import dataclass

from core.events.event import Action, ActionType


@dataclass
class MessageAction(Action):
    content: str
    image_urls: list[str] | None = None
    wait_for_response: bool = False
    action: str = ActionType.MESSAGE

    @property
    def message(self) -> str:
        return self.content

    @property
    def images_urls(self):
        # Deprecated alias for backward compatibility
        return self.image_urls

    @images_urls.setter
    def images_urls(self, value):
        self.image_urls = value

    def __str__(self) -> str:
        ret = f"**MessageAction** (source={self.source})\n"
        ret += f"CONTENT: {self.content}"
        if self.image_urls:
            for url in self.image_urls:
                ret += f"\nIMAGE_URL: {url}"
        return ret

from dataclasses import dataclass, field
from typing import Any

from browsergym.utils.obs import flatten_axtree_to_str

from core.events.event import ActionType, Observation, ObservationType


@dataclass
class BrowserOutputObservation(Observation):
    """This data class represents the output of a browser."""

    url: str
    trigger_by_action: str
    screenshot: str = field(repr=False, default="")  # don't show in repr
    screenshot_path: str | None = field(default=None)  # path to saved screenshot file
    set_of_marks: str = field(default="", repr=False)  # don't show in repr
    error: bool = False
    observation: str = ObservationType.BROWSE
    goal_image_urls: list[str] = field(default_factory=list)
    # do not include in the memory
    open_pages_urls: list[str] = field(default_factory=list)
    active_page_index: int = -1
    dom_object: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    axtree_object: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    extra_element_properties: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    last_browser_action: str = ""
    last_browser_action_error: str = ""
    focused_element_bid: str = ""

    @property
    def message(self) -> str:
        return "Visited " + self.url

    def __str__(self) -> str:
        ret = (
            "**BrowserOutputObservation**\n"
            f"URL: {self.url}\n"
            f"Error: {self.error}\n"
            f"Open pages: {self.open_pages_urls}\n"
            f"Active page index: {self.active_page_index}\n"
            f"Last browser action: {self.last_browser_action}\n"
            f"Last browser action error: {self.last_browser_action_error}\n"
            f"Focused element bid: {self.focused_element_bid}\n"
        )
        if self.screenshot_path:
            ret += f"Screenshot saved to: {self.screenshot_path}\n"
        ret += "--- Agent Observation ---\n"
        ret += self.get_agent_obs_text()
        return ret

    def get_agent_obs_text(self) -> str:
        """Get a concise text that will be shown to the agent."""
        if self.trigger_by_action == ActionType.BROWSE_INTERACTIVE:
            text = f"[Current URL: {self.url}]\n"
            text += f"[Focused element bid: {self.focused_element_bid}]\n"

            # Add screenshot path information if available
            if self.screenshot_path:
                text += f"[Screenshot saved to: {self.screenshot_path}]\n"

            text += "\n"

            if self.error:
                text += (
                    "================ BEGIN error message ===============\n"
                    "The following error occurred when executing the last action:\n"
                    f"{self.last_browser_action_error}\n"
                    "================ END error message ===============\n"
                )
            else:
                text += "[Action executed successfully.]\n"
            try:
                # We do not filter visible only here because we want to show the full content
                # of the web page to the agent for simplicity.
                # FIXME: handle the case when the web page is too large
                cur_axtree_txt = self.get_axtree_str(filter_visible_only=False)
                text += (
                    f"============== BEGIN accessibility tree ==============\n"
                    f"{cur_axtree_txt}\n"
                    f"============== END accessibility tree ==============\n"
                )
            except Exception as e:
                text += (
                    f"\n[Error encountered when processing the accessibility tree: {e}]"
                )
            return text

        elif self.trigger_by_action == ActionType.BROWSE:
            text = f"[Current URL: {self.url}]\n"

            if self.error:
                text += (
                    "================ BEGIN error message ===============\n"
                    "The following error occurred when trying to visit the URL:\n"
                    f"{self.last_browser_action_error}\n"
                    "================ END error message ===============\n"
                )
            text += "============== BEGIN webpage content ==============\n"
            text += self._truncate_large_string(self.content)
            text += "\n============== END webpage content ==============\n"
            return text
        else:
            raise ValueError(f"Invalid trigger_by_action: {self.trigger_by_action}")

    def get_axtree_str(self, filter_visible_only: bool = False) -> str:
        cur_axtree_txt = flatten_axtree_to_str(
            self.axtree_object,
            extra_properties=self.extra_element_properties,
            with_clickable=True,
            skip_generic=False,
            filter_visible_only=filter_visible_only,
        )
        # Apply additional filtering for accessibility tree
        filtered_text = self._filter_accessibility_tree(str(cur_axtree_txt))
        return self._truncate_large_string(filtered_text)

    def _filter_accessibility_tree(self, text: str) -> str:
        """Filter accessibility tree to reduce noise and keep important elements."""
        lines = text.split('\n')
        filtered_lines = []
        
        # Keep only the most important elements
        for line in lines:
            # Keep lines with important interactive content
            if any(keyword in line.lower() for keyword in [
                'button', 'link', 'textbox', 'heading', 'listitem', 'image', 
                'checkbox', 'combobox', 'search', 'sign in', 'register', 'download',
                'clickable', 'focused'
            ]):
                filtered_lines.append(line)
            # Keep lines with actual meaningful text content
            elif 'StaticText' in line and len(line.strip()) > 20:
                filtered_lines.append(line)
            # Skip most generic elements entirely
            elif 'generic' in line.lower():
                continue
            # Keep some structural elements but very selectively
            elif any(keyword in line.lower() for keyword in ['list', 'heading', 'main']):
                if len(filtered_lines) % 20 == 0:  # Only keep every 20th structural element
                    filtered_lines.append(line)
        
        # Limit total lines to prevent memory explosion
        if len(filtered_lines) > 200:
            # Keep first 60% and last 40% of important lines
            keep_start = int(200 * 0.6)
            keep_end = 200 - keep_start
            filtered_lines = (
                filtered_lines[:keep_start] + 
                [f"\n... (truncated {len(lines) - 200} lines) ...\n"] + 
                filtered_lines[-keep_end:]
            )
        
        return '\n'.join(filtered_lines)

    def _truncate_large_string(self, text: str, max_chars: int = 5000) -> str:
        """Truncate large strings to prevent memory explosion."""
        if len(text) <= max_chars:
            return text
        
        # Keep first 70% and last 30% of the content
        keep_start = int(max_chars * 0.7)
        keep_end = max_chars - keep_start
        
        return (
            text[:keep_start] + 
            f"\n... (truncated {len(text) - max_chars} characters) ...\n" + 
            text[-keep_end:]
        )

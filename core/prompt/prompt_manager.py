import json
import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, Template

from core.utils.types.message import Message, TextContent


class PromptManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self._init_templates()

    def _init_templates(self):
        os.makedirs(self.template_dir, exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir), autoescape=True
        )

        # default to image_user_prompt; fallback to generic if missing
        try:
            self.user_prompt_template = self._load_template("image_user_prompt.j2")
        except FileNotFoundError:
            self.user_prompt_template = self._load_template("user_prompt.j2")

        self.tool_guidance_template = self._load_template("tool_guidance.j2")

        self.tool_description_template = self._load_template("tool_description.j2")
        
        self.continue_prompt_template = self._load_template("continue_prompt.j2")

    def _load_template(self, template_name: str) -> Optional[Template]:
        template_path = os.path.join(self.template_dir, template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_name}")
        return self.env.get_template(template_name)

    def get_system_prompt(self, template_name: Optional[str] = None) -> str:
        template = self._load_template(template_name)
        return template.render()

    def get_user_prompt(self, task: str, paper_path: Optional[str] = None) -> str:
        return self.user_prompt_template.render(task=task, paper_path=paper_path)

    def get_continue_prompt(self) -> str:
        return self.continue_prompt_template.render()

    def save_template(self, template_name: str, content: str):
        template_path = os.path.join(self.template_dir, template_name)
        with open(template_path, "w") as f:
            f.write(content)

    def load_template(self, template_name: str) -> str:
        template_path = os.path.join(self.template_dir, template_name)
        with open(template_path, "r") as f:
            return f.read()

    def list_templates(self) -> List[str]:
        return [f for f in os.listdir(self.template_dir) if f.endswith(".j2")]

    def add_examples_to_initial_message(
        self, message: Message, task: str, paper_path: str
    ) -> None:
        example_message = self.get_user_prompt(task, paper_path) or None

        if example_message:
            message.content.insert(0, TextContent(text=example_message))

    def add_env_setup_to_initial_message(self, message: Message, env_setup_name: str, type_of_processor: str, max_time_in_hours: int, workspace_base: str) -> None:
        try:
            # Use Jinja2 template engine for proper variable substitution
            template = self._load_template(env_setup_name)
            if template:
                env_setup_message = template.render(
                    type_of_processor=type_of_processor,
                    max_time_in_hours=max_time_in_hours,
                    workspace_base=workspace_base
                )
                # message.content.insert(0, TextContent(text=env_setup_message))
                message.content.append(TextContent(text=env_setup_message))
        except Exception as e:
            # Fallback to simple string loading if template loading fails
            env_setup_message = self.load_template(env_setup_name) or None
            if env_setup_message:
                # Use string.Template for ${variable} syntax
                from string import Template
                template_obj = Template(env_setup_message)
                env_setup_message = template_obj.substitute(
                    type_of_processor=type_of_processor,
                    max_time_in_hours=max_time_in_hours,
                    workspace_base=workspace_base
                )
                message.content.append(TextContent(text=env_setup_message))

    def craft_user_prompt(self, task: str, devai_path: str) -> str:
        with open(devai_path, "r") as f:
            instance_data = json.load(f)
            task = instance_data.get("query", task)

        return task

    def add_query_to_initial_message(
        self, message: Message, task: str, devai_path: str
    ) -> None:
        query_message = self.craft_user_prompt(task, devai_path) or None

        if query_message:
            message.content.insert(0, TextContent(text=query_message))

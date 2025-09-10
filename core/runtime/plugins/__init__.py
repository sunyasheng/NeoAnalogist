# Requirements
from core.runtime.plugins.agent_skills import (AgentSkillsPlugin,
                                               AgentSkillsRequirement)
from core.runtime.plugins.requirement import Plugin, PluginRequirement
from core.runtime.plugins.jupyter import JupyterPlugin, JupyterRequirement

__all__ = [
    "Plugin",
    "PluginRequirement",
    "AgentSkillsRequirement",
    "AgentSkillsPlugin",
    "JupyterRequirement",
    "JupyterPlugin",
]

ALL_PLUGINS = {
    "agent_skills": AgentSkillsPlugin,
    "jupyter": JupyterPlugin,
}

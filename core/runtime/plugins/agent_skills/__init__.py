from dataclasses import dataclass

from core.runtime.plugins.agent_skills import agentskills
from core.runtime.plugins.requirement import Plugin, PluginRequirement


@dataclass
class AgentSkillsRequirement(PluginRequirement):
    name: str = "agent_skills"
    documentation: str = agentskills.DOCUMENTATION


class AgentSkillsPlugin(Plugin):
    name: str = "agent_skills"

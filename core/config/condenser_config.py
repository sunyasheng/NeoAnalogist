from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class NoOpCondenserConfig(BaseModel):
    """Configuration for NoOpCondenser."""
    type: Literal['noop'] = Field('noop')

class LLMSummarizingCondenserConfig(BaseModel):
    """Configuration for LLMCondenser."""
    type: Literal['llm_summarizing'] = Field('llm_summarizing')
    max_size: int = Field(
        default=100,
        description='Maximum size of the condensed history before triggering forgetting.',
        ge=2,
    )
    keep_first: int = Field(
        default=1,
        description='Number of initial events to always keep in history.',
        ge=0,
    )
    max_event_length: int = Field(
        default=10_000,
        description='Maximum length of the event representations to be passed to the LLM.',
    )

class BrowserOutputCondenserConfig(BaseModel):
    """Configuration for the BrowserOutputCondenser."""
    type: Literal['browser_output_masking'] = Field('browser_output_masking')
    attention_window: int = Field(
        default=1,
        description='The number of most recent browser output observations that will not be masked.',
        ge=1,
    )

# Union of all supported condenser configs
CondenserConfig = LLMSummarizingCondenserConfig | NoOpCondenserConfig | BrowserOutputCondenserConfig

class CondenserPipelineConfig(BaseModel):
    """Configuration for the CondenserPipeline.

    Not currently supported by the TOML or ENV_VAR configuration strategies.
    """
    type: Literal['pipeline'] = Field('pipeline')
    condensers: list[CondenserConfig] = Field(
        default_factory=list,
        description='List of condenser configurations to be used in the pipeline.',
    )
    model_config = {'extra': 'forbid'} 
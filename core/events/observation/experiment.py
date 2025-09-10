from dataclasses import dataclass
from typing import List, Optional, Dict

from core.events.observation import Observation
from core.events.event import ObservationType


@dataclass
class ExperimentInfo:
    """Information about an experiment"""
    name: str
    created_at: str
    status: str
    wrapper_path: str


@dataclass
class ExperimentManagerObservation(Observation):
    """Observation for experiment manager operations"""
    
    success: bool
    execution_time: float
    experiments: List[Dict]
    mode: str
    error_message: Optional[str] = None
    observation: str = ObservationType.EXPERIMENT_MANAGER

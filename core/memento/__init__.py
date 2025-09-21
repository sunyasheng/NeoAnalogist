"""
Memento Integration for NeoAnalogist
===================================

This package provides a complete integration of Memento's Planner-Executor
architecture into the NeoAnalogist agent system.

Key Components:
- HierarchicalClient: Main coordinator for Planner-Executor architecture
- NonParametricMemory: Memory system for case retrieval and storage
- ScientistExecutor: Adapter for using Scientist agent as Executor
- MementoToolAdapter: Tool calling interface adapter

Usage:
    from core.memento import MementoPlanner
    
    # Create Memento planner with Scientist agent as executor
    planner = MementoPlanner(
        config=config,
        scientist_agent=scientist_agent,
        working_dir=working_dir
    )
    
    # Process a task
    result = await planner.process_query("Edit an image to remove a person")
"""

from .client.hierarchical_client import HierarchicalClient
from .memory.np_memory import NonParametricMemory
from .tools.scientist_executor import ScientistExecutor, MementoToolAdapter
from .memento_planner import MementoPlanner

__all__ = [
    "HierarchicalClient",
    "NonParametricMemory", 
    "ScientistExecutor",
    "MementoToolAdapter",
    "MementoPlanner"
]

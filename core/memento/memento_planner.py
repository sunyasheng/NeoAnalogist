"""
Memento Planner - Main Integration Entry Point
==============================================

This module provides the main entry point for using Memento's Planner-Executor
architecture with NeoAnalogist. It combines all the components into a single
easy-to-use interface.

Key Features:
- Complete Memento integration
- Scientist agent as Executor
- Memory-augmented learning
- Progressive planning with feedback loops
- Easy-to-use interface for NeoAnalogist
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from .client.hierarchical_client import HierarchicalClient, QueryRecord
from .tools.scientist_executor import ScientistExecutor, MementoToolAdapter


class MementoPlanner:
    """
    Main entry point for Memento integration with NeoAnalogist.
    
    This class provides a simple interface for using Memento's Planner-Executor
    architecture with the existing Scientist agent as the Executor.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        scientist_agent: Any,
        working_dir: Optional[str] = None,
        memory_path: Optional[str] = None
    ):
        """
        Initialize the Memento Planner.
        
        Args:
            config: Configuration dictionary
            scientist_agent: NeoAnalogist Scientist agent to use as Executor
            working_dir: Working directory for the planner
            memory_path: Path to memory file (JSONL format)
        """
        self.config = config
        self.scientist_agent = scientist_agent
        self.working_dir = working_dir or "/tmp/memento"
        self.memory_path = memory_path or "core/memento/memory/cases.jsonl"
        
        # Initialize logging
        self.logger = logging.getLogger("MementoPlanner")
        
        # Initialize components
        self.scientist_executor = None
        self.tool_adapter = None
        self.hierarchical_client = None
        
        # Initialize the planner
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing Memento Planner")
        
        # Create Scientist executor
        self.scientist_executor = ScientistExecutor(
            scientist_agent=self.scientist_agent,
            max_steps_per_task=self.config.get("max_steps_per_task", 5),
            timeout_per_task=self.config.get("timeout_per_task", 300)
        )
        
        # Create tool adapter
        self.tool_adapter = MementoToolAdapter(self.scientist_executor)
        
        # Create hierarchical client
        self.hierarchical_client = HierarchicalClient(
            config=self.config,
            scientist_agent=self.scientist_agent,
            working_dir=self.working_dir,
            memory_path=self.memory_path
        )
        
        self.logger.info("Memento Planner initialized successfully")
    
    async def process_query(self, query: str, task_id: Optional[str] = None) -> QueryRecord:
        """
        Process a query using the Memento Planner-Executor architecture.
        
        Args:
            query: The task query to process
            task_id: Optional task ID for tracking
            
        Returns:
            QueryRecord with execution results
        """
        if not task_id:
            task_id = f"task_{int(asyncio.get_event_loop().time())}"
        
        self.logger.info(f"Processing query: {query}")
        
        # Process query using hierarchical client
        result = await self.hierarchical_client.process_query(query, task_id)
        
        # Save successful cases to memory
        if result.model_output and not result.model_output.startswith("["):
            self.hierarchical_client.save_case_to_memory(
                query=query,
                plan_json=result.plan_json,
                reward=1  # Successful execution
            )
        
        return result
    
    def process_query_sync(self, query: str, task_id: Optional[str] = None) -> QueryRecord:
        """
        Synchronous version of process_query.
        
        Args:
            query: The task query to process
            task_id: Optional task ID for tracking
            
        Returns:
            QueryRecord with execution results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_query(query, task_id))
        finally:
            loop.close()
    
    def save_memory(self, output_path: Optional[str] = None):
        """Save memory to file"""
        path = output_path or self.memory_path
        self.hierarchical_client.save_memory(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics"""
        stats = {
            "planner_type": "MementoPlanner",
            "working_dir": self.working_dir,
            "memory_path": self.memory_path
        }
        
        # Add hierarchical client stats
        if self.hierarchical_client:
            stats.update(self.hierarchical_client.get_stats())
        
        # Add executor stats
        if self.scientist_executor:
            stats.update(self.scientist_executor.get_execution_stats())
        
        return stats
    
    def reset(self):
        """Reset the planner state"""
        if self.hierarchical_client:
            # Reset is handled internally by the client
            pass
        
        if self.scientist_executor:
            self.scientist_executor.reset()
        
        self.logger.info("Memento Planner reset")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        if self.scientist_executor:
            return self.scientist_executor.get_available_tools()
        return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        if self.hierarchical_client and self.hierarchical_client.memory_system:
            return self.hierarchical_client.memory_system.get_stats()
        return {"memory_enabled": False}


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Example configuration
    config = {
        "max_cycles": 3,
        "max_tasks_per_plan": 5,
        "memory_enabled": True,
        "max_steps_per_task": 5,
        "timeout_per_task": 300,
        "memory_model": "sentence-transformers/all-MiniLM-L6-v2",
        "memory_device": "auto",
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2000
        }
    }
    
    # Mock scientist agent for testing
    class MockScientist:
        def __init__(self):
            self.agent_id = "test_scientist"
            self.agent_name = "Test Scientist"
            self.event_history = []
        
        def reset(self):
            self.event_history = []
        
        def run(self, task, max_steps=5, controller=None):
            return {"status": "success", "steps": 1}
        
        def execute(self, actions):
            return ["Mock execution result"]
    
    async def test_memento_planner():
        # Create planner
        mock_scientist = MockScientist()
        planner = MementoPlanner(
            config=config,
            scientist_agent=mock_scientist,
            working_dir="/tmp/memento_test",
            memory_path="core/memento/memory/cases.jsonl"
        )
        
        # Test query processing
        query = "Edit an image to remove a person and replace the background with a beach scene"
        result = await planner.process_query(query, "test_001")
        
        print(f"Query: {query}")
        print(f"Result: {result.model_output}")
        print(f"Plan: {result.plan_json}")
        print(f"Meta trace length: {len(result.meta_trace)}")
        print(f"Executor trace length: {len(result.executor_trace)}")
        
        # Test stats
        stats = planner.get_stats()
        print(f"Stats: {stats}")
        
        # Test memory stats
        memory_stats = planner.get_memory_stats()
        print(f"Memory stats: {memory_stats}")
    
    # Run test
    asyncio.run(test_memento_planner())

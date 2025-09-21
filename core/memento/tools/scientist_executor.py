"""
Scientist Executor for Memento Integration
=========================================

This module provides an adapter that allows the NeoAnalogist Scientist agent
to act as an Executor in the Memento Planner-Executor architecture.

Key Features:
- Wraps Scientist agent as Memento Executor
- Handles task execution and result reporting
- Maintains compatibility with Memento's execution model
- Provides tool calling interface for the Planner
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.agent.scientist import Scientist
from core.agent.function_calling import get_tools, response_to_actions
from core.events.action import Action
from core.events.observation import Observation


@dataclass
class ExecutorResult:
    """Result of task execution"""
    success: bool
    result: str
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = None
    execution_time: float = 0.0


class ScientistExecutor:
    """
    Adapter that makes the Scientist agent work as a Memento Executor.
    
    This class wraps the existing Scientist agent and provides the interface
    expected by the Memento HierarchicalClient.
    """
    
    def __init__(
        self,
        scientist_agent: Scientist,
        max_steps_per_task: int = 5,
        timeout_per_task: int = 300
    ):
        self.scientist_agent = scientist_agent
        self.max_steps_per_task = max_steps_per_task
        self.timeout_per_task = timeout_per_task
        self.logger = logging.getLogger("ScientistExecutor")
        
        # Get available tools
        self.tools = get_tools(codeact_enable_browsing=True)
        
        self.logger.info("ScientistExecutor initialized")
    
    async def execute_task(self, task_description: str) -> ExecutorResult:
        """Execute a single task using the Scientist agent"""
        self.logger.info(f"Executing task: {task_description}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Reset scientist agent state for new task
            self.scientist_agent.reset()
            
            # Execute task with timeout
            result = await asyncio.wait_for(
                self._execute_with_scientist(task_description),
                timeout=self.timeout_per_task
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ExecutorResult(
                success=result.get("status") == "success",
                result=result.get("result", "Task completed"),
                error=result.get("error"),
                tool_calls=result.get("tool_calls", []),
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ExecutorResult(
                success=False,
                result="Task execution timed out",
                error=f"Task exceeded timeout of {self.timeout_per_task} seconds",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Error executing task: {e}")
            return ExecutorResult(
                success=False,
                result=f"Task execution failed: {str(e)}",
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_with_scientist(self, task_description: str) -> Dict[str, Any]:
        """Execute task using the Scientist agent"""
        try:
            # Run the scientist agent
            result = self.scientist_agent.run(
                task=task_description,
                max_steps=self.max_steps_per_task,
                controller=None
            )
            
            # Extract tool calls from event history
            tool_calls = []
            for event in self.scientist_agent.event_history:
                if hasattr(event, 'tool_call_metadata') and event.tool_call_metadata:
                    tool_calls.append({
                        "tool": event.tool_call_metadata.function_name,
                        "arguments": getattr(event, 'arguments', {}),
                        "result": str(event) if hasattr(event, '__str__') else "Executed"
                    })
            
            return {
                "status": result.get("status", "success"),
                "result": f"Task '{task_description}' completed successfully",
                "error": None,
                "tool_calls": tool_calls,
                "steps": result.get("steps", 0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "result": f"Task '{task_description}' failed",
                "error": str(e),
                "tool_calls": [],
                "steps": 0
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "parameters": tool["function"].get("parameters", {})
            }
            for tool in self.tools
        ]
    
    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """Get tool schema for LLM"""
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool directly"""
        self.logger.info(f"Calling tool: {tool_name} with args: {arguments}")
        
        try:
            # Find the tool
            tool = None
            for t in self.tools:
                if t["function"]["name"] == tool_name:
                    tool = t
                    break
            
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found",
                    "result": None
                }
            
            # Create action from tool call
            actions = response_to_actions({
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{tool_name}",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments)
                            }
                        }]
                    }
                }]
            })
            
            if not actions:
                return {
                    "success": False,
                    "error": "Failed to create action from tool call",
                    "result": None
                }
            
            # Execute action
            observations = self.scientist_agent.execute(actions)
            
            # Extract result
            if observations:
                result = str(observations[0])
                return {
                    "success": True,
                    "error": None,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": "No observation returned",
                    "result": None
                }
                
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "max_steps_per_task": self.max_steps_per_task,
            "timeout_per_task": self.timeout_per_task,
            "available_tools": len(self.tools),
            "scientist_agent_id": self.scientist_agent.agent_id,
            "scientist_agent_name": getattr(self.scientist_agent, 'agent_name', 'Unknown')
        }
    
    def reset(self):
        """Reset the executor state"""
        if hasattr(self.scientist_agent, 'reset'):
            self.scientist_agent.reset()
        self.logger.info("ScientistExecutor reset")


class MementoToolAdapter:
    """
    Adapter for Memento's tool calling interface.
    
    This class provides the interface that Memento expects for tool execution,
    while using the NeoAnalogist tool system underneath.
    """
    
    def __init__(self, scientist_executor: ScientistExecutor):
        self.scientist_executor = scientist_executor
        self.logger = logging.getLogger("MementoToolAdapter")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool and return the result"""
        result = await self.scientist_executor.call_tool(tool_name, arguments)
        
        # Return result in format expected by Memento
        if result["success"]:
            return type('ToolResult', (), {
                'content': result["result"],
                'success': True
            })()
        else:
            return type('ToolResult', (), {
                'content': f"Error: {result['error']}",
                'success': False
            })()
    
    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """Get tool schema for Memento"""
        return self.scientist_executor.get_tool_schema()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # This would be used in the main integration
    async def test_executor():
        # Create a mock scientist agent (in real usage, this would be a real Scientist)
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
        
        # Create executor
        mock_scientist = MockScientist()
        executor = ScientistExecutor(mock_scientist)
        
        # Test task execution
        result = await executor.execute_task("Test task description")
        print(f"Execution result: {result}")
        
        # Test tool calling
        tool_result = await executor.call_tool("test_tool", {"arg1": "value1"})
        print(f"Tool result: {tool_result}")
    
    asyncio.run(test_executor())

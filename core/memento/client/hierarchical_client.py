"""
Hierarchical Client for Memento Integration
==========================================

This module implements the main Memento client logic, adapted for integration
with NeoAnalogist. It coordinates between the Planner and Executor (Scientist agent).

Key Features:
- Planner-Executor architecture
- Memory-augmented learning
- Progressive planning with feedback loops
- Integration with NeoAnalogist's Scientist agent
"""

import json
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from core.llm.interface import LLMInterface
from core.memory.conversation_memory import ConversationMemory
from core.prompt.prompt_manager import PromptManager
from core.memento.memory.np_memory import NonParametricMemory


@dataclass
class MetaCycle:
    """Represents a single planning cycle"""
    cycle: int
    messages: List[str]
    response: str


@dataclass
class ExecStep:
    """Represents a single execution step"""
    task_id: str
    input: str
    output: str


@dataclass
class ToolCallRecord:
    """Record of a tool call"""
    tool: str
    arguments: Dict[str, Any]
    result: str


@dataclass
class QueryRecord:
    """Record of a complete query execution"""
    task_id: str
    query: str
    model_output: str
    plan_json: str
    meta_trace: List[MetaCycle]
    executor_trace: List[ExecStep]
    tool_history: List[ToolCallRecord]


class HierarchicalClient:
    """
    Hierarchical client that coordinates Planner-Executor architecture.
    
    This is the main entry point for Memento-style task execution,
    adapted to work with NeoAnalogist's Scientist agent as the Executor.
    """
    
    # System prompts adapted for image editing tasks
    META_SYSTEM_PROMPT = """You are the META-PLANNER for an image editing agent system. Your role is to:

1. Break down complex image editing tasks into smaller, manageable subtasks
2. Create detailed plans with specific task descriptions
3. Coordinate with the Executor (Scientist agent) to complete tasks
4. Learn from execution results and adjust plans accordingly

IMPORTANT INSTRUCTIONS:
- Always output plans in valid JSON format
- Each task should have a unique ID and clear description
- Focus on image editing, manipulation, and generation tasks
- Consider the available tools: image editing, object detection, inpainting, etc.
- If a task is completed successfully, don't include it in subsequent plans
- If a task fails, provide alternative approaches in new plans
- When all tasks are complete, output "FINAL ANSWER: [result]"

PLAN FORMAT:
```json
{
  "plan": [
    {"id": "task_1", "description": "Specific task description"},
    {"id": "task_2", "description": "Another specific task description"}
  ]
}
```

Remember: You are coordinating with an Executor that has access to various image editing tools. Make your task descriptions specific and actionable."""

    EXEC_SYSTEM_PROMPT = """You are the EXECUTOR for an image editing agent system. Your role is to:

1. Execute individual tasks assigned by the Planner
2. Use available tools to complete image editing tasks
3. Report results back to the Planner
4. Focus on one task at a time

Available tools include:
- Image editing and manipulation
- Object detection and segmentation
- Inpainting and outpainting
- Image generation
- File operations
- And more...

When executing a task:
- Be precise and follow the task description exactly
- Use the most appropriate tools for the job
- Report success or failure clearly
- Provide detailed results when possible"""

    def __init__(
        self,
        config: Dict[str, Any],
        scientist_agent: Optional[Any] = None,
        working_dir: Optional[str] = None,
        memory_path: Optional[str] = None
    ):
        self.config = config
        self.scientist_agent = scientist_agent
        self.working_dir = working_dir or "/tmp/memento"
        
        # Initialize logging
        self.logger = logging.getLogger("HierarchicalClient")
        
        # Configuration
        self.max_cycles = config.get("max_cycles", 3)
        self.max_tasks_per_plan = config.get("max_tasks_per_plan", 5)
        self.memory_enabled = config.get("memory_enabled", True)
        
        # Initialize components
        self.meta_llm = None  # Planner LLM
        self.exec_llm = None  # Executor LLM (will use scientist_agent)
        self.memory_system = None
        self.shared_history: List[Dict[str, str]] = []
        
        # Initialize the client
        self._initialize()
        
        # Load memory if enabled
        if self.memory_enabled and memory_path:
            self._load_memory(memory_path)
    
    def _initialize(self):
        """Initialize the hierarchical client"""
        self.logger.info("Initializing Hierarchical Client")
        
        # Create proper LLM config
        llm_config = self._create_llm_config()
        
        # Initialize Planner LLM
        self.meta_llm = LLMInterface(llm_config, working_dir=self.working_dir)
        
        # Initialize Executor LLM (same as Planner for now, but could be different)
        self.exec_llm = LLMInterface(llm_config, working_dir=self.working_dir)
        
        # Initialize memory system
        if self.memory_enabled:
            self.memory_system = NonParametricMemory(
                model_name=self.config.get("memory_model", "sentence-transformers/all-MiniLM-L6-v2"),
                device=self.config.get("memory_device", "auto")
            )
        
        self.logger.info("Hierarchical Client initialized successfully")
    
    def _create_llm_config(self) -> Dict[str, Any]:
        """Create proper LLM configuration"""
        # Get base config
        base_config = self.config.get("llm", {})
        
        # Create proper config structure for LLMInterface
        llm_config = {
            "llm": {
                "temperature": base_config.get("temperature", 0.7),
                "max_tokens": base_config.get("max_tokens", 2000),
                "providers": {
                    "litellm": {
                        "use": True,
                        "model": base_config.get("model", "gpt-4o"),
                        "api_key": base_config.get("api_key", ""),
                        "base_url": base_config.get("base_url", "")
                    }
                }
            }
        }
        
        return llm_config
    
    def _load_memory(self, memory_path: str):
        """Load memory system from file"""
        try:
            if Path(memory_path).exists():
                success = self.memory_system.load_cases(memory_path)
                if success:
                    self.logger.info(f"Memory system loaded from {memory_path}")
                else:
                    self.logger.warning(f"Failed to load memory from {memory_path}")
            else:
                self.logger.warning(f"Memory file not found: {memory_path}")
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
    
    def _add_to_history(self, role: str, content: str):
        """Add a message to the shared history"""
        self.shared_history.append({"role": role, "content": content})
    
    def _memory_prompt_for(self, query: str) -> Optional[str]:
        """Get memory guidance for the query"""
        if not self.memory_enabled or not self.memory_system or not self.memory_system.is_loaded:
            return None
        
        try:
            # Retrieve similar cases
            results = self.memory_system.retrieve(query, top_k=3, min_score=0.3)
            
            if not results:
                return None
            
            # Format memory guidance
            positive_examples = []
            negative_examples = []
            
            for result in results:
                if result.get("reward", 0) == 1:
                    positive_examples.append({
                        "question": result["question"],
                        "plan": result["plan"]
                    })
                else:
                    negative_examples.append({
                        "question": result["question"],
                        "plan": result["plan"]
                    })
            
            # Create memory prompt
            memory_prompt = "Here are some similar cases from memory:\n\n"
            
            if positive_examples:
                memory_prompt += "POSITIVE EXAMPLES (successful approaches):\n"
                for i, example in enumerate(positive_examples[:2], 1):
                    memory_prompt += f"{i}. Question: {example['question']}\n"
                    memory_prompt += f"   Plan: {example['plan']}\n\n"
            
            if negative_examples:
                memory_prompt += "NEGATIVE EXAMPLES (approaches to avoid):\n"
                for i, example in enumerate(negative_examples[:1], 1):
                    memory_prompt += f"{i}. Question: {example['question']}\n"
                    memory_prompt += f"   Plan: {example['plan']}\n\n"
            
            memory_prompt += "Use these examples to guide your planning, but adapt them to the current task."
            
            return memory_prompt
            
        except Exception as e:
            self.logger.error(f"Error in memory retrieval: {e}")
            return None
    
    def _strip_fences(self, text: str) -> str:
        """Strip code fences from text"""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()
    
    async def _execute_task_with_scientist(self, task_description: str) -> Dict[str, Any]:
        """Execute a single task using the Scientist agent"""
        if not self.scientist_agent:
            return {
                "success": False,
                "error": "No Scientist agent available",
                "result": "No Scientist agent available"
            }
        
        try:
            # Use the Scientist agent to execute the task
            result = self.scientist_agent.run(
                task=task_description,
                max_steps=5,  # Limit steps for individual tasks
                controller=None
            )
            
            success = result.get("status") == "success"
            
            return {
                "success": success,
                "result": f"Task executed: {task_description}",
                "details": result,
                "error": None if success else "Task execution failed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": f"Error executing task: {e}"
            }
    
    async def process_query(self, query: str, task_id: str) -> QueryRecord:
        """Process a query using the Planner-Executor architecture"""
        self.logger.info(f"Processing query: {query}")
        
        # Reset shared history
        self.shared_history = []
        
        # Add query to history
        self._add_to_history("user", query)
        
        # Get memory guidance
        mem_prompt = self._memory_prompt_for(query)
        if mem_prompt:
            self._add_to_history("user", mem_prompt)
        
        # Prepare Planner messages
        planner_msgs = [{"role": "system", "content": self.META_SYSTEM_PROMPT}] + self.shared_history
        
        # Initialize tracking
        meta_trace: List[MetaCycle] = []
        executor_trace: List[ExecStep] = []
        tool_history: List[ToolCallRecord] = []
        final_answer: str = ""
        latest_plan_json: str = ""
        
        # Main planning-execution loop
        for cycle in range(self.max_cycles):
            self.logger.info(f"Starting cycle {cycle + 1}/{self.max_cycles}")
            
            # Planner generates plan
            try:
                meta_reply = self.meta_llm.chat(
                    messages=planner_msgs,
                    temperature=0.7,
                    tools=None,
                    tool_choice=None
                )
                meta_content = meta_reply.get("content", "")
                
                # Record meta cycle
                meta_trace.append(MetaCycle(
                    cycle=cycle,
                    messages=[m["content"] for m in planner_msgs],
                    response=meta_content
                ))
                
                self._add_to_history("assistant", meta_content)
                
                # Check for final answer
                if meta_content.startswith("FINAL ANSWER:"):
                    final_answer = meta_content[len("FINAL ANSWER:"):].strip()
                    break
                
                # Parse plan JSON
                try:
                    # First try to strip code fences
                    stripped = self._strip_fences(meta_content)
                    
                    # Clean up the content - remove any formatting artifacts
                    cleaned_content = stripped.strip()
                    
                    # Try to find JSON object in the content
                    start = cleaned_content.find('{')
                    end = cleaned_content.rfind('}') + 1
                    
                    if start != -1 and end > start:
                        json_str = cleaned_content[start:end]
                        plan_data = json.loads(json_str)
                        
                        # Validate plan structure
                        if "plan" not in plan_data:
                            raise ValueError("Plan structure missing 'plan' key")
                        
                        latest_plan_json = json.dumps(plan_data)
                        self.logger.info(f"Successfully parsed plan with {len(plan_data['plan'])} tasks")
                    else:
                        raise ValueError("No valid JSON object found in response")
                    
                except Exception as e:
                    self.logger.error(f"JSON parsing error: {e}")
                    self.logger.error(f"Content to parse: {meta_content[:200]}...")
                    final_answer = f"[planner error] {e}: {meta_content}"
                    break
                
                # Execute tasks in the plan
                tasks = json.loads(latest_plan_json)["plan"]
                for task in tasks:
                    task_desc = f"Task {task['id']}: {task['description']}"
                    self.logger.info(f"Executing: {task_desc}")
                    
                    # Execute task using Scientist agent
                    exec_result = await self._execute_task_with_scientist(task_desc)
                    
                    # Record execution step
                    executor_trace.append(ExecStep(
                        task_id=task["id"],
                        input=task_desc,
                        output=exec_result["result"]
                    ))
                    
                    # Add to shared history
                    self._add_to_history("assistant", f"Task {task['id']} result: {exec_result['result']}")
                
                # Update Planner messages for next cycle
                planner_msgs = [{"role": "system", "content": self.META_SYSTEM_PROMPT}] + self.shared_history
                
            except Exception as e:
                self.logger.error(f"Error in cycle {cycle + 1}: {e}")
                final_answer = f"[execution error] {e}"
                break
        
        # If we exhausted cycles without final answer
        if not final_answer:
            final_answer = meta_content.strip() if meta_content else "No final answer generated"
        
        # Clear shared history
        self.shared_history.clear()
        
        # Create query record
        query_record = QueryRecord(
            task_id=task_id,
            query=query,
            model_output=final_answer,
            plan_json=latest_plan_json,
            meta_trace=meta_trace,
            executor_trace=executor_trace,
            tool_history=tool_history
        )
        
        self.logger.info(f"Query processing completed. Success: {bool(final_answer)}")
        return query_record
    
    def save_case_to_memory(self, query: str, plan_json: str, reward: int = 0):
        """Save a case to memory for future retrieval"""
        if self.memory_enabled and self.memory_system:
            try:
                self.memory_system.add_case(query, plan_json, reward)
                self.logger.info(f"Saved case to memory: {query[:50]}...")
            except Exception as e:
                self.logger.error(f"Error saving case to memory: {e}")
    
    def save_memory(self, output_path: str):
        """Save memory system to file"""
        if self.memory_enabled and self.memory_system:
            try:
                self.memory_system.save_cases(output_path)
                self.logger.info(f"Memory saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {
            "max_cycles": self.max_cycles,
            "memory_enabled": self.memory_enabled,
            "scientist_agent_available": self.scientist_agent is not None
        }
        
        if self.memory_system:
            stats.update(self.memory_system.get_stats())
        
        return stats


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "max_cycles": 3,
        "max_tasks_per_plan": 5,
        "memory_enabled": True,
        "memory_model": "sentence-transformers/all-MiniLM-L6-v2",
        "memory_device": "auto",
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2000
        }
    }
    
    # Create client
    client = HierarchicalClient(
        config=config,
        scientist_agent=None,  # Would be provided in real usage
        working_dir="/tmp/memento_test",
        memory_path="core/memento/memory/cases.jsonl"
    )
    
    # Example task
    task = "Edit an image to remove a person and replace the background with a beach scene"
    
    # Process task
    async def main():
        result = await client.process_query(task, "test_001")
        print(f"Result: {result.model_output}")
        print(f"Plan: {result.plan_json}")
    
    asyncio.run(main())

"""
Example Usage of Memento Integration
===================================

This file demonstrates how to use the Memento integration with NeoAnalogist.
It shows how to set up the MementoPlanner and use it for image editing tasks.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Import NeoAnalogist components
from core.agent.scientist import Scientist
from core.memento.memento_planner import MementoPlanner


def setup_logging():
    """Setup logging for the example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_example_config() -> Dict[str, Any]:
    """Create example configuration for MementoPlanner"""
    return {
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
        },
        "agent": {
            "task_type": "image_editing"
        }
    }


def create_scientist_agent(config: Dict[str, Any], working_dir: str) -> Scientist:
    """Create a Scientist agent for use as Executor"""
    scientist_config = {
        "agent_name": "MementoExecutor",
        "agent_description": "Scientist agent acting as Memento Executor",
        "context_window": 5,
        "enable_som_visual_browsing": False,
        "condenser": {
            "type": "llm_summarizing",
            "max_size": 10,
            "keep_first": 2,
            "max_event_length": 1000
        },
        **config
    }
    
    return Scientist(
        agent_id="memento_executor",
        config=scientist_config,
        working_dir=working_dir
    )


async def example_image_editing_task():
    """Example of using MementoPlanner for image editing tasks"""
    print("=== Memento Image Editing Example ===\n")
    
    # Setup
    setup_logging()
    config = create_example_config()
    working_dir = "/tmp/memento_example"
    Path(working_dir).mkdir(exist_ok=True)
    
    # Create Scientist agent
    print("Creating Scientist agent...")
    scientist_agent = create_scientist_agent(config, working_dir)
    
    # Create MementoPlanner
    print("Creating MementoPlanner...")
    planner = MementoPlanner(
        config=config,
        scientist_agent=scientist_agent,
        working_dir=working_dir,
        memory_path="core/memento/memory/cases.jsonl"
    )
    
    # Example tasks
    tasks = [
        "Edit an image to remove a person and replace the background with a beach scene",
        "Generate a realistic image of a cat sitting on a windowsill",
        "Remove unwanted objects from a photo while preserving the original lighting"
    ]
    
    # Process each task
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}: {task} ---")
        
        try:
            result = await planner.process_query(task, f"task_{i}")
            
            print(f"✅ Task completed successfully!")
            print(f"📋 Final answer: {result.model_output}")
            print(f"📝 Plan used: {result.plan_json}")
            print(f"🔄 Planning cycles: {len(result.meta_trace)}")
            print(f"⚙️  Execution steps: {len(result.executor_trace)}")
            
            # Show execution trace
            if result.executor_trace:
                print("📊 Execution trace:")
                for step in result.executor_trace:
                    print(f"   - {step.task_id}: {step.output[:100]}...")
            
        except Exception as e:
            print(f"❌ Task failed: {e}")
    
    # Show final stats
    print(f"\n=== Final Statistics ===")
    stats = planner.get_stats()
    print(f"📈 Total tasks processed: {len(tasks)}")
    print(f"🧠 Memory enabled: {stats.get('memory_enabled', False)}")
    print(f"🔧 Available tools: {len(planner.get_available_tools())}")
    
    memory_stats = planner.get_memory_stats()
    if memory_stats.get('memory_enabled'):
        print(f"💾 Total cases in memory: {memory_stats.get('total_cases', 0)}")
    
    # Save memory
    planner.save_memory("core/memento/memory/updated_cases.jsonl")
    print("💾 Memory saved to updated_cases.jsonl")


async def example_memory_retrieval():
    """Example of memory retrieval functionality"""
    print("\n=== Memory Retrieval Example ===\n")
    
    # Setup
    setup_logging()
    config = create_example_config()
    working_dir = "/tmp/memento_memory_test"
    Path(working_dir).mkdir(exist_ok=True)
    
    # Create components
    scientist_agent = create_scientist_agent(config, working_dir)
    planner = MementoPlanner(
        config=config,
        scientist_agent=scientist_agent,
        working_dir=working_dir,
        memory_path="core/memento/memory/cases.jsonl"
    )
    
    # Test memory retrieval
    if planner.hierarchical_client.memory_system and planner.hierarchical_client.memory_system.is_loaded:
        print("🧠 Testing memory retrieval...")
        
        test_queries = [
            "Edit an image to remove a person",
            "Generate a realistic image of an animal",
            "Remove objects from a photo"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = planner.hierarchical_client.memory_system.retrieve(query, top_k=2)
            
            if results:
                print("📚 Retrieved cases:")
                for result in results:
                    print(f"   - {result['question']} (score: {result['score']})")
            else:
                print("   No similar cases found")
    else:
        print("⚠️  Memory system not loaded")


def example_sync_usage():
    """Example of synchronous usage"""
    print("\n=== Synchronous Usage Example ===\n")
    
    # Setup
    setup_logging()
    config = create_example_config()
    working_dir = "/tmp/memento_sync_test"
    Path(working_dir).mkdir(exist_ok=True)
    
    # Create components
    scientist_agent = create_scientist_agent(config, working_dir)
    planner = MementoPlanner(
        config=config,
        scientist_agent=scientist_agent,
        working_dir=working_dir,
        memory_path="core/memento/memory/cases.jsonl"
    )
    
    # Process task synchronously
    task = "Edit an image to add a sunset background"
    print(f"🔄 Processing task synchronously: {task}")
    
    try:
        result = planner.process_query_sync(task, "sync_task_001")
        print(f"✅ Task completed: {result.model_output}")
    except Exception as e:
        print(f"❌ Task failed: {e}")


async def main():
    """Main function to run all examples"""
    print("🚀 Memento Integration Examples\n")
    
    try:
        # Run async examples
        await example_image_editing_task()
        await example_memory_retrieval()
        
        # Run sync example
        example_sync_usage()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

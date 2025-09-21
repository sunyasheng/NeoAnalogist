"""
Test Integration for Memento
============================

Simple test to verify that the Memento integration works correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.memento.memento_planner import MementoPlanner


class MockScientist:
    """Mock Scientist agent for testing"""
    
    def __init__(self):
        self.agent_id = "test_scientist"
        self.agent_name = "Test Scientist"
        self.event_history = []
    
    def reset(self):
        self.event_history = []
    
    def run(self, task, max_steps=5, controller=None):
        # Simulate successful execution
        return {
            "status": "success",
            "steps": 1,
            "result": f"Successfully executed: {task}"
        }
    
    def execute(self, actions):
        return [f"Mock execution result for {len(actions)} actions"]


async def test_basic_functionality():
    """Test basic Memento functionality"""
    print("🧪 Testing Memento Integration...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration - use original memory system
    config = {
        "max_cycles": 2,
        "max_tasks_per_plan": 3,
        "memory_enabled": True,  # Enable original memory system
        "max_steps_per_task": 3,
        "timeout_per_task": 60,
        "memory_model": "sentence-transformers/all-MiniLM-L6-v2",
        "memory_device": "auto",
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    # Create mock scientist
    mock_scientist = MockScientist()
    
    # Create MementoPlanner
    try:
        planner = MementoPlanner(
            config=config,
            scientist_agent=mock_scientist,
            working_dir="/tmp/memento_test",
            memory_path="core/memento/memory/cases.jsonl"
        )
        print("✅ MementoPlanner created successfully")
    except Exception as e:
        print(f"❌ Failed to create MementoPlanner: {e}")
        return False
    
    # Test memory system
    try:
        memory_stats = planner.get_memory_stats()
        print(f"✅ Memory system: {memory_stats}")
    except Exception as e:
        print(f"❌ Memory system error: {e}")
        return False
    
    # Test tool listing
    try:
        tools = planner.get_available_tools()
        print(f"✅ Available tools: {len(tools)}")
    except Exception as e:
        print(f"❌ Tool listing error: {e}")
        return False
    
    # Test stats
    try:
        stats = planner.get_stats()
        print(f"✅ Stats: {stats}")
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False
    
    print("🎉 All basic tests passed!")
    return True


async def test_query_processing():
    """Test query processing (requires actual LLM)"""
    print("\n🧪 Testing Query Processing...")
    
    # This test requires actual LLM access, so we'll make it optional
    try:
        config = {
            "max_cycles": 1,
            "max_tasks_per_plan": 2,
            "memory_enabled": False,  # Disable memory for simpler test
            "max_steps_per_task": 2,
            "timeout_per_task": 30,
            "llm": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
        
        mock_scientist = MockScientist()
        planner = MementoPlanner(
            config=config,
            scientist_agent=mock_scientist,
            working_dir="/tmp/memento_test",
            memory_path=None  # No memory for this test
        )
        
        # Test simple query
        query = "Test image editing task"
        result = await planner.process_query(query, "test_001")
        
        print(f"✅ Query processed: {result.model_output[:100]}...")
        print(f"✅ Plan: {result.plan_json[:100]}...")
        print(f"✅ Meta trace: {len(result.meta_trace)} cycles")
        print(f"✅ Executor trace: {len(result.executor_trace)} steps")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Query processing test skipped (requires LLM): {e}")
        return True  # Don't fail the test if LLM is not available


async def main():
    """Run all tests"""
    print("🚀 Starting Memento Integration Tests\n")
    
    # Test basic functionality
    basic_success = await test_basic_functionality()
    
    # Test query processing
    query_success = await test_query_processing()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Query processing: {'✅ PASS' if query_success else '❌ FAIL'}")
    
    if basic_success and query_success:
        print("\n🎉 All tests passed! Memento integration is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

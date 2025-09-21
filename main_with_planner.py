import argparse
import json
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv

from core.agent.scientist import Scientist
from core.events.serialization import event_to_trajectory
from core.utils.logger import get_logger, log_task_completion, setup_logging
from core.utils.workspace import WorkspaceManager
from core.controller.agent_controller import AgentController
from core.events.stream import EventStream
from core.storage.local import LocalFileStore
from core.memento.memento_planner import MementoPlanner

load_dotenv()


def run_agent_with_planner(
    planner: MementoPlanner, task: str, max_steps: int = 10, controller=None
) -> Dict[str, Any]:
    """
    Run the agent with Memento planner and case bank mechanism
    """
    logger = get_logger("memento-planner")
    
    # Process the query using Memento planner
    logger.info(f"Processing task with Memento planner: {task}")
    result = planner.process_query(task)
    
    # Extract execution statistics
    execution_stats = {
        "planner_cycles": result.get("meta_trace", 0),
        "executor_steps": result.get("executor_trace", 0),
        "success": result.get("success", False),
        "final_answer": result.get("final_answer", ""),
        "plan": result.get("plan", {}),
        "shared_history": result.get("shared_history", [])
    }
    
    # Get LLM usage stats from the planner's LLM interface
    usage_stats = {}
    if hasattr(planner, 'hierarchical_client') and hasattr(planner.hierarchical_client, 'llm_interface'):
        if hasattr(planner.hierarchical_client.llm_interface, 'get_usage_stats'):
            usage_stats = planner.hierarchical_client.llm_interface.get_usage_stats()
    
    # Save trajectory (convert shared_history to trajectory format)
    trajectory = []
    for entry in result.get("shared_history", []):
        if isinstance(entry, dict):
            trajectory.append({
                "type": "planner_cycle",
                "content": entry.get("content", ""),
                "timestamp": time.time()
            })
    
    trajectory_path = os.path.join(planner.working_dir, "trajectory.json")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    
    return {
        "task": task,
        "result": result,
        "execution_stats": execution_stats,
        "usage_stats": usage_stats,
        "trajectory_path": trajectory_path,
    }


def main(config, working_dir):
    logger = get_logger("main_with_planner")
    
    # If working_dir is the default, create a new timestamped workspace as before
    if working_dir == "workspace/current":
        working_dir, config, log_file = WorkspaceManager.initialize(config)
    else:
        log_file = os.path.join(working_dir, "data", "logs", "agent.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.info(f"Workspace directory: {working_dir}")
    setup_logging(
        log_file=log_file, level="DEBUG" if config["agent"]["debug"] else "INFO"
    )
    
    # Initialize FileStore and EventStream
    file_store = LocalFileStore(working_dir)
    event_stream = EventStream(sid="session_id", file_store=file_store, user_id="user_id")
    
    # Create Scientist agent (will be used as executor by Memento planner)
    scientist_agent = Scientist(
        agent_id=config["agent"]["agent_id"], 
        config=config, 
        working_dir=working_dir, 
        event_stream=event_stream
    )
    
    # Create AgentController for the scientist agent
    agent_controller = AgentController(
        agent=scientist_agent,
        event_stream=event_stream,
        iteration_delta=config["max_steps"],
        file_store=file_store,
        user_id="user_id",
        sid="session_id"
    )
    
    # Restore state for the scientist agent
    restored_state = agent_controller.restore_state("session_id", "user_id")
    agent_controller.state = restored_state
    scientist_agent.event_history = list(event_stream.search_events(start_id=0, reverse=False, filter=None))
    print("Session state restored.")
    print("[DEBUG] event_history after restore:", len(scientist_agent.event_history))
    
    # Initialize Memento Planner with case bank
    logger.info("Initializing Memento Planner with case bank...")
    planner = MementoPlanner(
        scientist_agent=scientist_agent,
        working_dir=working_dir,
        memory_enabled=True,  # Enable case bank
        memory_path="core/memento/memory/cases.jsonl",  # Path to case bank
        max_cycles=config.get("max_cycles", 3),  # Maximum planning cycles
        max_steps_per_task=config.get("max_steps_per_task", 5),  # Max steps per task
        timeout_per_task=config.get("timeout_per_task", 60)  # Timeout per task
    )
    
    logger.info("Memento Planner initialized successfully")
    logger.info(f"Case bank loaded: {planner.memory_system.get('total_cases', 0)} cases")
    
    # Process the task with Memento planner
    results = run_agent_with_planner(
        planner=planner,
        task=config["agent"]["task"],
        max_steps=config["max_steps"],
        controller=agent_controller,
    )
    
    # Save results
    output_file = config["output_file"]
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    # Log execution statistics
    execution_stats = results.get("execution_stats", {})
    logger.info(f"Planner cycles: {execution_stats.get('planner_cycles', 0)}")
    logger.info(f"Executor steps: {execution_stats.get('executor_steps', 0)}")
    logger.info(f"Task success: {execution_stats.get('success', False)}")
    
    # Log LLM usage
    usage_stats = results.get("usage_stats", {})
    if usage_stats:
        logger.info(
            f"LLM usage: {usage_stats.get('total_prompt_tokens', 0)} prompt tokens, "
            f"{usage_stats.get('total_completion_tokens', 0)} completion tokens"
        )
        logger.info(f"Total tokens: {usage_stats.get('total_tokens', 0)}")
        logger.info(f"Total cost: ${usage_stats.get('total_cost', 0):.5f}")
    
    # Save state at the end
    agent_controller.save_state()
    print("Session state saved.")
    
    # Log final results
    logger.info("=" * 50)
    logger.info("MEMENTO PLANNER EXECUTION COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Task: {config['agent']['task']}")
    logger.info(f"Success: {execution_stats.get('success', False)}")
    logger.info(f"Final Answer: {execution_stats.get('final_answer', '')[:200]}...")
    logger.info(f"Case Bank Cases Used: {planner.memory_system.get('total_cases', 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The First AI PhD with Memento Planner and Case Bank.")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file"
    )
    args, remaining_args = parser.parse_known_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Allow command-line overrides after config is loaded
    parser = argparse.ArgumentParser(description="Run scientist agent with Memento planner on a task")
    parser.add_argument("--task", type=str, help="Task for the agent to complete")
    parser.add_argument("--agent-id", type=str, help="Agent ID")
    parser.add_argument("--max-steps", type=int, help="Maximum steps")
    parser.add_argument("--max-cycles", type=int, help="Maximum planning cycles")
    parser.add_argument("--max-steps-per-task", type=int, help="Maximum steps per task")
    parser.add_argument("--timeout-per-task", type=int, help="Timeout per task in seconds")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["user", "paperbench", "devai"],
        default="user",
        help="Input type: user custom input, paperbench, or devai",
    )
    parser.add_argument("--paper-id", type=str, help="Paper ID for the task")
    parser.add_argument("--paper-path", type=str, help="Path to the paper directory")
    parser.add_argument(
        "--devai-path", type=str, help="Path to the devai task instance directory"
    )
    parser.add_argument(
        "--work-dir", type=str, default="workspace/current", help="Working directory for session storage"
    )
    parser.add_argument(
        "--memory-path", type=str, default="core/memento/memory/cases.jsonl", 
        help="Path to case bank memory file"
    )
    parser.add_argument(
        "--disable-memory", action="store_true", 
        help="Disable case bank memory system"
    )
    
    args = parser.parse_args(remaining_args)
    
    if args.task is not None:
        config["agent"]["task"] = args.task
    if args.agent_id is not None:
        config["agent"]["agent_id"] = args.agent_id
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.max_cycles is not None:
        config["max_cycles"] = args.max_cycles
    if args.max_steps_per_task is not None:
        config["max_steps_per_task"] = args.max_steps_per_task
    if args.timeout_per_task is not None:
        config["timeout_per_task"] = args.timeout_per_task
    if args.output is not None:
        config["output_file"] = args.output
    if args.debug:
        config["agent"]["debug"] = True
    if args.paper_id is not None:
        config["agent"]["paper_id"] = args.paper_id
    if args.paper_path is not None:
        config["agent"]["paper_path"] = args.paper_path
    if args.devai_path is not None:
        config["agent"]["devai_path"] = args.devai_path
    
    # Handle input type and corresponding IDs
    config["agent"]["task_type"] = args.task_type
    
    if args.task_type in ["paperbench", "devai"]:
        if args.task_type == "paperbench":
            config["paper_path"] = config["agent"]["paper_path"]
        else:  # devai
            config["devai_path"] = config["agent"]["devai_path"]
    
    # Override memory settings if specified
    if args.disable_memory:
        config["memory_enabled"] = False
    if args.memory_path:
        config["memory_path"] = args.memory_path
    
    main(config, args.work_dir)

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

load_dotenv()


def run_agent(
    agent: Scientist, task: str, max_steps: int = 10, controller=None
) -> Dict[str, Any]:
    # logger = get_logger("scientist-agent")
    # agent.reset()
    agent.config = getattr(agent, "config", {})
    agent.config["user_prompt"] = task
    result = agent.run(task, max_steps, controller=controller)

    log_task_completion(
        completed=result.get("completed", False),
        stats=getattr(agent.llm, "get_usage_stats", lambda: {})()
        if hasattr(agent, "llm")
        else {},
    )

    usage_stats = {}
    if hasattr(agent, "llm") and hasattr(agent.llm, "get_usage_stats"):
        usage_stats = agent.llm.get_usage_stats()
    if hasattr(agent, "episodic_memory") and agent.episodic_memory is not None:
        agent.episodic_memory.save()

    trajectory = [event_to_trajectory(event) for event in agent.event_history]
    trajectory_path = os.path.join(agent.working_dir, "trajectory.json")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    return {
        "task": task,
        "result": result,
        "usage_stats": usage_stats,
        "trajectory_path": trajectory_path,
    }


def main(config, working_dir):
    logger = get_logger("main")

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

    # Create AgentController
    agent = Scientist(
        agent_id=config["agent"]["agent_id"], config=config, working_dir=working_dir, event_stream=event_stream
    )
    agent_controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        iteration_delta=config["max_steps"],
        file_store=file_store,
        user_id="user_id",
        sid="session_id"
    )

    restored_state = agent_controller.restore_state("session_id", "user_id")
    agent_controller.state = restored_state
    agent.event_history = list(event_stream.search_events(start_id=0, reverse=False, filter=None))  # Restore full event history
    print("Session state restored.")
    print("[DEBUG] event_history after restore:", len(agent.event_history))

    results = run_agent(
        agent=agent,
        task=config["agent"]["task"],
        max_steps=config["max_steps"],
        controller=agent_controller,
    )

    output_file = config["output_file"]
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The First AI PhD: `gpt-scientist`.")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file"
    )
    args, remaining_args = parser.parse_known_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    # Allow command-line overrides after config is loaded
    parser = argparse.ArgumentParser(description="Run scientist agent on a task")
    parser.add_argument("--task", type=str, help="Task for the agent to complete")
    parser.add_argument("--agent-id", type=str, help="Agent ID")
    parser.add_argument("--max-steps", type=int, help="Maximum steps")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    # parser.add_argument(
    #     "--verbose",
    #     action="store_false",
    #     dest="verbose",
    #     help="Disable detailed output",
    # )
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

    args = parser.parse_args(remaining_args)

    if args.task is not None:
        config["agent"]["task"] = args.task
    if args.agent_id is not None:
        config["agent"]["agent_id"] = args.agent_id
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.output is not None:
        config["output_file"] = args.output
    if args.debug:
        config["agent"]["debug"] = True
    # if not args.verbose:
    #     config["agent"]["verbose"] = False
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

    main(config, args.work_dir)

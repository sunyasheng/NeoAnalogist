import os
import shutil
import time
from typing import Any, Dict, Optional, Tuple


class WorkspaceManager:
    @staticmethod
    def initialize(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
        working_dir = WorkspaceManager.create_workspace(config)
        updated_config = WorkspaceManager.update_config_paths(config, working_dir)
        log_file_path = WorkspaceManager.get_log_file_path(working_dir)
        return working_dir, updated_config, log_file_path

    @staticmethod
    def create_workspace(config: Dict[str, Any]) -> str:
        workspace_dir = config.get("environment", {}).get("working_dir", "workspace")

        if not os.path.isabs(workspace_dir):
            workspace_dir = os.path.join(os.getcwd(), workspace_dir)

        os.makedirs(workspace_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        working_dir = os.path.join(workspace_dir, timestamp)

        # Create directory structure
        for subdir in ["data", "data/logs"]:
            os.makedirs(os.path.join(working_dir, subdir), exist_ok=True)

        # Copy paperbench papers to workspace if paper_path is provided
        paper_path = config.get("paper_path")
        if paper_path and os.path.exists(paper_path):
            target_path = os.path.join(working_dir, paper_path)
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)

            if os.path.isdir(paper_path):
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.copytree(paper_path, target_path)
            else:
                shutil.copy2(paper_path, target_path)

            print(f"Copied paper files from {paper_path} to {target_path}")
        
        env_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        shutil.copy2(env_key_path, os.path.join(working_dir, 'agent.env'))

        print(f"Created new workspace: {working_dir}")
        return working_dir

    @staticmethod
    def update_config_paths(config: Dict[str, Any], working_dir: str) -> Dict[str, Any]:
        updated_config = config.copy()

        if "memory" in updated_config:
            memory_path = os.path.join(working_dir, "data", "memory")
            updated_config["memory"]["memory_path"] = memory_path

        return updated_config

    @staticmethod
    def get_log_file_path(working_dir: str) -> str:
        return os.path.join(working_dir, "data", "logs", "agent.log")

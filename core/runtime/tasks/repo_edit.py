"""
Repository Edit Task for applying user-specified edits to a repository.

This task provides a framework for editing repository code based on user instructions.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import sys
import traceback
from dotenv import load_dotenv
from pathlib import Path
import os

# Aider imports
try:
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput
    from aider.repo import GitRepo
except ImportError:
    Coder = None
    Model = None
    InputOutput = None

from core.events.observation.repo import RepoEditObservation

logger = logging.getLogger(__name__)

@dataclass
class RepoEditInput:
    """Input data structure for repo edit task."""
    repo_path: str
    edit_description: str  # User's description of the edit to perform
    traceback: str = ""   # Optional: error message and traceback if this is a bug fix request

class RepoEditTask:
    """
    Task for editing repository code based on user instructions.
    Uses aider to apply edits to the codebase, following the official aider API example.
    """
    def __init__(self, repo_path: str, edit_description: str, traceback: str = ""):
        self.repo_path = repo_path
        self.edit_description = edit_description
        self.traceback = traceback

    async def run(self) -> RepoEditObservation:
        start_time = time.time()
        try:
            # Load environment variables from .env files (like other tasks)
            env_locations = [
                os.path.join(self.repo_path, ".env"),
                ".env",
                "../.env",
                "../../.env",
                os.path.expanduser("~/.env")
            ]
            for env_path in env_locations:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from: {env_path}")
                    break
            else:
                load_dotenv()
                logger.info("Attempted to load .env from current directory")

            logger.info(f"RepoEditTask started for repo: {self.repo_path}")
            logger.info(f"Edit description: {self.edit_description}")
            if self.traceback:
                logger.info(f"Traceback provided: {self.traceback}")

            if Coder is None or Model is None or InputOutput is None:
                raise ImportError("aider is not installed in the environment.")

            # Use official aider API example
            from pathlib import Path
            repo_path = Path(self.repo_path).resolve()
            # Ensure the directory is a git repo (for aider isolation)
            if not (repo_path / ".git").exists():
                import subprocess
                subprocess.run(["git", "init"], cwd=str(repo_path), check=True)
            cwd = os.getcwd()
            os.chdir(repo_path)
            try:
                # Find all relevant files (support .py, .sh, .md, .txt, .yaml, .yml; skip venv, .git, __pycache__)
                include_exts = [".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".MD"]
                files = []
                
                # Define directories and patterns to exclude
                exclude_dirs = {"venv", ".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
                exclude_patterns = {
                    "outputs",  # Exclude outputs directory completely
                    "logs",     # Exclude logs directory
                    "cache",    # Exclude cache directories
                    "tmp",      # Exclude temporary directories
                    "temp",     # Exclude temporary directories
                    ".cache",   # Exclude .cache directories
                    ".log"
                }
                
                # Define file patterns that are likely not code files
                non_code_patterns = {
                    ".hydra",   # Hydra config files
                    "lightning_logs",  # PyTorch Lightning logs
                    "wandb",    # Weights & Biases logs
                    "tensorboard",  # TensorBoard logs
                    "checkpoints",   # Model checkpoints
                    "weights",   # Weight files
                    "results",   # Results files
                    "experiments",  # Experiment outputs
                    "runs",      # ML experiment runs
                    "artifacts", # ML artifacts
                }
                
                for ext in include_exts:
                    for path in repo_path.rglob(f"*{ext}"):
                        # Skip if any part of the path contains excluded directories
                        if any(skip in path.parts for skip in exclude_dirs):
                            continue
                        
                        # Skip if any part of the path contains excluded patterns
                        if any(pattern in str(path) for pattern in exclude_patterns):
                            continue
                        
                        # Skip if any part of the path contains non-code patterns
                        if any(pattern in str(path) for pattern in non_code_patterns):
                            continue
                        
                        files.append(str(path.relative_to(repo_path)))
                if not files:
                    raise FileNotFoundError("No relevant files found in the repository to edit.")
                logger.info(f"[Aider] Found {len(files)} code files to edit: {files}")
                
                # model = Model("gpt-4o")
                model = Model("sonnet") # Aider expects short model id; Anthropic key from .env
                io = InputOutput(yes=True)
                workspace_root = str(repo_path)
                # Manually create GitRepo with subtree_only=True
                # repo = GitRepo(
                #     io=io,
                #     fnames=files,
                #     git_dname=None,
                #     # subtree_only=True,
                # )
                # Step 1: Planning with architect mode
                architect = Coder.create(main_model=model, io=io, fnames=files, edit_format="architect")
                architect.root = workspace_root
                plan_prompt = f"Please plan how to implement: {self.edit_description}"
                if self.traceback:
                    plan_prompt += f"\nError/Traceback:\n{self.traceback}"
                plan = architect.run(plan_prompt)

                # Step 2: Implementation with code mode
                # coder = Coder.create(main_model=model, io=io, fnames=files, edit_format="code", repo=repo)
                coder = Coder.create(main_model=model, io=io, fnames=files)
                coder.root = workspace_root
                code_prompt = f"I wish {self.edit_description}. Please implement the functionality according to the above plan: {plan}"
                if self.traceback:
                    code_prompt += f"\nError/Traceback:\n{self.traceback}"
                exec_result = coder.run(code_prompt)
                result = str(plan) + str(exec_result)
                logger.info("[Aider] Completed editing and committed the changes.")

            finally:
                os.chdir(cwd)

            execution_time = time.time() - start_time
            
            # Truncate modified files list to prevent overwhelming output
            max_files_to_show = 10
            if len(files) <= max_files_to_show:
                modified_files_str = ", ".join(files)
            else:
                shown_files = files[:max_files_to_show]
                remaining_count = len(files) - max_files_to_show
                modified_files_str = ", ".join(shown_files) + f" (and {remaining_count} more files)"
            
            return RepoEditObservation(
                content=str(result),
                success=True,
                output=str(result),
                modified_files="", # skip showing modified files
#                modified_files=modified_files_str,
                suggestions="",
                execution_time=execution_time,
                error_message="",
                summary=f"Edit applied: {self.edit_description}"
            )
        except Exception as e:
            logger.error(f"RepoEditTask failed: {e}")
            logger.error(traceback.format_exc())
            execution_time = time.time() - start_time
            return RepoEditObservation(
                content="",
                success=False,
                output="",
                modified_files="",
                suggestions="",
                execution_time=execution_time,
                error_message=str(e),
                summary=traceback.format_exc()
            ) 
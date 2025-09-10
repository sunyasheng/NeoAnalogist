import logging
import os
import time
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from core.events.action import ExperimentManagerAction
from core.events.observation.experiment import ExperimentManagerObservation, ExperimentInfo
from core.llm.interface import LLMInterface
from core.utils.logger import get_logger

logger = get_logger(__name__)

class ExperimentManagerTask:
    """Task for generating MLflow wrapper scripts for bash commands."""
    
    def __init__(self, output_dir: str = "workspace/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
    
    def _generate_experiment_name(self, command: str = "") -> str:
        """Generate a default experiment name if not provided"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if command:
            # Extract a meaningful name from the command
            cmd_parts = command.split()
            if cmd_parts:
                base_name = cmd_parts[0].replace('/', '_').replace('.', '_')
                return f"{base_name}_{timestamp}"
        return f"experiment_{timestamp}"
    
    def _parse_bash_command(self, command: str) -> Dict[str, Any]:
        """Parse bash command to extract parameters and arguments"""
        try:
            # Split command into parts
            parts = command.split()
            if not parts:
                return {"script": "", "args": [], "params": {}}
            
            script = parts[0]
            args = parts[1:]
            params = {}
            
            # Extract key-value pairs (--key value format)
            i = 0
            while i < len(args):
                if args[i].startswith('--') and i + 1 < len(args):
                    key = args[i][2:]  # Remove --
                    value = args[i + 1]
                    params[key] = value
                    i += 2
                else:
                    i += 1
            
            return {
                "script": script,
                "args": args,
                "params": params
            }
        except Exception as e:
            logger.error(f"Error parsing bash command: {e}")
            return {"script": command, "args": [], "params": {}}
    
    def _generate_mlflow_wrapper(self, experiment_name: str, command: str, repo_path: str = "") -> str:
        """Generate MLflow wrapper script for the given command"""
        parsed = self._parse_bash_command(command)
        script = parsed["script"]
        params = parsed["params"]
        args_all = parsed["args"]

        # Extract positional args (exclude --key value pairs)
        positional_args: list[str] = []
        i = 0
        while i < len(args_all):
            token = args_all[i]
            if token.startswith("--") and i + 1 < len(args_all):
                # skip key and its value
                i += 2
                continue
            positional_args.append(token)
            i += 1

        # Determine the primary python script (e.g., main.py) if present
        run_py = ""
        for tok in positional_args:
            if tok.endswith('.py'):
                run_py = tok
                break

        # Compute run working directory: directory of the wrapped script
        run_cwd = ""
        if run_py:
            # If repo_path provided, join with script's parent; else use script's parent as-is
            try:
                from pathlib import Path as _P
                if repo_path:
                    run_cwd = str((_P(repo_path) / _P(run_py).parent).resolve())
                else:
                    run_cwd = str(_P(run_py).parent.resolve())
            except Exception:
                run_cwd = repo_path or ""
        
        # Generate parameter list for function signature
        param_names = list(params.keys())
        param_signature = ", ".join(param_names)
        param_logging = "\n        ".join([f'mlflow.log_param("{key}", {key})' for key in param_names])
        param_args = " ".join([f'["--{key}", str({key})]' for key in param_names])
        
        # Generate the wrapper script
        wrapper_content = f'''import subprocess
import mlflow
import sys
from pathlib import Path

def run_experiment({param_signature}):
    """Run experiment with MLflow tracking"""
    experiment_name = "{experiment_name}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        {param_logging}
        
        # Log additional metadata
        mlflow.log_param("script", "{script}")
        mlflow.log_param("repo_path", "{repo_path}")
        
        # Build command arguments
        cmd_args = ["{script}"]
        {f"cmd_args.extend([{', '.join([repr(a) for a in positional_args])}])" if positional_args else ""}
        cmd_args.extend([item for sublist in [{param_args}] for item in sublist])
        
        print(f"Running command: {{' '.join(cmd_args)}}")
        
        # Execute the command (inherit current working directory)
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                cwd=None
            )
            
            # Log outputs
            if result.stdout:
                mlflow.log_text(result.stdout, "stdout.txt")
                print("STDOUT:", result.stdout)
            
            if result.stderr:
                mlflow.log_text(result.stderr, "stderr.txt")
                print("STDERR:", result.stderr)
            
            # Log return code
            mlflow.log_param("exit_code", result.returncode)
            
            if result.returncode == 0:
                print("✅ Experiment completed successfully")
                mlflow.log_param("status", "success")
                return True
            else:
                print(f"❌ Experiment failed with exit code {{result.returncode}}")
                mlflow.log_param("status", "failed")
                # Explicitly mark the MLflow run as FAILED
                mlflow.end_run(status="FAILED")
                return False
            
        except Exception as e:
            error_msg = f"Error running experiment: {{str(e)}}"
            mlflow.log_text(error_msg, "error.txt")
            mlflow.log_param("status", "error")
            # Explicitly mark the MLflow run as FAILED on exception
            mlflow.end_run(status="FAILED")
            print(f"❌ {{error_msg}}")
            return False

if __name__ == "__main__":
    # Example usage - modify these values as needed
    # You can also pass arguments from command line
    if len(sys.argv) > 1:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser()
        {chr(10).join([f'parser.add_argument("--{key}", type=str, default="{value}")' for key, value in params.items()])}
        args = parser.parse_args()
        
        # Run with command line arguments
        run_experiment({', '.join([f'args.{key}' for key in param_names])})
    else:
        # Run with default values
        run_experiment({', '.join([f'"{value}"' for key, value in params.items()])})
'''
        
        return wrapper_content
    
    def _create_wrapper_file(self, experiment_name: str, command: str, repo_path: str = "") -> str:
        """Create MLflow wrapper file"""
        try:
            # Generate wrapper content
            wrapper_content = self._generate_mlflow_wrapper(experiment_name, command, repo_path)
            
            # Create wrapper file path
            wrapper_filename = f"{experiment_name}_mlflow_wrapper.py"
            wrapper_path = self.output_dir / wrapper_filename
            
            # Write the wrapper file
            with open(wrapper_path, 'w', encoding='utf-8') as f:
                f.write(wrapper_content)
            
            logger.info(f"Created MLflow wrapper: {wrapper_path}")
            return str(wrapper_path)
            
        except Exception as e:
            logger.error(f"Error creating wrapper file: {e}")
            raise
    
    def _list_experiments_from_mlflow(self) -> List[ExperimentInfo]:
        """List experiments using MLflow API"""
        import mlflow
        
        experiments = []
        
        # Get all experiments
        experiment_list = mlflow.search_experiments()
        
        for exp in experiment_list:
            try:
                # Get experiment details
                experiment = mlflow.get_experiment(exp.experiment_id)
                
                # Search for runs in this experiment
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=10  # Limit to recent runs
                )
                
                # Skip MLflow default experiment if it has no runs
                exp_name = (experiment.name or "").strip()
                is_default_exp = str(experiment.experiment_id) in {"0", "0.0"} or exp_name.lower() == "default"
                if is_default_exp and (runs is None or getattr(runs, 'empty', True)):
                    # ignore default empty experiment
                    continue

                # Get the most recent run for status
                status = "no_runs"
                if not runs.empty:
                    latest_run = runs.iloc[0]
                    status = latest_run.get('status', 'unknown')
                    # Heuristic: if our wrapper logged params.status/exit_code, override to FAILED when appropriate
                    try:
                        p_status = latest_run.get('params.status', '')
                        p_exit = latest_run.get('params.exit_code', '')
                        if (isinstance(p_status, str) and p_status.lower() == 'failed') or (
                            isinstance(p_exit, str) and p_exit and p_exit != '0'
                        ):
                            status = 'FAILED'
                    except Exception:
                        pass
                
                # Compute created_at as ISO string (mlflow returns creation_time as int millis)
                _created_iso = ""
                try:
                    from datetime import datetime, timezone
                    if experiment.creation_time:
                        if isinstance(experiment.creation_time, (int, float)):
                            _created_iso = datetime.fromtimestamp(
                                float(experiment.creation_time) / 1000.0, tz=timezone.utc
                            ).isoformat()
                        else:
                            # Fallback if some provider returns datetime-like
                            _created_iso = str(experiment.creation_time)
                except Exception:
                    _created_iso = ""

                # Create experiment info
                exp_info = ExperimentInfo(
                    name=exp_name or f"experiment_{exp.experiment_id}",
                    created_at=_created_iso,
                    status=status,
                    wrapper_path=f"MLflow Experiment: {exp_name or exp.experiment_id}"
                )
                experiments.append(exp_info)
                
            except Exception as e:
                logger.warning(f"Error getting details for experiment {exp.experiment_id}: {e}")
                # On error, skip adding a placeholder to avoid noisy/ambiguous results
                continue
        
        return experiments
    
    def _list_experiments(self) -> List[ExperimentInfo]:
        """List all experiments using MLflow API"""
        return self._list_experiments_from_mlflow()
    
    def _create_experiment(self, name: str, command: str = "", repo_path: str = "") -> ExperimentInfo:
        """Create a new experiment and generate wrapper"""
        # Generate MLflow wrapper
        wrapper_path = self._create_wrapper_file(name, command, repo_path)
        
        experiment = ExperimentInfo(
            name=name,
            created_at=datetime.now().isoformat(),
            status="created",
            wrapper_path=wrapper_path
        )
        
        return experiment
    
    def run(self, action: ExperimentManagerAction) -> ExperimentManagerObservation:
        """
        Execute experiment manager task - generate MLflow wrapper
        
        Args:
            action: ExperimentManagerAction containing mode and parameters
            
        Returns:
            ExperimentManagerObservation with operation results
        """
        start_time = time.time()
        
        try:
            mode = action.mode
            command = action.command
            experiment_name = action.experiment_name
            repo_path = action.repo_path
            
            logger.info(f"Running experiment manager in mode: {mode}")
            logger.info(f"Command: {command}")
            logger.info(f"Experiment name: {experiment_name}")
            logger.info(f"Repo path: {repo_path}")
            
            if mode in ("list", "query"):
                # List experiments using MLflow API
                # If a repo_path is provided, point MLflow to that repo's mlruns directory
                try:
                    if repo_path:
                        import os as _os
                        import mlflow as _mlflow
                        mlruns_dir = repo_path
                        # If repo_path is a submission directory, use its mlruns subdir
                        if not mlruns_dir.endswith("mlruns"):
                            mlruns_dir = _os.path.join(mlruns_dir, "mlruns")
                        tracking_uri = f"file://{mlruns_dir}"
                        _mlflow.set_tracking_uri(tracking_uri)
                        logger.info(f"Set MLflow tracking URI to {tracking_uri}")
                except Exception as _e:
                    logger.warning(f"Failed to set tracking URI from repo_path: {repo_path}. Error: {_e}")

                experiments = self._list_experiments()

                # Build optional detailed summaries (all runs reason/metrics)
                detailed_summaries: dict[str, list[str]] = {}
                try:
                    import mlflow
                    import os
                    exp_list = mlflow.search_experiments()
                    for exp in exp_list:
                        try:
                            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=10, order_by=["attributes.start_time DESC"])
                            details: list[str] = []
                            if runs is not None and not runs.empty:
                                details.append(f"Total runs: {len(runs)}")
                                for i, run in runs.iterrows():
                                    run_id = run.get('run_id', 'unknown')
                                    run_name = run.get('tags.mlflow.runName', 'unnamed')
                                    status_str = run.get('status', 'unknown')
                                    details.append(f"  Run {i+1}: {run_name} ({run_id[:8]}...) - {status_str}")
                                    
                                    # Extract reason from stderr.txt if failed
                                    reason_lines = []
                                    art_uri = run.get('artifact_uri', '') or ''
                                    if isinstance(art_uri, str) and art_uri.startswith('file://'):
                                        stderr_path = art_uri[7:]  # strip file://
                                        stderr_path = os.path.join(stderr_path, 'stderr.txt')
                                        try:
                                            if os.path.exists(stderr_path):
                                                with open(stderr_path, 'r', encoding='utf-8', errors='ignore') as f:
                                                    content = f.read().strip()
                                                    if content:
                                                        # Split into lines and take last 10 lines (most relevant)
                                                        lines = content.splitlines()
                                                        reason_lines = lines[-10:] if len(lines) > 10 else lines
                                        except Exception:
                                            pass
                                    
                                    # Summarize common metrics if present
                                    metric_keys = [c for c in run.index if isinstance(c, str) and c.startswith('metrics.')]
                                    metric_items = []
                                    for mk in metric_keys:
                                        try:
                                            val = run.get(mk)
                                            if val is not None and val == val:
                                                metric_items.append(f"{mk[8:]}={val}")
                                        except Exception:
                                            continue
                                    
                                    if metric_items:
                                        details.append(f"    metrics: " + ", ".join(metric_items[:3]))
                                    if reason_lines:
                                        details.append(f"    error details:")
                                        for line in reason_lines:
                                            details.append(f"      {line}")
                                    
                                    # Only show first 3 runs to avoid too much output
                                    if i >= 2:
                                        break
                                
                                if details:
                                    # Map by experiment name
                                    name_key = exp.name or f"experiment_{exp.experiment_id}"
                                    detailed_summaries[name_key] = details
                        except Exception:
                            continue
                except Exception:
                    pass

                content_parts = []
                content_parts.append(f"Experiment Manager Task completed successfully in {time.time() - start_time:.2f} seconds")
                content_parts.append(f"Mode: {mode}")
                
                if experiments:
                    content_parts.append(f"\nExperiments ({len(experiments)}):")
                    for exp in experiments:
                        content_parts.append(f"- {exp.name}: {exp.status}")
                        if exp.wrapper_path:
                            content_parts.append(f"  Wrapper: {exp.wrapper_path}")
                        # Append details if available
                        det = detailed_summaries.get(exp.name, [])
                        for d in det:
                            content_parts.append(f"  {d}")
                else:
                    content_parts.append("\nNo experiments found.")
                
                content = "\n".join(content_parts)
                
                return ExperimentManagerObservation(
                    content=content,
                    success=True,
                    execution_time=time.time() - start_time,
                    experiments=experiments,
                    mode=mode
                )
                
            elif mode in ("auto", "wrap"):
                # Auto mode: create experiment and generate wrapper
                if not experiment_name:
                    experiment_name = self._generate_experiment_name(command)
                
                # Create experiment and generate wrapper
                experiment = self._create_experiment(experiment_name, command, repo_path)
                logger.info(f"Created experiment '{experiment_name}' with wrapper: {experiment.wrapper_path}")
                
                # Derive wrapped script and its directory for PYTHONPATH suggestion
                wrapped_script = ""
                wrapped_dir = ""
                try:
                    parsed = self._parse_bash_command(command)
                    args_all = parsed.get("args", [])
                    # collect positional tokens (exclude --k v)
                    positional_tokens: list[str] = []
                    i = 0
                    while i < len(args_all):
                        tok = args_all[i]
                        if tok.startswith("--") and i + 1 < len(args_all):
                            i += 2
                            continue
                        positional_tokens.append(tok)
                        i += 1
                    for tok in positional_tokens:
                        if tok.endswith('.py'):
                            wrapped_script = tok
                            break
                    if wrapped_script:
                        from pathlib import Path as _P
                        if repo_path:
                            wrapped_dir = str((_P(repo_path) / _P(wrapped_script).parent).resolve())
                        else:
                            wrapped_dir = str(_P(wrapped_script).parent.resolve())
                except Exception:
                    wrapped_script = ""
                    wrapped_dir = repo_path or ""

                content_parts = []
                content_parts.append(f"Experiment Manager Task completed successfully in {time.time() - start_time:.2f} seconds")
                content_parts.append(f"Mode: {mode}")
                content_parts.append(f"Created experiment: {experiment_name}")
                content_parts.append(f"Wrapper: {experiment.wrapper_path}")
                content_parts.append(f"\nNext step: Run the wrapper using cmd action:")
                content_parts.append(f"cmd: python {experiment.wrapper_path}")
                content_parts.append(f"\nTip: Add the WRAPPED SCRIPT directory to PYTHONPATH (directory of the .py you wrapped), e.g.:")
                if wrapped_dir:
                    content_parts.append(f"export PYTHONPATH=\\\"{wrapped_dir}\\\":$PYTHONPATH")
                if wrapped_script:
                    content_parts.append(f"(Wrapped file detected: {wrapped_script})")
                
                content = "\n".join(content_parts)
                
                return ExperimentManagerObservation(
                    content=content,
                    success=True,
                    execution_time=time.time() - start_time,
                    experiments=[],  # Don't include experiments list for wrap mode to avoid duplication
                    mode=mode
                )
                
            else:
                error_content = f"Experiment Manager Task failed: Unknown mode '{mode}'"
                return ExperimentManagerObservation(
                    content=error_content,
                    success=False,
                    execution_time=time.time() - start_time,
                    experiments=[],
                    mode=mode,
                    error_message=f"Unknown mode: {mode}"
                )
                
        except Exception as e:
            logger.error(f"Error in experiment manager task: {e}")
            error_content = f"Experiment Manager Task failed: {str(e)}"
            return ExperimentManagerObservation(
                content=error_content,
                success=False,
                execution_time=time.time() - start_time,
                experiments=[],
                mode=action.mode,
                error_message=str(e)
            )

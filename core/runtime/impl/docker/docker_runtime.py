import os
import subprocess
import threading

import docker
import requests
import tenacity
from docker.models.containers import Container

from core.events.action import CmdRunAction, FileEditAction
from core.events.action import RepoEditAction
from core.events.event import Observation
from core.events.serialization import event_to_dict
from core.runtime.impl.execution.client import (ActionExecutionClient,
                                                AgentRuntimeError)
from core.runtime.utils.system import find_available_tcp_port
from core.utils.logger import get_logger
from core.utils.tenacity_stop import stop_if_should_exit
from core.utils.env_utils import get_gpu_docker_args

# Import repo run functionality
from core.events.action import RepoRunAction
from core.runtime.tasks.repo_run import RepoRunTask
from core.events.observation.repo import RepoRunObservation
from core.events.observation.repo import RepoEditObservation
from core.runtime.tasks.repo_edit import RepoEditTask
from core.events.action import PaperRubricAction
from core.events.observation.repo import PaperRubricObservation
from core.events.observation.repo import GoTEditObservation
from core.events.action.image import ImageEntityExtractAction, GoTEditAction

# Import PDF query functionality
from core.events.action import PDFQueryAction
from core.runtime.tasks.pdf_query_task import PDFQueryTask
from core.runtime.tasks.got_edit import GoTEditClient
from core.events.observation import PDFQueryObservation

logger = get_logger(__name__)

CONTAINER_NAME_PREFIX = "gpt-scientist-runtime-"

EXECUTION_SERVER_PORT_RANGE = (30000, 39999)
VSCODE_PORT_RANGE = (40000, 49999)
APP_PORT_RANGE_1 = (50000, 54999)
APP_PORT_RANGE_2 = (55000, 59999)


class DockerRuntime(ActionExecutionClient):
    """Docker runtime implementation for managing Docker container lifecycle.

    Inherits from ActionExecutionClient, provides container management functionality.
    """

    def __init__(
        self,
        config,
        sid: str = "default",
        plugins: list = None,
        env_vars: dict = None,
        status_callback=None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        enable_logging: bool = True,
    ):
        super().__init__(
            config=config,
            sid=sid,
            plugins=plugins,
            env_vars=env_vars,
            status_callback=status_callback,
            attach_to_existing=attach_to_existing,
            headless_mode=headless_mode,
        )

        self._runtime_initialized: bool = False
        self.status_callback = status_callback
        self.env_vars = env_vars or {}  # Initialize env_vars

        self._host_port = -1
        self._container_port = -1
        self._vscode_port = -1
        self._app_ports: list[int] = []

        # 添加日志流相关属性
        self._log_thread = None
        self._stop_logging = False
        self._enable_logging = enable_logging

        # Ensure workspace path is absolute
        if not os.path.isabs(self.config.workspace_base):
            self.config.workspace_base = os.path.abspath(self.config.workspace_base)

        # Create necessary directories
        os.makedirs(self.config.workspace_base, exist_ok=True)
        os.makedirs(os.path.join(self.config.workspace_base, "logs"), exist_ok=True)

        self.docker_client: docker.DockerClient = self._init_docker_client()
        self.container_name = CONTAINER_NAME_PREFIX + sid
        self.container: Container | None = None

    def _init_docker_client(self) -> docker.DockerClient:
        """Initialize Docker client"""
        try:
            return docker.from_env()
        except Exception as ex:
            logger.error(
                "Failed to launch Docker client. Please ensure Docker is installed and Docker Desktop/daemon is running."
            )
            raise ex

    def _find_available_port(self, port_range: tuple[int, int]) -> int:
        """Find available port"""
        return find_available_tcp_port(port_range[0], port_range[1])

    def _init_container(self):
        """Initialize container configuration"""
        logger.debug("Preparing to start container...")
        self._host_port = self._find_available_port(EXECUTION_SERVER_PORT_RANGE)
        self._container_port = self._host_port
        self._vscode_port = self._find_available_port(VSCODE_PORT_RANGE)
        self._app_ports = [
            self._find_available_port(APP_PORT_RANGE_1),
            self._find_available_port(APP_PORT_RANGE_2),
        ]
        self.api_url = f"http://localhost:{self._host_port}"

    def _cleanup_existing_container(self):
        """Clean up existing container with the same name if it exists"""
        try:
            existing_container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Found existing container: {self.container_name}")
            try:
                existing_container.stop(timeout=10)
                existing_container.remove()
                logger.info(f"Removed existing container: {self.container_name}")
            except Exception as e:
                logger.error(f"Failed to remove existing container: {str(e)}")
                # Try force removal
                try:
                    existing_container.remove(force=True)
                    logger.info(
                        f"Force removed existing container: {self.container_name}"
                    )
                except Exception as e:
                    logger.error(f"Failed to force remove existing container: {str(e)}")
                    raise
        except docker.errors.NotFound:
            # No existing container found, that's fine
            pass
        except Exception as e:
            logger.error(f"Error checking for existing container: {str(e)}")
            raise

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception_type(
            (ConnectionError, requests.exceptions.ConnectionError, AgentRuntimeError)
        ),
        reraise=True,
        wait=tenacity.wait_fixed(2),
    )
    def _wait_for_server(self):
        """Wait for server to be ready"""
        logger.info(f"Waiting for server to be ready at {self.api_url}...")

        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status == "exited":
                logs = container.logs(tail=50).decode("utf-8")
                raise RuntimeError(
                    f"Container {self.container_name} has exited. Logs:\n{logs}"
                )
        except docker.errors.NotFound:
            raise RuntimeError(f"Container {self.container_name} not found.")

        self.check_if_alive()
        logger.info("Server is ready")
        return True

    def got_edit(self, action: GoTEditAction) -> GoTEditObservation:
        """Call GoT API to edit an image with a prompt via the action execution server."""
        return self.send_action_for_execution(action)

    def _log_stream(self):
        """处理容器日志流的函数"""
        try:
            buffer = ""
            for line in self.container.logs(stream=True, follow=True):
                if self._stop_logging:
                    break
                try:
                    # 尝试使用 utf-8 解码
                    decoded_line = line.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        # 如果 utf-8 失败，尝试使用 latin-1
                        decoded_line = line.decode("latin-1")
                    except Exception:
                        # 如果都失败了，使用 repr 显示原始字节
                        decoded_line = repr(line)

                # 将解码后的内容添加到缓冲区
                buffer += decoded_line

                # 如果遇到换行符，输出并清空缓冲区
                if "\n" in buffer:
                    lines = buffer.split("\n")
                    # 输出除最后一行外的所有行
                    for line in lines[:-1]:
                        if line.strip():  # 只输出非空行
                            print(f"Container log: {line}")
                            logger.debug(f"Container log: {line}")
                    # 保留最后一行（可能不完整）
                    buffer = lines[-1]

        except Exception as e:
            logger.error(f"Error in log stream: {str(e)}")

    def start_container(self, image: str):
        """Start container"""
        # Clean up any existing container first
        self._cleanup_existing_container()

        self._init_container()

        # Configure port mappings
        port_bindings = {
            f"{self._container_port}/tcp": self._host_port,
            f"{self._vscode_port}/tcp": self._vscode_port,
        }
        for port in self._app_ports:
            port_bindings[f"{port}/tcp"] = port

        # Verify workspace path exists and is accessible
        if not os.path.exists(self.config.workspace_base):
            raise RuntimeError(
                f"Workspace path does not exist: {self.config.workspace_base}"
            )

        if not os.access(self.config.workspace_base, os.R_OK | os.W_OK):
            raise RuntimeError(
                f"Workspace path is not accessible: {self.config.workspace_base}"
            )

        logger.info(f"Starting container with ports: {port_bindings}")
        logger.info(f"Workspace path: {self.config.workspace_base}")

        try:
            # Get project root directory
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            )

            # Get GPU support arguments if available
            gpu_args = get_gpu_docker_args()
            if gpu_args:
                logger.info("NVIDIA GPU detected, enabling GPU support in container")
            else:
                logger.info("No NVIDIA GPU detected, running in CPU mode")

            # Prepare environment variables
            environment_vars = {
                **self.env_vars,
                "DEBUG": "True",
                "PYTHONPATH": "/app_sci",
                "RUNTIME_MEMORY_MONITOR": "True",
                "LOG_LEVEL": "DEBUG",
            }
            
            # Add GPU environment variables if available
            if gpu_args.get("environment"):
                environment_vars.update(gpu_args["environment"])

            # Prepare container run arguments
            run_kwargs = {
                "image": image,
                "name": self.container_name,
                "command": [
                    "/opt/conda/envs/server/bin/python",
                    "-u",
                    "/app_sci/core/runtime/server.py",
                    "--working-dir",
                    self.config.workspace_base.replace(
                        os.path.join(os.getcwd(), "workspace"),
                        "/app_sci/workspace",
                    ),
                    "--port",
                    str(self._container_port),
                    "--username",
                    "scientist",
                    "--user-id",
                    "1000",
                    "--plugins", "jupyter",
                ],
                "detach": True,
                "ports": port_bindings,
                "environment": environment_vars,
                "volumes": {
                    project_root: {"bind": "/app_sci", "mode": "rw"},
                    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                },
                "remove": False,
                "tty": True,
                "stdin_open": True,
                "shm_size": "2g",  # Increase shared memory to 2GB for PyTorch operations
            }

            # Add GPU support if available
            if gpu_args.get("gpus"):
                # For Python docker library, we need to use device_requests for GPU support
                import docker.types
                run_kwargs["device_requests"] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]

            self.container = self.docker_client.containers.run(**run_kwargs)

            logger.info(f"Container started: {self.container_name}")

            # 根据enable_logging标志决定是否启动日志流处理
            if self._enable_logging:
                self._stop_logging = False
                self._log_thread = threading.Thread(
                    target=self._log_stream, daemon=True
                )
                self._log_thread.start()

            # Wait for server to be ready
            self._wait_for_server()

            return True
        except Exception as e:
            logger.error(f"Failed to start container: {str(e)}")
            if self.container:
                try:
                    logs = self.container.logs(tail=50).decode("utf-8")
                    logger.error(f"Container logs:\n{logs}")
                except Exception as log_e:
                    logger.error(f"Failed to get container logs: {str(log_e)}")
            raise

    def stop_container(self):
        """Stop and remove container"""
        if not self.container:
            logger.info(f"No container to stop: {self.container_name}")
            return

        try:
            # 停止日志流处理
            self._stop_logging = True
            if self._log_thread and self._log_thread.is_alive():
                self._log_thread.join(timeout=5)

            # First try to stop the container
            try:
                self.container.stop(timeout=10)
                logger.info(f"Container stopped: {self.container_name}")
            except docker.errors.NotFound:
                logger.info(
                    f"Container already stopped or removed: {self.container_name}"
                )
                return
            except Exception as e:
                logger.error(f"Failed to stop container: {str(e)}")
                return

            # Then try to remove it
            try:
                self.container.remove()
                logger.info(f"Container removed: {self.container_name}")
            except docker.errors.NotFound:
                logger.info(f"Container already removed: {self.container_name}")
            except Exception as e:
                logger.error(f"Failed to remove container: {str(e)}")
                # Try force removal if normal removal fails
                try:
                    self.container.remove(force=True)
                    logger.info(f"Container force removed: {self.container_name}")
                except docker.errors.NotFound:
                    logger.info(f"Container already removed: {self.container_name}")
                except Exception as e:
                    logger.error(f"Failed to force remove container: {str(e)}")
        finally:
            # Clear container reference
            self.container = None
            self._log_thread = None

    def get_container_status(self) -> str:
        """Get container status"""
        if not self.container:
            return "Not created"
        try:
            self.container.reload()
            return self.container.status
        except Exception:
            return "Unknown"

    def get_container_logs(self, tail: int = 100) -> str:
        """Get container logs"""
        if not self.container:
            return "Container not created"
        try:
            return self.container.logs(tail=tail).decode("utf-8")
        except Exception as e:
            return f"Failed to get logs: {str(e)}"

    def close(self):
        """Clean up resources"""
        try:
            # Clean up persistent containers
            if hasattr(self, 'repo_run_task'):
                self.repo_run_task.cleanup_persistent_containers()
            
            # Stop the action execution server
            if hasattr(self, '_action_execution_server') and self._action_execution_server:
                self._action_execution_server.stop()
                self._action_execution_server = None
            
            # Close Docker client
            if hasattr(self, 'docker_client'):
                self.docker_client.close()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_repo_run_task(self):
        """Get or create RepoRunTask instance"""
        if not hasattr(self, 'repo_run_task'):
            from core.runtime.tasks.repo_run import RepoRunTask
            self.repo_run_task = RepoRunTask(self)
        return self.repo_run_task

    def run_server(self, port: int = 8333, debug: bool = True):
        """Run server in Docker container

        Args:
            port: Server port number
            debug: Whether to enable debug mode
        """
        # Get GPU support arguments if available
        gpu_args = get_gpu_docker_args()
        
        # Build Docker run command
        cmd = [
            "docker",
            "run",
            "--rm",
        ]
        
        # Add GPU support if available
        if gpu_args.get("gpus"):
            cmd.extend(["--gpus", gpu_args["gpus"]])
        
        # Add port mapping
        cmd.append(f"-p {port}:{port}")
        
        # Add environment variables
        cmd.append(f"-e DEBUG={str(debug)}")
        
        # Add GPU environment variables if available
        if gpu_args.get("environment"):
            for key, value in gpu_args["environment"].items():
                cmd.append(f"-e {key}={value}")
        
        # Add volume mappings
        cmd.extend([
            f"-v {self.config.workspace_base}:/app_sci",
            f"-v {os.path.join(self.config.workspace_base, 'logs')}:/app_sci/logs",
            "mc-scientist:latest",
            "python",
            "-u",
            "core/runtime/server.py",
            "--working-dir",
            "/app_sci/workspace",
            "--port",
            str(port),
        ])

        # Convert command to string
        cmd_str = " ".join(cmd)

        # Set log file path
        log_file = os.path.join(self.config.workspace_base, "logs", "server.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Run command and write output to both console and log file
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            # Read output in real-time and write to log file
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                f.flush()

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(
                    f"Docker container failed with return code {process.returncode}"
                )

    def execute_command(self, command: str) -> str:
        """Execute a command in the container and return its output

        Args:
            command: Command to execute

        Returns:
            Command output as string
        """
        if not self.container:
            raise RuntimeError("Container is not running")

        try:
            # Create a CmdRunAction for the command
            action = CmdRunAction(command=command)

            # Send POST request to execute_action endpoint
            response = self._send_action_server_request(
                "POST",
                f"{self._get_action_execution_server_host()}/execute_action",
                json={"action": event_to_dict(action)},
                timeout=30,  # Add timeout
            )

            # Parse response
            result = response.json()
            if isinstance(result, dict):
                if "content" in result:
                    return result["content"]
                elif "error" in result:
                    raise RuntimeError(f"Command execution failed: {result['error']}")
                elif "stdout" in result:
                    return result["stdout"]
                elif "stderr" in result:
                    return result["stderr"]
            return str(result)

        except Exception as e:
            logger.error(f"Failed to execute command: {str(e)}")
            raise

    def run_image_entity_extract(
        self,
        image_path: str | None = None,
        model: str = "gpt-4o",
        timeout: int = 180,
    ):
        """Convenience method: ensure server, then run ImageEntityExtractAction.

        Returns the Observation produced by the server.
        """
        from core.events.action.image import ImageEntityExtractAction

        action = ImageEntityExtractAction(
            image_path=image_path,
            model=model,
        )
        if action.timeout is None:
            action.set_hard_timeout(timeout, blocking=False)
        return self.image_entity_extract(action)

    async def _repo_run_async(self, action: RepoRunAction) -> RepoRunObservation:
        """Run a repository's reproduce.sh script in an isolated environment (async impl)"""
        task = self._get_repo_run_task()
        return await task.run(action)

    def repo_run(self, action: RepoRunAction) -> RepoRunObservation:
        """同步包装，保证和 repo-verify 等接口一致"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # 嵌套 event loop 场景（如 Jupyter），用 asyncio.run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(self._repo_run_async(action), loop)
            return future.result()
        else:
            return loop.run_until_complete(self._repo_run_async(action))



def create_test_repo(repo_dir: str) -> None:
    """Create a test Git repository with some sample files.
    
    Args:
        repo_dir: Directory to create the repository in
    """
    os.makedirs(repo_dir, exist_ok=True)
    # Create some sample files
    sample_files = {
        "README.md": "# Test Repository\nThis is a test repository for snapshot testing.",
        "data/sample.txt": "This is a sample data file.",
        "src/main.py": "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()"
    }
    
    for file_path, content in sample_files.items():
        full_path = os.path.join(repo_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    
    print(f"Created test repository in {repo_dir}")

# python -m core.runtime.impl.docker.docker_runtime --browse "https://www.baidu.com"
# python -m core.runtime.impl.docker.docker_runtime --snapshot /submission/ --create-test-repo
# python -m core.runtime.impl.docker.docker_runtime --repo-plan-paper-path debug/data/semantic-self-consistency/paper.md
# python -m core.runtime.impl.docker.docker_runtime --repo-create-paper-path debug/data/lbcs/paper.md
# python -m core.runtime.impl.docker.docker_runtime --repo-verify-path /app_sci/workspace/repo_create/submission --repo-verify-requirement "The repository should be able to run the code and produce the expected output." --repo-verify-level "comprehensive"
# python -m core.runtime.impl.docker.docker_runtime --paper-reproduction-paper-path debug/data/semantic-self-consistency/paper.md --paper-reproduction-analysis-level detailed
# python -m core.runtime.impl.docker.docker_runtime --repo-debug-path /app_sci/workspace/20250624_171358/debug/data/semantic-self-consistency/submission --repo-debug-action "fix syntax errors"
# python -m core.runtime.impl.docker.docker_runtime --repo-run-path workspace/20250702_170640/debug/data/semantic-self-consistency/submission --repo-run-timeout 300 --repo-run-retry-threshold 60 --repo-run-docker-image "pb-reproducer:latest"
# python -m core.runtime.impl.docker.docker_runtime --repo-debug-path /app_sci/workspace/20250624_171358/debug/data/semantic-self-consistency/submission --repo-debug-action "fix the bug "
# cd /Users/suny0a/Proj/MC-Scientist && docker run --rm -v $(pwd):/app_sci mc-scientist:latest python /app_sci/core/runtime/thirdparty/refact_agent/engine/python_binding_and_cmdline/refact/cli_standalone.py /app_sci/workspace/20250624_171358/debug/data/semantic-self-consistency/submission -- -- add a eval function in model.py
# python -m core.runtime.impl.docker.docker_runtime --repo-analyzer-paper-path debug/data/semantic-self-consistency/paper.md --repo-analyzer-codebase-path workspace/20250705_195616/debug/data/semantic-self-consistency/submission
# python -m core.runtime.impl.docker.docker_runtime --repo-update-path workspace/20250707_123830/debug/data/lbcs/submission --repo-update-requirements "Add a README file" --repo-update-apply
# python -m core.runtime.impl.docker.docker_runtime --repo-edit-path workspace/20250721_115015/debug/data/stay-on-topic-with-classifier-free-guidance/submission --repo-edit-description "Fix the syntax erorr in submission"
# python -m core.runtime.impl.docker.docker_runtime --pdf-query-path "debug/data/lbcs/paper.pdf" --pdf-query-question "What is this document about?"
# python -m core.runtime.impl.docker.docker_runtime --ipython-code "import os; print(os.getcwd())"
# python -m core.runtime.impl.docker.docker_runtime --paper-rubric-path evaluation/preparedness/project/paperbench/data/papers/stay-on-topic-with-classifier-free-guidance/paper.pdf --paper-rubric-output-dir workspace/rubrics
# python -m core.runtime.impl.docker.docker_runtime --repo-judge-path workspace/20250721_115015/debug/data/stay-on-topic-with-classifier-free-guidance/submission --repo-judge-rubric-file workspace/rubrics/paper_rubric_20250729_041607.txt
# python -m core.runtime.impl.docker.docker_runtime --exp-manager-mode query --exp-manager-repo-dir "workspace/20250820_071036/debug/data/lbcs/submission"
# python -m core.runtime.impl.docker.docker_runtime --command "cd /app_sci/workspace/20250820_071036/debug/data/lbcs/submission && /opt/conda/envs/clean_env/bin/pip install -r requirements.txt && /opt/conda/envs/clean_env/bin/python /app_sci/workspace/20250820_071036/debug/data/lbcs/mlflow_scripts/lbcs_fashionmnist_experiment_mlflow_wrapper.py --config config.yaml"
# python -m core.runtime.impl.docker.docker_runtime --exp-manager-cmd "python main.py --config config.yaml" --exp-manager-mode wrap --exp-manager-exp-name "lbcs_fashionmnist_experiment" --exp-manager-repo-dir "workspace/20250820_071036/debug/data/lbcs/submission"
# python -m core.runtime.impl.docker.docker_runtime --image-entity-extract-path /app_sci/workspace/imgs/test.jpg
# python -m core.runtime.impl.docker.docker_runtime --image-entity-extract-path /app_sci/workspace/imgs/test.png --image-entity-extract-model gpt-4o


def main():
    """Test function for DockerRuntime class"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1:8333", help="Server address"
    )
    parser.add_argument("--command", type=str, help="Command to execute")
    parser.add_argument("--info", action="store_true", help="Get server info")
    parser.add_argument("--edit", type=str, help="File path to edit")
    parser.add_argument("--content", type=str, help="Content to write to file")
    parser.add_argument("--browse", type=str, help="URL to browse")
    parser.add_argument("--interactive", type=str, help="Interactive browser action")
    parser.add_argument("--task-graph", type=str, help="Task description for building task graph", metavar='DESCRIPTION')
    parser.add_argument("--test-rollback", action="store_true", help="Test snapshot and rollback functionality")
    parser.add_argument("--snapshot", type=str, default="submission", help="Snapshot directory")
    parser.add_argument("--repo-plan-paper-path", type=str, help="Paper path for repository planning", metavar='PAPER_PATH')
    parser.add_argument("--repo-create-paper-path", type=str, help="Paper path for repository creation with code generation", metavar='PAPER_PATH')
    parser.add_argument("--repo-analyzer-paper-path", type=str, help="Paper path for repository analysis", metavar='PAPER_PATH')
    parser.add_argument("--repo-analyzer-codebase-path", type=str, help="Codebase path for repository analysis", metavar='CODEBASE_PATH')
    parser.add_argument("--repo-update-path", type=str, help="Repository path for code updates", metavar='REPO_PATH')
    parser.add_argument("--repo-update-requirements", type=str, help="Requirements for repository updates", metavar='REQUIREMENTS')
    parser.add_argument("--repo-update-apply", action="store_true", help="Apply the generated changes to files")
    parser.add_argument("--repo-update-snapshot", action="store_true", default=True, help="Save snapshot after applying changes")
    parser.add_argument("--repo-verify-path", type=str, help="Repository path to verify", metavar='REPO_PATH')
    parser.add_argument("--repo-verify-requirement", type=str, help="Requirement description for verification", metavar='REQUIREMENT')
    parser.add_argument("--repo-verify-level", type=str, choices=["basic", "functional", "comprehensive"], default="comprehensive", help="Verification level")
    parser.add_argument("--repo-debug-path", type=str, help="Repository path to debug", metavar='REPO_PATH')
    parser.add_argument("--repo-debug-action", type=str, help="Action description for debugging", metavar='ACTION_DESCRIPTION')
    parser.add_argument("--paper-reproduction-paper-path", type=str, help="Paper path for reproduction analysis", metavar='PAPER_PATH')
    parser.add_argument("--paper-reproduction-analysis-level", type=str, choices=["basic", "detailed", "comprehensive"], default="detailed", help="Analysis level for paper reproduction")
    parser.add_argument("--paper-reproduction-output-file", type=str, help="Output file for paper reproduction analysis results", metavar='OUTPUT_FILE')
    parser.add_argument("--repo-run-path", type=str, help="Repository path to run reproduce.sh script", metavar='REPO_PATH')
    parser.add_argument("--repo-run-timeout", type=int, default=3600, help="Timeout for reproduce.sh script execution in seconds")
    parser.add_argument("--repo-run-retry-threshold", type=int, default=600, help="Retry threshold for reproduce.sh script execution in seconds")
    parser.add_argument("--repo-run-docker-image", type=str, default="pb-reproducer:latest", help="Docker image to use for isolation")
    parser.add_argument("--repo-run-memory-limit", type=str, default="4g", help="Memory limit for container (e.g., 4g, 8g)")
    parser.add_argument("--repo-run-network", action="store_true", default=True, help="Enable network access in container (default: True)")
    parser.add_argument("--repo-run-no-network", action="store_true", help="Disable network access in container")
    parser.add_argument("--repo-run-gpu", action="store_true", default=True, help="Enable GPU access in container (default: True)")
    parser.add_argument("--repo-run-no-gpu", action="store_true", help="Disable GPU access in container")
    parser.add_argument("--repo-run-log-file", type=str, default="experiment.log", help="Path to save execution logs with timestamp (default: experiment.log)", metavar='LOG_FILE')
    parser.add_argument("--repo-edit-path", type=str, help="Repository path for code edits", metavar='REPO_PATH')
    parser.add_argument("--repo-edit-description", type=str, help="Edit description for repository edits", metavar='EDIT_DESCRIPTION')
    parser.add_argument("--repo-edit-traceback", type=str, help="Traceback string for repository edit (for bugfix requests)", metavar='TRACEBACK')
    parser.add_argument("--repo-judge-path", type=str, help="Repository path to judge", metavar='REPO_PATH')
    parser.add_argument("--repo-judge-rubrics", type=str, nargs='+', help="Rubric questions to evaluate the repository", metavar='RUBRIC')
    parser.add_argument("--repo-judge-rubric-file", type=str, help="Path to rubric file to read and parse", metavar='RUBRIC_FILE')
    parser.add_argument("--paper-rubric-path", type=str, help="PDF paper path to extract rubrics", metavar='PAPER_PATH')
    parser.add_argument("--paper-rubric-output-dir", type=str, default="workspace/rubrics", help="Output directory for rubric files", metavar='OUTPUT_DIR')
    parser.add_argument("--paper-rubric-static", action="store_true", default=True, help="Include static rubrics (code requirements)")
    parser.add_argument("--paper-rubric-dynamic", action="store_true", default=True, help="Include dynamic rubrics (experimental results)")
    parser.add_argument("--pdf-query-path", type=str, help="PDF file path to query", metavar='PDF_PATH')
    parser.add_argument("--pdf-query-question", type=str, help="Question to ask about the PDF content", metavar='QUESTION')
    parser.add_argument("--pdf-query-embedding-model", type=str, default="openai", help="Embedding model to use (openai, huggingface, bedrock)", metavar='MODEL')
    parser.add_argument("--pdf-query-top-k", type=int, default=5, help="Number of top results to retrieve", metavar='TOP_K')
    parser.add_argument("--ipython-code", type=str, help="Python code to execute in IPython kernel")
    # Image entity extraction (one-line CLI)
    parser.add_argument("--image-entity-extract-path", type=str, help="Image path inside container or repo (will be mapped)", metavar='IMG_PATH')
    parser.add_argument("--image-entity-extract-model", type=str, default="gpt-4o", help="Vision model (e.g., gpt-4o)", metavar='MODEL')
    parser.add_argument("--image-entity-extract-timeout", type=int, default=180, help="Timeout seconds", metavar='SEC')
    # GoT image edit (one-line CLI)
    parser.add_argument("--got-edit-image-path", type=str, help="Image path for GoT edit (absolute or container path)", metavar='IMG_PATH')
    parser.add_argument("--got-edit-prompt", type=str, help="Edit prompt for GoT", metavar='PROMPT')
    parser.add_argument("--got-edit-height", type=int, default=1024, help="Output height", metavar='H')
    parser.add_argument("--got-edit-width", type=int, default=1024, help="Output width", metavar='W')
    # Experiment Manager (wrap experiments into MLflow scripts or query experiment history)
    parser.add_argument("--exp-manager-cmd", type=str, help="Bash command to wrap into MLflow experiment (e.g., python xxx.py)", metavar='CMD')
    parser.add_argument("--exp-manager-exp-name", type=str, help="Experiment name for the wrapper script", metavar='EXP_NAME')
    parser.add_argument("--exp-manager-runs", type=str, help="Run identifier(s) under the experiment (for future use)", metavar='RUNS')
    parser.add_argument("--exp-manager-repo-dir", type=str, help="Repository directory for the experiment", metavar='REPO_PATH')
    parser.add_argument("--exp-manager-mode", type=str, choices=["wrap", "query"], default="query", help="Experiment manager mode: wrap (create MLflow script) or query (list experiments)", metavar='MODE')

    args = parser.parse_args()

    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.sandbox = type("Sandbox", (), {"timeout": 30})()
            # Use current working directory
            self.workspace_base = os.getcwd()
            self.workspace_mount_path = None
            self.workspace_mount_path_in_sandbox = None
            self.debug = True

    runtime = DockerRuntime(
        config=SimpleConfig(),
        sid="test-session",
        env_vars={"PYTHONUNBUFFERED": "1", "TEST_ENV": "test_value"},
        enable_logging=True,
    )

    # After runtime is created and before entering the main command logic, add:
    if args.ipython_code:
        from core.events.action import IPythonRunCellAction
        # Ensure the container is started and api_url is set, with Jupyter plugin loaded
        runtime.start_container(image="mc-scientist:latest")
        # Patch the server command to always include --plugins jupyter
        if '--plugins' not in runtime.container.attrs['Config']['Cmd']:
            runtime.container.exec_run('supervisorctl stop server')
            cmd = runtime.container.attrs['Config']['Cmd']
            # Insert --plugins jupyter before other args
            if '--plugins' not in cmd:
                cmd += ['--plugins', 'jupyter']
            runtime.container.exec_run(' '.join(cmd))
            runtime._wait_for_server()
        action = IPythonRunCellAction(code=args.ipython_code)
        result = runtime.run_ipython(action)
        import pdb; pdb.set_trace()
        print(result.content if hasattr(result, 'content') else result)

        exit(0)

    try:
        # Start container
        print("Starting container...")
        success = runtime.start_container(image="mc-scientist:latest")
        if not success:
            print("Failed to start container")
            return

        print(f"\nServer is running at: {runtime.api_url}")

        if args.info:
            # Get server information
            info = runtime.get_server_info()
            print("\nServer Information:")
            print(f"Uptime: {info['uptime']:.2f} seconds")
            print(f"Idle Time: {info['idle_time']:.2f} seconds")
            print("\nSystem Resources:")
            for key, value in info["resources"].items():
                print(f"{key}: {value}")
        elif args.command:
            # Create CmdRunAction
            from core.events.action import CmdRunAction

            action = CmdRunAction(command=args.command)
            result = runtime.run(action)
            print(
                f"Command output:\n{result.content if hasattr(result, 'content') else result}"
            )
        elif args.edit:
            if not args.content:
                print("Error: --content is required when using --edit")
                return
            # Convert to absolute path
            edit_path = os.path.join("/app_sci", args.edit)
            print(f"Editing file: {edit_path}")
            # Create FileEditAction
            from core.events.action import FileEditAction
            from core.events.event import FileEditSource

            # Create new file
            action = FileEditAction(
                path=edit_path,
                file_text=args.content,
                impl_source=FileEditSource.OH_ACI,
                command="create",
            )
            result = runtime.edit(action)
            print(
                f"Edit result:\n{result.content if hasattr(result, 'content') else result}"
            )
        elif args.browse:
            from core.events.action.browse import BrowseURLAction

            print(f"\nTesting browser URL: {args.browse}")
            action = BrowseURLAction(url=args.browse, thought="Testing browser URL")
            result = runtime.browse(action)
            print(f"Browser Result:\n{result}")
        elif args.interactive:
            from core.events.action.browse import BrowseInteractiveAction

            print(f"\nTesting interactive browser action: {args.interactive}")
            action = BrowseInteractiveAction(
                browser_actions=args.interactive, thought="Testing interactive browser"
            )
            result = runtime.browse_interactive(action)
            print(f"Interactive Browser Result:\n{result}")
        elif args.image_entity_extract_path:
            # Ensure absolute path in container if using path
            img_path = args.image_entity_extract_path
            if img_path:
                if not img_path.startswith('/app_sci'):
                    img_path = os.path.join('/app_sci', img_path)
            result = runtime.run_image_entity_extract(
                image_path=img_path,
                model=args.image_entity_extract_model,
                timeout=args.image_entity_extract_timeout,
            )
            # Print caption + entities (and fallback raw)
            try:
                output = {
                    'content': getattr(result, 'content', None),
                    'entities': getattr(result, 'entities', None),
                    'image_size': getattr(result, 'image_size', None),
                    'model': getattr(result, 'model', None),
                    'time_ms': getattr(result, 'time_ms', None),
                }
                print(output)
            except Exception:
                print(result)
        elif args.got_edit_image_path or args.got_edit_prompt:
            from core.events.action.image import GoTEditAction
            # Map container path back to host path for GoT API
            img_path = args.got_edit_image_path or ""
            if img_path:
                if img_path.startswith('/app_sci/'):
                    # Map container path back to host workspace path
                    relative_path = img_path[9:]  # Remove '/app_sci/'
                    img_path = os.path.join(os.getcwd(), relative_path)
                elif not img_path.startswith('/'):
                    # Relative path - make it absolute in host workspace
                    img_path = os.path.join(os.getcwd(), img_path)
            action = GoTEditAction(
                image_path=img_path,
                prompt=args.got_edit_prompt or "",
                height=args.got_edit_height,
                width=args.got_edit_width,
            )
            result = runtime.got_edit(action)
            print({
                'success': getattr(result, 'success', False),
                'error': getattr(result, 'error_message', ''),
                'got_text': getattr(result, 'got_text', ''),
                'image_paths': getattr(result, 'image_paths', []),
            })
        elif args.task_graph:
            from core.events.action.tasks import TaskGraphBuildAction

            print(f"\nTesting task graph build with description: {args.task_graph}")
            action = TaskGraphBuildAction(task_description=args.task_graph)
            result = runtime.task_graph_build(action)
            print(f"Task Graph Build Result:\n{result.task_graph_str if hasattr(result, 'task_graph_str') else result}")
        elif args.test_rollback:
            from core.events.action.snapshot import SnapshotAction
            from core.events.action.rollback import RollbackAction
            from core.events.action import CmdRunAction
            from core.events.action import FileEditAction
            from core.events.event import FileEditSource

            # 1. Create test repository
            repo_dir = os.path.join(os.getcwd(), 'workspace', args.snapshot)
            print(f"\nCreating test repository in: {repo_dir}")
            create_test_repo(repo_dir)

            # 2. Create first snapshot
            snapshot_dir = os.path.join('/app_sci/workspace', args.snapshot)
            print(f"\nCreating snapshot for repository: {snapshot_dir}")
            action = SnapshotAction(repo_directory=snapshot_dir)
            result = runtime.snapshot(action)

            # 3. Get first snapshot tag
            first_tag = result.tag if hasattr(result, 'tag') else None
            print(f"First snapshot tag: {first_tag}")

            # 3. Modify file
            print("\nModifying README.md...")
            edit_path = os.path.join(snapshot_dir, 'README.md')
            action = FileEditAction(
                path=edit_path,
                insert_line=1,
                file_text="Modified content for testing rollback",
                impl_source=FileEditSource.OH_ACI,
                command="insert",
            )
            result = runtime.edit(action)
            print(f"Edit result: {result.content if hasattr(result, 'content') else result}")

            # 4. Create second snapshot
            print("\nCreating second snapshot...")
            action = SnapshotAction(repo_directory=snapshot_dir)
            result = runtime.snapshot(action)
            second_tag = result.tag if hasattr(result, 'tag') else None
            print(f"Second snapshot tag: {second_tag}")

            # 5. Rollback to first snapshot
            print(f"\nRolling back to first snapshot: {first_tag}")
            action = RollbackAction(repo_directory=snapshot_dir, tag=first_tag)
            result = runtime.rollback(action)
            print(f"Rollback result: {result.content if hasattr(result, 'content') else result}")

            # 6. Verify rollback result
            print("\nVerifying rollback result...")
            action = CmdRunAction(command=f"cat {os.path.join(snapshot_dir, 'README.md')}")
            result = runtime.run(action)
            print(f"README.md content after rollback:\n{result.content if hasattr(result, 'content') else result}")
        elif args.repo_plan_paper_path:
            from core.events.action.repo import RepoPlanAction

            print(f"\nTesting repository planning with paper: {args.repo_plan_paper_path}")
            # Convert to absolute path in container
            paper_path = os.path.join('/app_sci', args.repo_plan_paper_path)
            action = RepoPlanAction(paper_path=paper_path, output_dir=os.path.join('/app_sci', 'workspace', 'repo_plan'))
            # Set a longer timeout for repo planning
            action.set_hard_timeout(300)  # 5 minutes timeout
            result = runtime.repo_plan(action)
            print(f"Repository Plan Result:\n{result.plan_content if hasattr(result, 'plan_content') else result}")
        elif args.repo_create_paper_path:
            from core.events.action.repo import RepoCreateAction

            print(f"\nTesting repository creation with code generation for paper: {args.repo_create_paper_path}")
            # Convert to absolute path in container
            paper_path = os.path.join('/app_sci', args.repo_create_paper_path)
            action = RepoCreateAction(
                paper_path=paper_path, 
                output_dir="",  # 不再使用 output_dir
                output_repo_dir=os.path.join('/app_sci', 'workspace', 'repo_create', 'submission')
            )
            # Set a longer timeout for repo creation (includes coding phase)
            action.set_hard_timeout(1200)  # 20 minutes timeout
            result = runtime.repo_create(action)
            print(f"Repository Creation Result:\n{result.plan_content if hasattr(result, 'plan_content') else result}")
            if hasattr(result, 'generated_files') and result.generated_files:
                print(f"\nGenerated Files:\n{result.generated_files}")
                if hasattr(result, 'repo_path') and result.repo_path:
                    print(f"\nGenerated Repository Path: {result.repo_path}")
                    # List the generated files
                    from core.events.action import CmdRunAction
                    list_action = CmdRunAction(command=f"find {result.repo_path} -type f -name '*.py' -o -name '*.yaml' -o -name '*.md' -o -name '*.txt' | head -20")
                    list_result = runtime.run(list_action)
                    print(f"\nGenerated Files List:\n{list_result.content if hasattr(list_result, 'content') else list_result}")
        elif args.repo_verify_path:
            if not args.repo_verify_requirement:
                print("Error: --repo-verify-requirement is required when using --repo-verify-path")
                return

            print(f"\n🔍 Verifying repository: {args.repo_verify_path}")
            print(f"Requirement: {args.repo_verify_requirement}")
            print(f"Level: {args.repo_verify_level}")

            from core.events.action.repo import RepoVerifyAction
            # 构造 Action
            action = RepoVerifyAction(
                repo_path=os.path.join('/app_sci', args.repo_verify_path),
                requirement=args.repo_verify_requirement,
                verification_level=args.repo_verify_level
            )
            # 通过 runtime 发送
            result = runtime.repo_verify(action)
            print(result)
        elif args.repo_debug_path:
            if not args.repo_debug_action:
                print("Error: --repo-debug-action is required when using --repo-debug-path")
                return

            print(f"\n🔧 Debugging repository: {args.repo_debug_path}")
            print(f"Action: {args.repo_debug_action}")

            from core.events.action.repo import RepoDebugAction
            # Create Action
            action = RepoDebugAction(
                repo_path=os.path.join('/app_sci', args.repo_debug_path),
                action_description=args.repo_debug_action
            )
            # Set timeout for debug operation
            action.set_hard_timeout(300)  # 5 minutes timeout
            # Send through runtime
            result = runtime.repo_debug(action)
            
            print(f"\n🔧 Repository Debug Result:")
            print(f"Success: {result.success}")
            print(f"Execution Time: {result.execution_time:.2f} seconds")
            
            if result.summary:
                print(f"\n📊 COMPREHENSIVE SUMMARY:")
                print("=" * 60)
                print(result.summary)
                print("=" * 60)
            
            if result.error_message:
                print(f"Error: {result.error_message}")
            
            if result.fixed_files:
                print(f"\n📝 Fixed Files:\n{result.fixed_files}")
            
            if result.suggestions:
                print(f"\n💡 Suggestions:\n{result.suggestions}")
            
            if result.output:
                print(f"\n📋 Output:\n{result.output}")
        elif args.repo_analyzer_paper_path:
            if not args.repo_analyzer_codebase_path:
                print("Error: --repo-analyzer-codebase-path is required when using --repo-analyzer-paper-path")
                return
            from core.events.action.repo import RepoAnalyzerAction

            print(f"\nTesting repository analysis comparing paper: {args.repo_analyzer_paper_path} with codebase: {args.repo_analyzer_codebase_path}")
            # Convert to absolute paths in container
            paper_path = os.path.join('/app_sci', args.repo_analyzer_paper_path)
            codebase_path = os.path.join('/app_sci', args.repo_analyzer_codebase_path)
            action = RepoAnalyzerAction(
                paper_path=paper_path, 
                codebase_path=codebase_path,
                output_dir=os.path.join('/app_sci', 'workspace', 'repo_analyzer')
            )
            # Set a longer timeout for repo analysis
            action.set_hard_timeout(450)  # 7.5 minutes timeout
            result = runtime.repo_analyzer(action)
            print(f"Repository Analysis Result:\n{result.analysis_content if hasattr(result, 'analysis_content') else result}")
            if hasattr(result, 'analysis_report') and result.analysis_report:
                print(f"\nStructured Analysis Report:\n{result.analysis_report}")
            if hasattr(result, 'missing_functionalities') and result.missing_functionalities:
                print(f"\nMissing Functionalities:\n{result.missing_functionalities}")
        elif args.repo_update_path:
            if not args.repo_update_requirements:
                print("Error: --repo-update-requirements is required when using --repo-update-path")
                return
            from core.events.action.repo import RepoUpdateAction

            print(f"\nTesting repository update for path: {args.repo_update_path}")
            print(f"Requirements: {args.repo_update_requirements}")
            
            # Convert to absolute path in container
            repo_path = os.path.join('/app_sci', args.repo_update_path)
            action = RepoUpdateAction(
                repo_path=repo_path,
                requirements=args.repo_update_requirements,
                apply_changes=args.repo_update_apply,
                save_snapshot=args.repo_update_snapshot
            )
            # Set a longer timeout for repo updates
            action.set_hard_timeout(600)  # 10 minutes timeout
            result = runtime.repo_update(action)
            
            print(f"Repository Update Result:")
            print(f"Success: {result.success}")
            print(f"Applied: {result.applied}")
            
            if result.plan:
                print(f"\nImplementation Plan:\n{result.plan}")
            
            if result.modified_files:
                print(f"\nModified Files:\n{result.modified_files}")
            
            if result.changes_summary:
                print(f"\nChanges Summary:\n{result.changes_summary}")
            
            if result.detailed_changes:
                print(f"\nDetailed Changes:\n{result.detailed_changes}")
            
            if result.file_diffs:
                print(f"\nFile Diffs:\n{result.file_diffs}")
            
            if result.error_message:
                print(f"\nError: {result.error_message}")
        elif args.paper_reproduction_paper_path:
            from core.events.action.repo import PaperReproductionAnalyzerAction

            print(f"\nTesting paper reproduction analysis with paper: {args.paper_reproduction_paper_path}")
            print(f"Analysis level: {args.paper_reproduction_analysis_level}")
            
            # Convert to absolute path in container
            # The project root is mounted at /app_sci, so we need to map paths correctly
            if args.paper_reproduction_paper_path.startswith('workspace/'):
                # Remove 'workspace/' prefix and use /app_sci/workspace as base
                paper_path = os.path.join('/app_sci/workspace', args.paper_reproduction_paper_path[10:])
            elif args.paper_reproduction_paper_path.startswith('debug/'):
                # Map debug/ paths directly to /app_sci/debug/
                paper_path = os.path.join('/app_sci', args.paper_reproduction_paper_path)
            else:
                # Use the path as is, assuming it's already relative to project root
                paper_path = os.path.join('/app_sci', args.paper_reproduction_paper_path)
            
            # Create action
            action = PaperReproductionAnalyzerAction(
                paper_path=paper_path,
                paper_content="",  # Empty content, will be read from paper_path
                analysis_level=args.paper_reproduction_analysis_level
            )
            
            # Run analysis
            result = runtime.paper_reproduction_analyzer(action)
            
            if result.success:
                print("✅ Paper reproduction analysis completed successfully!")
                print("\n📋 Analysis Result:")
                print("=" * 60)
                print(result.analysis_result)
                
                # Save to output file if specified
                if args.paper_reproduction_output_file:
                    try:
                        output_path = os.path.join('/app_sci', args.paper_reproduction_output_file)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"Paper Reproduction Analysis Result\n")
                            f.write(f"Analysis Level: {result.analysis_level}\n")
                            f.write(f"Success: {result.success}\n")
                            f.write("=" * 60 + "\n")
                            f.write(result.analysis_result)
                        print(f"\n💾 Analysis result saved to: {output_path}")
                    except Exception as e:
                        print(f"❌ Error saving to output file: {e}")
            else:
                print("❌ Paper reproduction analysis failed:")
                print(result.error_message)
        elif args.repo_run_path:
            print(f"\n🚀 Running repository reproduce.sh script: {args.repo_run_path}")
            print(f"Timeout: {args.repo_run_timeout} seconds")
            print(f"Retry threshold: {args.repo_run_retry_threshold} seconds")
            
            # Convert to absolute path in container
            repo_path = os.path.join('/app_sci', args.repo_run_path)
            
            # Determine network and GPU settings
            network_enabled = args.repo_run_network and not args.repo_run_no_network
            gpu_enabled = args.repo_run_gpu and not args.repo_run_no_gpu
            from core.events.action.repo import RepoRunAction
            action = RepoRunAction(
                repo_path=repo_path,
                timeout=args.repo_run_timeout,
                retry_threshold=args.repo_run_retry_threshold,
                docker_image=args.repo_run_docker_image,
                memory_limit=args.repo_run_memory_limit,
                network_enabled=network_enabled,
                gpu_enabled=gpu_enabled,
                log_file=args.repo_run_log_file
            )
            # 通过 runtime 发送到 server（远程调用）
            result = runtime.repo_run(action)
            print(f"\n🚀 Repository Run Result:")
            print(f"Success: {result.success}")
            print(f"Execution Time: {result.execution_time:.2f} seconds")
            print(f"Exit Code: {result.exit_code}")
            print(f"Timedout: {result.timedout}")
            print(f"Container ID: {result.container_id}")
            if result.error_message:
                print(f"❌ Error: {result.error_message}")
            if result.output:
                print(f"\n📋 Output:")
                print("=" * 60)
                print(result.output)
                print("=" * 60)
            if result.retried_results:
                print(f"\n🔄 Retry Results:")
                for i, retry in enumerate(result.retried_results):
                    print(f"  Retry {i+1}: {retry}")
        elif args.repo_edit_path:
            if not args.repo_edit_description:
                print("Error: --repo-edit-description is required when using --repo-edit-path")
                return
            from core.events.action.repo import RepoEditAction

            print(f"\nTesting repository edit for path: {args.repo_edit_path}")
            print(f"Edit Description: {args.repo_edit_description}")
            if args.repo_edit_traceback:
                print(f"Traceback: {args.repo_edit_traceback}")
            # Convert to absolute path in container
            repo_path = os.path.join('/app_sci', args.repo_edit_path)
            action = RepoEditAction(
                repo_path=repo_path,
                edit_description=args.repo_edit_description,
                traceback=args.repo_edit_traceback or ""
            )
            # Set a reasonable timeout for repo edits
            # (You can adjust this as needed)
            # action.set_hard_timeout(600)
            result = runtime.repo_edit(action)
            print(f"Repository Edit Result:")
            if hasattr(result, 'success'):
                print(f"Success: {result.success}")
            if hasattr(result, 'output'):
                print(f"Output: {result.output}")
            if hasattr(result, 'modified_files'):
                print(f"Modified Files: {result.modified_files}")
            if hasattr(result, 'suggestions'):
                print(f"Suggestions: {result.suggestions}")
            if hasattr(result, 'error_message'):
                print(f"Error: {result.error_message}")
            if hasattr(result, 'summary'):
                print(f"Summary: {result.summary}")
        elif args.repo_judge_path:
            if not args.repo_judge_rubrics and not args.repo_judge_rubric_file:
                print("Error: --repo-judge-rubrics or --repo-judge-rubric-file is required when using --repo-judge-path")
                return
            from core.events.action.repo import RepoJudgeAction

            print(f"\n👨‍💻 Judging repository: {args.repo_judge_path}")
            
            # Convert to absolute path in container
            repo_path = os.path.join('/app_sci', args.repo_judge_path)
            
            if args.repo_judge_rubric_file:
                print(f"Using rubric file: {args.repo_judge_rubric_file}")
                rubric_file_path = os.path.join('/app_sci', args.repo_judge_rubric_file)
                action = RepoJudgeAction(
                    repo_path=repo_path,
                    rubric_file_path=rubric_file_path
                )
            else:
                print(f"Using rubric list: {args.repo_judge_rubrics}")
                action = RepoJudgeAction(
                    repo_path=repo_path,
                    rubric_list=args.repo_judge_rubrics
                )
            
            # Set a longer timeout for repo judge
            action.set_hard_timeout(900) # 15 minutes timeout
            result = runtime.repo_judge(action)

            print(f"\n👨‍💻 Repository Judgement Result:")
            print(f"Success: {result.success}")
            print(f"Execution Time: {result.execution_time:.2f} seconds")
            print(f"Rubric Count: {len(result.rubric_results)}")
            
            if result.summary:
                print(f"\n📊 OVERALL SUMMARY:")
                print("=" * 60)
                print(result.summary)
                print("=" * 60)
            
            if result.rubric_results:
                print(f"\n📋 RUBRIC RESULTS:")
                for i, rubric_result in enumerate(result.rubric_results):
                    print(f"\n--- Rubric {i+1} ---")
                    print(f"Question: {rubric_result.rubric}")
                    print(f"Answer: {rubric_result.result}")
            
            if result.error_message:
                print(f"❌ Error: {result.error_message}")
        elif args.exp_manager_cmd or args.exp_manager_mode:
            # Experiment Manager entry (shell)
            print("\n🧪 Experiment Manager")
            if args.exp_manager_cmd:
                print(f"Command: {args.exp_manager_cmd}")
            if args.exp_manager_mode:
                print(f"Mode: {args.exp_manager_mode}")
            print(f"Output Dir: workspace/experiments (default)")
            if args.exp_manager_repo_dir:
                print(f"Repo Dir: {args.exp_manager_repo_dir}")
            if args.exp_manager_exp_name:
                print(f"Experiment Name: {args.exp_manager_exp_name}")
            if args.exp_manager_runs:
                print(f"Runs: {args.exp_manager_runs}")

            from core.events.action.experiment import ExperimentManagerAction  # type: ignore

            # Map paths into container mount
            if args.exp_manager_repo_dir:
                # Create mlflow_scripts directory at the same level as submission
                repo_dir = os.path.dirname(args.exp_manager_repo_dir)
                output_dir = os.path.join('/app_sci', repo_dir, 'mlflow_scripts')
                repo_path = os.path.join('/app_sci', args.exp_manager_repo_dir)
            else:
                # Fallback to default location
                output_dir = os.path.join('/app_sci', 'workspace', 'experiments')
                repo_path = ''

            # Determine mode - map new mode names to internal mode names
            mode = args.exp_manager_mode or 'query'
            if mode == 'wrap':
                mode = 'auto'  # Map 'wrap' to internal 'auto' mode
            elif mode == 'query':
                mode = 'list'  # Map 'query' to internal 'list' mode

            # Construct action and set timeout
            action = ExperimentManagerAction(
                command=args.exp_manager_cmd or "",
                output_dir=output_dir,
                mode=mode,
                repo_path=repo_path,
                experiment_name=args.exp_manager_exp_name or "",
                runs=args.exp_manager_runs or "",
            )
            # Allow longer timeout for experiment orchestration
            action.set_hard_timeout(900)

            # Send through runtime (expects runtime.experiment_manager to be available server-side)
            try:
                result = runtime.experiment_manager(action)  # type: ignore[attr-defined]
            except AttributeError:
                print("Error: runtime.experiment_manager is not implemented. Ensure server and client support this action.")
                return

            print(f"\n🧪 Experiment Manager Result:")
            # Print content first (contains the main information)
            if hasattr(result, 'content') and result.content:
                print(f"\n📋 Content:\n{result.content}")
            
            # Print commonly expected fields if present
            if hasattr(result, 'success'):
                print(f"Success: {result.success}")
            if hasattr(result, 'execution_time'):
                try:
                    print(f"Execution Time: {result.execution_time:.2f} seconds")
                except Exception:
                    print(f"Execution Time: {result.execution_time}")
            if hasattr(result, 'experiments') and result.experiments:
                print(f"\n📁 Experiments ({len(result.experiments)}):")
                for exp in result.experiments:
                    # experiments are dicts (by serialization); read safely
                    if isinstance(exp, dict):
                        name = exp.get('name', 'Unknown')
                        status = exp.get('status', 'Unknown')
                        print(f"  - {name}: {status}")
                        wrapper_path = exp.get('wrapper_path')
                        if wrapper_path:
                            print(f"    Wrapper: {wrapper_path}")
                    else:
                        # fallback for unexpected types
                        try:
                            print(f"  - {getattr(exp, 'name', 'Unknown')}: {getattr(exp, 'status', 'Unknown')}")
                            wp = getattr(exp, 'wrapper_path', None)
                            if wp:
                                print(f"    Wrapper: {wp}")
                        except Exception:
                            print(f"  - {str(exp)}")
            if hasattr(result, 'plan') and getattr(result, 'plan'):
                print("\n📝 Plan:\n" + str(result.plan))
            if hasattr(result, 'runs') and getattr(result, 'runs'):
                print("\n🚀 Runs:\n" + str(result.runs))
            if hasattr(result, 'summary') and getattr(result, 'summary'):
                print("\n📊 SUMMARY:\n" + str(result.summary))
            if hasattr(result, 'error_message') and getattr(result, 'error_message'):
                print(f"\n❌ Error: {result.error_message}")
        elif args.paper_rubric_path:
            from core.events.action.repo import PaperRubricAction

            # Validate required parameters
            if not args.paper_rubric_output_dir:
                print("Error: --paper-rubric-output-dir is required when using --paper-rubric-path")
                return

            print(f"\n📋 Extracting rubrics from paper: {args.paper_rubric_path}")
            print(f"Output Directory: {args.paper_rubric_output_dir}")
            print(f"Include Static: {args.paper_rubric_static}")
            print(f"Include Dynamic: {args.paper_rubric_dynamic}")

            # Convert to absolute path in container
            paper_path = os.path.join('/app_sci', args.paper_rubric_path)
            output_dir = os.path.join('/app_sci', args.paper_rubric_output_dir)
            action = PaperRubricAction(
                paper_path=paper_path,
                output_dir=output_dir,
                include_static=args.paper_rubric_static,
                include_dynamic=args.paper_rubric_dynamic
            )
            # Set timeout for paper rubric extraction
            action.set_hard_timeout(600)  # 10 minutes timeout
            result = runtime.paper_rubric(action)

            print(f"\n📋 Paper Rubric Extraction Result:")
            if hasattr(result, 'success'):
                print(f"Success: {result.success}")
                print(f"Execution Time: {result.execution_time:.2f} seconds")
                print(f"Static Rubrics: {len(result.static_rubrics)}")
                print(f"Dynamic Rubrics: {len(result.dynamic_rubrics)}")
            else:
                print(f"Error: Unexpected result type: {type(result)}")
                print(f"Result content: {result.content if hasattr(result, 'content') else result}")
                print(f"Result attributes: {dir(result)}")
                if hasattr(result, 'content'):
                    print(f"Content: '{result.content}'")
                return
            
            if result.paper_analysis:
                print(f"\n📄 PAPER ANALYSIS:")
                print("=" * 60)
                print(result.paper_analysis)
                print("=" * 60)
            
            if result.rubric_summary:
                print(f"\n📋 RUBRIC SUMMARY:")
                print("=" * 60)
                print(result.rubric_summary)
                print("=" * 60)
            
            if result.static_rubrics:
                print(f"\n🔧 STATIC RUBRICS (Code Requirements):")
                for i, rubric in enumerate(result.static_rubrics):
                    print(f"\n--- Static Rubric {i+1} ---")
                    if isinstance(rubric, dict):
                        print(f"Category: {rubric.get('category', 'Unknown')}")
                        print(f"Requirement: {rubric.get('requirement', 'Unknown')}")
                        print(f"Description: {rubric.get('description', 'No description')}")
                        if 'code_examples' in rubric:
                            print(f"Code Examples: {rubric['code_examples']}")
                    else:
                        print(f"Requirement: {str(rubric)}")
            
            if result.dynamic_rubrics:
                print(f"\n📊 DYNAMIC RUBRICS (Experimental Results):")
                for i, rubric in enumerate(result.dynamic_rubrics):
                    print(f"\n--- Dynamic Rubric {i+1} ---")
                    if isinstance(rubric, dict):
                        print(f"Category: {rubric.get('category', 'Unknown')}")
                        print(f"Requirement: {rubric.get('requirement', 'Unknown')}")
                        print(f"Expected Result: {rubric.get('expected_result', 'Unknown')}")
                        print(f"Description: {rubric.get('description', 'No description')}")
                        if 'table_data' in rubric:
                            print(f"Table Data: {rubric['table_data']}")
                    else:
                        print(f"Requirement: {str(rubric)}")
            
            if result.error_message:
                print(f"❌ Error: {result.error_message}")
        elif args.pdf_query_path:
            if not args.pdf_query_question:
                print("Error: --pdf-query-question is required when using --pdf-query-path")
                return
            from core.events.action import PDFQueryAction

            print(f"\n📄 Querying PDF: {args.pdf_query_path}")
            print(f"Question: {args.pdf_query_question}")
            print(f"Embedding Model: {args.pdf_query_embedding_model}")
            print(f"Top-K: {args.pdf_query_top_k}")
            
            # Convert to absolute path in container
            pdf_path = os.path.join('/app_sci', args.pdf_query_path)
            action = PDFQueryAction(
                pdf_path=pdf_path,
                query=args.pdf_query_question,
                embedding_model=args.pdf_query_embedding_model,
                top_k=args.pdf_query_top_k
            )
            # Set timeout for PDF query operation
            action.set_hard_timeout(300)  # 5 minutes timeout
            # Send through runtime
            result = runtime.pdf_query(action)
            
            print(f"\n📄 PDF Query Result:")
            print(f"Success: {result.success}")
            print(f"Execution Time: {result.execution_time:.2f} seconds")
            
            if result.answer:
                print(f"\n📋 ANSWER:")
                print("=" * 60)
                print(result.answer)
                print("=" * 60)
            
            if result.source_documents:
                print(f"\n📄 SOURCE DOCUMENTS:")
                print("=" * 60)
                print(result.source_documents)
                print("=" * 60)
            
            if result.search_results:
                print(f"\n🔍 SEARCH RESULTS:")
                print("=" * 60)
                print(result.search_results)
                print("=" * 60)
            
            if result.metadata:
                print(f"\n📊 METADATA:")
                print("=" * 60)
                print(result.metadata)
                print("=" * 60)
            
            if result.error_message:
                print(f"❌ Error: {result.error_message}")
        else:
            print("Please provide arguments:")
            print("  --command <command>  Execute command in container")
            print("  --info              Get server information")
            print("  --edit <file>       Edit file")
            print(
                "  --content <text>    Content to write to file (required with --edit)"
            )
            print("  --browse <url>      Browse a URL")
            print("  --interactive <action> Run an interactive browser action")
            print("  --task-graph <description> Build task graph from description")
            print("  --test-rollback     Test snapshot and rollback functionality")
            print("  --snapshot <directory> Test snapshot and rollback functionality")
            print("  --repo-plan-paper-path <path>   Generate repository plan from paper")
            print("  --repo-create-paper-path <path> Generate full repository with code from paper")
            print("  --repo-analyzer-paper-path <path> --repo-analyzer-codebase-path <path> Analyze repository by comparing paper with codebase")
            print("  --repo-update-path <path> --repo-update-requirements <requirements> Update repository code based on requirements")
            print("    Optional flags: --repo-update-apply (apply changes), --repo-update-snapshot (save snapshot)")
            print("  --repo-verify-path <path> --repo-verify-requirement <requirement> --repo-verify-level <level> Verify repository")
            print("  --repo-debug-path <path> --repo-debug-action <action> Debug repository")
            print("  --paper-reproduction-paper-path <path> --paper-reproduction-analysis-level <level> Analyze paper reproduction")
            print("    Optional flags: --paper-reproduction-output-file <output_file>")
            print("  --repo-run-path <path> --repo-run-timeout <seconds> --repo-run-retry-threshold <seconds> Run repository reproduce.sh script")
            print("    Optional flags: --repo-run-docker-image <image>, --repo-run-memory-limit <limit>, --repo-run-no-network, --repo-run-no-gpu, --repo-run-log-file <log_file> (default: experiment.log)")
            print("  --repo-edit-path <path> --repo-edit-description <description> Edit repository code")
            print("  --repo-edit-traceback <traceback> Traceback for repository edit (for bugfix requests)")
            print("  --repo-judge-path <path> --repo-judge-rubrics <rubric1> <rubric2> Judge repository based on rubrics")
            print("  --repo-judge-path <path> --repo-judge-rubric-file <rubric_file> Judge repository using rubric file")
            print("  --paper-rubric-path <path> Extract rubrics from PDF paper")
            print("    Required: --paper-rubric-output-dir <output_dir>")
            print("    Optional flags: --paper-rubric-static, --paper-rubric-dynamic")
            print("  --pdf-query-path <path> --pdf-query-question <question> Query PDF document with semantic search")
            print("    Optional flags: --pdf-query-embedding-model <model>, --pdf-query-top-k <number>")
            print("  --exp-manager-cmd <bash_cmd> Wrap bash command into MLflow experiment script (e.g., python xxx.py)")
            print("  --exp-manager-mode <wrap|query> Experiment manager mode: wrap (create MLflow script) or query (list experiments) (default: query)")
            print("    Optional: --exp-manager-exp-name <exp_name> (experiment name for the wrapper)")
            print("    Optional: --exp-manager-runs <runs> (run identifiers for future use)")
            print("    Optional: --exp-manager-repo-dir <path> (repository directory for the experiment)")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        runtime.close()


if __name__ == "__main__":
    main()

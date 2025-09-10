import asyncio
import logging
import os
import time
import docker
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import sys

from core.events.action import RepoRunAction
from core.events.observation import Observation
from core.events.observation.repo import RepoRunObservation
from core.utils.logger import get_logger

logger = get_logger(__name__)

# 避免循环导入
if TYPE_CHECKING:
    from core.runtime.impl.docker.docker_runtime import DockerRuntime



class RepoRunTask:
    """Task for running repository reproduce.sh scripts in isolated Docker containers"""
    
    def __init__(self, runtime: 'DockerRuntime'):
        self.runtime = runtime
        self.logger = logging.getLogger(__name__)
        self.docker_client = docker.from_env()
        self.persistent_containers = {}  # Cache for persistent containers
        
    def _check_host_gpu_availability(self) -> bool:
        """Check if GPU is available on the host system"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    async def run(self, action: RepoRunAction) -> RepoRunObservation:
        """
        Run the reproduce.sh script from a repository in an isolated Docker container
        
        Args:
            action: RepoRunAction containing repo path and configuration
            
        Returns:
            RepoRunObservation with execution results
        """
        start_time = time.time()
        
        try:
            # Validate repo path in current container environment
            repo_path = Path(action.repo_path)
            print('repo_path', repo_path)
            print('Current working directory:', Path.cwd())
            print('Path exists:', repo_path.exists())
            print('Path absolute:', repo_path.absolute())
            
            # If the path doesn't exist, try to resolve it relative to /app_sci
            if not repo_path.exists() and str(repo_path).startswith('/app_sci'):
                # The path should exist in container, let's proceed without validation
                print('Path appears to be container path, proceeding with execution...')
            elif not repo_path.exists():
                return RepoRunObservation(
                    success=False,
                    execution_time=0,
                    output="",
                    error_message=f"Repository path does not exist: {action.repo_path}"
                )
            
            # Check if reproduce.sh exists (skip validation for container paths)
            reproduce_script = repo_path / "reproduce.sh"
            if not reproduce_script.exists() and not str(repo_path).startswith('/app_sci'):
                return RepoRunObservation(
                    success=False,
                    execution_time=0,
                    output="",
                    error_message=f"reproduce.sh script not found in repository: {reproduce_script}"
                )
            elif not reproduce_script.exists() and str(repo_path).startswith('/app_sci'):
                print('reproduce.sh validation skipped for container path, proceeding with execution...')
            
            # Use persistent containers to avoid reinstalling dependencies
            use_persistent = getattr(action, 'use_persistent_containers', True)
            
            if use_persistent:
                # Get or create persistent container
                container_id = self._get_or_create_persistent_container(repo_path, action)
                
                # Run in persistent container
                run_result = await self._run_in_persistent_container(container_id, repo_path, action)
            else:
                # Use original isolated container approach
                run_result = await self._run_in_isolated_container(action)
            
            execution_time = time.time() - start_time
            
            return RepoRunObservation(
                success=run_result.success,
                execution_time=execution_time,
                output=run_result.output,
                error_message=run_result.error_message,
                exit_code=run_result.exit_code,
                timedout=run_result.timedout,
                retried_results=run_result.retried_results,
                container_id=run_result.container_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error running repo: {str(e)}")
            return RepoRunObservation(
                success=False,
                execution_time=execution_time,
                output="",
                error_message=str(e)
            )
    
    async def _run_in_isolated_container(self, action: RepoRunAction):
        """Run the reproduce.sh script in an isolated Docker container"""
        container = None
        start_time = time.time()
        
        try:
            # Get absolute path for mounting - map container path to host path
            repo_path = Path(action.repo_path)
            
            # If it's a container path, map it to host path for Docker mounting
            if str(repo_path).startswith('/app_sci'):
                # Map container path to host path: /app_sci/workspace/... -> workspace/...
                host_path = str(repo_path).replace('/app_sci/', '')
                # Get absolute host path
                host_path = os.path.abspath(host_path)
                print(f'Mapped container path {repo_path} to host path {host_path}')
            else:
                host_path = str(repo_path)
            
            # Prepare environment variables
            env_vars = action.environment_vars or {}
            env_vars.update({
                'PYTHONUNBUFFERED': '1',
                'REPO_PATH': '/submission'  # 容器内的仓库路径
            })
            
            # Prepare container configuration - 使用 paperbench 的架构
            container_config = {
                'image': action.docker_image,
                'command': self._build_container_command(repo_path),
                'volumes': {
                    host_path: {
                        'bind': '/submission',  # 使用 paperbench 的标准路径
                        'mode': 'rw'  # Read-write mount to allow chmod
                    }
                },
                'detach': True,
                'remove': False,  # Don't auto-remove to get logs
                'mem_limit': action.memory_limit,
                'working_dir': '/submission',  # 使用 paperbench 的工作目录
                'environment': env_vars
            }
            
            # Add GPU support if enabled and available on host
            if action.gpu_enabled:
                host_gpu_available = self._check_host_gpu_availability()
                if host_gpu_available:
                    container_config['device_requests'] = [
                        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                    ]
                    self.logger.info(f"GPU enabled for container (host GPU detected)")
                    print(f"🎮 GPU enabled for container (host GPU detected)")
                else:
                    self.logger.info(f"GPU requested but not available on host, running without GPU")
                    print(f"⚠️  GPU requested but not available on host, running without GPU")
            else:
                self.logger.info(f"GPU disabled for container")
                print(f"🎮 GPU disabled for container")
            
            # Debug: Print container configuration
            self.logger.info(f"Container config - repo_path: {repo_path}")
            self.logger.info(f"Container config - volumes: {container_config['volumes']}")
            
            # Set network mode based on configuration
            if not action.network_enabled:
                container_config['network_mode'] = 'none'
                self.logger.info(f"Network disabled, using network_mode: none")
            else:
                # Force host network mode for better connectivity
                container_config['network_mode'] = 'host'
                self.logger.info(f"Network enabled, using host network mode")
            
            # Create and start container
            container = self.docker_client.containers.run(**container_config)
            container_id = container.id[:12]  # Short container ID
            
            self.logger.info(f"Started container {container_id} for repo {repo_path.name}")
            
            # Start real-time log streaming
            print(f"\n🚀 Container {container_id} started. Streaming logs in real-time:")
            print("=" * 60)
            
            # Function to stream logs with real-time file writing
            def stream_logs():
                try:
                    for log in container.logs(stream=True, follow=True):
                        log_line = log.decode('utf-8', errors='ignore')
                        print(log_line, end='')  # Print to console
                except Exception as e:
                    error_msg = f"Error streaming logs: {e}"
                    print(error_msg)
            
            # Start log streaming in a separate thread
            import threading
            log_thread = threading.Thread(target=stream_logs, daemon=True)
            log_thread.start()
            
            # Wait for container to complete with timeout
            try:
                # Use a longer timeout for Docker client operations
                result = container.wait(timeout=action.timeout + 300)  # Add 5 minutes buffer
                exit_code = result['StatusCode']
                timedout = False
                self.logger.info(f"Container {container_id} completed with exit code {exit_code}")
                print(f"\n✅ Container {container_id} completed with exit code {exit_code}")
            except Exception as e:
                # Container timed out or failed
                self.logger.warning(f"Container {container_id} timed out or failed: {str(e)}")
                print(f"\n❌ Container {container_id} timed out or failed: {str(e)}")
                try:
                    container.stop(timeout=30)  # Increase stop timeout
                except:
                    pass
                exit_code = None
                timedout = True
            
            # Get final container logs
            try:
                logs = container.logs().decode('utf-8', errors='ignore')
            except Exception as e:
                self.logger.warning(f"Failed to get container logs: {str(e)}")
                logs = f"Failed to get logs: {str(e)}"
            
            # Write final log information
            execution_time = time.time() - start_time
            return RepoRunObservation(
                success=exit_code == 0 if exit_code is not None else False,
                execution_time=execution_time,
                output=logs,
                error_message=None if exit_code == 0 else f"Script failed with exit code {exit_code}",
                exit_code=exit_code,
                timedout=timedout,
                retried_results=[],
                container_id=container_id
            )
            
        except Exception as e:
            self.logger.error(f"Error in isolated container execution: {str(e)}")
            return RepoRunObservation(
                success=False,
                execution_time=time.time() - start_time, # Use start_time from run()
                output='',
                error_message=str(e),
                exit_code=None,
                timedout=False,
                retried_results=[],
                container_id=None
            )
        finally:
            # Ensure container is cleaned up
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
    
    def _get_or_create_persistent_container(self, repo_path: Path, action: RepoRunAction) -> str:
        """Get or create a persistent container for the repository to avoid reinstalling dependencies"""
        container_key = f"persistent-{repo_path.name}-{hash(str(repo_path))}"
        
        # Check if we already have a persistent container for this repo
        if container_key in self.persistent_containers:
            container_id = self.persistent_containers[container_key]
            try:
                # Check if container still exists and is running
                container = self.docker_client.containers.get(container_id)
                if container.status == 'running':
                    self.logger.info(f"Reusing existing persistent container {container_id}")
                    return container_id
                else:
                    # Container exists but not running, remove it
                    container.remove(force=True)
                    del self.persistent_containers[container_key]
            except docker.errors.NotFound:
                # Container doesn't exist anymore
                del self.persistent_containers[container_key]
        
        # Create new persistent container
        self.logger.info(f"Creating new persistent container for {repo_path.name}")
        
        # Get host path for mounting
        if str(repo_path).startswith('/app_sci'):
            host_path = str(repo_path).replace('/app_sci/', '')
            host_path = os.path.abspath(host_path)
        else:
            host_path = str(repo_path)
        
        # Prepare environment variables
        env_vars = action.environment_vars or {}
        env_vars.update({
            'PYTHONUNBUFFERED': '1',
            'REPO_PATH': '/submission'
        })
        
        # Container configuration for persistent container
        container_config = {
            'image': action.docker_image,
            'command': ['tail', '-f', '/dev/null'],  # Keep container running
            'volumes': {
                host_path: {
                    'bind': '/submission',
                    'mode': 'rw'
                }
            },
            'detach': True,
            'remove': False,
            'mem_limit': action.memory_limit,
            'working_dir': '/submission',
            'environment': env_vars,
            'name': container_key  # Give it a predictable name
        }
        
        # Add GPU support if enabled
        if action.gpu_enabled and self._check_host_gpu_availability():
            container_config['device_requests'] = [
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        
        # Set network mode
        if not action.network_enabled:
            container_config['network_mode'] = 'none'
        else:
            container_config['network_mode'] = 'host'
        
        # Create and start persistent container
        container = self.docker_client.containers.run(**container_config)
        container_id = container.id[:12]
        
        # Install dependencies in the persistent container
        self._install_dependencies_in_container(container_id, repo_path)
        
        # Cache the container ID
        self.persistent_containers[container_key] = container_id
        
        self.logger.info(f"Created persistent container {container_id} for {repo_path.name}")
        return container_id
    
    def _install_dependencies_in_container(self, container_id: str, repo_path: Path):
        """Install dependencies in a persistent container"""
        try:
            container = self.docker_client.containers.get(container_id)
            
            # Check if requirements.txt exists
            result = container.exec_run(['test', '-f', 'requirements.txt'])
            if result.exit_code == 0:
                self.logger.info(f"Installing dependencies in persistent container {container_id}")
                
                # Install dependencies
                install_result = container.exec_run([
                    'bash', '-c', 
                    'export PIP_BREAK_SYSTEM_PACKAGES=1 && pip install -r requirements.txt'
                ])
                
                if install_result.exit_code == 0:
                    self.logger.info(f"Dependencies installed successfully in {container_id}")
                else:
                    self.logger.warning(f"Dependency installation failed in {container_id}: {install_result.output.decode()}")
            else:
                self.logger.info(f"No requirements.txt found, skipping dependency installation in {container_id}")
                
        except Exception as e:
            self.logger.error(f"Error installing dependencies in container {container_id}: {e}")
    
    async def _run_in_persistent_container(self, container_id: str, repo_path: Path, action: RepoRunAction):
        """Run reproduce.sh in a persistent container"""
        start_time = time.time()
        
        try:
            container = self.docker_client.containers.get(container_id)
            
            # 移除所有 log_file 相关的 open/write/flush/close 逻辑
            # 只保留 shell tee 方式
            
            # First check if reproduce.sh exists
            check_result = container.exec_run(['test', '-f', 'reproduce.sh'])
            if check_result.exit_code != 0:
                self.logger.warning(f"reproduce.sh not found in container {container_id}")
                return RepoRunObservation(
                    success=False,
                    execution_time=time.time() - start_time,
                    output="",
                    error_message="reproduce.sh not found in repository",
                    exit_code=None,
                    timedout=False,
                    retried_results=[],
                    container_id=container_id
                )
            
            # Check if reproduce.sh is empty
            size_result = container.exec_run(['wc', '-c', 'reproduce.sh'])
            if size_result.exit_code == 0:
                try:
                    file_size = int(size_result.output.decode().strip().split()[0])
                    if file_size == 0:
                        self.logger.warning(f"reproduce.sh exists but is empty in container {container_id}")
                        return RepoRunObservation(
                            success=False,
                            execution_time=time.time() - start_time,
                            output="",
                            error_message="reproduce.sh exists but is empty",
                            exit_code=None,
                            timedout=False,
                            retried_results=[],
                            container_id=container_id
                        )
                except (ValueError, IndexError):
                    pass  # Continue if we can't parse the size
            
            # Ensure reproduce.sh is executable
            chmod_result = container.exec_run(['chmod', '+x', 'reproduce.sh'])
            if chmod_result.exit_code != 0:
                return RepoRunObservation(
                    success=False,
                    execution_time=time.time() - start_time,
                    output="",
                    error_message="Failed to make reproduce.sh executable",
                    exit_code=None,
                    timedout=False,
                    retried_results=[],
                    container_id=container_id
                )
            
            # Run reproduce.sh in the persistent container with real-time output
            self.logger.info(f"Running reproduce.sh in persistent container {container_id}")
            
            # 直接在容器内重定向输出到 experiment.log
            log_filename = action.log_file if action.log_file else "experiment.log"
            exec_result = container.exec_run(['bash', '-c', f'bash reproduce.sh | tee {log_filename}'], stream=True)
            
            # Check exec_result and its output
            if exec_result is None or not hasattr(exec_result, 'output') or exec_result.output is None:
                self.logger.error("exec_run returned None or has no output attribute.")
                return RepoRunObservation(
                    success=False,
                    execution_time=time.time() - start_time,
                    output="",
                    error_message="exec_run returned None or has no output attribute.",
                    exit_code=None,
                    timedout=False,
                    retried_results=[],
                    container_id=container_id
                )
            
            # Combine all output
            output = ''.join(line.decode('utf-8', errors='ignore') if isinstance(line, bytes) else str(line) for line in exec_result.output)
            
            # Get exit code
            exit_code = exec_result.exit_code if hasattr(exec_result, 'exit_code') else None
            
            # Write final log information
            execution_time = time.time() - start_time
            return RepoRunObservation(
                success=exit_code == 0,
                execution_time=execution_time,
                output=output,
                error_message=None if exit_code == 0 else f"Script failed with exit code {exit_code}",
                exit_code=exit_code,
                timedout=False,
                retried_results=[],
                container_id=container_id
            )
            
        except Exception as e:
            self.logger.error(f"Error running in persistent container {container_id}: {e}")
            return RepoRunObservation(
                success=False,
                execution_time=time.time() - start_time,
                output="",
                error_message=str(e),
                exit_code=None,
                timedout=False,
                retried_results=[],
                container_id=container_id
            )
    
    def cleanup_persistent_containers(self):
        """Clean up all persistent containers"""
        for container_key, container_id in self.persistent_containers.items():
            try:
                container = self.docker_client.containers.get(container_id)
                container.stop(timeout=30)
                container.remove(force=True)
                self.logger.info(f"Cleaned up persistent container {container_id}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up container {container_id}: {e}")
        
        self.persistent_containers.clear()

    def _build_container_command(self, repo_path: Path) -> list:
        """Build the command to run inside the container - 适配 pb-reproducer 镜像"""
        return [
            "bash", "-c", 
            f"""
            set -e  # Exit on any error
            
            echo "🚀 Starting reproduction in pb-reproducer container..."
            echo "Repository: {repo_path.name}"
            echo "Working directory: $(pwd)"
            
            # 检查 Python 版本
            echo "🐍 Python versions available:"
            python3.11 --version 2>/dev/null || echo "Python 3.11 not available"
            python3.12 --version 2>/dev/null || echo "Python 3.12 not available"
            python3 --version
            
            # 设置 Python 和 pip 命令别名
            echo "🔗 Setting up Python command aliases..."
            alias python='/usr/bin/python3.12'
            alias pip='/usr/bin/pip3.12'
            export PATH="/usr/bin:$PATH"
            
            # 禁用 PEP 668 限制
            export PIP_BREAK_SYSTEM_PACKAGES=1
            
            # 测试网络连接（简化版）
            echo "🌐 Testing network connectivity..."
            curl -s --connect-timeout 5 https://pypi.org/simple/ >/dev/null && echo "✅ Network connectivity OK" || echo "❌ Network connectivity failed"
            
            # 验证 python 命令
            echo "✅ Python command verification:"
            python --version || echo "python command still not available"
            
            # 检查 GPU 可用性
            echo "🎮 GPU availability check:"
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "✅ NVIDIA GPU detected:"
                nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "nvidia-smi command failed"
            else
                echo "❌ NVIDIA GPU not available or nvidia-smi not found"
            fi
            
            # 显示仓库内容
            echo "📄 Repository contents:"
            ls -la
            
            # 检查 reproduce.sh 是否存在
            if [ ! -f "reproduce.sh" ]; then
                echo "❌ Error: reproduce.sh not found in repository"
                exit 1
            fi
            
            # 确保 reproduce.sh 可执行
            chmod +x reproduce.sh
            echo "🔧 Made reproduce.sh executable"
            
            # 安装依赖包（PEP 668 限制已禁用）
            echo "📦 Installing dependencies from requirements.txt..."
            if [ -f "requirements.txt" ]; then
                pip install -r requirements.txt || echo "Warning: Some packages may not have installed correctly"
            else
                echo "No requirements.txt found, skipping dependency installation"
            fi
            
            # 运行 reproduce.sh 脚本
            echo "🚀 Executing reproduce.sh..."
            bash reproduce.sh
            
            echo "✅ Reproduction script completed!"
            """
        ]


 
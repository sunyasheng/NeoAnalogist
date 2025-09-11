"""
This is the main file for the runtime client.
It is responsible for executing actions received from OpenHands backend and producing observations.

NOTE: this will be executed inside the docker sandbox.
"""

import argparse
import asyncio
import base64
import mimetypes
import os
import shutil
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug: Check if OPENAI_API_KEY is loaded
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OPENAI_API_KEY loaded successfully, length: {len(openai_api_key)}")
else:
    print("WARNING: OPENAI_API_KEY not found after load_dotenv()")

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from uvicorn import run

from core.environment.openhands_aci.editor.editor import OHEditor
from core.environment.openhands_aci.editor.exceptions import ToolError
from core.environment.openhands_aci.editor.results import ToolResult
from core.environment.openhands_aci.utils.diff import get_diff
from core.events.action import (BrowseInteractiveAction, BrowseURLAction,
                                CmdRunAction, FileEditAction, FileReadAction, IPythonRunCellAction,
                                FileWriteAction, TaskGraphBuildAction, SnapshotAction, RollbackAction,
                                RepoPlanAction, RepoCreateAction, RepoAnalyzerAction, RepoUpdateAction, RepoVerifyAction, RepoRunAction, PaperReproductionAnalyzerAction, RepoDebugAction, RepoEditAction, PDFQueryAction, RepoJudgeAction, PaperRubricAction, ExperimentManagerAction)
from core.events.event import (Action, FileEditSource, FileReadSource,
                               Observation)
from core.events.observation import (CmdOutputObservation, ErrorObservation,
                                     FileEditObservation, FileReadObservation, IPythonRunCellObservation,
                                     FileWriteObservation, TaskGraphBuildObservation, SnapshotObservation, RollbackObservation,
                                     RepoPlanObservation, RepoCreateObservation, RepoAnalyzerObservation, RepoUpdateObservation, RepoVerifyObservation, PaperReproductionAnalyzerObservation, RepoDebugObservation, PDFQueryObservation, RepoJudgeObservation, PaperRubricObservation)
from core.events.observation.experiment import ExperimentManagerObservation
from core.events.observation.repo import RepoRunObservation
from core.events.serialization import event_from_dict, event_to_dict
from core.runtime.browser import browse
from core.runtime.browser.browser_env import BrowserEnv
from core.runtime.plugins import ALL_PLUGINS, Plugin
from core.runtime.utils.bash import BashSession
from core.runtime.utils.files import insert_lines, read_lines
from core.runtime.utils.memory_monitor import MemoryMonitor
from core.runtime.utils.runtime_init import init_user_and_working_directory
from core.runtime.utils.system_stats import get_system_stats
from core.utils.async_utils import call_sync_from_async, wait_all
from core.utils.logger import get_logger
from core.utils.types.exceptions import BrowserUnavailableException
from core.runtime.tasks.taskgraph import TaskGraphBuilder
from core.runtime.tasks.rollback import TaskRollback
from core.runtime.tasks.repo_plan import RepoPlan
from core.runtime.tasks.repo_create import RepoCreate
from core.runtime.tasks.repo_analyzer import TaskRepoAnalyzer
from core.runtime.tasks.repo_update import RepoUpdate, RepoUpdateInput
from core.runtime.tasks.repo_verify import evaluate_repository
from core.runtime.tasks.repo_run import RepoRunTask
from core.runtime.tasks.paper_reproduction_analyzer import TaskPaperReproductionAnalyzer, PaperReproductionAnalyzerInput
from core.runtime.tasks.repo_debug import RepoDebugTask
from core.runtime.tasks.repo_edit import RepoEditTask
from core.runtime.tasks.pdf_query_task import PDFQueryTool
from core.events.observation.repo import RepoEditObservation
from core.runtime.plugins.jupyter import JupyterPlugin
from core.events.action.image import ImageEntityExtractAction
from core.events.observation.image import ImageEntityExtractObservation
from core.runtime.tasks.image_entity_extract import ImageEntityExtractTask


logger = get_logger(__name__)


def _execute_file_editor(
    editor: OHEditor,
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
    enable_linting: bool = False,
) -> tuple[str, tuple[str | None, str | None]]:
    """Execute file editor command and handle exceptions.

    Args:
        editor: The OHEditor instance
        command: Editor command to execute
        path: File path
        file_text: Optional file text content
        view_range: Optional view range tuple (start, end)
        old_str: Optional string to replace
        new_str: Optional replacement string
        insert_line: Optional line number for insertion
        enable_linting: Whether to enable linting

    Returns:
        tuple: A tuple containing the output string and a tuple of old and new file content
    """
    result: ToolResult | None = None
    try:
        result = editor(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            enable_linting=enable_linting,
        )
    except ToolError as e:
        result = ToolResult(error=e.message)

    if result.error:
        return f"ERROR:\n{result.error}", (None, None)

    if not result.output:
        # logger.warning(f'No output from file_editor for {path}')
        return "", (None, None)

    return result.output, (result.old_content, result.new_content)


class ActionExecutor:
    """ActionExecutor is running inside docker sandbox.
    It is responsible for executing actions received from OpenHands backend and producing observations.
    """

    def __init__(
        self,
        plugins_to_load: list[Plugin],
        work_dir: str,
        username: str,
        user_id: int,
        browsergym_eval_env: str | None,
    ) -> None:
        self.plugins_to_load = plugins_to_load
        self._initial_cwd = work_dir
        self.username = username
        self.user_id = user_id
        _updated_user_id = init_user_and_working_directory(
            username=username, user_id=self.user_id, initial_cwd=work_dir
        )
        if _updated_user_id is not None:
            self.user_id = _updated_user_id

        self.bash_session: BashSession | None = None
        self.lock = asyncio.Lock()
        self.plugins: dict[str, Plugin] = {}
        self.file_editor = OHEditor(workspace_root=self._initial_cwd)
        self.browser: BrowserEnv | None = None
        self.browser_init_task: asyncio.Task | None = None
        self.browsergym_eval_env = browsergym_eval_env
        self.start_time = time.time()
        self.last_execution_time = self.start_time
        self._initialized = False

        self.max_memory_gb: int | None = None
        if _override_max_memory_gb := os.environ.get("RUNTIME_MAX_MEMORY_GB", None):
            self.max_memory_gb = int(_override_max_memory_gb)
            logger.info(
                f"Setting max memory to {self.max_memory_gb}GB (according to the RUNTIME_MAX_MEMORY_GB environment variable)"
            )
        else:
            logger.info("No max memory limit set, using all available system memory")
        self.memory_monitor = MemoryMonitor(
            enable=os.environ.get("RUNTIME_MEMORY_MONITOR", "False").lower()
            in ["true", "1", "yes"]
        )
        self.memory_monitor.start_monitoring()

    async def run_action(self, action) -> Observation:
        async with self.lock:
            action_type = action.action
            logger.debug(f"Running action:\n{action}")
            observation = await getattr(self, action_type)(action)
            logger.debug(f"Action output:\n{observation}")
            return observation

    async def run(
        self, action: CmdRunAction
    ) -> CmdOutputObservation | ErrorObservation:
        assert self.bash_session is not None
        # import pdb; pdb.set_trace()
        obs = await call_sync_from_async(self.bash_session.execute, action)
        return obs

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        assert self.bash_session is not None
        if 'jupyter' in self.plugins:
            _jupyter_plugin: JupyterPlugin = self.plugins['jupyter']  # type: ignore
            # This is used to make AgentSkills in Jupyter aware of the
            # current working directory in Bash
            jupyter_cwd = getattr(self, '_jupyter_cwd', None)
            if self.bash_session.cwd != jupyter_cwd:
                logger.debug(
                    f'{self.bash_session.cwd} != {jupyter_cwd} -> reset Jupyter PWD'
                )
                # escape windows paths
                cwd = self.bash_session.cwd.replace('\\', '/')
                reset_jupyter_cwd_code = f'import os; os.chdir("{cwd}")'
                _aux_action = IPythonRunCellAction(code=reset_jupyter_cwd_code)
                _reset_obs: IPythonRunCellObservation = await _jupyter_plugin.run(
                    _aux_action
                )
                logger.debug(
                    f'Changed working directory in IPython to: {self.bash_session.cwd}. Output: {_reset_obs}'
                )
                self._jupyter_cwd = self.bash_session.cwd

            obs: IPythonRunCellObservation = await _jupyter_plugin.run(action)
            obs.content = obs.content.rstrip()

            if action.include_extra:
                obs.content += (
                    f'\n[Jupyter current working directory: {self.bash_session.cwd}]'
                )
                obs.content += f'\n[Jupyter Python interpreter: {_jupyter_plugin.python_interpreter_path}]'
            return obs
        else:
            raise RuntimeError(
                'JupyterRequirement not found. Unable to run IPython action.'
            )

    async def _init_plugin(self, plugin: Plugin):
        assert self.bash_session is not None
        await plugin.initialize(self.username)
        self.plugins[plugin.name] = plugin
        logger.debug(f'Initializing plugin: {plugin.name}')

        if isinstance(plugin, JupyterPlugin):
            # Escape backslashes in Windows path
            cwd = self.bash_session.cwd.replace('\\', '/')
            await self.run_ipython(
                IPythonRunCellAction(code=f'import os; os.chdir(r"{cwd}")')
            )

    def _resolve_path(self, path: str, working_dir: str) -> str:
        filepath = Path(path)
        if not filepath.is_absolute():
            return str(Path(working_dir) / filepath)
        return str(filepath)

    async def read(self, action: FileReadAction) -> Observation:
        assert self.bash_session is not None
        if action.impl_source == FileReadSource.OH_ACI:
            result_str, _ = _execute_file_editor(
                self.file_editor,
                command="view",
                path=action.path,
                view_range=action.view_range,
            )

            return FileReadObservation(
                content=result_str,
                path=action.path,
                impl_source=FileReadSource.OH_ACI,
            )

        # NOTE: the client code is running inside the sandbox,
        # so there's no need to check permission
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)
        try:
            if filepath.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                with open(filepath, "rb") as file:
                    image_data = file.read()
                    encoded_image = base64.b64encode(image_data).decode("utf-8")
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = "image/png"  # default to PNG if mime type cannot be determined
                    encoded_image = f"data:{mime_type};base64,{encoded_image}"

                return FileReadObservation(path=filepath, content=encoded_image)
            elif filepath.lower().endswith(".pdf"):
                with open(filepath, "rb") as file:
                    pdf_data = file.read()
                    encoded_pdf = base64.b64encode(pdf_data).decode("utf-8")
                    encoded_pdf = f"data:application/pdf;base64,{encoded_pdf}"
                return FileReadObservation(path=filepath, content=encoded_pdf)
            elif filepath.lower().endswith((".mp4", ".webm", ".ogg")):
                with open(filepath, "rb") as file:
                    video_data = file.read()
                    encoded_video = base64.b64encode(video_data).decode("utf-8")
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = "video/mp4"  # default to MP4 if MIME type cannot be determined
                    encoded_video = f"data:{mime_type};base64,{encoded_video}"

                return FileReadObservation(path=filepath, content=encoded_video)

            with open(filepath, "r", encoding="utf-8") as file:
                lines = read_lines(file.readlines(), action.start, action.end)
        except FileNotFoundError:
            return ErrorObservation(
                f"File not found: {filepath}. Your current working directory is {working_dir}."
            )
        except UnicodeDecodeError:
            return ErrorObservation(f"File could not be decoded as utf-8: {filepath}.")
        except IsADirectoryError:
            return ErrorObservation(
                f"Path is a directory: {filepath}. You can only read files"
            )

        code_view = "".join(lines)
        return FileReadObservation(path=filepath, content=code_view)

    async def ainit(self):
        # bash needs to be initialized first
        logger.debug("Initializing bash session")
        self.bash_session = BashSession(
            work_dir=self._initial_cwd,
            username=self.username,
            no_change_timeout_seconds=int(
                os.environ.get("NO_CHANGE_TIMEOUT_SECONDS", 30)
            ),
            max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
        )
        self.bash_session.initialize()
        logger.debug("Bash session initialized")

        # Start browser initialization in the background
        self.browser_init_task = asyncio.create_task(self._init_browser_async())
        logger.debug("Browser initialization started in background")

        await wait_all(
            (self._init_plugin(plugin) for plugin in self.plugins_to_load),
            timeout=30,
        )
        logger.debug("All plugins initialized")

        # This is a temporary workaround
        # TODO: refactor AgentSkills to be part of JupyterPlugin
        # AFTER ServerRuntime is deprecated
        logger.debug("Initializing AgentSkills")
        if "agent_skills" in self.plugins and "jupyter" in self.plugins:
            obs = await self.run_ipython(
                IPythonRunCellAction(
                    code="from core.runtime.plugins.agent_skills.agentskills import *\n"
                )
            )
            logger.debug(f"AgentSkills initialized: {obs}")

        # logger.debug('Initializing bash commands')
        await self._init_bash_commands()
        # logger.debug('Runtime client initialized.')
        self._initialized = True

    async def _init_browser_async(self):
        """Initialize the browser asynchronously."""
        logger.debug("Initializing browser asynchronously")
        try:
            self.browser = BrowserEnv(self.browsergym_eval_env)
            logger.debug("Browser initialized asynchronously")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.browser = None

    async def _ensure_browser_ready(self):
        """Ensure the browser is ready for use."""
        logger.debug("Ensuring browser is ready...")
        if self.browser is None:
            if self.browser_init_task is None:
                # Start browser initialization if it hasn't been started
                self.browser_init_task = asyncio.create_task(self._init_browser_async())
            elif self.browser_init_task.done():
                # If the task is done but browser is still None, restart initialization
                self.browser_init_task = asyncio.create_task(self._init_browser_async())

            # Wait for browser to be initialized
            if self.browser_init_task:
                logger.debug("Waiting for browser to be ready...")
                await self.browser_init_task

            # Check if browser was successfully initialized
            if self.browser is None:
                raise BrowserUnavailableException("Browser initialization failed")

        # If we get here, the browser is ready
        logger.debug("Browser is ready")

    async def _init_bash_commands(self):
        INIT_COMMANDS = [
            'git config --file ./.git_config user.name "yashengsun" && git config --file ./.git_config user.email "1044705955@qq.com" && alias git="git --no-pager" && export GIT_CONFIG=$(pwd)/.git_config'
            if os.environ.get("LOCAL_RUNTIME_MODE") == "1"
            else 'git config --global user.name "yashengsun" && git config --global user.email "1044705955@qq.com" && alias git="git --no-pager"'
        ]
        logger.debug(f"Initializing by running {len(INIT_COMMANDS)} bash commands...")
        for command in INIT_COMMANDS:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(300)
            logger.debug(f"Executing init command: {command}")
            obs = await self.run(action)
            assert isinstance(obs, CmdOutputObservation)
            logger.debug(
                f"Init command outputs (exit code: {obs.exit_code}): {obs.content}"
            )
            assert obs.exit_code == 0
        logger.debug("Bash init commands completed")
        pass

    async def snapshot(self, action: SnapshotAction) -> Observation:
        """Create a snapshot of the current repository state.
        
        Args:
            action: The SnapshotAction containing the repository directory
            
        Returns:
            SnapshotObservation containing the snapshot tag
        """
        try:
            rollback = TaskRollback()
            tag = rollback.save_snapshot(action.repo_directory)
            return SnapshotObservation(
                content=f"Created snapshot with tag: {tag}",
                tag=tag
            )
        except Exception as e:
            logger.error(f"Failed to create snapshot: {str(e)}")
            return ErrorObservation(f"Failed to create snapshot: {str(e)}")

    async def rollback(self, action: RollbackAction) -> Observation:
        """Rollback to a previous snapshot.
        
        Args:
            action: The RollbackAction containing the repository directory and tag
            
            Returns:
                RollbackObservation containing the rollback status  
        """
        try:
            rollback = TaskRollback()
            rollback.rollback(action.repo_directory, action.tag)
            return RollbackObservation(tag=action.tag)
        except Exception as e:
            logger.error(f"Failed to rollback: {str(e)}")
            return ErrorObservation(f"Failed to rollback: {str(e)}")

    async def write(self, action: FileWriteAction) -> Observation:
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        insert = action.content.split("\n")
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        file_exists = os.path.exists(filepath)
        if file_exists:
            file_stat = os.stat(filepath)
        else:
            file_stat = None

        mode = "w" if not file_exists else "r+"
        try:
            with open(filepath, mode, encoding="utf-8") as file:
                if mode != "w":
                    all_lines = file.readlines()
                    new_file = insert_lines(insert, all_lines, action.start, action.end)
                else:
                    new_file = [i + "\n" for i in insert]

                file.seek(0)
                file.writelines(new_file)
                file.truncate()

        except FileNotFoundError:
            return ErrorObservation(f"File not found: {filepath}")
        except IsADirectoryError:
            return ErrorObservation(
                f"Path is a directory: {filepath}. You can only write to files"
            )
        except UnicodeDecodeError:
            return ErrorObservation(f"File could not be decoded as utf-8: {filepath}")

        # Attempt to handle file permissions
        try:
            if file_exists:
                assert file_stat is not None
                # restore the original file permissions if the file already exists
                os.chmod(filepath, file_stat.st_mode)
                os.chown(filepath, file_stat.st_uid, file_stat.st_gid)
            else:
                # set the new file permissions if the file is new
                os.chmod(filepath, 0o664)
                os.chown(filepath, self.user_id, self.user_id)
        except PermissionError as e:
            return ErrorObservation(
                f"File {filepath} written, but failed to change ownership and permissions: {e}"
            )
        return FileWriteObservation(content="", path=filepath)

    async def edit(self, action: FileEditAction) -> Observation:
        assert action.impl_source == FileEditSource.OH_ACI
        result_str, (old_content, new_content) = _execute_file_editor(
            self.file_editor,
            command=action.command,
            path=action.path,
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            enable_linting=False,
        )

        return FileEditObservation(
            content=result_str,
            path=action.path,
            old_content=action.old_str,
            new_content=action.new_str,
            impl_source=FileEditSource.OH_ACI,
            diff=get_diff(
                old_contents=old_content or "",
                new_contents=new_content or "",
                filepath=action.path,
            ),
        )

    async def browse(self, action: BrowseURLAction) -> Observation:
        await self._ensure_browser_ready()
        return await browse(action, self.browser)

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        await self._ensure_browser_ready()
        return await browse(action, self.browser)

    async def task_graph_build(self, action: TaskGraphBuildAction) -> TaskGraphBuildObservation:
        """Build a task graph from a task description.
        
        This method takes a task description and builds a task graph representation
        showing the execution topology of the task.
        
        Args:
            action: The TaskGraphBuildAction containing the task description
            
        Returns:
            TaskGraphBuildObservation containing the task graph representation
        """
        try:
            # Create task graph builder and build graph
            builder = TaskGraphBuilder()
            task_graph_str = builder.build_graph(action.task_description)
            
            return TaskGraphBuildObservation(
                content=task_graph_str,
                task_description=action.task_description,
                task_graph_str=task_graph_str
            )
        except Exception as e:
            logger.error(f"Error building task graph: {str(e)}")
            return ErrorObservation(f"Failed to build task graph: {str(e)}")

    async def repo_plan(self, action: RepoPlanAction) -> RepoPlanObservation:
        """Generate a repository implementation plan based on a paper.
        
        This method takes a paper path and generates a detailed plan for implementing
        the repository based on the paper's content.
        
        Args:
            action: The RepoPlanAction containing the paper path and output directory
            
        Returns:
            RepoPlanObservation containing the planning results
        """
        try:
            # Create RepoPlan instance and execute full pipeline
            repo_plan = RepoPlan(
                paper_path=action.paper_path,
                output_dir=action.output_dir
            )
            repo_plan.execute_full_pipeline()
            
            # Collect content from responses
            plan_content = []
            for response in repo_plan.responses:
                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    plan_content.append(content)
            
            # Join all content with newlines
            full_plan_content = "\n\n".join(plan_content)
            
            return RepoPlanObservation(
                content=full_plan_content,
                plan_content=full_plan_content
            )
        except Exception as e:
            logger.error(f"Error generating repository plan: {str(e)}")
            return ErrorObservation(f"Failed to generate repository plan: {str(e)}")

    async def repo_create(self, action: RepoCreateAction) -> RepoCreateObservation:
        """Generate a full repository implementation with code generation based on a paper.
        
        This method takes a paper path and generates both a detailed plan and complete
        working code for implementing the repository based on the paper's content.
        
        Args:
            action: The RepoCreateAction containing the paper path and output directories
            
        Returns:
            RepoCreateObservation containing the planning results and generated code info
        """
        try:
            # Create RepoCreate instance and execute full pipeline
            repo_create = RepoCreate(
                paper_path=action.paper_path,
                output_dir=action.output_dir,
                output_repo_dir=action.output_repo_dir
            )
            repo_create.execute_full_pipeline()
            
            # Collect content from responses
            plan_content = []
            for response in repo_create.responses:
                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    plan_content.append(content)
            
            # Join all content with newlines
            full_plan_content = "\n\n".join(plan_content)
            
            # Generate info about created files
            generated_files_info = []
            if os.path.exists(repo_create.output_repo_dir):
                for root, dirs, files in os.walk(repo_create.output_repo_dir):
                    for file in files:
                        file_path = os.path.relpath(os.path.join(root, file), repo_create.output_repo_dir)
                        generated_files_info.append(file_path)
            
            generated_files_text = "\n".join(generated_files_info) if generated_files_info else "No files generated"
            
            return RepoCreateObservation(
                content=full_plan_content,
                plan_content=full_plan_content,
                generated_files=generated_files_text,
                repo_path=repo_create.output_repo_dir
            )
        except Exception as e:
            logger.error(f"Error creating repository: {str(e)}")
            return ErrorObservation(f"Failed to create repository: {str(e)}")

    async def repo_analyzer(self, action: RepoAnalyzerAction) -> RepoAnalyzerObservation:
        """Analyze repository implementation by comparing paper with existing codebase.
        
        This method takes a paper path and codebase path, then generates a detailed
        analysis report comparing what's described in the paper versus what's
        actually implemented in the codebase.
        
        Args:
            action: The RepoAnalyzerAction containing the paper path, codebase path and output directory
            
        Returns:
            RepoAnalyzerObservation containing the analysis results and implementation status
        """
        try:
            # Create TaskRepoAnalyzer instance with default constructor
            repo_analyzer = TaskRepoAnalyzer()
            
            # Execute paper vs codebase analysis
            analysis_result = repo_analyzer.analyze_paper_vs_codebase(
                paper_path=action.paper_path,
                repo_path=action.codebase_path
            )
            
            # The analysis_result is already a formatted string report
            return RepoAnalyzerObservation(
                content=analysis_result,
                analysis_content=analysis_result,
                analysis_report=analysis_result,
                missing_functionalities=""  # This info is included in the main report
            )
        except Exception as e:
            logger.error(f"Error analyzing repository: {str(e)}")
            return ErrorObservation(f"Failed to analyze repository: {str(e)}")

    async def repo_update(self, action: RepoUpdateAction) -> RepoUpdateObservation:
        """Update repository code based on user requirements.
        
        This method takes a repository path and user requirements, then generates
        modifications to implement new features, improve code quality, and fix issues.
        
        Args:
            action: The RepoUpdateAction containing the repository path, requirements and options
            
        Returns:
            RepoUpdateObservation containing the update results with detailed changes
        """
        try:
            # Create RepoUpdate instance
            repo_updater = RepoUpdate()
            
            # Create input for the repo update task
            update_input = RepoUpdateInput(
                repo_path=action.repo_path,
                requirements=action.requirements,
                target_files=action.target_files if action.target_files else None,
                context=action.context,
                preserve_structure=True
            )
            
            # Execute repo update
            update_result = repo_updater.update_repo(update_input)
            
            # Apply changes if requested
            applied = False
            if update_result.success and action.apply_changes:
                applied = repo_updater.apply_changes(
                    update_result, 
                    action.repo_path, 
                    save_snapshot=action.save_snapshot
                )
            
            # Format detailed changes and diffs for observation
            detailed_changes_str = ""
            file_diffs_str = ""
            
            if update_result.detailed_changes:
                changes_parts = []
                for file_path, changes in update_result.detailed_changes.items():
                    changes_parts.append(f"📄 {file_path}:")
                    changes_parts.append(f"  • Lines added: {changes['lines_added']}")
                    changes_parts.append(f"  • Lines deleted: {changes['lines_deleted']}")
                    changes_parts.append(f"  • Total changes: {changes['total_changes']}")
                    if changes['change_types']:
                        changes_parts.append(f"  • Change types: {', '.join(changes['change_types'])}")
                    changes_parts.append("")
                detailed_changes_str = "\n".join(changes_parts)
            
            if update_result.file_diffs:
                diff_parts = []
                for file_path, diff in update_result.file_diffs.items():
                    diff_parts.append(f"📝 Diff for {file_path}:")
                    diff_parts.append("-" * 40)
                    # Show first 30 lines of diff to avoid overwhelming output
                    diff_lines = diff.split('\n')
                    for line in diff_lines[:30]:
                        diff_parts.append(line)
                    if len(diff_lines) > 30:
                        diff_parts.append(f"... (showing first 30 lines of {len(diff_lines)} total)")
                    diff_parts.append("-" * 40)
                    diff_parts.append("")
                file_diffs_str = "\n".join(diff_parts)
            
            # Format modified files info
            modified_files_str = ""
            if update_result.modified_files:
                modified_files_str = "\n".join([
                    f"📂 {file_path} ({len(content.splitlines())} lines)" 
                    for file_path, content in update_result.modified_files.items()
                ])
            
            # Create content summary for the observation
            content_parts = []
            if update_result.success:
                content_parts.append("✅ Repository update completed successfully")
                if applied:
                    content_parts.append("🔧 Changes have been applied to files")
                    content_parts.append("📸 Snapshot saved after applying changes")
                else:
                    content_parts.append("📋 Changes generated but not applied")
            else:
                content_parts.append("❌ Repository update failed")
                if update_result.error_message:
                    content_parts.append(f"Error: {update_result.error_message}")
            
            content = "\n".join(content_parts)
            
            return RepoUpdateObservation(
                content=content,
                plan=update_result.plan,
                modified_files=modified_files_str,
                changes_summary=update_result.changes_summary,
                success=update_result.success,
                repo_analysis=update_result.repo_analysis or "",
                error_message=update_result.error_message or "",
                applied=applied,
                file_diffs=file_diffs_str,
                detailed_changes=detailed_changes_str
            )
            
        except Exception as e:
            logger.error(f"Error updating repository: {str(e)}")
            error_content = f"❌ Repository update failed with error: {str(e)}"
            return RepoUpdateObservation(
                content=error_content,
                plan="",
                modified_files="",
                changes_summary="",
                success=False,
                repo_analysis="",
                error_message=f"Failed to update repository: {str(e)}",
                applied=False,
                file_diffs="",
                detailed_changes=""
            )

    async def repo_verify(self, action: RepoVerifyAction) -> RepoVerifyObservation:
        """Verify repository implementation and functionality, only return LLM summary report as content."""
        try:
            verify_result = evaluate_repository(
                action.repo_path, 
                action.requirement, 
                action.verification_level
            )
            summary = verify_result.get('llm_summary_report', '')
            return RepoVerifyObservation(
                content=summary
            )
        except Exception as e:
            logger.error(f"Error verifying repository: {str(e)}")
            error_content = f"❌ Repository verification failed with error: {str(e)}"
            return RepoVerifyObservation(
                content=error_content,
                error_message=f"Failed to verify repository: {str(e)}"
            )

    async def repo_run(self, action: RepoRunAction) -> RepoRunObservation:
        """Execute a repository's reproduce.sh script in an isolated Docker container."""
        try:
            # Create and run the repo run task
            repo_run_task = RepoRunTask(self)
            result = await repo_run_task.run(action)
            
            # Create content summary for the observation
            content_parts = []
            if result.success:
                content_parts.append("✅ Repository reproduce.sh script executed successfully")
                content_parts.append(f"⏱️  Execution time: {result.execution_time:.2f} seconds")
                if result.container_id:
                    content_parts.append(f"🐳 Container ID: {result.container_id}")
            else:
                content_parts.append("❌ Repository reproduce.sh script execution failed")
                if result.error_message:
                    content_parts.append(f"Error: {result.error_message}")
                    logger.warning(f"repo_run failed with error: {result.error_message}")
                if result.timedout:
                    content_parts.append("⏰ Execution timed out")
            
            # Add output summary
            if result.output:
                # Truncate output if too long
                output_summary = result.output[:1000] + "..." if len(result.output) > 1000 else result.output
                content_parts.append(f"📋 Output: {output_summary}")
            
            content = "\n".join(content_parts)
            
            # Update the result content
            result.content = content
            
            return result
            
        except Exception as e:
            logger.error(f"Error running repository: {str(e)}")
            error_content = f"❌ Repository run failed with error: {str(e)}"
            return RepoRunObservation(
                success=False,
                execution_time=0,
                output="",
                error_message=f"Failed to run repository: {str(e)}",
                content=error_content
            )

    async def repo_debug(self, action: RepoDebugAction) -> RepoDebugObservation:
        """Debug and fix code issues in repositories using refact agent."""
        try:
            # Create and run the debug task
            debug_task = RepoDebugTask(action.repo_path, action.action_description)
            result = await debug_task.run()
            
            # Create content summary
            content_parts = []
            if result.success:
                content_parts.append("✅ Repository debug operation completed successfully")
                if result.fixed_files:
                    content_parts.append(f"📝 Fixed {len(result.fixed_files)} files")
                if result.suggestions:
                    content_parts.append(f"💡 Generated {len(result.suggestions)} suggestions")
            else:
                content_parts.append("❌ Repository debug operation failed")
                if result.error_message:
                    content_parts.append(f"Error: {result.error_message}")
            
            content = "\n".join(content_parts)
            
            return RepoDebugObservation(
                content=content,
                success=result.success,
                output=result.output,
                fixed_files="\n".join(result.fixed_files) if result.fixed_files else "",
                suggestions="\n".join(result.suggestions) if result.suggestions else "",
                execution_time=result.execution_time,
                error_message=result.error_message,
                summary=result.summary
            )
            
        except Exception as e:
            logger.error(f"Error debugging repository: {str(e)}")
            error_content = f"❌ Repository debug failed with error: {str(e)}"
            return RepoDebugObservation(
                content=error_content,
                success=False,
                error_message=f"Failed to debug repository: {str(e)}"
            )

    async def paper_reproduction_analyzer(self, action: PaperReproductionAnalyzerAction) -> PaperReproductionAnalyzerObservation:
        """Analyze paper reproduction requirements and implementation guidance.
        
        This method takes a paper path or content and analysis level, then generates
        detailed implementation requirements for reproducing the paper.
        
        Args:
            action: The PaperReproductionAnalyzerAction containing the paper path/content and analysis level
            
        Returns:
            PaperReproductionAnalyzerObservation containing the analysis results
        """
        try:
            # Create analyzer instance
            analyzer = TaskPaperReproductionAnalyzer()
            
            # Debug logging
            logger.info(f"Paper reproduction analysis - paper_path: {action.paper_path}")
            logger.info(f"Paper reproduction analysis - paper_content length: {len(action.paper_content)}")
            logger.info(f"Paper reproduction analysis - analysis_level: {action.analysis_level}")
            
            # Create input
            analysis_input = PaperReproductionAnalyzerInput(
                paper_content=action.paper_content,
                paper_path=action.paper_path,
                analysis_level=action.analysis_level
            )
            
            # Execute analysis
            result = analyzer.analyze_paper(analysis_input)
            
            # Create content summary for the observation
            content_parts = []
            if result.success:
                content_parts.append("✅ Paper reproduction analysis completed successfully")
                content_parts.append(f"📊 Analysis Level: {result.analysis_level}")
            else:
                content_parts.append("❌ Paper reproduction analysis failed")
                if result.error_message:
                    content_parts.append(f"Error: {result.error_message}")
            
            content = "\n".join(content_parts)
            
            return PaperReproductionAnalyzerObservation(
                content=content,
                analysis_result=result.analysis_result,
                analysis_level=result.analysis_level,
                success=result.success,
                error_message=result.error_message or ""
            )
            
        except Exception as e:
            logger.error(f"Error analyzing paper reproduction: {str(e)}")
            error_content = f"❌ Paper reproduction analysis failed with error: {str(e)}"
            return PaperReproductionAnalyzerObservation(
                content=error_content,
                analysis_result="",
                analysis_level=action.analysis_level,
                success=False,
                error_message=f"Failed to analyze paper reproduction: {str(e)}"
            )

    async def paper_rubric(self, action: PaperRubricAction) -> PaperRubricObservation:
        """Extract rubrics from PDF papers.
        
        This method takes a paper path or content and extracts both static and dynamic
        rubric requirements that need to be implemented in the code.
        
        Args:
            action: The PaperRubricAction containing the paper path/content and extraction parameters
            
        Returns:
            PaperRubricObservation containing the extracted rubrics
        """
        try:
            # Create and run the rubric extraction task
            from core.runtime.tasks.paper_rubric import PaperRubricTask
            rubric_task = PaperRubricTask(
                paper_path=action.paper_path,
                include_static=action.include_static,
                include_dynamic=action.include_dynamic,
                rubric_categories=action.rubric_categories,
                save_to_file=getattr(action, 'save_to_file', True),  # Default to True for backward compatibility
                output_dir=action.output_dir  # Required parameter - directory where rubric file will be saved
            )
            
            result = await rubric_task.run()
            return result
            
        except Exception as e:
            logger.error(f"Error extracting paper rubrics: {str(e)}")
            error_content = f"❌ Paper rubric extraction failed with error: {str(e)}"
            return PaperRubricObservation(
                content=error_content,
                success=False,
                static_rubrics=[],
                dynamic_rubrics=[],
                rubric_summary="",
                paper_analysis="",
                execution_time=0.0,
                error_message=f"Failed to extract paper rubrics: {str(e)}"
            )

    async def repo_edit(self, action: RepoEditAction) -> RepoEditObservation:
        """Edit repository code based on user instructions."""
        try:
            # Create and run the edit task
            edit_task = RepoEditTask(action.repo_path, action.edit_description)
            result = await edit_task.run()

            # Build observation
            return RepoEditObservation(
                content=result.content,
                success=result.success,
                output=result.output,
                modified_files=result.modified_files,
                suggestions=result.suggestions,
                execution_time=result.execution_time,
                error_message=result.error_message,
                summary=result.summary
            )
        except Exception as e:
            logger.error(f"Error editing repository: {str(e)}")
            error_content = f"❌ Repository edit failed with error: {str(e)}"
            return RepoEditObservation(
                content=error_content,
                success=False,
                error_message=f"Failed to edit repository: {str(e)}"
            )

    async def image_entity_extract(self, action: ImageEntityExtractAction) -> ImageEntityExtractObservation | ErrorObservation:
        """Run image entity extraction task.

        Loads image from path or base64, runs detection, returns entities and metadata.
        """
        try:
            task = ImageEntityExtractTask(self)
            obs: ImageEntityExtractObservation = task.run(action)
            return obs
        except Exception as e:
            logger.error(f"Error in image_entity_extract: {str(e)}")
            return ErrorObservation(f"Failed to extract entities: {str(e)}")

    async def repo_judge(self, action: RepoJudgeAction) -> RepoJudgeObservation:
        """Judge repository code based on rubric questions or rubric file."""
        try:
            # Create and run the judge task
            from core.runtime.tasks.repo_judge import RepoJudgeTask
            
            # Create RepoJudgeTask with rubric file
            judge_task = RepoJudgeTask(
                repo_path=action.repo_path,
                rubric_file_path=action.rubric_file_path
            )
            
            result = await judge_task.run()

            # Return the observation directly since RepoJudgeTask.run() now returns RepoJudgeObservation
            return result
        except Exception as e:
            logger.error(f"Error judging repository: {str(e)}")
            error_content = f"❌ Repository judge failed with error: {str(e)}"
            return RepoJudgeObservation(
                content=error_content,
                success=False,
                error_message=f"Failed to judge repository: {str(e)}"
            )

    async def experiment_manager(self, action: ExperimentManagerAction) -> ExperimentManagerObservation:
        """Manage and execute experiments.
        
        This method handles experiment management operations including:
        - Listing existing experiments
        - Creating new experiments
        - Running experiments with commands
        
        Args:
            action: The ExperimentManagerAction containing mode and parameters
            
        Returns:
            ExperimentManagerObservation with operation results
        """
        try:
            # Create and run the experiment manager task
            from core.runtime.tasks.experiment_manager import ExperimentManagerTask
            
            # Create ExperimentManagerTask
            task = ExperimentManagerTask(output_dir=action.output_dir)
            
            # Run the task and return the observation directly
            return task.run(action)
            
        except Exception as e:
            logger.error(f"Error in experiment manager: {str(e)}")
            error_content = f"❌ Experiment manager failed with error: {str(e)}"
            return ExperimentManagerObservation(
                content=error_content,
                success=False,
                execution_time=0.0,
                experiments=[],
                mode=action.mode,
                error_message=f"Failed to manage experiments: {str(e)}"
            )

    def _get_pdf_query_task(self):
        """Get PDF query task instance."""
        return PDFQueryTool()

    async def pdf_query(self, action: PDFQueryAction) -> PDFQueryObservation:
        """Query PDF documents using semantic search and retrieval.
        
        This method takes a PDF path and question, then performs semantic search
        to find relevant content and generates an answer based on the retrieved context.
        
        Args:
            action: The PDFQueryAction containing PDF path and query parameters
            
        Returns:
            PDFQueryObservation with query results
        """
        try:
            # Create PDFQueryTool instance and execute query
            pdf_tool = self._get_pdf_query_task()
            
            # Create input for the query
            from core.runtime.tasks.pdf_query_task import PDFQueryInput
            query_input = PDFQueryInput(
                pdf_path=action.pdf_path,
                question=action.query,
                max_source_docs=action.top_k,
                chunk_size=action.chunk_size,
                chunk_overlap=action.chunk_overlap,
                embedding_model=action.embedding_model
            )
            
            # Execute query
            result = pdf_tool.query_pdf(query_input)
            
            # Build observation with consistent format
            return PDFQueryObservation(
                content=f"PDF query completed in {result.processing_time:.2f} seconds",
                success=result.success,
                answer=result.answer,
                source_documents=str(result.sources),
                error_message=result.error_message or "",
                execution_time=result.processing_time
            )
        except Exception as e:
            logger.error(f"Error querying PDF: {str(e)}")
            error_content = f"❌ PDF query failed with error: {str(e)}"
            return PDFQueryObservation(
                content=error_content,
                success=False,
                answer="",
                source_documents="",
                error_message=f"Failed to query PDF: {str(e)}"
            )

    def close(self):
        self.memory_monitor.stop_monitoring()
        if self.bash_session is not None:
            self.bash_session.close()
        if self.browser is not None:
            self.browser.close()


class ActionRequest(BaseModel):
    action: dict


if __name__ == "__main__":
    # PYTHONPATH=./ python core/runtime/server.py --working-dir /Users/yashengsun/Proj2025/MC-Scientist --port 8333
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8333, help="Port to listen on")
    parser.add_argument(
        "--working-dir",
        type=str,
        default="/Users/yashengsun/Proj2025/MC-Scientist",
        help="Working directory",
    )
    parser.add_argument(
        "--plugins", type=str, help="Plugins to initialize", default="", nargs="+"
    )
    parser.add_argument(
        "--username", type=str, help="User to run as", default="yashengsun"
    )
    parser.add_argument("--user-id", type=int, help="User ID to run as", default=1000)
    parser.add_argument(
        "--browsergym-eval-env",
        type=str,
        help="BrowserGym environment used for browser evaluation",
        default=None,
    )
    # example: python client.py 8000 --working-dir /workspace --plugins JupyterRequirement
    args = parser.parse_args()

    plugins_to_load: list[Plugin] = []
    if args.plugins:
        for plugin in args.plugins:
            if plugin not in ALL_PLUGINS:
                raise ValueError(f"Plugin {plugin} not found")
            plugins_to_load.append(ALL_PLUGINS[plugin]())  # type: ignore

    client: ActionExecutor | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global client
        client = ActionExecutor(
            plugins_to_load,
            work_dir=args.working_dir,
            username=args.username,
            user_id=args.user_id,
            browsergym_eval_env=args.browsergym_eval_env,
        )
        await client.ainit()
        yield
        # Clean up & release the resources
        client.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/alive")
    async def alive():
        """Health check endpoint that verifies client initialization status"""
        if client is None or not client._initialized:
            return {"status": "not initialized"}
        return {"status": "ok"}

    @app.get("/server_info")
    async def get_server_info():
        assert client is not None
        current_time = time.time()
        uptime = current_time - client.start_time
        idle_time = current_time - client.last_execution_time

        response = {
            "uptime": uptime,
            "idle_time": idle_time,
            "resources": get_system_stats(),
        }
        # logger.info('Server info endpoint response: %s', response)
        return response

    @app.post("/execute_action")
    async def execute_action(action_request: ActionRequest):
        logger.info("action: ", action_request.action)
        assert client is not None
        try:
            action = event_from_dict(action_request.action)
            if not isinstance(action, Action):
                raise HTTPException(status_code=400, detail="Invalid action type")
            client.last_execution_time = time.time()

            observation = await client.run_action(action)
            return event_to_dict(observation)
        except Exception as e:
            logger.error(f"Error while running /execute_action: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=traceback.format_exc(),
            )

    @app.post("/list_files")
    async def list_files(request: Request):
        """List files in the specified path.

        This function retrieves a list of files from the agent's runtime file store,
        excluding certain system and hidden files/directories.

        To list files:
        ```sh
        curl http://localhost:3000/api/list-files
        ```

        Args:
            request (Request): The incoming request object.
            path (str, optional): The path to list files from. Defaults to '/'.

        Returns:
            list: A list of file names in the specified path.

        Raises:
            HTTPException: If there's an error listing the files.
        """

        assert client is not None
        # get request as dict
        request_dict = await request.json()
        path = request_dict.get("path", None)

        # Get the full path of the requested directory
        if path is None:
            full_path = client.initial_cwd
        elif os.path.isabs(path):
            full_path = path
        else:
            full_path = os.path.join(client.initial_cwd, path)

        if not os.path.exists(full_path):
            # if user just removed a folder, prevent server error 500 in UI
            return []

        try:
            # Check if the directory exists
            if not os.path.exists(full_path) or not os.path.isdir(full_path):
                return []

            entries = os.listdir(full_path)

            # Separate directories and files
            directories = []
            files = []
            for entry in entries:
                # Remove leading slash and any parent directory components
                entry_relative = entry.lstrip("/").split("/")[-1]

                # Construct the full path by joining the base path with the relative entry path
                full_entry_path = os.path.join(full_path, entry_relative)
                if os.path.exists(full_entry_path):
                    is_dir = os.path.isdir(full_entry_path)
                    if is_dir:
                        # add trailing slash to directories
                        # required by FE to differentiate directories and files
                        entry = entry.rstrip("/") + "/"
                        directories.append(entry)
                    else:
                        files.append(entry)

            # Sort directories and files separately
            directories.sort(key=lambda s: s.lower())
            files.sort(key=lambda s: s.lower())

            # Combine sorted directories and files
            sorted_entries = directories + files
            return sorted_entries

        except Exception as e:
            # logger.error(f'Error listing files: {e}')
            return []

    run(app, host="0.0.0.0", port=args.port)

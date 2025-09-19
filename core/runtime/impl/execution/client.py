import os
import shutil
import tempfile
import threading
from abc import abstractmethod
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests

from core.events.action import (Action, AgentThinkAction,
                                BrowseInteractiveAction, BrowseURLAction,
                                CmdRunAction, FileEditAction, FileReadAction,
                                FileWriteAction, MessageAction, NullAction,
                                TaskGraphBuildAction, SnapshotAction, RollbackAction,
                                RepoPlanAction, RepoCreateAction, RepoAnalyzerAction, RepoUpdateAction, RepoVerifyAction, PaperReproductionAnalyzerAction, RepoDebugAction, RepoEditAction, PDFQueryAction, IPythonRunCellAction, RepoJudgeAction, PaperRubricAction, ExperimentManagerAction)
from core.events.event import FileEditSource, FileReadSource
from core.events.observation import (AgentThinkObservation,
                                     CmdOutputObservation, ErrorObservation,
                                     FileEditObservation, FileReadObservation,
                                     NullObservation, Observation,
                                     TaskGraphBuildObservation, SnapshotObservation, RollbackObservation,
                                     RepoPlanObservation, RepoDebugObservation)
from core.events.serialization import (ACTION_TYPE_TO_CLASS, event_to_dict,
                                       observation_from_dict)
from core.runtime.plugins import PluginRequirement
from core.runtime.utils.request import send_request
from core.utils.http_session import HttpSession
from core.utils.types.exceptions import (AgentRuntimeError,
                                         AgentRuntimeTimeoutError)
from core.events.action.image import ImageEntityExtractAction, GroundingSAMAction, InpaintRemoveAction, SDXLInpaintAction, LAMARemoveAction


class ActionExecutionClient:
    """Base class for runtimes that interact with the action execution server.

    This class contains shared logic between DockerRuntime and RemoteRuntime
    for interacting with the HTTP server defined in action_execution_server.py.
    """

    def __init__(
        self,
        config,
        sid: str = "default",
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Any | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
    ):
        self.config = config
        self.action_semaphore = threading.Semaphore(1)  # Ensure one action at a time
        self._runtime_closed: bool = False
        self.session = HttpSession()

    def run_action(self, action: Action) -> Observation:
        """Run an action and return the resulting observation.
        If the action is not runnable in any runtime, a NullObservation is returned.
        If the action is not supported by the current runtime, an ErrorObservation is returned.
        """
        if not action.runnable:
            if isinstance(action, AgentThinkAction):
                return AgentThinkObservation("Your thought has been logged.")
            return NullObservation("")
        action_type = action.action  # type: ignore[attr-defined]
        if action_type not in ACTION_TYPE_TO_CLASS:
            return ErrorObservation(f"Action {action_type} does not exist.")
        if not hasattr(self, action_type):
            return ErrorObservation(
                f"Action {action_type} is not supported in the current runtime."
            )
        observation = getattr(self, action_type)(action)
        return observation

    def _get_action_execution_server_host(self):
        return self.api_url

    def _send_action_server_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """Send a request to the action execution server.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to send the request to
            **kwargs: Additional arguments to pass to requests.request()

        Returns:
            Response from the server

        Raises:
            AgentRuntimeError: If the request fails
        """
        return send_request(self.session, method, url, **kwargs)

    # ---- Generic helper: image entity extraction via standard flow ----
    def image_entity_extract(self, action: "ImageEntityExtractAction") -> Observation:
        if action.timeout is None:
            action.set_hard_timeout(180, blocking=False)
        return self.send_action_for_execution(action)

    # ---- GroundingSAM passthrough (server will call external FastAPI) ----
    def grounding_sam(self, action: "GroundingSAMAction") -> Observation:
        if action.timeout is None:
            action.set_hard_timeout(600, blocking=False)
        return self.send_action_for_execution(action)

    def inpaint_remove(self, action: "InpaintRemoveAction") -> Observation:
        if action.timeout is None:
            action.set_hard_timeout(600, blocking=False)
        return self.send_action_for_execution(action)

    def send_action_for_execution(self, action: Action) -> Observation:
        if (
            isinstance(action, FileEditAction)
            and action.impl_source == FileEditSource.LLM_BASED_EDIT
        ):
            return self.llm_based_edit(action)

        # set timeout to default if not set
        if action.timeout is None:
            # We don't block the command if this is a default timeout action
            action.set_hard_timeout(self.config.sandbox.timeout, blocking=False)

        with self.action_semaphore:
            if not action.runnable:
                if isinstance(action, AgentThinkAction):
                    return AgentThinkObservation("Your thought has been logged.")
                return NullObservation("")
            # if (
            #     hasattr(action, 'confirmation_state')
            #     and action.confirmation_state
            #     == ActionConfirmationStatus.AWAITING_CONFIRMATION
            # ):
            #     return NullObservation('')
            action_type = action.action  # type: ignore[attr-defined]
            if action_type not in ACTION_TYPE_TO_CLASS:
                raise ValueError(f"Action {action_type} does not exist.")
            if not hasattr(self, action_type):
                return ErrorObservation(
                    f"Action {action_type} is not supported in the current runtime.",
                    error_id="AGENT_ERROR$BAD_ACTION",
                )
            # if (
            #     getattr(action, 'confirmation_state', None)
            #     == ActionConfirmationStatus.REJECTED
            # ):
            #     return UserRejectObservation(
            #         'Action has been rejected by the user! Waiting for further user input.'
            #     )

            assert action.timeout is not None
            try:
                with self._send_action_server_request(
                    "POST",
                    f"{self._get_action_execution_server_host()}/execute_action",
                    json={"action": event_to_dict(action)},
                    # wait a few more seconds to get the timeout error from client side
                    timeout=action.timeout + 5,
                ) as response:
                    output = response.json()
                    obs = observation_from_dict(output)
                    obs._cause = action.id  # type: ignore[attr-defined]
            except requests.Timeout:
                raise AgentRuntimeTimeoutError(
                    f"Runtime failed to return execute_action before the requested timeout of {action.timeout}s"
                )
            return obs

    def run(self, action: CmdRunAction) -> Observation:
        return self.send_action_for_execution(action)

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        return self.send_action_for_execution(action)

    def snapshot(self, action: SnapshotAction) -> Observation:
        return self.send_action_for_execution(action)

    def rollback(self, action: RollbackAction) -> Observation:
        return self.send_action_for_execution(action)

    def read(self, action: FileReadAction) -> Observation:
        return self.send_action_for_execution(action)

    def write(self, action: FileWriteAction) -> Observation:
        return self.send_action_for_execution(action)

    def edit(self, action: FileEditAction) -> Observation:
        return self.send_action_for_execution(action)

    def browse(self, action: BrowseURLAction) -> Observation:
        return self.send_action_for_execution(action)

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return self.send_action_for_execution(action)

    def task_graph_build(self, action: TaskGraphBuildAction) -> Observation:
        action.set_hard_timeout(300, blocking=False)
        return self.send_action_for_execution(action)

    def repo_plan(self, action: RepoPlanAction) -> Observation:
        # Ensure action.timeout is not None before adding 30 seconds
        if action.timeout is None:
            action.set_hard_timeout(350, blocking=False)
        else:
            action.set_hard_timeout(action.timeout + 300, blocking=False)
        return self.send_action_for_execution(action)

    def repo_create(self, action: RepoCreateAction) -> Observation:
        # Ensure action.timeout is not None before adding timeout for coding phase
        if action.timeout is None:
            action.set_hard_timeout(1200, blocking=False)  # 20 minutes for full pipeline with coding
        else:
            action.set_hard_timeout(action.timeout + 600, blocking=False)
        return self.send_action_for_execution(action)

    def repo_analyzer(self, action: RepoAnalyzerAction) -> Observation:
        # Ensure action.timeout is not None before adding timeout for analysis
        if action.timeout is None:
            action.set_hard_timeout(450, blocking=False)  # 7+ minutes for analysis
        else:
            action.set_hard_timeout(action.timeout + 300, blocking=False)
        return self.send_action_for_execution(action)

    def repo_update(self, action: RepoUpdateAction) -> Observation:
        # Ensure action.timeout is not None before adding timeout for repo update
        if action.timeout is None:
            action.set_hard_timeout(300, blocking=False)  # 5 minutes for repo update
        else:
            action.set_hard_timeout(action.timeout + 180, blocking=False)
        return self.send_action_for_execution(action)

    def repo_verify(self, action: RepoVerifyAction) -> Observation:
        # Ensure action.timeout is not None before adding timeout for repo verification
        if action.timeout is None:
            action.set_hard_timeout(350, blocking=False)
        else:
            action.set_hard_timeout(action.timeout + 300, blocking=False)
        return self.send_action_for_execution(action)

    def repo_debug(self, action: RepoDebugAction) -> Observation:
        # Ensure action.timeout is not None before adding timeout for repo debugging
        if action.timeout is None:
            action.set_hard_timeout(350, blocking=False)
        else:
            action.set_hard_timeout(action.timeout + 300, blocking=False)
        return self.send_action_for_execution(action)

    def paper_reproduction_analyzer(self, action: PaperReproductionAnalyzerAction) -> Observation:
        # Ensure action.timeout is not None before adding timeout for paper reproduction analysis
        if action.timeout is None:
            action.set_hard_timeout(450, blocking=False)  # 7.5 minutes for analysis
        else:
            action.set_hard_timeout(action.timeout + 300, blocking=False)
        return self.send_action_for_execution(action)

    def paper_rubric(self, action: PaperRubricAction) -> Observation:
        # Set timeout to 600 seconds (10 minutes) for rubric extraction
        if action.timeout is None:
            action.set_hard_timeout(600, blocking=False)
        else:
            action.set_hard_timeout(action.timeout + 300, blocking=False)
        return self.send_action_for_execution(action)

    def repo_edit(self, action: RepoEditAction) -> Observation:
        # Set timeout to 600 seconds (10 minutes) to give aider more time for complex edits
        action.set_hard_timeout(600, blocking=False)
        return self.send_action_for_execution(action)

    def repo_judge(self, action: RepoJudgeAction) -> Observation:
        # Set timeout to 900 seconds (15 minutes) to give aider more time to analyze multiple rubrics
        action.set_hard_timeout(900, blocking=False)
        return self.send_action_for_execution(action)

    def pdf_query(self, action: PDFQueryAction) -> Observation:
        # Set timeout to 300 seconds
        action.set_hard_timeout(300, blocking=False)
        return self.send_action_for_execution(action)

    def experiment_manager(self, action: ExperimentManagerAction) -> Observation:
        # Set timeout to 900 seconds (15 minutes) for experiment orchestration
        action.set_hard_timeout(900, blocking=False)
        return self.send_action_for_execution(action)

    def close(self) -> None:
        # Make sure we don't close the session multiple times
        # Can happen in evaluation
        if self._runtime_closed:
            return
        self._runtime_closed = True
        self.session.close()

    def get_server_info(self) -> dict:
        """Get server information"""
        try:
            response = self._send_action_server_request(
                "GET",
                f"{self._get_action_execution_server_host()}/server_info",
                timeout=10,
            )
            return response.json()
        except Exception as e:
            print(f"Request failed: {str(e)}")
            raise AgentRuntimeError(f"Failed to get server info: {str(e)}")

    def check_if_alive(self) -> None:
        """Check if the server is alive by making a request to the /alive endpoint"""
        try:
            with self._send_action_server_request(
                "GET",
                f"{self._get_action_execution_server_host()}/alive",
                timeout=5,
            ) as response:
                if response.status_code != 200:
                    raise AgentRuntimeError(
                        f"Server returned status code {response.status_code}"
                    )
        except Exception as e:
            raise AgentRuntimeError(f"Server is not alive: {str(e)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1:8333", help="Server address"
    )
    parser.add_argument("--file", type=str, help="File path to read")
    parser.add_argument("--info", action="store_true", help="Get server info")
    parser.add_argument("--browse", type=str, help="URL to browse")
    parser.add_argument("--interactive", type=str, help="Interactive browser action")
    args = parser.parse_args()

    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.sandbox = type("Sandbox", (), {"timeout": 30})()

    client = ActionExecutionClient(
        config=SimpleConfig(),
    )

    try:
        if args.info:
            # Get server information
            info = client.get_server_info()
            print("\nServer Information:")
            print(f"Uptime: {info['uptime']:.2f} seconds")
            print(f"Idle Time: {info['idle_time']:.2f} seconds")
            print("\nSystem Resources:")
            for key, value in info["resources"].items():
                print(f"{key}: {value}")
        elif args.file:
            # Create FileReadAction
            action = FileReadAction(path=args.file)
            result = client.run(action)
            print(f"File Content:\n{result.content}")
        elif args.browse:
            # Test browser URL
            print(f"\nTesting browser URL: {args.browse}")
            action = BrowseURLAction(url=args.browse, thought="Testing browser URL")
            result = client.browse(action)
            print(f"Browser Result:\n{result}")
        elif args.interactive:
            # Test interactive browser
            print(f"\nTesting interactive browser action: {args.interactive}")
            action = BrowseInteractiveAction(
                browser_actions=args.interactive, thought="Testing interactive browser"
            )
            result = client.browse_interactive(action)
            print(f"Interactive Browser Result:\n{result}")
        else:
            print("Please provide arguments:")
            print("  --file <file_path>     Read file content")
            print("  --info                Get server information")
            print("  --browse <url>        Browse a URL")
            print("  --interactive <action> Run an interactive browser action")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

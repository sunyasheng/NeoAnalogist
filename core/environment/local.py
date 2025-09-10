"""
Local environment implementation for direct file and command operations.
"""

import difflib
import logging
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, TypeVar

from core.environment.base import Environment
from core.environment.openhands_aci.editor.editor import OHEditor

T = TypeVar("T")


def get_diff_with_context(
    old_contents: str, new_contents: str, filepath: str = "file", context_lines: int = 3
) -> str:
    """Generate a unified diff with context lines using difflib directly."""
    old_lines = old_contents.strip().splitlines(keepends=True)
    new_lines = new_contents.strip().splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"--- {filepath}",
        tofile=f"+++ {filepath}",
        n=context_lines,
        lineterm="",
    )
    return "\n".join(diff)


class LocalEnvironment(Environment):
    """Local environment implementation for system interaction"""

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__()
        self.working_dir = working_dir or os.getcwd()
        self._setup_logger()
        self.file_editor = OHEditor(workspace_root=self.working_dir)

    def _setup_logger(self):
        data_dir = os.path.join(self.working_dir, "data")
        log_dir = os.path.join(data_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "environment.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _handle_operation(
        self,
        op_type: str,
        op_func: Callable[[], T],
        log_msg: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle operations with consistent error handling and logging"""
        self.logger.info(log_msg)

        try:
            result = op_func()
            self._record_history(op_type, {**context, "result": result})

            if "stderr" in result and result["stderr"]:
                self.logger.warning(f"{op_type} stderr: {result['stderr']}")
            if "success" in result:
                self.logger.info(f"{op_type} result: {result['success']}")

            return result

        except Exception as e:
            import traceback

            traceback.print_exc()
            error_msg = str(e)
            error_result = {"success": False, "error": error_msg, **context}

            # Add specific fields for command errors
            if op_type == "command":
                error_result.update(
                    {"stdout": "", "stderr": error_msg, "return_code": -1}
                )

            self._record_history(op_type, {**context, "result": error_result})
            self.logger.error(f"Error in {op_type}: {e}")
            return error_result

    def execute_command(self, command: str) -> Dict[str, Any]:
        def run_command():
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
            )
            stdout, stderr = process.communicate()
            return_code = process.returncode

            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "success": return_code == 0,
            }

        return self._handle_operation(
            "command",
            run_command,
            f"Executing command: {command}",
            {"command": command},
        )

    def read_file(self, file_path: str) -> Dict[str, Any]:
        def do_read_file():
            working_dir_file_path = None
            actual_path = file_path

            if not os.path.isabs(file_path):
                working_dir_file_path = os.path.join(self.working_dir, file_path)
                if not os.path.exists(working_dir_file_path):
                    os.makedirs(os.path.dirname(working_dir_file_path), exist_ok=True)
                    cmd = f"cp -r {file_path} {os.path.dirname(working_dir_file_path)}"
                    os.system(cmd)
                actual_path = working_dir_file_path

            with open(actual_path, "r") as f:
                content = f.read()

            return {"success": True, "content": content, "file_path": actual_path}

        return self._handle_operation(
            "read_file",
            do_read_file,
            f"Reading file: {file_path}",
            {"file_path": file_path},
        )

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        def do_write_file():
            actual_path = file_path
            if not os.path.isabs(file_path):
                actual_path = os.path.join(self.working_dir, file_path)

            directory = os.path.dirname(actual_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(actual_path, "w") as f:
                f.write(content)

            return {"success": True, "file_path": actual_path}

        return self._handle_operation(
            "write_file",
            do_write_file,
            f"Writing to file: {file_path}",
            {"file_path": file_path},
        )

    def str_replace_editor(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Edit a file with the given action.

        Args:
            action: Dictionary containing edit action details
                - path: File path
                - command: Edit command
                - file_text: Optional file text
                - old_str: Optional string to replace
                - new_str: Optional replacement string
                - insert_line: Optional line number

        Returns:
            Dict containing:
                - success: Boolean indicating success
                - message: Result message
                - old_content: Previous file content
                - new_content: New file content
        """

        def do_edit():
            try:
                result = self.file_editor(
                    command=action["command"],
                    path=action["path"],
                    file_text=action.get("file_text"),
                    old_str=action.get("old_str"),
                    new_str=action.get("new_str"),
                    insert_line=action.get("insert_line"),
                    view_range=action.get("view_range"),
                    enable_linting=False,
                )
                # import pdb; pdb.set_trace()
                if result.error:
                    return {
                        "success": False,
                        "message": f"ERROR:\n{result.error}",
                        "old_content": None,
                        "new_content": None,
                        "file_path": action["path"],
                    }

                if not result.output:
                    self.logger.warning(
                        f'No output from file_editor for {action["path"]}'
                    )
                    return {
                        "success": True,
                        "message": "",
                        "old_content": result.old_content,
                        "new_content": result.new_content,
                        "file_path": action["path"],
                    }

                if action["command"] == "view" or action["command"] == "create":
                    diff_msg = result.output
                else:
                    diff_msg = get_diff_with_context(
                        result.old_content, result.new_content, action["path"]
                    )

                return {
                    "success": True,
                    # "message": result.output,
                    "message": diff_msg,
                    "old_content": result.old_content,
                    "new_content": result.new_content,
                    "file_path": action["path"],
                }

            except Exception as e:
                # import traceback; traceback.print_exc()
                # import pdb; pdb.set_trace()
                return {
                    "success": False,
                    "message": f"ERROR:\n{str(e)}",
                    "old_content": None,
                    "new_content": None,
                    "file_path": action["path"],
                }

        return self._handle_operation(
            "str_replace_editor",
            do_edit,
            f"Str-Replace: {action['path']}",
            {"action": action},
        )

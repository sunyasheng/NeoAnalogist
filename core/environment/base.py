"""
Base class for environment interactions.
"""

import abc
import logging
from typing import Any, Dict, List, Optional


class Environment(abc.ABC):
    """
    Abstract base class for environment interactions.
    Defines the core interface for environment implementations.
    """

    def __init__(self):
        """Initialize the environment."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        self.history = []

    @abc.abstractmethod
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a shell command and return the result.

        Args:
            command: The command to execute

        Returns:
            Dictionary containing execution results
        """
        pass

    @abc.abstractmethod
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a file and return its contents.

        Args:
            file_path: Path to the file to read

        Returns:
            Dictionary containing file contents or error
        """
        pass

    @abc.abstractmethod
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file.

        Args:
            file_path: Path to the file to write
            content: Content to write to the file

        Returns:
            Dictionary indicating success or failure
        """
        pass

    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the n most recent history entries.

        Args:
            n: Number of entries to return

        Returns:
            List of history entries
        """
        return self.history[-n:] if len(self.history) >= n else self.history.copy()

    def _record_history(self, entry_type: str, data: Dict[str, Any]) -> None:
        """
        Record an entry in the history.

        Args:
            entry_type: Type of history entry
            data: Data for the history entry
        """
        self.history.append({"type": entry_type, **data})

from dataclasses import dataclass
from typing import ClassVar

from core.events.event import Action, ActionType


@dataclass
class RepoPlanAction(Action):
    """Action for planning repository implementation based on a paper.
    
    This action takes a paper content and generates a plan for implementing
    the repository based on the paper's methodology and experiments.
    
    Attributes:
        paper_path (str): Path to the paper file
        paper_format (str): Format of the paper (JSON, LaTeX, or MARKDOWN)
        output_dir (str): Directory to save the planning results
        thought (str): The reasoning behind the planning
        action (str): The action type, namely ActionType.REPO_PLAN
    """
    
    paper_path: str
    paper_format: str = "MARKDOWN"  # Options: "JSON", "LaTeX", "MARKDOWN"
    output_dir: str = ""
    thought: str = ""
    action: str = ActionType.REPO_PLAN
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Planning repository implementation for paper at: {self.paper_path}"

    def __str__(self) -> str:
        ret = "**RepoPlanAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"PAPER_PATH: {self.paper_path}\n"
        ret += f"PAPER_FORMAT: {self.paper_format}\n"
        ret += f"OUTPUT_DIR: {self.output_dir}"
        return ret


@dataclass
class RepoCreateAction(Action):
    """Action for creating a full repository implementation with code generation based on a paper.
    
    This action takes a paper content and generates both planning and actual code files
    for implementing the repository based on the paper's methodology and experiments.
    
    Attributes:
        paper_path (str): Path to the paper file
        paper_format (str): Format of the paper (JSON, LaTeX, or MARKDOWN)
        output_dir (str): Directory to save the planning results and artifacts
        output_repo_dir (str): Directory to save the generated code repository
        thought (str): The reasoning behind the creation
        action (str): The action type, namely ActionType.REPO_CREATE
    """
    
    paper_path: str
    paper_format: str = "MARKDOWN"  # Options: "JSON", "LaTeX", "MARKDOWN"
    output_dir: str = ""
    output_repo_dir: str = ""
    thought: str = ""
    action: str = ActionType.REPO_CREATE
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Creating full repository implementation with code generation for paper at: {self.paper_path}"

    def __str__(self) -> str:
        ret = "**RepoCreateAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"PAPER_PATH: {self.paper_path}\n"
        ret += f"PAPER_FORMAT: {self.paper_format}\n"
        ret += f"OUTPUT_DIR: {self.output_dir}\n"
        ret += f"OUTPUT_REPO_DIR: {self.output_repo_dir}"
        return ret


@dataclass
class RepoAnalyzerAction(Action):
    """Action for analyzing repository implementation by comparing paper with existing codebase.
    
    This action takes a paper content and an existing codebase directory, then generates
    a detailed analysis report comparing what's described in the paper versus what's
    actually implemented in the codebase.
    
    Attributes:
        paper_path (str): Path to the paper file
        codebase_path (str): Path to the existing codebase directory to analyze
        paper_format (str): Format of the paper (JSON, LaTeX, or MARKDOWN)
        output_dir (str): Directory to save the analysis results
        thought (str): The reasoning behind the analysis
        action (str): The action type, namely ActionType.REPO_ANALYZER
    """
    
    paper_path: str
    codebase_path: str
    paper_format: str = "MARKDOWN"  # Options: "JSON", "LaTeX", "MARKDOWN"
    output_dir: str = ""
    thought: str = ""
    action: str = ActionType.REPO_ANALYZER
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Analyzing repository implementation by comparing paper at: {self.paper_path} with codebase at: {self.codebase_path}"

    def __str__(self) -> str:
        ret = "**RepoAnalyzerAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"PAPER_PATH: {self.paper_path}\n"
        ret += f"CODEBASE_PATH: {self.codebase_path}\n"
        ret += f"PAPER_FORMAT: {self.paper_format}\n"
        ret += f"OUTPUT_DIR: {self.output_dir}"
        return ret


@dataclass
class RepoUpdateAction(Action):
    """Action for updating repository code based on user requirements.
    
    This action takes a repository path and user requirements, then generates
    modifications to implement new features and improve existing functionality.
    
    Attributes:
        repo_path (str): Path to the repository to be updated
        requirements (str): Detailed requirements describing what changes need to be made
        target_files (list): Optional list of specific files to modify
        context (str): Optional additional context about the changes needed
        apply_changes (bool): Whether to apply the changes to actual files

        save_snapshot (bool): Whether to save a snapshot after applying changes
        thought (str): The reasoning behind the update
        action (str): The action type, namely ActionType.REPO_UPDATE
    """
    
    repo_path: str
    requirements: str
    target_files: list = None
    context: str = ""
    apply_changes: bool = False
    save_snapshot: bool = True
    thought: str = ""
    action: str = ActionType.REPO_UPDATE
    runnable: ClassVar[bool] = True

    def __post_init__(self):
        if self.target_files is None:
            self.target_files = []

    @property
    def message(self) -> str:
        return f"Updating repository at: {self.repo_path} with requirements: {self.requirements[:100]}..."

    def __str__(self) -> str:
        ret = "**RepoUpdateAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"REQUIREMENTS: {self.requirements}\n"
        if self.target_files:
            ret += f"TARGET_FILES: {self.target_files}\n"
        if self.context:
            ret += f"CONTEXT: {self.context}\n"
        ret += f"APPLY_CHANGES: {self.apply_changes}\n"
        ret += f"SAVE_SNAPSHOT: {self.save_snapshot}"
        return ret


@dataclass
class RepoVerifyAction(Action):
    """Action for verifying repository implementation and functionality.
    
    This action takes a repository path and requirement description, then performs
    comprehensive analysis to verify the repository's structure, dependencies,
    entry points, and execution capabilities.
    
    Attributes:
        repo_path (str): Path to the repository to be verified
        requirement (str): Description of what the repository should accomplish
        verification_level (str): Level of verification (basic, functional, comprehensive)
        thought (str): The reasoning behind the verification
        action (str): The action type, namely ActionType.REPO_VERIFY
    """
    
    repo_path: str
    requirement: str
    verification_level: str = "comprehensive"  # Options: "basic", "functional", "comprehensive"
    thought: str = ""
    action: str = ActionType.REPO_VERIFY
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Verifying repository at: {self.repo_path} with requirement: {self.requirement[:100]}..."

    def __str__(self) -> str:
        ret = "**RepoVerifyAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"REQUIREMENT: {self.requirement}\n"
        ret += f"VERIFICATION_LEVEL: {self.verification_level}"
        return ret


@dataclass
class PaperReproductionAnalyzerAction(Action):
    """Action for analyzing paper reproduction requirements.
    
    This action takes a paper path and analysis level, then generates
    detailed implementation requirements for reproducing the paper.
    
    Attributes:
        paper_path (str): Path to the paper file to analyze
        paper_content (str): Direct paper content (alternative to paper_path)
        analysis_level (str): Analysis level ("basic", "detailed", "comprehensive")
        thought (str): The reasoning behind the analysis
        action (str): The action type, namely ActionType.PAPER_REPRODUCTION_ANALYZER
    """
    
    paper_path: str = ""
    paper_content: str = ""
    analysis_level: str = "detailed"
    thought: str = ""
    action: str = ActionType.PAPER_REPRODUCTION_ANALYZER
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Analyzing paper reproduction requirements (level: {self.analysis_level})"

    def __str__(self) -> str:
        ret = "**PaperReproductionAnalyzerAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        if self.paper_path:
            ret += f"PAPER_PATH: {self.paper_path}\n"
        if self.paper_content:
            ret += f"PAPER_CONTENT: {self.paper_content[:100]}...\n"
        ret += f"ANALYSIS_LEVEL: {self.analysis_level}"
        return ret


@dataclass
class RepoDebugAction(Action):
    """Action for debugging and fixing code issues in repositories using refact agent.
    
    This action takes a repository path and action description, then uses refact agent
    to automatically analyze and fix code issues including syntax errors, bugs, and
    optimization problems.
    
    Attributes:
        repo_path (str): Path to the repository to be debugged
        action_description (str): For errors: paste exact error message. For editing: describe what to add/modify
        thought (str): The reasoning behind the debug operation
        action (str): The action type, namely ActionType.REPO_DEBUG
    """
    
    repo_path: str
    action_description: str
    thought: str = ""
    action: str = ActionType.REPO_DEBUG
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Debugging repository at: {self.repo_path} with action: {self.action_description[:100]}..."

    def __str__(self) -> str:
        ret = "**RepoDebugAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"ACTION_DESCRIPTION: {self.action_description}"
        return ret


@dataclass
class RepoEditAction(Action):
    """Action for editing repository code based on user instructions (not using refact agent).
    
    This action takes a repository path and an edit description, then applies the specified edits to the repository codebase.
    
    Attributes:
        repo_path (str): Path to the repository to be edited
        edit_description (str): Description of the edit to perform (e.g., what to add/modify)
        traceback (str, optional): Error message and traceback if this is a bug fix request
        thought (str): The reasoning behind the edit operation
        action (str): The action type, namely ActionType.REPO_EDIT
    """
    repo_path: str
    edit_description: str
    traceback: str = ""
    thought: str = ""
    action: str = ActionType.REPO_EDIT
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Editing repository at: {self.repo_path} with edit: {self.edit_description[:100]}..."

    def __str__(self) -> str:
        ret = "**RepoEditAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"EDIT_DESCRIPTION: {self.edit_description}"
        if self.traceback:
            ret += f"\nTRACEBACK: {self.traceback}"
        return ret


@dataclass
class RepoRunAction(Action):
    """Action for running a repository's reproduce.sh script in an isolated environment.
    
    This action takes a repository path and runs its reproduce.sh script in an isolated
    Docker container to ensure reproducible execution environment.
    
    Attributes:
        repo_path (str): Path to the repository to be executed
        timeout (int): Timeout in seconds for the execution
        retry_threshold (int): Threshold in seconds for retrying failed executions
        output_dir (str): Optional directory to save output files
        log_file (str): Optional path to save execution logs with timestamp
        docker_image (str): Docker image to use for execution
        memory_limit (str): Memory limit for the container
        network_enabled (bool): Whether to enable network access
        gpu_enabled (bool): Whether to enable GPU access
        environment_vars (dict): Optional environment variables to pass to container
        use_persistent_containers (bool): Whether to use persistent containers to avoid reinstalling dependencies
        thought (str): The reasoning behind the run operation
        action (str): The action type, namely ActionType.REPO_RUN
    """
    
    repo_path: str
    timeout: int = 3600  # 1 hour default
    retry_threshold: int = 600  # 10 minutes default
    output_dir: str = ""
    log_file: str = "experiment.log"  # Default path to save execution logs with timestamp
    docker_image: str = "pb-reproducer:latest"  # Use paperbench reproducer image
    memory_limit: str = "4g"  # Memory limit for container
    network_enabled: bool = True  # Whether to enable network access
    gpu_enabled: bool = True  # Whether to enable GPU access
    environment_vars: dict = None  # Environment variables to pass
    use_persistent_containers: bool = True  # Whether to use persistent containers
    thought: str = ""
    action: str = ActionType.REPO_RUN
    runnable: ClassVar[bool] = True

    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}

    def set_hard_timeout(self, timeout: int):
        """Set hard timeout for the action"""
        self.timeout = timeout

    @property
    def message(self) -> str:
        return f"Running repository reproduce.sh script at: {self.repo_path} with timeout: {self.timeout}s"

    def __str__(self) -> str:
        ret = "**RepoRunAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"TIMEOUT: {self.timeout} seconds\n"
        ret += f"RETRY_THRESHOLD: {self.retry_threshold} seconds\n"
        ret += f"DOCKER_IMAGE: {self.docker_image}\n"
        ret += f"MEMORY_LIMIT: {self.memory_limit}\n"
        ret += f"NETWORK_ENABLED: {self.network_enabled}\n"
        ret += f"GPU_ENABLED: {self.gpu_enabled}\n"
        if self.output_dir:
            ret += f"OUTPUT_DIR: {self.output_dir}\n"
        if self.log_file:
            ret += f"LOG_FILE: {self.log_file}\n"
        if self.environment_vars:
            ret += f"ENVIRONMENT_VARS: {self.environment_vars}"
        return ret


@dataclass
class RepoJudgeAction(Action):
    """Action for judging repository code based on a rubric file.
    
    This action takes a repository path and a rubric file path,
    then uses aider to analyze the repo and answer each rubric question. Finally, calls LLM to summarize
    whether the repo satisfies the rubric overall.
    
    Attributes:
        repo_path (str): Path to the repository to be judged
        rubric_file_path (str): Path to rubric file to read and parse
        thought (str): The reasoning behind the judge operation
        action (str): The action type, namely ActionType.REPO_JUDGE
    """
    
    repo_path: str
    rubric_file_path: str
    thought: str = ""
    action: str = ActionType.REPO_JUDGE
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Judging repository at: {self.repo_path} using rubric file: {self.rubric_file_path}"

    def __str__(self) -> str:
        ret = "**RepoJudgeAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"RUBRIC_FILE_PATH: {self.rubric_file_path}\n"
        return ret


@dataclass
class PaperRubricAction(Action):
    """Action for extracting rubrics from a PDF paper.
    
    This action analyzes a PDF paper and extracts both static and dynamic rubric requirements
    that need to be implemented in the code.
    
    Attributes:
        paper_path (str): Path to the PDF paper file
        paper_content (str): Content of the paper (optional, will be read from paper_path if not provided)
        include_static (bool): Whether to include static rubrics (code structure, functions, etc.)
        include_dynamic (bool): Whether to include dynamic rubrics (experimental results, tables, etc.)
        rubric_categories (list): Specific categories to extract (e.g., ["experiments", "evaluation", "results"])
        save_to_file (bool): Whether to save extracted rubrics to a text file
        output_dir (str): Directory to save rubric files (required parameter)
        timeout (int): Timeout in seconds for the action
    """
    
    paper_path: str
    output_dir: str  # Required parameter - no default value
    paper_content: str = ""
    include_static: bool = True
    include_dynamic: bool = True
    rubric_categories: list = None
    save_to_file: bool = True
    timeout: int = 600  # 10 minutes default
    action: str = ActionType.PAPER_RUBRIC
    runnable: ClassVar[bool] = True

    def __post_init__(self):
        if self.rubric_categories is None:
            self.rubric_categories = ["experiments", "evaluation", "results", "data", "models", "metrics"]

    def set_hard_timeout(self, timeout: int, blocking: bool = True):
        """Set hard timeout for this action"""
        self.timeout = timeout

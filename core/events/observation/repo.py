from dataclasses import dataclass
from typing import List
from core.events.event import Observation, ObservationType


@dataclass
class RepoPlanObservation(Observation):
    """This data class represents the result of a repository planning action.
    
    The observation contains the planning results for repository implementation.
    
    Attributes:
        plan_content (str): The content of the planning results
        observation (str): The observation type, namely ObservationType.REPO_PLAN
    """
    
    plan_content: str
    observation: str = ObservationType.REPO_PLAN

    @property
    def message(self) -> str:
        """Get a human-readable message describing the planning operation."""
        return "Successfully generated repository implementation plan"

    def __str__(self) -> str:
        """Get a string representation of the planning observation."""
        ret = "**RepoPlanObservation**\n"
        ret += "PLAN_CONTENT:\n"
        ret += self.plan_content
        return ret


@dataclass
class RepoCreateObservation(Observation):
    """This data class represents the result of a repository creation action.
    
    The observation contains both planning results and generated code files
    for repository implementation.
    
    Attributes:
        plan_content (str): The content of the planning results
        generated_files (str): Information about generated code files
        repo_path (str): Path to the generated repository
        observation (str): The observation type, namely ObservationType.REPO_CREATE
    """
    
    plan_content: str
    generated_files: str = ""
    repo_path: str = ""
    observation: str = ObservationType.REPO_CREATE

    @property
    def message(self) -> str:
        """Get a human-readable message describing the creation operation."""
        return "Successfully created full repository implementation with code generation"

    def __str__(self) -> str:
        """Get a string representation of the creation observation."""
        ret = "**RepoCreateObservation**\n"
        ret += "PLAN_CONTENT:\n"
        ret += self.plan_content
        if self.generated_files:
            ret += "\n\nGENERATED_FILES:\n"
            ret += self.generated_files
        if self.repo_path:
            ret += f"\n\nREPO_PATH: {self.repo_path}"
        return ret


@dataclass
class RepoAnalyzerObservation(Observation):
    """This data class represents the result of a repository analysis action.
    
    The observation contains the analysis results comparing paper with codebase
    implementation, including hierarchical functionality analysis and implementation status.
    
    Attributes:
        analysis_content (str): The content of the analysis results
        analysis_report (str): Structured analysis report with implementation status
        missing_functionalities (str): List of functionalities not implemented
        observation (str): The observation type, namely ObservationType.REPO_ANALYZER
    """
    
    analysis_content: str
    analysis_report: str = ""
    missing_functionalities: str = ""
    observation: str = ObservationType.REPO_ANALYZER

    @property
    def message(self) -> str:
        """Get a human-readable message describing the analysis operation."""
        return "Successfully analyzed repository implementation against paper"

    def __str__(self) -> str:
        """Get a string representation of the analysis observation."""
        ret = "**RepoAnalyzerObservation**\n"
        ret += "ANALYSIS_CONTENT:\n"
        ret += self.analysis_content
        if self.analysis_report:
            ret += "\n\nANALYSIS_REPORT:\n"
            ret += self.analysis_report
        if self.missing_functionalities:
            ret += "\n\nMISSING_FUNCTIONALITIES:\n"
            ret += self.missing_functionalities
        return ret


@dataclass
class RepoUpdateObservation(Observation):
    """This data class represents the result of a repository update action.
    
    The observation contains the results of updating repository code based on
    user requirements, including the implementation plan and modified files.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        plan (str): The implementation plan for the updates
        modified_files (str): Information about files that were modified
        changes_summary (str): Summary of changes made
        success (bool): Whether the update was successful
        repo_analysis (str): Analysis of the repository structure
        error_message (str): Error message if update failed
        applied (bool): Whether changes were applied to actual files
        file_diffs (str): Detailed diff information for changed files
        detailed_changes (str): Detailed analysis of what was changed
        observation (str): The observation type, namely ObservationType.REPO_UPDATE
    """
    
    content: str = ""
    plan: str = ""
    modified_files: str = ""
    changes_summary: str = ""
    success: bool = False
    repo_analysis: str = ""
    error_message: str = ""
    applied: bool = False
    file_diffs: str = ""
    detailed_changes: str = ""
    observation: str = ObservationType.REPO_UPDATE

    @property
    def message(self) -> str:
        """Get a human-readable message describing the update operation."""
        if self.success:
            return "Successfully updated repository code based on requirements"
        else:
            return f"Failed to update repository: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the update observation."""
        ret = "**RepoUpdateObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        
        if self.plan:
            ret += "\n📋 IMPLEMENTATION PLAN:\n"
            ret += self.plan + "\n"
        
        if self.changes_summary:
            ret += "\n📊 CHANGES SUMMARY:\n"
            ret += self.changes_summary + "\n"
        
        if self.detailed_changes:
            ret += "\n🔍 DETAILED CHANGES:\n"
            ret += self.detailed_changes + "\n"
        
        if self.file_diffs:
            ret += "\n📝 FILE DIFFS:\n"
            ret += self.file_diffs + "\n"
        
        if self.modified_files:
            ret += "\n📂 MODIFIED FILES:\n"
            ret += self.modified_files + "\n"
        
        if self.repo_analysis:
            ret += "\n🔬 REPO ANALYSIS:\n"
            ret += self.repo_analysis + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        ret += f"\n🔧 APPLIED: {self.applied}"
        return ret


@dataclass
class RepoVerifyObservation(Observation):
    """This data class represents the result of a repository verification action.
    
    The observation contains comprehensive verification results including
    repository analysis, execution attempts, and overall status assessment.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        overall_status (str): Overall verification status (FULLY_COMPLIANT, MOSTLY_COMPLIANT, etc.)
        overall_score (float): Numerical score from 0.0 to 1.0
        status_description (str): Human-readable description of the status
        analysis_steps (str): Summary of analysis steps performed
        execution_attempts (str): Summary of execution attempts and results
        dependencies (str): Information about dependencies and missing packages
        entry_points (str): Information about identified entry points
        summary (str): Overall summary of verification results
        error_message (str): Error message if verification failed
        observation (str): The observation type, namely ObservationType.REPO_VERIFY
    """
    
    content: str = ""
    overall_status: str = ""
    overall_score: float = 0.0
    status_description: str = ""
    analysis_steps: str = ""
    execution_attempts: str = ""
    dependencies: str = ""
    entry_points: str = ""
    summary: str = ""
    error_message: str = ""
    observation: str = ObservationType.REPO_VERIFY

    @property
    def message(self) -> str:
        """Get a human-readable message describing the verification operation."""
        if self.overall_status:
            return f"Repository verification completed with status: {self.overall_status}"
        else:
            return f"Repository verification failed: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the verification observation."""
        ret = "**RepoVerifyObservation**\n"
        
        if self.content:
            ret += "LLM SUMMARY REPORT:\n"
            ret += self.content + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret


@dataclass
class PaperReproductionAnalyzerObservation(Observation):
    """This data class represents the result of a paper reproduction analysis action.
    
    The observation contains the analysis results for paper reproduction requirements,
    including detailed implementation guidance and technical specifications.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        analysis_result (str): The detailed analysis result
        analysis_level (str): The analysis level used ("basic", "detailed", "comprehensive")
        success (bool): Whether the analysis was successful
        error_message (str): Error message if analysis failed
        observation (str): The observation type, namely ObservationType.PAPER_REPRODUCTION_ANALYZER
    """
    
    content: str = ""
    analysis_result: str = ""
    analysis_level: str = "detailed"
    success: bool = False
    error_message: str = ""
    observation: str = ObservationType.PAPER_REPRODUCTION_ANALYZER

    @property
    def message(self) -> str:
        """Get a human-readable message describing the analysis operation."""
        if self.success:
            return f"Successfully analyzed paper reproduction requirements (level: {self.analysis_level})"
        else:
            return f"Failed to analyze paper reproduction: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the analysis observation."""
        ret = "**PaperReproductionAnalyzerObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"ANALYSIS_LEVEL: {self.analysis_level}\n"
        
        if self.analysis_result:
            ret += "\n📋 ANALYSIS RESULT:\n"
            ret += "=" * 60 + "\n"
            ret += self.analysis_result + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret


@dataclass
class RepoDebugObservation(Observation):
    """This data class represents the result of a repository debug action.
    
    The observation contains the results of debugging and fixing code issues
    using refact agent, including what was fixed and any suggestions.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        success (bool): Whether the debug operation was successful
        output (str): Raw output from refact agent
        fixed_files (str): List of files that were fixed
        suggestions (str): Suggestions for further improvements
        execution_time (float): Time taken for the debug operation
        error_message (str): Error message if debug failed
        summary (str): Comprehensive summary of the debug operation
        observation (str): The observation type, namely ObservationType.REPO_DEBUG
    """
    
    content: str = ""
    success: bool = False
    output: str = ""
    fixed_files: str = ""
    suggestions: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    summary: str = ""
    observation: str = ObservationType.REPO_DEBUG

    @property
    def message(self) -> str:
        """Get a human-readable message describing the debug operation."""
        if self.success:
            return "Successfully debugged and fixed code issues in repository"
        else:
            return f"Failed to debug repository: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the debug observation."""
        ret = "**RepoDebugObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"EXECUTION_TIME: {self.execution_time:.2f} seconds\n"
        
        # Show summary first if available
        if self.summary:
            ret += "\n📊 COMPREHENSIVE SUMMARY:\n"
            ret += "=" * 60 + "\n"
            ret += self.summary + "\n"
            ret += "=" * 60 + "\n"
        

        
        if self.fixed_files:
            ret += "\n📝 FIXED FILES:\n"
            ret += self.fixed_files + "\n"
        
        if self.suggestions:
            ret += "\n💡 SUGGESTIONS:\n"
            ret += self.suggestions + "\n"
        
        # if self.output:
        #     ret += "\n📋 RAW OUTPUT:\n"
        #     ret += self.output + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret


@dataclass
class RepoEditObservation(Observation):
    """This data class represents the result of a repository edit action.
    
    The observation contains the results of editing repository code based on user instructions.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        success (bool): Whether the edit operation was successful
        output (str): Raw output or log from the edit operation
        modified_files (str): List or description of files that were modified
        suggestions (str): Suggestions for further improvements
        execution_time (float): Time taken for the edit operation
        error_message (str): Error message if edit failed
        summary (str): Summary of the edit operation
        observation (str): The observation type, namely ObservationType.REPO_EDIT
    """
    content: str = ""
    success: bool = False
    output: str = ""
    modified_files: str = ""
    suggestions: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    summary: str = ""
    observation: str = ObservationType.REPO_EDIT

    @property
    def message(self) -> str:
        if self.success:
            return "Successfully edited repository code."
        else:
            return f"Failed to edit repository: {self.error_message}"

    def __str__(self) -> str:
        ret = "**RepoEditObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"EXECUTION_TIME: {self.execution_time:.2f} seconds\n"
        if self.summary:
            ret += "\n📊 SUMMARY:\n" + self.summary + "\n"
        if self.modified_files:
            ret += "\n📝 MODIFIED FILES:\n" + self.modified_files + "\n"
        if self.suggestions:
            ret += "\n💡 SUGGESTIONS:\n" + self.suggestions + "\n"
        if self.output:
            ret += "\n📋 OUTPUT:\n" + self.output + "\n"
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        return ret


@dataclass
class RepoRunObservation(Observation):
    """This data class represents the result of a repository run action.
    
    The observation contains the results of running a repository's reproduce.sh script
    in an isolated Docker environment, including execution logs and status.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        success (bool): Whether the repository execution was successful
        output (str): Raw output from the reproduce.sh script execution
        logs (str): Container logs and execution details
        execution_time (float): Time taken for the execution
        error_message (str): Error message if execution failed
        summary (str): Summary of the execution results
        repo_path (str): Path to the repository that was executed
        docker_image (str): Docker image used for execution
        timeout (int): Timeout used for execution
        retry_threshold (int): Retry threshold used for execution
        memory_limit (str): Memory limit used for execution
        network_enabled (bool): Whether network was enabled
        gpu_enabled (bool): Whether GPU was enabled
        observation (str): The observation type, namely ObservationType.REPO_RUN
    """
    
    content: str = ""
    success: bool = False
    output: str = ""
    logs: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    summary: str = ""
    repo_path: str = ""
    docker_image: str = ""
    timeout: int = 0
    retry_threshold: int = 0
    memory_limit: str = ""
    network_enabled: bool = False
    gpu_enabled: bool = False
    exit_code: int = None
    timedout: bool = False
    retried_results: list = None
    container_id: str = ""
    observation: str = ObservationType.REPO_RUN

    def __post_init__(self):
        if self.retried_results is None:
            self.retried_results = []

    @property
    def message(self) -> str:
        """Get a human-readable message describing the run operation."""
        if self.success:
            return f"Successfully executed repository reproduce.sh script in {self.execution_time:.2f} seconds"
        else:
            return f"Failed to execute repository: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the run observation."""
        ret = "**RepoRunObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"EXECUTION_TIME: {self.execution_time:.2f} seconds\n"
        ret += f"REPO_PATH: {self.repo_path}\n"
        ret += f"DOCKER_IMAGE: {self.docker_image}\n"
        ret += f"TIMEOUT: {self.timeout} seconds\n"
        ret += f"RETRY_THRESHOLD: {self.retry_threshold} seconds\n"
        ret += f"MEMORY_LIMIT: {self.memory_limit}\n"
        ret += f"NETWORK_ENABLED: {self.network_enabled}\n"
        ret += f"GPU_ENABLED: {self.gpu_enabled}\n"
        ret += f"EXIT_CODE: {self.exit_code}\n"
        ret += f"TIMEDOUT: {self.timedout}\n"
        ret += f"CONTAINER_ID: {self.container_id}\n"
        
        if self.summary:
            ret += "\n📊 EXECUTION SUMMARY:\n"
            ret += "=" * 60 + "\n"
            ret += self.summary + "\n"
            ret += "=" * 60 + "\n"
        
        if self.output:
            # Truncate and filter output to prevent memory explosion
            filtered_output = self._filter_and_truncate_output(self.output)
            ret += "\n📋 SCRIPT OUTPUT:\n"
            ret += filtered_output + "\n"
        
        if self.logs:
            # Truncate logs as well
            filtered_logs = self._filter_and_truncate_output(self.logs)
            ret += "\n📝 CONTAINER LOGS:\n"
            ret += filtered_logs + "\n"
        
        if self.retried_results:
            ret += "\n🔄 RETRIED RESULTS:\n"
            for i, result in enumerate(self.retried_results):
                ret += f"Attempt {i+1}: {result}\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret

    def _filter_and_truncate_output(self, output: str, max_lines: int = 50, max_chars: int = 5000) -> str:
        """Filter and truncate output to prevent memory explosion."""
        if not output:
            return ""
        
        lines = output.split('\n')
        
        # Filter out progress bars and excessive download output
        filtered_lines = []
        progress_bar_count = 0
        max_progress_bars = 3  # Limit progress bar lines
        
        for line in lines:
            # Skip excessive progress bar lines
            if any(pattern in line for pattern in ['%|', 'B/s', '▏▎▍▌▋▊▉', '█', '=']):
                progress_bar_count += 1
                if progress_bar_count <= max_progress_bars:
                    filtered_lines.append(line)
                # Skip additional progress bars to reduce noise
                continue
            
            # Keep other lines
            filtered_lines.append(line)
        
        # Truncate if too many lines
        if len(filtered_lines) > max_lines:
            # Keep first 60% and last 40% of lines
            keep_start = int(max_lines * 0.6)
            keep_end = max_lines - keep_start
            filtered_lines = (
                filtered_lines[:keep_start] + 
                [f"\n... (truncated {len(lines) - max_lines} lines) ...\n"] + 
                filtered_lines[-keep_end:]
            )
        
        result = '\n'.join(filtered_lines)
        
        # Truncate if too many characters
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n... (truncated {len(output) - max_chars} characters) ..."
        
        return result


@dataclass
class RepoJudgeObservation(Observation):
    """This data class represents the result of a repository judge action.
    
    The observation contains the results of judging repository code based on rubric questions,
    including individual rubric results and overall summary.
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        success (bool): Whether the judge operation was successful
        rubric_results (list): List of individual rubric results
        summary (str): Overall summary of whether repo satisfies rubric
        execution_time (float): Time taken for the judge operation
        error_message (str): Error message if judge failed
        observation (str): The observation type, namely ObservationType.REPO_JUDGE
    """
    
    content: str = ""
    success: bool = False
    rubric_results: list = None
    summary: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    observation: str = ObservationType.REPO_JUDGE

    def __post_init__(self):
        if self.rubric_results is None:
            self.rubric_results = []
        else:
            # Convert dict items to RepoJudgeResult instances if needed
            from core.runtime.tasks.repo_judge import RepoJudgeResult
            converted_results = []
            for result in self.rubric_results:
                if isinstance(result, dict):
                    converted_results.append(RepoJudgeResult(**result))
                elif isinstance(result, RepoJudgeResult):
                    converted_results.append(result)
                else:
                    # Fallback: create a RepoJudgeResult with string representation
                    converted_results.append(RepoJudgeResult(
                        rubric=str(result.get('rubric', 'Unknown')) if isinstance(result, dict) else 'Unknown',
                        result=str(result.get('result', str(result))) if isinstance(result, dict) else str(result)
                    ))
            self.rubric_results = converted_results

    @property
    def message(self) -> str:
        """Get a human-readable message describing the judge operation."""
        if self.success:
            return f"Successfully judged repository with {len(self.rubric_results)} rubric questions in {self.execution_time:.2f} seconds"
        else:
            return f"Failed to judge repository: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the judge observation."""
        ret = "**RepoJudgeObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"EXECUTION_TIME: {self.execution_time:.2f} seconds\n"
        ret += f"RUBRIC_COUNT: {len(self.rubric_results)}\n"
        
        if self.summary:
            ret += "\n📊 OVERALL SUMMARY:\n"
            ret += "=" * 60 + "\n"
            ret += self.summary + "\n"
            ret += "=" * 60 + "\n"
        
        # if self.rubric_results:
        #     ret += "\n📋 RUBRIC RESULTS:\n"
        #     for i, result in enumerate(self.rubric_results):
        #         ret += f"\n--- Rubric {i+1} ---\n"
        #         ret += f"Question: {result.rubric}\n"
        #         ret += f"Answer: {result.result}\n"
        
        if self.content:
            ret += "\n📝 DETAILED CONTENT:\n"
            ret += self.content + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret


@dataclass
class PaperRubricObservation(Observation):
    """This data class represents the result of a paper rubric extraction action.
    
    The observation contains the extracted rubrics from a PDF paper, including both
    static requirements (code structure, functions) and dynamic requirements (experimental results).
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        success (bool): Whether the rubric extraction was successful
        static_rubrics (list): List of static rubric requirements (code structure, functions, etc.)
        dynamic_rubrics (list): List of dynamic rubric requirements (experimental results, tables, etc.)
        rubric_summary (str): Summary of all extracted rubrics
        paper_analysis (str): Analysis of the paper content
        execution_time (float): Time taken for the rubric extraction
        error_message (str): Error message if extraction failed
        observation (str): The observation type, namely ObservationType.PAPER_RUBRIC
    """
    
    content: str = ""
    success: bool = False
    static_rubrics: List[str] = None
    dynamic_rubrics: List[str] = None
    rubric_summary: str = ""
    paper_analysis: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    saved_file_path: str = ""
    observation: str = ObservationType.PAPER_RUBRIC

    def __post_init__(self):
        if self.static_rubrics is None:
            self.static_rubrics = []
        if self.dynamic_rubrics is None:
            self.dynamic_rubrics = []

    @property
    def message(self) -> str:
        """Get a human-readable message describing the rubric extraction operation."""
        if self.success:
            return f"Successfully extracted {len(self.static_rubrics)} static and {len(self.dynamic_rubrics)} dynamic rubrics in {self.execution_time:.2f} seconds"
        else:
            return f"Failed to extract rubrics: {self.error_message}"

    def __str__(self) -> str:
        """Get a string representation of the rubric extraction observation."""
        ret = "**PaperRubricObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"EXECUTION_TIME: {self.execution_time:.2f} seconds\n"
        ret += f"STATIC_RUBRICS: {len(self.static_rubrics)}\n"
        ret += f"DYNAMIC_RUBRICS: {len(self.dynamic_rubrics)}\n"
        if self.saved_file_path:
            ret += f"SAVED_FILE_PATH: {self.saved_file_path}\n"
        
        if self.paper_analysis:
            ret += "\n📄 PAPER ANALYSIS:\n"
            ret += "=" * 60 + "\n"
            ret += self.paper_analysis + "\n"
            ret += "=" * 60 + "\n"
        
        if self.rubric_summary:
            ret += "\n📋 RUBRIC SUMMARY:\n"
            ret += "=" * 60 + "\n"
            ret += self.rubric_summary + "\n"
            ret += "=" * 60 + "\n"
        
        if self.static_rubrics:
            ret += "\n🔧 STATIC RUBRICS (Code Requirements):\n"
            for i, rubric in enumerate(self.static_rubrics):
                ret += f"\n--- Static Rubric {i+1} ---\n"
                ret += rubric + "\n"
        
        if self.dynamic_rubrics:
            ret += "\n📊 DYNAMIC RUBRICS (Experimental Results):\n"
            for i, rubric in enumerate(self.dynamic_rubrics):
                ret += f"\n--- Dynamic Rubric {i+1} ---\n"
                ret += rubric + "\n"
        
        if self.content:
            ret += "\n📝 DETAILED CONTENT:\n"
            ret += self.content + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret


@dataclass
class GoTEditObservation(Observation):
    """Observation for GoT image edit/generation."""

    content: str = ""
    got_text: str = ""
    image_paths: List[str] = None
    success: bool = False
    error_message: str = ""
    observation: str = "GOT_EDIT"

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []


@dataclass
class AnyDoorEditObservation(Observation):
    """Observation for AnyDoor edit operation."""

    content: str = ""
    output_path: str = ""
    success: bool = False
    error_message: str = ""
    observation: str = "ANYDOOR_EDIT"

@dataclass
class QwenAPIObservation(Observation):
    """Observation from Qwen2.5-VL API operation."""

    content: str = ""
    response: str = ""
    success: bool = False
    error_message: str = ""
    observation: str = "QWEN_API"

    @property
    def message(self) -> str:
        return "GoT edit completed" if self.success else f"GoT edit failed: {self.error_message}"
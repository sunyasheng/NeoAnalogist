import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import os
from pathlib import Path

# Aider imports
try:
    from aider.coders import Coder
    from aider.models import Model
    from aider.io import InputOutput
except ImportError:
    Coder = None
    Model = None
    InputOutput = None

from core.events.observation.repo import RepoJudgeObservation
from core.llm.interface import LLMInterface

logger = logging.getLogger(__name__)

@dataclass
class RepoJudgeResult:
    rubric: str
    result: str

class RepoJudgeTask:
    """Task for judging repository code based on rubric file.
    
    This task takes a repository path and a rubric file path,
    then uses aider to analyze the repo for each rubric question from the file,
    and finally uses LLM to provide an overall summary.
    """
    
    def __init__(self, repo_path: str, rubric_file_path: str):
        self.repo_path = repo_path
        self.rubric_file_path = rubric_file_path
        self.start_time = time.time()
        
        # Parse rubric file to get rubric_list
        logger.info(f"Parsing rubric file: {rubric_file_path}")
        self.rubric_list = self._parse_rubric_file(rubric_file_path)


    def _parse_rubric_file(self, rubric_file_path: str) -> List[str]:
        """Parse rubric file to extract evaluation criteria."""
        try:
            rubric_path = Path(rubric_file_path)
            if not rubric_path.exists():
                logger.error(f"Rubric file not found: {rubric_file_path}")
                return []
            
            with open(rubric_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the rubric content
            logger.info(f"=== RUBRIC FILE CONTENT ===")
            logger.info(content)
            logger.info(f"=== END RUBRIC FILE CONTENT ===")
            
            rubric_items = self._extract_rubric_items(content)
            logger.info(f"=== EXTRACTED RUBRIC ITEMS ===")
            for i, item in enumerate(rubric_items):
                logger.info(f"Rubric {i+1}: {item}")
            logger.info(f"=== END EXTRACTED RUBRIC ITEMS ===")
            logger.info(f"Extracted {len(rubric_items)} rubric items from {rubric_file_path}")
            
            return rubric_items
            
        except Exception as e:
            logger.error(f"Error parsing rubric file {rubric_file_path}: {str(e)}")
            return []
    
    def _extract_rubric_items(self, content: str) -> List[str]:
        """Extract rubric items from the content using LLM - only code-verifiable requirements."""
        try:
            # Create LLM interface for intelligent parsing
            llm = LLMInterface(self._get_common_llm_config())
            
            # Create prompt for LLM to extract implementation requirements
            prompt = f"""
You are an expert at analyzing research paper rubrics and extracting specific, concrete evaluation criteria.

Please analyze the following rubric content and extract implementation requirements that can be verified by examining the code structure.

RUBRIC CONTENT:
{content}

INSTRUCTIONS:
1. Extract implementation requirements that can be checked by examining the code structure
2. Focus on concrete technical details that can be found in the code:
   - Specific algorithm implementations (functions, classes, methods)
   - Parameter configurations and hyperparameters
   - Model architectures and components
   - Data preprocessing pipelines
   - Specific formulas or mathematical implementations
   - Loss function implementations
   - Optimizer configurations
   - Dataset loading and processing
   - Evaluation metrics implementation
   - Model initialization and setup

3. For requirements that mention evaluation or benchmarks, extract the implementation aspects:
   - If it mentions "evaluate using X metric", extract "Implement X metric calculation"
   - If it mentions "benchmark on X dataset", extract "Include X dataset loading and processing"
   - If it mentions "use X model", extract "Implement X model architecture or loading"

4. Make requirements as specific as possible - avoid vague statements
5. Skip any error messages or placeholder text
6. Only extract requirements from "Code Development Requirements" and "Code Execution Requirements" sections
7. Skip "Result Analysis Requirements" as these are about experimental results, not code implementation

OUTPUT FORMAT:
Return a JSON array of strings, where each string is a specific, implementation-focused evaluation criterion.
Example:
[
    "Implement ResNet-50 model architecture with batch normalization",
    "Use Adam optimizer with learning rate 0.001 and weight decay 0.01",
    "Include ImageNet dataset loading and preprocessing",
    "Implement cross-entropy loss function with label smoothing",
    "Use data augmentation with random crop and horizontal flip",
    "Implement accuracy and precision metric calculations",
    "Include validation split and early stopping mechanism"
]

IMPORTANT: 
- Only return the JSON array, no other text
- Focus on implementation aspects that can be found in the code
- Convert evaluation requirements into implementation requirements
- Make each requirement as specific and concrete as possible
- Only include code development and code execution requirements
"""
            
            # Generate rubric items using LLM
            messages = [{"role": "user", "content": prompt}]
            response = llm.chat(messages=messages, model="o3-mini")
            
            # Extract content from response
            response_text = self._extract_llm_response(response)
            logger.info(f"=== LLM RESPONSE ===")
            logger.info(response_text)
            logger.info(f"=== END LLM RESPONSE ===")
            
            # Parse JSON response
            import json
            try:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    logger.info(f"=== EXTRACTED JSON ===")
                    logger.info(json_str)
                    logger.info(f"=== END EXTRACTED JSON ===")
                    rubric_items = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the entire response
                    logger.info("No JSON array found, trying to parse entire response")
                    rubric_items = json.loads(response_text)
                
                # Ensure we have a list of strings
                if isinstance(rubric_items, list):
                    final_items = [str(item) for item in rubric_items if item]
                    logger.info(f"=== FINAL RUBRIC ITEMS ===")
                    for i, item in enumerate(final_items):
                        logger.info(f"Final Rubric {i+1}: {item}")
                    logger.info(f"=== END FINAL RUBRIC ITEMS ===")
                    return final_items
                else:
                    logger.warning("LLM response was not a list, returning empty list")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                logger.warning(f"LLM response: {response_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error using LLM to extract rubric items: {str(e)}")
            return []

    def _get_common_llm_config(self):
        """Get common LLM configuration."""
        return {
            "llm": {
                "temperature": 0.1,
                "providers": {
                    "litellm": {
                        "use": True,
                        "model": "o3-mini",
                        "api_key": os.getenv("OPENAI_API_KEY", "")
                    }
                }
            }
        }

    def _scan_repository_files(self):
        """Scan repository for relevant files."""
        try:
            repo_path = Path(self.repo_path).resolve()
            
            # Define file extensions to include
            include_exts = [".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".MD"]
            
            files = []
            
            # Define directories and patterns to exclude
            exclude_dirs = {"venv", ".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
            exclude_patterns = {
                "outputs", "logs", "cache", "tmp", "temp", ".cache",
                ".hydra", "lightning_logs", "wandb", "tensorboard", 
                "checkpoints", "weights", "results", "experiments", "runs", "artifacts"
            }
            
            for ext in include_exts:
                for path in repo_path.rglob(f"*{ext}"):
                    # Skip if any part of the path contains excluded directories
                    if any(skip in path.parts for skip in exclude_dirs):
                        continue
                    
                    # Skip if any part of the path contains excluded patterns
                    if any(pattern in str(path) for pattern in exclude_patterns):
                        continue
                    
                    files.append(str(path.relative_to(repo_path)))
            
            return files
            
        except Exception as e:
            logger.error(f"Error scanning repository files: {str(e)}")
            return []

    def _check_prerequisites(self):
        """Check if prerequisites are met."""
        if not self.rubric_list:
            return False, "No rubric items available"
        
        if Coder is None:
            return False, "Aider not installed"
        
        return True, None

    def _create_aider_coder(self, files: List[str]):
        """Create aider coder instance."""
        # model = Model("gpt-4o")
        model = Model("sonnet") # documents say the claude-3-7-sonnet is great for code editing
        io = InputOutput(yes=True)
        return Coder.create(main_model=model, io=io, fnames=files)

    def _create_static_prompt(self, rubric: str) -> str:
        """Create static analysis prompt."""
        return f"""
Please perform a STATIC ANALYSIS of this codebase to evaluate the following implementation requirement:

{rubric}

STATIC ANALYSIS FOCUS:
1. Check if the required code implementation exists in the codebase
2. Verify specific algorithm implementations, functions, and classes
3. Check parameter configurations and hyperparameters
4. Look for specific model architectures or components
5. Verify data preprocessing and pipeline implementations
6. Check for proper imports and dependencies
7. Identify any missing implementations or incomplete code
8. For "Code has been implemented" requirements: Look for the actual implementation
9. For "Code has been executed" requirements: Look for execution scripts, main functions, or run configurations

Provide a detailed static analysis with specific examples from the code, including:
- Function/class names and their implementations
- Parameter values and configurations found in code
- Algorithm implementations and their correctness
- Missing or incomplete implementations
- Code structure and organization
- Execution scripts or entry points if applicable

IMPORTANT: This is a READ-ONLY static analysis. Do NOT modify any files. Only analyze the code structure and implementation.
"""

    async def run(self) -> RepoJudgeObservation:
        """Run the repository judge task."""
        logger.info(f"Starting repo judge with {len(self.rubric_list)} rubrics")
        logger.info(f"Rubric list: {self.rubric_list}")
        
        return await self._run_static_analysis(self.rubric_list)

    async def _run_static_analysis(self, rubric_list: List[str]) -> RepoJudgeObservation:
        """Run static analysis using coder.run."""
        try:
            logger.info(f"Starting static judge for {self.repo_path} with {len(rubric_list)} rubrics")
            
            # Check prerequisites
            is_valid, error_msg = self._check_prerequisites()
            if not is_valid:
                return RepoJudgeObservation(
                    content=f"No rubric items found to evaluate." if "No rubric" in error_msg else error_msg,
                    success=False,
                    rubric_results=[],
                    summary="",
                    execution_time=time.time() - self.start_time,
                    error_message=error_msg
                )
            
            # Analyze each static rubric using coder.run
            rubric_results = []
            for i, rubric in enumerate(rubric_list):
                logger.info(f"Static analysis for rubric {i+1}/{len(rubric_list)}")
                
                try:
                    # Create aider coder instance for static analysis
                    repo_path = Path(self.repo_path).resolve()
                    
                    # Ensure the directory is a git repo (for aider to build index)
                    if not (repo_path / ".git").exists():
                        import subprocess
                        subprocess.run(["git", "init"], cwd=str(repo_path), check=True)
                    
                    cwd = os.getcwd()
                    os.chdir(repo_path)
                    
                    try:
                        # Scan for relevant files
                        files = self._scan_repository_files()
                        
                        if not files:
                            result = f"Static analysis for rubric {i+1}: No relevant files found in repository."
                        else:
                            # Create aider coder
                            coder = self._create_aider_coder(files)
                            
                            # Create static analysis prompt
                            static_prompt = self._create_static_prompt(rubric)
                            
                            # Run analysis
                            analysis_result = coder.run(static_prompt)
                            result = f"Static analysis for rubric {i+1}: {str(analysis_result)}"
                                
                    finally:
                        os.chdir(cwd)
                    
                    rubric_results.append(RepoJudgeResult(
                        rubric=rubric,
                        result=result
                    ))
                    
                except Exception as e:
                    logger.error(f"Error in static analysis for rubric {i+1}: {str(e)}")
                    rubric_results.append(RepoJudgeResult(
                        rubric=rubric,
                        result=f"Static analysis error: {str(e)}"
                    ))
            
            # Generate overall summary using LLM
            summary = await self._generate_summary(rubric_results)
            
            execution_time = time.time() - self.start_time
            
            return RepoJudgeObservation(
                content=f"Static judge completed in {execution_time:.2f} seconds",
                success=True,
                rubric_results=rubric_results,
                summary=summary,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in static judge task: {str(e)}")
            return RepoJudgeObservation(
                content=f"Static judge failed: {str(e)}",
                success=False,
                rubric_results=[],
                summary="",
                execution_time=time.time() - self.start_time,
                error_message=str(e)
            )

    async def _generate_summary(self, rubric_results: List[RepoJudgeResult]) -> str:
        """Generate overall summary using LLM."""
        try:
            llm = LLMInterface(self._get_common_llm_config())
            
            prompt = "Based on the following rubric analysis results, provide an overall summary of whether the repository satisfies the requirements:\n\n"
            
            for i, result in enumerate(rubric_results):
                prompt += f"Rubric {i+1}: {result.rubric}\n"
                prompt += f"Analysis: {result.result}\n\n"
            
            prompt += "Please provide a comprehensive summary of whether the repository meets the overall requirements."
            
            messages = [{"role": "user", "content": prompt}]
            response = llm.chat(messages=messages, model="o3-mini")
            
            return self._extract_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _extract_llm_response(self, response: Dict[str, Any]) -> str:
        """Extract content from LLM response."""
        if "text" in response:
            return response["text"].strip()
        elif "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"].strip()
        else:
            return str(response).strip() 
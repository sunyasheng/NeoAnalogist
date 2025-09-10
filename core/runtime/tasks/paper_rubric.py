import os
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.events.observation.repo import PaperRubricObservation
from core.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class StaticRubric:
    """Static rubric requirement for code implementation."""
    category: str
    requirement: str
    description: str
    code_examples: Optional[str] = None

@dataclass
class DynamicRubric:
    """Dynamic rubric requirement for experimental results."""
    category: str
    requirement: str
    expected_result: str
    description: str
    table_data: Optional[Dict[str, Any]] = None

class PaperRubricTask:
    """Task for extracting rubrics from markdown papers.
    
    This task analyzes a markdown paper and extracts both static and dynamic rubric requirements
    that need to be implemented in the code. It uses the same planning approach as repo_create.py
    to first create a plan, then generates rubrics based on that plan.
    """
    
    def __init__(self, paper_path: str = "", 
                 include_static: bool = True, include_dynamic: bool = True, 
                 rubric_categories: Optional[List[str]] = None, 
                 save_to_file: bool = True, output_dir: str = None):
        self.paper_path = paper_path
        self.include_static = include_static
        self.include_dynamic = include_dynamic
        self.rubric_categories = rubric_categories or ["experiments", "evaluation", "results", "data", "models", "metrics"]
        self.save_to_file = save_to_file
        
        # Use the provided output_dir (required parameter)
        self.output_dir = output_dir if output_dir is not None else "workspace/rubrics"
        
        # Validate output directory path
        if self.output_dir:
            try:
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create output directory {self.output_dir}: {e}")
                self.output_dir = "workspace/rubrics"  # Fallback to default
            
        self.start_time = time.time()
        
    def _load_paper_content(self) -> str:
        """Load paper content from the specified markdown path."""
        try:
            with open(self.paper_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading paper content from {self.paper_path}: {e}")
            raise
        
    def _create_plan_message(self, paper_content: str) -> List[Dict[str, str]]:
        """Create the initial planning message - same as repo_create.py."""
        return [
            {'role': "system", "content": """You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 
You will receive a research paper in markdown format. 
Your task is to create a detailed and efficient plan to reproduce the experiments and methodologies described in the paper.
This plan should align precisely with the paper's methodology, experimental setup, and evaluation metrics. 

Instructions:

1. Align with the Paper: Your plan must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present the plan in a well-organized and easy-to-follow format, breaking it down into actionable steps.
3. Prioritize Efficiency: Optimize the plan for clarity and practical implementation while ensuring fidelity to the original experiments."""},
            {"role": "user",
             "content": f"""## Paper
{paper_content}

## Task
1. We want to reproduce the method described in the attached paper. 
2. The authors did not release any official code, so we have to plan our own implementation.
3. Before writing any Python code, please outline a comprehensive plan that covers:
   - Key details from the paper's **Methodology**.
   - Important aspects of **Experiments**, including dataset requirements, experimental settings, hyperparameters, or evaluation metrics.
4. The plan should be as **detailed and informative** as possible to help us write the final code later.

## Requirements
- You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.
- If something is unclear from the paper, mention it explicitly.

## Instruction
The response should give us a strong roadmap, making it easier to write the code later."""}
        ]

    def _create_rubric_from_plan_message(self, plan_content: str, paper_content: str) -> List[Dict[str, str]]:
        """Create message to generate rubrics from the plan and paper content."""
        return [
            {"role": "system", "content": """You are an expert code reviewer and rubric generator specializing in scientific paper reproduction. 
Your task is to analyze a detailed implementation plan and the original paper content to extract specific, actionable rubric requirements that can be used to verify the correctness of the implementation.

You will generate rubrics in the following categories:
1. CODE DEVELOPMENT: Specific code implementation requirements that can be checked in the source code
2. CODE EXECUTION: Code execution and runtime requirements that can be verified through testing
3. RESULT ANALYSIS: Experimental results and performance requirements that can be verified through analysis

Focus on extracting concrete, measurable requirements that can be objectively verified."""},
            {"role": "user", "content": f"""## Original Paper Content
{paper_content}

## Implementation Plan
{plan_content}

## Task
Based on the above implementation plan AND the original paper content, generate comprehensive rubrics for verifying the correctness of the paper reproduction.

## Requirements

### CODE DEVELOPMENT (Static Implementation Requirements)
Extract specific, actionable code requirements that can be verified in the source code:
- Specific model names, architectures, and configurations
- Specific dataset names and preprocessing requirements  
- Specific hyperparameters and training configurations
- Specific evaluation metrics and procedures
- Specific implementation details and algorithms
- Specific file structures and dependencies
- Specific mathematical formulas and equations to implement

### CODE EXECUTION (Dynamic Runtime Requirements)
Extract specific code execution and runtime requirements:
- Code execution steps and procedures
- Runtime verification requirements
- Data loading and processing verification
- Model training execution verification
- Evaluation execution verification

### RESULT ANALYSIS (Dynamic Results Requirements)
Extract specific experimental results and performance requirements from the paper:
- Expected accuracy, precision, recall, F1 scores with exact values from the paper
- Expected performance on specific benchmarks with exact metrics from the paper
- Expected comparison results with baselines with exact performance differences from the paper
- Expected ablation study outcomes with exact findings from the paper
- Expected statistical significance levels with exact p-values from the paper
- Expected error analysis findings with exact patterns from the paper

## Output Format
Provide your response in the following JSON format:

```json
{{
    "code_development": [
        "Code has been implemented such that [specific implementation detail]",
        "Code has been implemented for [specific functionality]",
        "Code has been implemented such that [specific model/algorithm] can be [specific action]"
    ],
    "code_execution": [
        "Code has been executed such that [specific execution step]",
        "[Specific process] has been run for [specific purpose]",
        "[Specific experiment] has been executed with [specific parameters]"
    ],
    "result_analysis": [
        "The [specific metric] measured shows that [specific expected result with exact values from paper]",
        "The results show that [specific performance requirement with exact values from paper]",
        "The [specific analysis] demonstrates [specific finding with exact values from paper]"
    ]
}}
```

## Instructions
1. Extract requirements from BOTH the plan and the original paper content
2. For CODE DEVELOPMENT and CODE EXECUTION: Focus on requirements from the plan
3. For RESULT ANALYSIS: Extract specific experimental results, metrics, and values directly from the paper content
4. Make requirements specific and measurable with exact values when available
5. Use the exact format shown in the examples above
6. Focus on implementation details that can be verified in code
7. Include expected experimental outcomes that can be tested
8. Ensure all requirements are actionable and checkable
9. Follow the rubric.json format style with "Code has been implemented such that..." and "Code has been executed such that..." patterns
10. For result analysis, include exact numerical values, percentages, and metrics reported in the paper

Generate comprehensive rubrics that cover all aspects of the implementation plan and verify against the original paper's experimental results."""}
        ]

    async def run(self) -> PaperRubricObservation:
        """Run the paper rubric extraction task."""
        try:
            logger.info(f"Starting paper rubric extraction from markdown file: {self.paper_path}")
            
            # Step 1: Load paper content from markdown file
            content_start_time = time.time()
            logger.info("Loading paper content from markdown file...")
            paper_content = self._load_paper_content()
            content_time = time.time() - content_start_time
            logger.info(f"Content loading completed in {content_time:.2f} seconds")
            
            # Step 2: Create implementation plan using the same method as repo_create.py
            plan_start_time = time.time()
            logger.info("Creating implementation plan...")
            plan_content = await self._create_implementation_plan(paper_content)
            plan_time = time.time() - plan_start_time
            logger.info(f"Plan creation completed in {plan_time:.2f} seconds")
            
            # Step 3: Generate rubrics from the plan and paper content
            rubric_start_time = time.time()
            logger.info("Generating rubrics from plan and paper content...")
            static_rubrics, dynamic_rubrics = await self._generate_rubrics_from_plan(plan_content, paper_content)
            rubric_time = time.time() - rubric_start_time
            logger.info(f"Rubric generation completed in {rubric_time:.2f} seconds")
            
            # Step 4: Generate summary
            summary_start_time = time.time()
            rubric_summary = await self._generate_rubric_summary(static_rubrics, dynamic_rubrics)
            summary_time = time.time() - summary_start_time
            logger.info(f"Summary generation completed in {summary_time:.2f} seconds")
            
            # Step 5: Save rubrics to file if requested
            save_start_time = time.time()
            saved_file_path = None
            if self.save_to_file:
                logger.info(f"Attempting to save rubrics to directory: {self.output_dir}")
                saved_file_path = await self._save_rubrics_to_file(static_rubrics, dynamic_rubrics, rubric_summary)
                save_time = time.time() - save_start_time
                if saved_file_path:
                    logger.info(f"File saving completed in {save_time:.2f} seconds - saved to: {saved_file_path}")
                else:
                    logger.warning(f"File saving failed after {save_time:.2f} seconds")
            else:
                logger.info("File saving disabled by configuration")
            
            execution_time = time.time() - self.start_time
            logger.info(f"Total execution time: {execution_time:.2f} seconds")
            
            return PaperRubricObservation(
                content=f"Paper rubric extraction completed in {execution_time:.2f} seconds",
                success=True,
                static_rubrics=static_rubrics,
                dynamic_rubrics=dynamic_rubrics,
                rubric_summary=rubric_summary,
                paper_analysis="",
                execution_time=execution_time,
                saved_file_path=saved_file_path
            )
            
        except Exception as e:
            logger.error(f"Error in paper rubric task: {str(e)}")
            return PaperRubricObservation(
                content=f"Paper rubric extraction failed: {str(e)}",
                success=False,
                static_rubrics=[],
                dynamic_rubrics=[],
                rubric_summary="",
                paper_analysis="",
                execution_time=time.time() - self.start_time,
                error_message=str(e)
            )

    async def _create_implementation_plan(self, paper_content: str) -> str:
        """Create implementation plan using the same method as repo_create.py."""
        try:
            from core.llm.interface import LLMInterface
            
            # Create LLM interface
            config = {
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
            llm = LLMInterface(config)
            
            # Create plan message using the same method as repo_create.py
            messages = self._create_plan_message(paper_content)
            
            # Generate plan
            response = llm.chat(messages=messages, model="o3-mini")
            
            # Extract content from response
            if "text" in response:
                return response["text"].strip()
            elif "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"].strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error creating implementation plan: {str(e)}")
            raise

    async def _generate_rubrics_from_plan(self, plan_content: str, paper_content: str) -> tuple[List[str], List[str]]:
        """Generate rubrics from the implementation plan and paper content."""
        try:
            from core.llm.interface import LLMInterface
            
            # Create LLM interface
            config = {
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
            llm = LLMInterface(config)
            
            # Create rubric generation message
            messages = self._create_rubric_from_plan_message(plan_content, paper_content)
            
            # Generate rubrics
            response = llm.chat(messages=messages, model="o3-mini")
            
            # Extract content from response
            if "text" in response:
                content = response["text"].strip()
            elif "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"].strip()
            else:
                content = str(response).strip()
            
            # Parse JSON response
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    rubric_data = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the entire response
                    rubric_data = json.loads(content)
                
                # Extract rubrics from the new format
                static_rubrics = []
                dynamic_rubrics = []
                
                # Code Development -> Static Rubrics
                if "code_development" in rubric_data:
                    for rubric in rubric_data["code_development"]:
                        if isinstance(rubric, str) and rubric.strip():
                            static_rubrics.append(rubric.strip())
                
                # Code Execution -> Dynamic Rubrics
                if "code_execution" in rubric_data:
                    for rubric in rubric_data["code_execution"]:
                        if isinstance(rubric, str) and rubric.strip():
                            dynamic_rubrics.append(rubric.strip())
                
                # Result Analysis -> Dynamic Rubrics (separate category)
                if "result_analysis" in rubric_data:
                    for rubric in rubric_data["result_analysis"]:
                        if isinstance(rubric, str) and rubric.strip():
                            dynamic_rubrics.append(rubric.strip())
                
                return static_rubrics, dynamic_rubrics
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse rubric JSON: {e}")
                # Fallback: return empty lists
                return [], []
                
        except Exception as e:
            logger.error(f"Error generating rubrics from plan: {str(e)}")
            return [], []

    async def _generate_rubric_summary(self, static_rubrics: List[str], dynamic_rubrics: List[str]) -> str:
        """Generate a summary of all extracted rubrics."""
        try:
            from core.llm.interface import LLMInterface
            
            # Create LLM interface
            config = {
                "llm": {
                    "temperature": 0.7,
                    "providers": {
                        "litellm": {
                            "use": True,
                            "model": "o3-mini",
                            "api_key": os.getenv("OPENAI_API_KEY", "")
                        }
                    }
                }
            }
            llm = LLMInterface(config)
            
            # Build summary prompt with separated categories
            static_summary = "\n".join([f"- {r}" for r in static_rubrics])
            
            # Separate dynamic rubrics into code execution and result analysis
            code_execution_rubrics = []
            result_analysis_rubrics = []
            
            for rubric in dynamic_rubrics:
                if rubric.startswith("Code has been executed") or "has been run" in rubric or "has been executed" in rubric:
                    code_execution_rubrics.append(rubric)
                else:
                    result_analysis_rubrics.append(rubric)
            
            code_execution_summary = "\n".join([f"- {r}" for r in code_execution_rubrics])
            result_analysis_summary = "\n".join([f"- {r}" for r in result_analysis_rubrics])
            
            prompt = f"""
Based on the following extracted rubrics from a research paper, provide a comprehensive summary:

CODE DEVELOPMENT REQUIREMENTS:
{static_summary}

CODE EXECUTION REQUIREMENTS:
{code_execution_summary}

RESULT ANALYSIS REQUIREMENTS:
{result_analysis_summary}

Please provide a clear, structured summary that:
1. Groups requirements by category (Code Development, Code Execution, Result Analysis)
2. Highlights the most critical implementation needs
3. Summarizes expected experimental outcomes
4. Provides implementation priorities
5. Identifies any missing requirements

Format the summary in a clear, readable way with sections and bullet points.
"""
            
            # Generate summary using LLM
            messages = [{"role": "user", "content": prompt}]
            response = llm.chat(messages=messages, model="o3-mini")
            
            # Extract content from response
            if "text" in response:
                return response["text"].strip()
            elif "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"].strip()
            else:
                return str(response).strip()
            
        except Exception as e:
            logger.error(f"Error generating rubric summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    async def _save_rubrics_to_file(self, static_rubrics: List[str], dynamic_rubrics: List[str], rubric_summary: str) -> Optional[str]:
        """Save extracted rubrics to a text file in a format suitable for repo-judge."""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename based on paper path
            paper_name = Path(self.paper_path).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{paper_name}_rubric_{timestamp}.txt"
            file_path = output_path / filename
            
            # Prepare content for the file - simple list format for repo-judge
            content_lines = []
            
            # Add header comment
            content_lines.append(f"# Markdown Paper Rubric Requirements for {paper_name}")
            content_lines.append(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append(f"# Paper: {self.paper_path}")
            content_lines.append("")
            
            # Add static rubrics (code development requirements)
            if static_rubrics:
                content_lines.append("# Code Development Requirements:")
                for rubric in static_rubrics:
                    if rubric.strip():
                        content_lines.append(rubric.strip())
                content_lines.append("")
            
            # Add dynamic rubrics (code execution and result analysis)
            if dynamic_rubrics:
                # Separate code execution and result analysis
                code_execution_rubrics = []
                result_analysis_rubrics = []
                
                for rubric in dynamic_rubrics:
                    if rubric.strip():
                        if rubric.startswith("Code has been executed") or "has been run" in rubric or "has been executed" in rubric:
                            code_execution_rubrics.append(rubric.strip())
                        else:
                            result_analysis_rubrics.append(rubric.strip())
                
                # Add code execution rubrics
                if code_execution_rubrics:
                    content_lines.append("# Code Execution Requirements:")
                    for rubric in code_execution_rubrics:
                        content_lines.append(rubric.strip())
                    content_lines.append("")
                
                # Add result analysis rubrics
                if result_analysis_rubrics:
                    content_lines.append("# Result Analysis Requirements:")
                    for rubric in result_analysis_rubrics:
                        content_lines.append(rubric.strip())
                    content_lines.append("")
            
            # Add summary if available
            if rubric_summary and "Error generating summary" not in rubric_summary:
                content_lines.append("# Summary:")
                content_lines.append(rubric_summary.strip())
                content_lines.append("")
            
            # Write to file with proper error handling
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(content_lines))
                logger.info(f"Rubrics saved to: {file_path}")
                return str(file_path)
            except IOError as io_error:
                logger.error(f"IO Error saving rubrics to file: {io_error}")
                return None
            except Exception as write_error:
                logger.error(f"Unexpected error writing to file: {write_error}")
                return None
            
        except Exception as e:
            logger.error(f"Error saving rubrics to file: {str(e)}")
            return None 
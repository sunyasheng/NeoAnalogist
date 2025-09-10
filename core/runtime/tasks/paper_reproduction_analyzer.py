"""Tool for analyzing paper content and summarizing reproduction requirements."""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from ii_researcher.reasoning.agent import ReasoningAgent
from ii_researcher.reasoning.builders.report import ReportType
import asyncio
from ii_researcher.reasoning.config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class PaperReproductionAnalyzerInput:
    """Input data structure for paper reproduction analyzer task."""
    paper_content: str
    paper_path: str = ""
    analysis_level: str = "detailed"  # "basic", "detailed", "comprehensive"


@dataclass
class PaperReproductionAnalyzerOutput:
    """Output data structure for paper reproduction analyzer task."""
    analysis_result: str
    success: bool
    error_message: Optional[str] = None
    analysis_level: str = "detailed"


def on_token(token: str):
    """Callback for processing streamed tokens."""
    print(token, end="", flush=True)


def get_event_loop():
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class TaskPaperReproductionAnalyzer:
    """
    A task that analyzes research paper content and extracts implementation requirements for reproduction.
    
    This task provides intelligent analysis of research papers to identify what needs to be implemented
    to reproduce the paper's methodology and results.
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1, env_file: Optional[str] = None):
        """
        Initialize the paper reproduction analyzer task.
        
        Args:
            model: The OpenAI model to use for paper analysis
            temperature: The temperature for the model
            env_file: Path to .env file (optional, will try to load from default locations)
        """
        # Load environment variables
        self._load_environment_variables(env_file)
        
        self.model = model
        self.temperature = temperature
        
        # Initialize OpenAI client with API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def _load_environment_variables(self, env_file: Optional[str] = None):
        """
        Load environment variables from .env file.
        
        Args:
            env_file: Specific path to .env file, or None to use default search
        """
        if env_file:
            # Load specific .env file
            if os.path.exists(env_file):
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from: {env_file}")
            else:
                logger.warning(f"Specified .env file not found: {env_file}")
        else:
            # Try to load from common locations
            env_locations = [
                ".env",                    # Current directory
                "../.env",                 # Parent directory
                "../../.env",              # Grandparent directory
                os.path.expanduser("~/.env")  # Home directory
            ]
            
            for env_path in env_locations:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from: {env_path}")
                    break
            else:
                # Try to load from current directory without specifying path
                load_dotenv()
                logger.info("Attempted to load .env from current directory")
    
    def _read_paper_file(self, paper_path: str) -> str:
        """Read paper content from file."""
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Unable to read paper file {paper_path}: {e}")

    def _build_analysis_prompt(self, paper_content: str, analysis_level: str) -> str:
        """Build the analysis prompt based on the paper content and analysis level."""
        
        if analysis_level == "basic":
            prompt = f"""You are an expert research paper analyst specializing in implementation requirements extraction. Your task is to analyze the following research paper and provide a high-level summary of what needs to be implemented to reproduce the paper.

## Paper Content:
{paper_content}

## Task:
Provide a concise, high-level summary of the main implementation requirements for reproducing this paper. Focus on the most critical components.

## What to Extract:
1. **Main Problem/Objective**: What is the paper trying to solve?
2. **Key Datasets**: What datasets are used (names, sources, sizes)?
3. **Core Algorithms/Methods**: What are the main algorithms or methods proposed?
4. **Evaluation Metrics**: How is the performance measured?
5. **High-level Architecture**: What are the main components needed?

## Output Format:
Provide a structured summary with clear sections for each requirement category. Be concise but comprehensive.

Focus on the essential elements needed to understand what must be implemented."""
        
        elif analysis_level == "detailed":
            prompt = f"""You are an expert research paper analyst specializing in detailed implementation requirements extraction. Your task is to analyze the following research paper and provide a comprehensive breakdown of what needs to be implemented to reproduce the paper.

## Paper Content:
{paper_content}

## Task:
Provide a detailed analysis of implementation requirements for reproducing this paper. Include specific technical details, configurations, and implementation guidance.

## What to Extract:

### 1. DATA REQUIREMENTS
- Exact dataset names, sources, and sizes
- Data preprocessing steps and requirements
- Input/output data formats and structures
- Data loading and handling requirements

### 2. ALGORITHM/METHOD REQUIREMENTS
- Detailed algorithm descriptions and pseudocode
- Mathematical formulations and equations
- Model architectures and components
- Key parameters and hyperparameters
- Implementation details and considerations

### 3. EXPERIMENTAL SETUP
- Training configurations and hyperparameters
- Evaluation protocols and metrics
- Baseline methods and comparisons
- Experimental conditions and settings
- Hardware/software requirements

### 4. IMPLEMENTATION COMPONENTS
- Core modules and functions needed
- Dependencies and libraries required
- Configuration files and settings
- File structure and organization
- Integration requirements

### 5. EVALUATION AND VALIDATION
- Performance metrics and benchmarks
- Validation procedures and tests
- Expected results and comparisons
- Reproducibility requirements

## Output Format:
Provide a structured, detailed analysis with clear sections for each requirement category. Include specific technical details, code structure suggestions, and implementation guidance where appropriate.

Focus on providing actionable implementation guidance that a developer can follow to reproduce the paper's results."""
        
        else:  # comprehensive
            prompt = f"""You are an expert research paper analyst specializing in comprehensive reproduction planning. Your task is to analyze the following research paper and provide a complete reproduction plan with detailed implementation guidance.

## Paper Content:
{paper_content}

## Task:
Provide a comprehensive reproduction plan that includes all aspects needed to fully reproduce this paper, from data preparation to final evaluation.

## What to Extract:

### 1. PROJECT OVERVIEW
- Research problem and objectives
- Key contributions and innovations
- Overall methodology and approach
- Expected outcomes and significance

### 2. DATA REQUIREMENTS (Detailed)
- Complete dataset specifications (names, sources, sizes, formats)
- Data preprocessing pipeline and requirements
- Data validation and quality checks
- Data storage and management requirements
- Data augmentation or generation procedures

### 3. ALGORITHM/METHOD IMPLEMENTATION (Detailed)
- Complete algorithm descriptions with pseudocode
- Mathematical formulations and derivations
- Model architecture specifications
- Parameter initialization and optimization
- Implementation considerations and best practices
- Potential challenges and solutions

### 4. EXPERIMENTAL SETUP (Complete)
- Full experimental design and methodology
- Training procedures and protocols
- Hyperparameter tuning strategies
- Baseline implementations and comparisons
- Hardware and software requirements
- Computational complexity and resource needs

### 5. IMPLEMENTATION ARCHITECTURE
- Complete file structure and organization
- Module dependencies and interfaces
- Configuration management
- Logging and monitoring requirements
- Error handling and validation
- Testing and debugging procedures

### 6. EVALUATION FRAMEWORK
- Complete evaluation methodology
- Performance metrics and benchmarks
- Statistical analysis requirements
- Comparison with baselines and state-of-the-art
- Ablation studies and analysis
- Reproducibility validation

### 7. REPRODUCTION CHECKLIST
- Step-by-step implementation guide
- Critical checkpoints and validations
- Common pitfalls and solutions
- Quality assurance procedures
- Documentation requirements

## Output Format:
Provide a comprehensive, structured reproduction plan with detailed sections for each requirement category. Include specific implementation guidance, code structure suggestions, configuration examples, and practical considerations.

Focus on creating a complete roadmap that enables full reproduction of the paper's methodology and results."""

        return prompt

    async def analyze_paper_async(self, analysis_input: PaperReproductionAnalyzerInput) -> PaperReproductionAnalyzerOutput:
        """
        Analyze a research paper and extract implementation requirements for reproduction (async version using ReasoningAgent).
        
        Args:
            analysis_input: The input containing paper content and analysis parameters
            
        Returns:
            PaperReproductionAnalyzerOutput containing the analysis results
        """
        try:
            # Get paper content
            paper_content = analysis_input.paper_content
            paper_path = analysis_input.paper_path
            analysis_level = analysis_input.analysis_level
            
            # If paper_path is provided, read the file
            if paper_path and not paper_content:
                try:
                    paper_content = self._read_paper_file(paper_path)
                except Exception as e:
                    return PaperReproductionAnalyzerOutput(
                        analysis_result="",
                        success=False,
                        error_message=f"Error reading paper file: {e}",
                        analysis_level=analysis_level
                    )
            
            if not paper_content:
                return PaperReproductionAnalyzerOutput(
                    analysis_result="",
                    success=False,
                    error_message="No paper content provided",
                    analysis_level=analysis_level
                )
            
            logger.info(f"Analyzing paper for reproduction requirements (level: {analysis_level})")
            
            # Build the analysis prompt
            analysis_prompt = self._build_analysis_prompt(paper_content, analysis_level)
            
            # Use ReasoningAgent for analysis
            new_llm_config = LLMConfig(
                model='gpt-4o',
                temperature=0.2,
                top_p=0.95,
                presence_penalty=0.0,
                stop_sequence=['<end_code>'],
                api_key='empty',
                base_url='http://localhost:4000',
                report_model='gpt-4o'
            )
            agent = ReasoningAgent(
                question=analysis_prompt, 
                report_type=ReportType.BASIC,
                override_config={'llm': new_llm_config}
            )
            result = await agent.run(on_token=on_token, is_stream=True)

            if not result:
                return PaperReproductionAnalyzerOutput(
                    analysis_result="",
                    success=False,
                    error_message="Model returned empty answer",
                    analysis_level=analysis_level
                )
            
            return PaperReproductionAnalyzerOutput(
                analysis_result=result,
                success=True,
                analysis_level=analysis_level
            )
            
        except Exception as e:
            logger.error(f"Paper reproduction analysis failed: {e}")
            return PaperReproductionAnalyzerOutput(
                analysis_result="",
                success=False,
                error_message=f"Analysis failed: {str(e)}",
                analysis_level=analysis_input.analysis_level
            )

    def analyze_paper(self, analysis_input: PaperReproductionAnalyzerInput) -> PaperReproductionAnalyzerOutput:
        """
        Analyze a research paper and extract implementation requirements for reproduction (sync version).
        
        Args:
            analysis_input: The input containing paper content and analysis parameters
            
        Returns:
            PaperReproductionAnalyzerOutput containing the analysis results
        """
        try:
            # Get the event loop
            loop = get_event_loop()
            
            # Run the async version
            if loop.is_running():
                # If we're already in an event loop, we need to create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.analyze_paper_async(analysis_input))
                    return future.result()
            else:
                # We can run the async function directly
                return loop.run_until_complete(self.analyze_paper_async(analysis_input))
                
        except Exception as e:
            logger.error(f"Paper reproduction analysis failed: {e}")
            return PaperReproductionAnalyzerOutput(
                analysis_result="",
                success=False,
                error_message=f"Analysis failed: {str(e)}",
                analysis_level=analysis_input.analysis_level
            )


async def main():
    """Main function for testing the paper reproduction analyzer."""
    
    # Use the real paper path directly
    paper_path = "workspace/20250623_162853/debug/data/semantic-self-consistency/paper.md"
    
    # Create analyzer instance
    analyzer = TaskPaperReproductionAnalyzer()
    
    # Create input with paper_path (the analyzer will read the file internally)
    analysis_input = PaperReproductionAnalyzerInput(
        paper_content="",  # Empty content, will be read from paper_path
        paper_path=paper_path,
        analysis_level="comprehensive"
    )
    
    print("Testing Paper Reproduction Analyzer with real paper content...")
    print(f"📄 Paper: {paper_path}")
    print(f"📊 Analysis Level: {analysis_input.analysis_level}")
    print("=" * 60)
    
    try:
        # Run analysis
        result = await analyzer.analyze_paper_async(analysis_input)
        
        if result.success:
            print("✅ Analysis completed successfully!")
            print("\n📋 Analysis Result:")
            print("=" * 60)
            print(result.analysis_result)
        else:
            print("❌ Analysis failed:")
            print(result.error_message)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    import asyncio
    asyncio.run(main()) 
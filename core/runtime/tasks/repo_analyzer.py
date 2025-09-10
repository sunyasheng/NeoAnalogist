"""
Repo Analyzer Task for analyzing and querying codebase content.

This task provides functionality to analyze repositories and answer questions about
the codebase structure, functionality, dependencies, and implementation details.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from openai import OpenAI
import os
import glob
import subprocess
import ast
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class RepoAnalyzerInput:
    """Input data structure for repo analyzer task."""
    repo_path: str
    query: str  # User's question about the codebase
    include_code_snippets: bool = True  # Whether to include relevant code snippets in response


@dataclass
class RepoAnalyzerOutput:
    """Output data structure for repo analyzer task."""
    answer: str
    relevant_files: List[str]
    code_snippets: Dict[str, str]  # filename -> code snippet
    success: bool
    error_message: Optional[str] = None
    repo_map: Optional[str] = None
    call_graph: Optional[str] = None


class TaskRepoAnalyzer:
    """
    A task that handles analyzing and querying codebase content in repositories.
    
    This task provides intelligent analysis of repository structure, code functionality,
    dependencies, and can answer specific questions about the codebase.
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1, env_file: Optional[str] = None):
        """
        Initialize the repo analyzer task.
        
        Args:
            model: The OpenAI model to use for code analysis
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
    
    def get_environment_status(self) -> Dict[str, Any]:
        """
        Get the status of loaded environment variables (without revealing sensitive values).
        
        Returns:
            Dictionary with environment variable status
        """
        status = {
            "openai_api_key_loaded": bool(os.environ.get("OPENAI_API_KEY")),
            "available_env_vars": [],
            "missing_required_vars": []
        }
        
        # Check for common environment variables (without revealing values)
        common_vars = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", 
            "HUGGINGFACE_API_KEY", "COHERE_API_KEY", "REPLICATE_API_TOKEN"
        ]
        
        for var in common_vars:
            if os.environ.get(var):
                status["available_env_vars"].append(var)
        
        # Required variables
        required_vars = ["OPENAI_API_KEY"]
        for var in required_vars:
            if not os.environ.get(var):
                status["missing_required_vars"].append(var)
        
        return status
    
    def _build_repo_map(self, repo_path: str) -> str:
        """
        Build a repository structure map.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            String representation of the repository structure
        """
        repo_path = Path(repo_path)
        repo_map = []
        
        def add_tree_structure(path: Path, prefix: str = "", is_last: bool = True):
            """Recursively build tree structure."""
            if path.name.startswith('.') or path.name in ['venv', '__pycache__', 'node_modules', '.git']:
                return
                
            connector = "└── " if is_last else "├── "
            repo_map.append(f"{prefix}{connector}{path.name}")
            
            if path.is_dir():
                try:
                    children = [p for p in path.iterdir() if not p.name.startswith('.') and p.name not in ['venv', '__pycache__', 'node_modules', '.git']]
                    # Sort: directories first, then files
                    children.sort(key=lambda p: (p.is_file(), p.name.lower()))
                    
                    for i, child in enumerate(children):
                        is_last_child = i == len(children) - 1
                        extension = "    " if is_last else "│   "
                        add_tree_structure(child, prefix + extension, is_last_child)
                except PermissionError:
                    pass
        
        repo_map.append(f"📁 Repository Structure: {repo_path.name}")
        repo_map.append("")
        
        try:
            children = [p for p in repo_path.iterdir() if not p.name.startswith('.')]
            children.sort(key=lambda p: (p.is_file(), p.name.lower()))
            
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                add_tree_structure(child, "", is_last_child)
                
        except Exception as e:
            repo_map.append(f"Error reading repository: {e}")
        
        return "\n".join(repo_map)
    
    def _extract_python_structure(self, file_path: str) -> Dict[str, List[str]]:
        """
        Extract classes and functions from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with 'classes' and 'functions' lists
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            return {'classes': classes, 'functions': functions}
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            return {'classes': [], 'functions': []}
    
    def _build_call_graph_with_pyan3(self, repo_path: str) -> str:
        """
        Build call graph using pyan3 if available.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Call graph representation
        """
        try:
            # Check if pyan3 is available
            result = subprocess.run(['pyan3', '--version'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return self._build_simple_call_graph(repo_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._build_simple_call_graph(repo_path)
        
        try:
            # Find all Python files (exclude venv and other system directories)
            python_files = [f for f in Path(repo_path).rglob("*.py") if not any(exclude in str(f) for exclude in ['venv', '__pycache__', 'node_modules', '.git'])]
            if not python_files:
                return "No Python files found for call graph analysis."
            
            # Run pyan3 to generate call graph
            cmd = ['pyan3', '--uses', '--no-defines', '--colored'] + [str(f) for f in python_files[:10]]  # Limit to first 10 files
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=repo_path)
            
            if result.returncode == 0 and result.stdout:
                return f"📊 Call Graph (via pyan3):\n\n{result.stdout}"
            else:
                return self._build_simple_call_graph(repo_path)
                
        except Exception as e:
            logger.warning(f"pyan3 analysis failed: {e}")
            return self._build_simple_call_graph(repo_path)
    
    def _build_simple_call_graph(self, repo_path: str) -> str:
        """
        Build a simple call graph by analyzing imports and function definitions.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Simple call graph representation
        """
        repo_path = Path(repo_path)
        call_graph = []
        call_graph.append("\U0001F4CA Program Structure Analysis:")
        call_graph.append("")
        
        # Exclude venv, __pycache__, node_modules, .git
        python_files = [f for f in repo_path.rglob("*.py") if not any(exclude in str(f) for exclude in ['venv', '__pycache__', 'node_modules', '.git'])]
        if not python_files:
            return "No Python files found for analysis."
        
        for py_file in python_files:
            try:
                relative_path = py_file.relative_to(repo_path)
                call_graph.append(f"\U0001F4C4 {relative_path}")
                
                # Extract structure
                structure = self._extract_python_structure(str(py_file))
                
                if structure['classes']:
                    call_graph.append(f"  \U0001F3DB️  Classes: {', '.join(structure['classes'])}")
                
                if structure['functions']:
                    call_graph.append(f"  \U0001F527 Functions: {', '.join(structure['functions'])}")
                
                # Extract imports
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    imports = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                    
                    if imports:
                        call_graph.append(f"  \U0001F4E6 Imports: {', '.join(set(imports))}")
                
                except Exception:
                    pass
                
                call_graph.append("")
                
            except Exception as e:
                call_graph.append(f"  ❌ Error analyzing {py_file}: {e}")
                call_graph.append("")
        
        return "\n".join(call_graph)
    
    def _find_relevant_files(self, repo_path: str, query: str) -> List[str]:
        """
        Find relevant Python files based on the query.
        
        Args:
            repo_path: Path to the repository
            query: User's query to find relevant files
            
        Returns:
            List of relevant Python file paths
        """
        repo_path = Path(repo_path)
        # Exclude venv, __pycache__, node_modules, .git
        python_files = [f for f in repo_path.rglob("*.py") if not any(exclude in str(f) for exclude in ['venv', '__pycache__', 'node_modules', '.git'])]
        
        if not python_files:
            return []
        
        relevant_files = []
        query_lower = query.lower()
        
        # Score files based on query relevance
        scored_files = []
        for py_file in python_files:
            score = 0
            file_name = py_file.name.lower()
            
            # Check filename relevance
            for word in query_lower.split():
                if word in file_name:
                    score += 2
            
            # Check content relevance (basic keyword matching)
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for word in query_lower.split():
                        score += content.count(word)
            except Exception:
                pass
            
            if score > 0:
                scored_files.append((py_file, score))
        
        # Sort by score and return top files
        scored_files.sort(key=lambda x: x[1], reverse=True)
        return [str(f[0]) for f in scored_files[:5]]  # Return top 5 relevant files
    

    
    def _extract_code_content(self, file_path: str) -> str:
        """
        Extract code content from a file.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Content of the code file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Could not read code file {file_path}: {e}")
            return ""
    
    def _get_project_context_from_repo(self, repo_path: str) -> str:
        """
        Try to extract project context from repository files.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Project context extracted from README or other files
        """
        repo_path = Path(repo_path)
        
        # Look for README files
        readme_patterns = ["README.md", "README.txt", "readme.md", "readme.txt"]
        for readme_file in readme_patterns:
            readme_path = repo_path / readme_file
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Take first few paragraphs as project context
                        lines = content.split('\n')
                        description_lines = []
                        for line in lines[:20]:  # First 20 lines
                            if line.strip() and not line.startswith('#'):
                                description_lines.append(line.strip())
                            if len(description_lines) >= 5:  # Stop after 5 content lines
                                break
                        return ' '.join(description_lines)
                except Exception as e:
                    logger.warning(f"Could not read README file {readme_path}: {e}")
        
        return "Machine learning project"
        
    def _get_prompt_environment(self) -> Dict[str, str]:
        """Get the prompt environment description."""
        return {
            "Development Environment": (
                "You can use any standard Python packages and libraries that are commonly available. "
                "Choose the most appropriate tools and libraries for the specific task at hand. "
                "Consider using well-established packages for reliability and maintainability. "
                "If you need specific packages, mention them in comments so they can be installed."
            )
        }
    
    def _get_prompt_analysis_guideline(self) -> Dict[str, list]:
        """Get the analysis guidelines."""
        return {
            "Analysis guideline": [
                "Provide a comprehensive and accurate analysis of the codebase based on the user's query.",
                "Use the repository structure and call graph to understand the overall architecture.",
                "Include relevant code snippets to support your analysis when appropriate.",
                "Explain complex concepts in clear, understandable terms.",
                "Identify key components, patterns, and relationships in the code.",
                "Point out potential issues, best practices, or areas for improvement if relevant.",
                "Be specific and reference actual file names and functions when discussing the code.",
                "If the query cannot be fully answered, explain what information is missing or unclear.",
            ]
        }
    
    def _get_prompt_resp_fmt(self) -> Dict[str, str]:
        """Get the response format requirements."""
        return {
            "Response format": (
                "Provide a comprehensive answer to the user's query about the codebase. "
                "Structure your response clearly with appropriate sections if needed. "
                "Include relevant code snippets in markdown code blocks when they help illustrate your points. "
                "Use clear headings and bullet points to organize complex information. "
                "Always reference specific files and functions when discussing implementation details."
            )
        }
    
    def _build_query_prompt(self, user_query: str, project_context: str, relevant_code: Dict[str, str], repo_map: str = "", call_graph: str = "") -> Dict[str, Any]:
        """
        Build the query prompt based on extracted information and user query.
        
        Args:
            user_query: The user's question about the codebase
            project_context: The project context from README
            relevant_code: Dictionary of filename -> code content for relevant files
            repo_map: Repository structure map
            call_graph: Program call graph
            
        Returns:
            The structured prompt for codebase analysis
        """
        prompt = {
            "Introduction": (
                "You are an experienced software engineer and code architect with deep expertise in code analysis. "
                "The user has a question about a codebase and you need to provide a comprehensive, accurate answer. "
                "Use the repository structure, call graph analysis, and relevant code snippets to understand the codebase thoroughly. "
                "Provide detailed insights, explanations, and specific references to help the user understand the code."
            ),
            "User Query": user_query,
            "Project Context": project_context,
            "Instructions": {}
        }
        
        # Add repository analysis if available
        if repo_map:
            prompt["Repository Structure"] = f"```\n{repo_map}\n```"
        
        if call_graph:
            prompt["Program Analysis"] = f"```\n{call_graph}\n```"
        
        # Add relevant code snippets
        if relevant_code:
            code_section = []
            for filename, code_content in relevant_code.items():
                code_section.append(f"### {filename}")
                code_section.append(f"```python\n{code_content}\n```")
            prompt["Relevant Code"] = "\n\n".join(code_section)
        
        # Add instruction components
        prompt["Instructions"].update(self._get_prompt_resp_fmt())
        prompt["Instructions"].update({
            "Code analysis guideline": [
                "Analyze the codebase comprehensively to answer the user's specific question.",
                "Use the repository structure to understand the overall project organization.",
                "Reference the program analysis to understand component relationships and dependencies.",
                "Examine the relevant code snippets carefully to provide accurate technical details.",
                "Explain code functionality, design patterns, and architectural decisions when relevant.",
                "Identify key functions, classes, and modules that relate to the user's query.",
                "Point out interesting implementation details, potential issues, or best practices observed.",
                "If the query involves troubleshooting, provide specific guidance based on the code analysis.",
            ]
        })
        prompt["Instructions"].update(self._get_prompt_analysis_guideline())
        prompt["Instructions"].update(self._get_prompt_environment())
            
        return prompt
    
    def _format_prompt_for_api(self, prompt: Dict[str, Any]) -> str:
        """
        Format the structured prompt into a string for API call.
        
        Args:
            prompt: The structured prompt dictionary
            
        Returns:
            Formatted prompt string
        """
        formatted_parts = []
        
        for key, value in prompt.items():
            if key == "Instructions":
                formatted_parts.append(f"## {key}")
                for sub_key, sub_value in value.items():
                    formatted_parts.append(f"### {sub_key}")
                    if isinstance(sub_value, list):
                        for item in sub_value:
                            formatted_parts.append(f"- {item}")
                    else:
                        formatted_parts.append(str(sub_value))
            else:
                formatted_parts.append(f"## {key}")
                formatted_parts.append(str(value))
            formatted_parts.append("")  # Add blank line
        
        return "\n".join(formatted_parts)
    

    
    def query_repo(self, query_input: RepoAnalyzerInput, retries: int = 3) -> RepoAnalyzerOutput:
        """
        Analyze the repository and answer the user's query about the codebase.
        
        Args:
            query_input: The query input containing repository path and user question
            retries: Number of retries if extraction fails
            
        Returns:
            RepoAnalyzerOutput containing the analysis and answer
        """
        try:
            # Extract information from repository
            repo_path = query_input.repo_path
            if not os.path.exists(repo_path):
                return RepoAnalyzerOutput(
                    answer="",
                    relevant_files=[],
                    code_snippets={},
                    success=False,
                    error_message=f"Repository path does not exist: {repo_path}"
                )
            
            # Find relevant files based on query
            relevant_files = self._find_relevant_files(repo_path, query_input.query)
            if not relevant_files:
                return RepoAnalyzerOutput(
                    answer="No relevant Python files found for the query.",
                    relevant_files=[],
                    code_snippets={},
                    success=False,
                    error_message="No relevant files found in repository"
                )
            
            # Extract code content from relevant files
            relevant_code = {}
            code_snippets = {}
            for file_path in relevant_files:
                code_content = self._extract_code_content(file_path)
                if code_content:
                    relative_path = str(Path(file_path).relative_to(Path(repo_path)))
                    relevant_code[relative_path] = code_content
                    # Store snippet (first 50 lines for output)
                    lines = code_content.split('\n')
                    snippet = '\n'.join(lines[:50])
                    if len(lines) > 50:
                        snippet += f"\n... ({len(lines) - 50} more lines)"
                    code_snippets[relative_path] = snippet
            
            # Get project context
            project_context = self._get_project_context_from_repo(repo_path)
            
            logger.info(f"Analyzing {len(relevant_files)} relevant files")
            logger.info(f"User query: {query_input.query[:100]}...")
            logger.info(f"Project context: {project_context[:100]}...")
            
            # Build repository structure map
            logger.info("Building repository structure map...")
            repo_map = self._build_repo_map(repo_path)
            
            # Build call graph
            logger.info("Building program call graph...")
            call_graph = self._build_call_graph_with_pyan3(repo_path)
            
            # Build the prompt with enhanced analysis
            prompt = self._build_query_prompt(query_input.query, project_context, relevant_code, repo_map, call_graph)
            formatted_prompt = self._format_prompt_for_api(prompt)
            
            # Make API call with retries
            for attempt in range(retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": formatted_prompt},
                            {"role": "user", "content": "Please analyze the codebase and answer the query following the guidelines above."}
                        ],
                        temperature=self.temperature
                    )
                    
                    completion_text = response.choices[0].message.content
                    
                    if completion_text:
                        logger.info(f"Successfully generated codebase analysis on attempt {attempt + 1}")
                        return RepoAnalyzerOutput(
                            answer=completion_text,
                            relevant_files=[str(Path(f).relative_to(Path(repo_path))) for f in relevant_files],
                            code_snippets=code_snippets,
                            success=True,
                            repo_map=repo_map,
                            call_graph=call_graph
                        )
                    else:
                        logger.warning(f"Empty response on attempt {attempt + 1}")
                        if attempt == retries - 1:
                            return RepoAnalyzerOutput(
                                answer="",
                                relevant_files=[],
                                code_snippets={},
                                success=False,
                                error_message="Received empty response from AI model"
                            )
                
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {str(e)}")
                    if attempt == retries - 1:
                        return RepoAnalyzerOutput(
                            answer="",
                            relevant_files=[],
                            code_snippets={},
                            success=False,
                            error_message=f"API call failed: {str(e)}"
                        )
        
        except Exception as e:
            logger.error(f"Repo analyzer task failed with error: {str(e)}")
            return RepoAnalyzerOutput(
                answer="",
                relevant_files=[],
                code_snippets={},
                success=False,
                error_message=f"Repo analyzer task failed: {str(e)}"
            )

    def analyze_paper_vs_codebase(self, paper_path: str, repo_path: str) -> str:
        """
        Generic method to read any paper.md and analyze what's implemented vs missing in codebase.
        
        Args:
            paper_path: Path to paper.md file
            repo_path: Path to repository
            
        Returns:
            Analysis report as string
        """
        print("🚀 Starting paper vs codebase implementation analysis...")
        
        # 1. Read paper content
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            print(f"✅ Successfully read paper file: {paper_path}")
        except Exception as e:
            return f"❌ Unable to read paper file: {e}"
        
        # 2. Extract key functionalities from paper
        print("🔍 Extracting key functionalities from paper...")
        paper_functionalities = self._extract_key_functionalities_generic(paper_content)
        
        # 3. Analyze codebase structure
        print("📁 Analyzing codebase structure...")
        codebase_info = self._analyze_codebase_simple(repo_path)
        
        # 4. Comparative analysis
        print("⚖️ Performing implementation status comparison...")
        comparison_result = self._compare_paper_with_code_generic(paper_functionalities, codebase_info)
        
        # 5. Generate report
        report = self._generate_comparison_report_generic(paper_functionalities, codebase_info, comparison_result)
        
        print("✅ Analysis completed!")
        return report
    
    def _extract_key_functionalities_generic(self, paper_content: str) -> Dict[str, List[str]]:
        """Extract key functionalities using hierarchical analysis approach"""
        
        # Extract hierarchical functionality structure
        hierarchical_analysis = self._analyze_paper_hierarchical_functionalities(paper_content)
        
        # Convert hierarchical structure to flat categories for comparison
        return self._convert_hierarchical_to_flat(hierarchical_analysis)
    
    def _analyze_paper_hierarchical_functionalities(self, paper_content: str) -> Dict[str, Any]:
        """Analyze paper to extract CONCRETE, VERIFIABLE implementation requirements"""
        
        try:
            # Use template-based prompt
            concrete_prompt = self._get_prompt_template("paper_analysis", paper_content=paper_content)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a software engineer specializing in concrete requirements analysis for research implementation."},
                    {"role": "user", "content": concrete_prompt}
                ],
                temperature=0.1
            )
            
            return self._parse_json_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Hierarchical functionality analysis failed: {e}")
            return {}
    
    def _convert_hierarchical_to_flat(self, concrete_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Convert concrete requirements structure to flat categories for comparison"""
        
        flat_structure = {
            "data_requirements": [],
            "algorithm_requirements": [],
            "experimental_setup": [],
            "implementation_requirements": [],
            "expected_outputs": [],
            "datasets": [],
            "models": [],
            "algorithms": [],
            "evaluation_metrics": []
        }
        
        if not concrete_data:
            return flat_structure
        
        # Direct mapping from new concrete format
        for category in ["data_requirements", "algorithm_requirements", "experimental_setup", 
                        "implementation_requirements", "expected_outputs"]:
            if category in concrete_data:
                flat_structure[category] = concrete_data[category]
                
                # Also categorize into traditional categories for backward compatibility
                for item in concrete_data[category]:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        
                        # Extract specific dataset names, sizes, formats
                        if any(keyword in item_lower for keyword in ['dataset', 'samples', 'test', 'json', 'csv']):
                            flat_structure["datasets"].append(item)
                        
                        # Extract specific models mentioned
                        if any(keyword in item_lower for keyword in ['bert', 'gpt', 'llama', 'mistral', 'roberta', 'scibert', 'openai']):
                            flat_structure["models"].append(item)
                        
                        # Extract specific algorithms and methods
                        if any(keyword in item_lower for keyword in ['algorithm', 'method', 'weighting', 'similarity', 'clustering', 'forest', 'svm']):
                            flat_structure["algorithms"].append(item)
                        
                        # Extract specific metrics and evaluation methods
                        if any(keyword in item_lower for keyword in ['accuracy', 'metric', 'score', 'evaluation', 'improvement', 'performance']):
                            flat_structure["evaluation_metrics"].append(item)
        
        # Clean and deduplicate
        for key in flat_structure:
            seen = set()
            cleaned = []
            for item in flat_structure[key]:
                if item and item.strip() and item.strip() not in seen:
                    cleaned.append(item.strip())
                    seen.add(item.strip())
            flat_structure[key] = cleaned
        
        return flat_structure
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling various formats"""
        import json
        
        try:
            # Try direct JSON parsing first
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON content between code blocks
        lines = response_text.split('\n')
        json_lines = []
        in_json_block = False
        
        for line in lines:
            if line.strip().startswith('```') and ('json' in line.lower() or '{' in line):
                in_json_block = True
                continue
            elif line.strip() == '```' and in_json_block:
                break
            elif in_json_block:
                json_lines.append(line)
        
        if json_lines:
            try:
                json_text = '\n'.join(json_lines)
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object by looking for { and }
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                try:
                    json_text = response_text[start_idx:end_idx]
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
        
        logger.warning(f"Failed to parse JSON from response")
        return {}
    
    def _analyze_codebase_simple(self, repo_path: str) -> Dict[str, any]:
        """Enhanced analysis of codebase structure using repo map and call graph"""
        python_files = list(Path(repo_path).rglob("*.py"))
        
        info = {
            "file_count": len(python_files),
            "all_functions": [],
            "all_classes": [],
            "imported_modules": [],
            "code_content": "",
            "repo_map": "",
            "call_graph": "",
            "file_structure": {},
            "class_methods": {},
            "function_calls": []
        }
        
        # Build repo map for better structure understanding
        try:
            info["repo_map"] = self._build_repo_map(repo_path)
        except Exception as e:
            logger.warning(f"Failed to build repo map: {e}")
            info["repo_map"] = ""
        
        # Build call graph for understanding function relationships
        try:
            info["call_graph"] = self._build_call_graph_with_pyan3(repo_path)
            if not info["call_graph"]:  # Fallback to simple call graph
                info["call_graph"] = self._build_simple_call_graph(repo_path)
        except Exception as e:
            logger.warning(f"Failed to build call graph: {e}")
            info["call_graph"] = ""
        
        all_code = []
        
        # Enhanced file analysis
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_code.append(content.lower())
                
                # Extract structure information
                structure = self._extract_python_structure(str(py_file))
                info["all_functions"].extend(structure['functions'])
                info["all_classes"].extend(structure['classes'])
                
                # Store file-specific structure
                relative_path = str(py_file.relative_to(Path(repo_path)))
                info["file_structure"][relative_path] = {
                    "classes": structure['classes'],
                    "functions": structure['functions']
                }
                
                # Extract class methods relationships
                for class_name in structure['classes']:
                    if class_name not in info["class_methods"]:
                        info["class_methods"][class_name] = []
                    # Simple heuristic to find methods in class
                    lines = content.split('\n')
                    in_class = False
                    for line in lines:
                        if f"class {class_name}" in line:
                            in_class = True
                        elif line.startswith('class ') and in_class:
                            in_class = False
                        elif in_class and line.strip().startswith('def '):
                            method_name = line.strip().split('(')[0].replace('def ', '')
                            if method_name not in info["class_methods"][class_name]:
                                info["class_methods"][class_name].append(method_name)
                
                # Extract imports with more detail
                for line in content.split('\n')[:50]:  # Check more lines for imports
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        info["imported_modules"].append(line)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze file {py_file}: {e}")
                continue
        
        # Clean and deduplicate
        info["code_content"] = ' '.join(all_code)
        info["all_functions"] = list(set(info["all_functions"]))
        info["all_classes"] = list(set(info["all_classes"]))
        info["imported_modules"] = list(set(info["imported_modules"]))
        
        # Extract function call relationships from call graph
        if info["call_graph"]:
            try:
                # Parse call graph to extract function relationships
                call_lines = info["call_graph"].split('\n')
                for line in call_lines:
                    if '->' in line:
                        # Extract caller -> callee relationships
                        parts = line.split('->')
                        if len(parts) == 2:
                            caller = parts[0].strip().strip('"')
                            callee = parts[1].strip().strip('"').rstrip(';')
                            info["function_calls"].append(f"{caller} -> {callee}")
            except Exception as e:
                logger.warning(f"Failed to parse call graph: {e}")
        
        return info
    
    def _compare_paper_with_code_generic(self, paper_functionalities: Dict[str, List[str]], 
                                       codebase_info: Dict[str, any]) -> Dict[str, Dict[str, List[str]]]:
        """LLM-driven semantic comparison of paper functionalities with code implementation"""
        
        # Prepare codebase summary for LLM analysis
        codebase_summary = self._prepare_codebase_summary(codebase_info)
        
        # Use LLM to perform intelligent comparison
        try:
            comparison_prompt = self._get_prompt_template(
                "code_comparison",
                codebase_summary=codebase_summary,
                paper_functionalities=self._format_functionalities_for_llm(paper_functionalities)
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code analyst with deep understanding of research implementations. Analyze code semantically, not just by keywords."},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.1
            )
            
            llm_result = self._parse_json_response(response.choices[0].message.content)
            return self._convert_llm_result_to_standard_format(llm_result)
            
        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            # Fallback to a simple structural analysis
            return self._fallback_structural_analysis(paper_functionalities, codebase_info)
    
    def _prepare_codebase_summary(self, codebase_info: Dict[str, any]) -> str:
        """Prepare a comprehensive summary of the codebase for LLM analysis"""
        summary = []
        
        summary.append(f"## Codebase Overview:")
        summary.append(f"- Python files: {codebase_info['file_count']}")
        summary.append(f"- Total classes: {len(codebase_info['all_classes'])}")
        summary.append(f"- Total functions: {len(codebase_info['all_functions'])}")
        summary.append("")
        
        # Include repo map for structure understanding
        if codebase_info.get("repo_map"):
            summary.append("## Repository Structure:")
            repo_map_lines = codebase_info["repo_map"].split('\n')[:20]  # Limit to avoid token overflow
            for line in repo_map_lines:
                if line.strip():
                    summary.append(line)
            summary.append("")
        
        # Include file-specific structure
        if codebase_info.get("file_structure"):
            summary.append("## File Structure Details:")
            for file_path, structure in list(codebase_info["file_structure"].items())[:5]:  # Limit files
                if structure["classes"] or structure["functions"]:
                    summary.append(f"### {file_path}:")
                    if structure["classes"]:
                        summary.append(f"  Classes: {', '.join(structure['classes'][:5])}")
                    if structure["functions"]:
                        summary.append(f"  Functions: {', '.join(structure['functions'][:5])}")
            summary.append("")
        
        # Include class-method relationships
        if codebase_info.get("class_methods"):
            summary.append("## Class-Method Relationships:")
            for class_name, methods in list(codebase_info["class_methods"].items())[:8]:  # Limit classes
                if methods:
                    summary.append(f"- {class_name}: {', '.join(methods[:5])}")
            summary.append("")
        
        # Include function call relationships
        if codebase_info.get("function_calls"):
            summary.append("## Function Call Relationships:")
            for call_rel in codebase_info["function_calls"][:10]:  # Limit to avoid overflow
                summary.append(f"- {call_rel}")
            summary.append("")
        
        # Traditional class and function lists (condensed)
        if codebase_info["all_classes"]:
            summary.append("## All Classes:")
            summary.append(f"- {', '.join(codebase_info['all_classes'][:10])}")
            if len(codebase_info['all_classes']) > 10:
                summary.append(f"- ... and {len(codebase_info['all_classes']) - 10} more")
            summary.append("")
        
        if codebase_info["all_functions"]:
            summary.append("## All Functions:")
            summary.append(f"- {', '.join(codebase_info['all_functions'][:15])}")
            if len(codebase_info['all_functions']) > 15:
                summary.append(f"- ... and {len(codebase_info['all_functions']) - 15} more")
            summary.append("")
        
        # Key imports (enhanced filtering)
        if codebase_info["imported_modules"]:
            summary.append("## Key Imports:")
            # Enhanced filtering for more relevant imports
            relevant_imports = []
            for imp in codebase_info["imported_modules"][:20]:
                if any(keyword in imp.lower() for keyword in 
                      ['torch', 'sklearn', 'numpy', 'pandas', 'transformers', 'scipy', 
                       'matplotlib', 'tensorflow', 'keras', 'huggingface', 'openai',
                       'anthropic', 'langchain', 'sentence_transformers']):
                    relevant_imports.append(imp)
            
            for imp in relevant_imports[:10]:  # Limit to avoid overflow
                summary.append(f"- {imp}")
            summary.append("")
        
        # Include partial call graph if available
        if codebase_info.get("call_graph"):
            summary.append("## Call Graph (Sample):")
            call_graph_lines = codebase_info["call_graph"].split('\n')[:15]  # Limit lines
            for line in call_graph_lines:
                if line.strip() and ('->' in line or 'digraph' in line or '}' in line):
                    summary.append(f"  {line.strip()}")
            summary.append("")
        
        return "\n".join(summary)
    
    def _format_functionalities_for_llm(self, paper_functionalities: Dict[str, List[str]]) -> str:
        """Format paper functionalities for LLM analysis"""
        formatted = []
        
        for category, functionalities in paper_functionalities.items():
            if functionalities:  # Only include non-empty categories
                formatted.append(f"### {category.replace('_', ' ').title()}:")
                for func in functionalities:
                    formatted.append(f"- {func}")
                formatted.append("")
        
        return "\n".join(formatted)
    
    def _convert_llm_result_to_standard_format(self, llm_result: Dict) -> Dict[str, Dict[str, List[str]]]:
        """Convert LLM analysis result to standard format"""
        result = {}
        
        # Key mapping to normalize LLM response keys to our expected format
        key_mapping = {
            "core functionalities": "core_functionalities",
            "sub functionalities": "sub_functionalities", 
            "implementation details": "implementation_details",
            "datasets": "datasets",
            "models": "models",
            "algorithms": "algorithms",
            "evaluation metrics": "evaluation_metrics"
        }
        
        for category, analysis in llm_result.items():
            # Normalize the category key
            normalized_key = category.lower().strip()
            final_key = key_mapping.get(normalized_key, normalized_key.replace(' ', '_'))
            
            category_result = {
                "implemented": [],
                "possibly_implemented": [],
                "not_implemented": [],
                "evidence": {}
            }
            
            # Process implemented items
            if "implemented" in analysis:
                for item in analysis["implemented"]:
                    if isinstance(item, dict):
                        func_name = item.get("functionality", "")
                        evidence = item.get("evidence", "")
                        if func_name:
                            category_result["implemented"].append(func_name)
                            if evidence:
                                category_result["evidence"][func_name] = [evidence]
                    elif isinstance(item, str):
                        category_result["implemented"].append(item)
            
            # Process possibly implemented items
            if "possibly_implemented" in analysis:
                for item in analysis["possibly_implemented"]:
                    if isinstance(item, dict):
                        func_name = item.get("functionality", "")
                        evidence = item.get("evidence", "")
                        if func_name:
                            category_result["possibly_implemented"].append(func_name)
                            if evidence:
                                category_result["evidence"][func_name] = [evidence]
                    elif isinstance(item, str):
                        category_result["possibly_implemented"].append(item)
            
            # Process not implemented items
            if "not_implemented" in analysis:
                for item in analysis["not_implemented"]:
                    if isinstance(item, dict):
                        func_name = item.get("functionality", "")
                        reason = item.get("reason", "")
                        if func_name:
                            category_result["not_implemented"].append(func_name)
                            if reason:
                                category_result["evidence"][func_name] = [f"Not found: {reason}"]
                    elif isinstance(item, str):
                        category_result["not_implemented"].append(item)
            
            result[final_key] = category_result
        
        return result
    
    def _fallback_structural_analysis(self, paper_functionalities: Dict[str, List[str]], 
                                    codebase_info: Dict[str, any]) -> Dict[str, Dict[str, List[str]]]:
        """Fallback structural analysis when LLM fails"""
        result = {}
        
        all_code_names = set()
        all_code_names.update([name.lower() for name in codebase_info["all_classes"]])
        all_code_names.update([name.lower() for name in codebase_info["all_functions"]])
        
        for category, functionalities in paper_functionalities.items():
            category_result = {
                "implemented": [],
                "possibly_implemented": [],
                "not_implemented": [],
                "evidence": {}
            }
            
            for func in functionalities:
                func_lower = func.lower().strip()
                
                # Simple structural check - only for very specific matches
                found_exact = False
                found_partial = False
                
                for code_name in all_code_names:
                    if func_lower == code_name:
                        found_exact = True
                        category_result["evidence"][func] = [f"Exact match: {code_name}"]
                        break
                    elif len(func_lower) > 4 and (func_lower in code_name or code_name in func_lower):
                        found_partial = True
                        category_result["evidence"][func] = [f"Partial match: {code_name}"]
                
                if found_exact:
                    category_result["implemented"].append(func)
                elif found_partial:
                    category_result["possibly_implemented"].append(func)
                else:
                    category_result["not_implemented"].append(func)
            
            result[category] = category_result
        
        return result
    
    def _generate_comparison_report_generic(self, paper_functionalities: Dict[str, List[str]], 
                                          codebase_info: Dict[str, any], 
                                          comparison_result: Dict[str, Dict[str, List[str]]]) -> str:
        """Generate hierarchical comparison analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("📋 Paper vs Codebase Implementation Comparison Report (Concrete Requirements Analysis)")
        report.append("=" * 80)
        report.append("")
        
        # Codebase overview
        report.append("📁 Codebase Overview:")
        report.append(f"  • Python files count: {codebase_info['file_count']}")
        report.append(f"  • Classes count: {len(codebase_info['all_classes'])}")
        report.append(f"  • Functions count: {len(codebase_info['all_functions'])}")
        report.append(f"  • Imported modules count: {len(codebase_info['imported_modules'])}")
        report.append("")
        
        # Concrete requirements analysis
        total_implemented = 0
        total_possible = 0
        total_missing = 0
        
        report.append("🔍 Concrete Requirements Implementation Analysis:")
        report.append("")
        
        # Define requirement categories for better presentation
        requirement_order = [
            ("data_requirements", "📊 DATA REQUIREMENTS"),
            ("algorithm_requirements", "⚙️ ALGORITHM REQUIREMENTS"), 
            ("experimental_setup", "🧪 EXPERIMENTAL SETUP"),
            ("implementation_requirements", "🔧 IMPLEMENTATION REQUIREMENTS"),
            ("expected_outputs", "📈 EXPECTED OUTPUTS"),
            ("datasets", "📂 Datasets (Legacy)"),
            ("models", "🤖 Models (Legacy)"),
            ("algorithms", "⚙️ Algorithms (Legacy)"),
            ("evaluation_metrics", "📊 Metrics (Legacy)")
        ]
        
        for category_key, category_title in requirement_order:
            if category_key not in comparison_result:
                continue
                
            result = comparison_result[category_key]
            if not any(result[key] for key in ["implemented", "possibly_implemented", "not_implemented"]):
                continue
            
            report.append(f"{category_title}:")
            
            if result["implemented"]:
                report.append(f"  ✅ Implemented ({len(result['implemented'])}):")
                for item in result["implemented"]:
                    report.append(f"    • {item}")
                    if item in result.get("evidence", {}):
                        for evidence in result["evidence"][item][:2]:
                            report.append(f"      → {evidence}")
                total_implemented += len(result["implemented"])
            
            if result["possibly_implemented"]:
                report.append(f"  🔶 Possibly Implemented ({len(result['possibly_implemented'])}):")
                for item in result["possibly_implemented"]:
                    report.append(f"    • {item}")
                    if item in result.get("evidence", {}):
                        for evidence in result["evidence"][item][:2]:
                            report.append(f"      → {evidence}")
                total_possible += len(result["possibly_implemented"])
            
            if result["not_implemented"]:
                report.append(f"  ❌ Not Implemented ({len(result['not_implemented'])}):")
                for item in result["not_implemented"]:
                    report.append(f"    • {item}")
                total_missing += len(result["not_implemented"])
            
            report.append("")
        
        # Statistical summary
        total_items = total_implemented + total_possible + total_missing
        if total_items > 0:
            report.append("📊 Implementation Status Statistics:")
            report.append(f"  • Total functionalities analyzed: {total_items}")
            report.append(f"  • ✅ Implemented: {total_implemented} ({total_implemented/total_items*100:.1f}%)")
            report.append(f"  • 🔶 Possibly implemented: {total_possible} ({total_possible/total_items*100:.1f}%)")
            report.append(f"  • ❌ Not implemented: {total_missing} ({total_missing/total_items*100:.1f}%)")
            report.append("")
        
        # Concrete requirements analysis insights
        report.append("🎯 Concrete Requirements Analysis Insights:")
        
        # Data requirements coverage
        data_result = comparison_result.get("data_requirements", {})
        data_implemented = len(data_result.get("implemented", []))
        data_possible = len(data_result.get("possibly_implemented", []))
        data_total = data_implemented + data_possible + len(data_result.get("not_implemented", []))
        
        if data_total > 0:
            data_coverage = (data_implemented + data_possible) / data_total * 100
            if data_coverage >= 70:
                report.append("  • ✅ Strong data requirements coverage - datasets and formats largely implemented")
            elif data_coverage >= 40:
                report.append("  • 🔶 Moderate data requirements coverage - some data handling implemented")
            else:
                report.append("  • ❌ Weak data requirements coverage - major data requirements missing")
        
        # Algorithm requirements analysis
        algo_result = comparison_result.get("algorithm_requirements", {})
        algo_implemented = len(algo_result.get("implemented", []))
        algo_total = algo_implemented + len(algo_result.get("possibly_implemented", [])) + len(algo_result.get("not_implemented", []))
        
        if algo_total > 0:
            algo_coverage = (algo_implemented + len(algo_result.get("possibly_implemented", []))) / algo_total * 100
            report.append(f"  • ⚙️ Algorithm requirements: {algo_implemented}/{algo_total} implemented ({algo_coverage:.1f}% coverage)")
        
        # Implementation requirements analysis
        impl_result = comparison_result.get("implementation_requirements", {})
        impl_implemented = len(impl_result.get("implemented", []))
        impl_total = impl_implemented + len(impl_result.get("possibly_implemented", [])) + len(impl_result.get("not_implemented", []))
        
        if impl_total > 0:
            impl_coverage = (impl_implemented + len(impl_result.get("possibly_implemented", []))) / impl_total * 100
            report.append(f"  • 🔧 Implementation requirements: {impl_implemented}/{impl_total} implemented ({impl_coverage:.1f}% coverage)")
        
        report.append("")
        
        # Recommendations based on concrete requirements
        report.append("💡 Concrete Requirements Implementation Recommendations:")
        
        # Priority recommendations based on missing data requirements
        missing_data = data_result.get("not_implemented", [])
        if missing_data:
            report.append("  📊 HIGH PRIORITY - Missing Data Requirements:")
            for req in missing_data[:3]:  # Show top 3
                report.append(f"    • Implement: {req}")
        
        # Algorithm requirements recommendations
        missing_algos = algo_result.get("not_implemented", [])
        if missing_algos:
            report.append("  ⚙️ HIGH PRIORITY - Missing Algorithm Requirements:")
            for algo in missing_algos[:3]:  # Show top 3
                report.append(f"    • Implement: {algo}")
        
        # Implementation requirements recommendations
        missing_impl = impl_result.get("not_implemented", [])
        if missing_impl:
            report.append("  🔧 MEDIUM PRIORITY - Missing Implementation Requirements:")
            for impl in missing_impl[:3]:  # Show top 3
                report.append(f"    • Implement: {impl}")
        
        # Verification recommendations
        possible_items = []
        for category in ["data_requirements", "algorithm_requirements", "implementation_requirements", "expected_outputs"]:
            if category in comparison_result:
                possible_items.extend(comparison_result[category].get("possibly_implemented", []))
        
        if possible_items:
            report.append("  🔍 VERIFICATION NEEDED - Possibly Implemented:")
            for item in possible_items[:3]:  # Show top 3
                report.append(f"    • Verify implementation completeness: {item}")
        
        report.append("")
        
        # # Code structure summary
        # if codebase_info["all_classes"]:
        #     report.append("📂 Main Classes:")
        #     for cls in codebase_info["all_classes"][:10]:
        #         report.append(f"  • {cls}")
        #     report.append("")
        
        # if codebase_info["all_functions"]:
        #     report.append("🔧 Main Functions:")
        #     for func in codebase_info["all_functions"][:15]:
        #         report.append(f"  • {func}")
        #     report.append("")
        
        return "\n".join(report)

    def _get_prompt_templates(self) -> Dict[str, str]:
        """Get prompt templates for different analysis tasks"""
        try:
            # Load templates from external file
            template_path = Path(__file__).parent.parent.parent / "prompt" / "templates" / "repo_analyzer_prompts.j2"
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the YAML-like format
                templates = {}
                current_template = None
                current_content = []
                
                for line in content.split('\n'):
                    if line.strip().endswith(': |'):
                        if current_template and current_content:
                            templates[current_template] = '\n'.join(current_content).strip()
                        current_template = line.strip()[:-2].rstrip(':')  # Remove ': |' and any trailing colon
                        current_content = []
                    elif current_template and line.startswith(' '):
                        current_content.append(line[2:])  # Remove leading spaces
                    elif current_template and line.strip() == '':
                        current_content.append('')
                
                if current_template and current_content:
                    templates[current_template] = '\n'.join(current_content).strip()
                
                # Convert Jinja2 placeholders to Python format placeholders
                for template_name, template_content in templates.items():
                    # Replace {{ variable }} with {variable}
                    import re
                    template_content = re.sub(r'\{\{\s*(\w+)\s*\}\}', r'{\1}', template_content)
                    templates[template_name] = template_content
                
                return templates
            else:
                # Fallback to hardcoded templates if file doesn't exist
                logger.warning(f"Template file not found: {template_path}, using fallback templates")
                return self._get_fallback_templates()
                
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}, using fallback")
            return self._get_fallback_templates()
    
    def _get_fallback_templates(self) -> Dict[str, str]:
        """Fallback templates if external file loading fails"""
        return {
            "paper_analysis": "You are a software engineer reading a research paper to understand EXACTLY what needs to be implemented. Focus on CONCRETE, TESTABLE requirements.\n\n## Paper Content:\n{paper_content}\n\n## Task:\nExtract SPECIFIC implementation requirements that can be verified in code. Avoid abstract concepts.\n\n## Instructions:\n- Include EXACT numbers, names, and values from the paper\n- Focus on what can be TESTED or VERIFIED in actual code\n- Extract specific technical details, not abstract descriptions\n- Include any code snippets, formulas, or exact specifications\n- If the paper mentions specific tools, libraries, or frameworks, include them\n\nExtract ONLY what is mentioned in the paper, do not fabricate or assume details.",
            "code_comparison": "You are an expert code analyst and research paper reviewer. Your task is to analyze whether the functionalities described in a research paper are implemented in the given codebase.\n\n## Codebase Summary:\n{codebase_summary}\n\n## Paper Functionalities to Check:\n{paper_functionalities}\n\n## Task:\nFor each functionality listed above, determine its implementation status in the codebase:\n\n1. **IMPLEMENTED**: The functionality is clearly implemented in the code with evidence\n2. **POSSIBLY_IMPLEMENTED**: There are signs the functionality might be implemented but it's not certain\n3. **NOT_IMPLEMENTED**: No evidence of this functionality in the codebase\n\n## Instructions:\n- Focus on semantic understanding, not just keyword matching\n- Consider class names, function names, and code logic\n- Look for actual implementation evidence, not just mentions\n- Be conservative: only mark as IMPLEMENTED if you have clear evidence\n- Provide specific evidence for your decisions",
            "query_analysis": "You are an experienced software engineer and code architect with deep expertise in code analysis. The user has a question about a codebase and you need to provide a comprehensive, accurate answer.\n\n## User Query:\n{user_query}\n\n## Project Context:\n{project_context}\n\n## Repository Structure:\n{repo_map}\n\n## Program Analysis:\n{call_graph}\n\n## Relevant Code:\n{relevant_code}\n\n## Instructions:\n- Analyze the codebase comprehensively to answer the user's specific question\n- Use the repository structure to understand the overall project organization\n- Reference the program analysis to understand component relationships and dependencies\n- Examine the relevant code snippets carefully to provide accurate technical details\n- Explain code functionality, design patterns, and architectural decisions when relevant\n- Identify key functions, classes, and modules that relate to the user's query\n- Point out interesting implementation details, potential issues, or best practices observed\n- If the query involves troubleshooting, provide specific guidance based on the code analysis"
        }

    def _get_prompt_template(self, template_name: str, **kwargs) -> str:
        """Get a prompt template and format it with the provided parameters"""
        templates = self._get_prompt_templates()
        if template_name not in templates:
            raise ValueError(f"Unknown prompt template: {template_name}")
        
        template = templates[template_name]
        return template.format(**kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize repo analyzer task
    repo_analyzer_task = TaskRepoAnalyzer(
        model="gpt-4o", 
        temperature=0.1,
        env_file=".env"
    )
    
    # Define paths
    paper_path = "debug/data/semantic-self-consistency/paper.md"
    repo_path = "evaluation/preparedness/project/paperbench/data/judge_eval/semantic-self-consistency/0/submission"
    
    print("🧪 Paper vs Codebase Implementation Analysis")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(paper_path):
        print(f"❌ Paper file not found: {paper_path}")
        print("Please ensure paper.md file exists at the specified path")
    elif not os.path.exists(repo_path):
        print(f"❌ Codebase path not found: {repo_path}")
        print("Please ensure codebase exists at the specified path")
    else:
        print(f"✅ Found paper file: {paper_path}")
        print(f"✅ Found codebase: {repo_path}")
        print()
        
        # Run simple paper vs codebase analysis
        print("🚀 Starting simple comparison analysis...")
        print("=" * 60)
        
        try:
            # Use new simple analysis method
            report = repo_analyzer_task.analyze_paper_vs_codebase(paper_path, repo_path)
            print("\n" + report)
            
            # Save report to file
            output_file = "paper_vs_codebase_analysis.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n💾 Report saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✅ Analysis completed!") 
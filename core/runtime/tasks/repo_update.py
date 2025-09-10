"""
Advanced Repo Update Task for implementing new features and code modifications.

This task provides comprehensive repo editing capabilities based on user requirements,
allowing for intelligent code modifications, feature additions, and improvements.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from openai import OpenAI
import os
import ast
from pathlib import Path
from dotenv import load_dotenv
import json
import re
import difflib
from .rollback import TaskRollback

logger = logging.getLogger(__name__)


@dataclass
class RepoUpdateInput:
    """Input data structure for repo update task."""
    repo_path: str
    requirements: str  # User's specific requirements for code changes
    target_files: Optional[List[str]] = None  # Specific files to edit (optional)
    context: Optional[str] = None  # Additional context about the changes needed
    preserve_structure: bool = True  # Whether to preserve existing code structure


@dataclass
class RepoUpdateOutput:
    """Output data structure for repo update task."""
    success: bool
    plan: str
    modified_files: Dict[str, str]  # filename -> new content
    changes_summary: str
    file_diffs: Dict[str, str] = None  # filename -> diff content
    detailed_changes: Dict[str, Dict[str, Any]] = None  # filename -> change details
    error_message: Optional[str] = None
    repo_analysis: Optional[str] = None


class RepoUpdate:
    """
    Advanced repo update task that can modify multiple files based on user requirements.
    
    This class provides comprehensive repo editing capabilities including:
    - Multi-file analysis and modification
    - Feature addition based on requirements
    - Code refactoring and improvement
    - Structure-aware code changes
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1, env_file: Optional[str] = None):
        """
        Initialize the repo update task.
        
        Args:
            model: The OpenAI model to use for code generation
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
        """Load environment variables from .env file."""
        if env_file:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from: {env_file}")
            else:
                logger.warning(f"Specified .env file not found: {env_file}")
        else:
            env_locations = [
                ".env", "../.env", "../../.env", os.path.expanduser("~/.env")
            ]
            for env_path in env_locations:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from: {env_path}")
                    break
            else:
                load_dotenv()
                logger.info("Attempted to load .env from current directory")
    
    def _analyze_repo_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and extract key information."""
        repo_path = Path(repo_path)
        analysis = {
            "repo_path": str(repo_path),  # Add repo_path for reference
            "python_files": [],
            "main_files": [],
            "config_files": [],
            "test_files": [],
            "dependencies": set(),
            "classes": {},
            "functions": {},
            "imports": set()
        }
        
        # Find all Python files
        for py_file in repo_path.rglob("*.py"):
            if py_file.name.startswith('.'):
                continue
                
            relative_path = str(py_file.relative_to(repo_path))
            analysis["python_files"].append(relative_path)
            
            # Categorize files
            if py_file.name in ["main.py", "run.py", "train.py", "app.py"]:
                analysis["main_files"].append(relative_path)
            elif "test" in py_file.name.lower() or "test" in str(py_file.parent).lower():
                analysis["test_files"].append(relative_path)
                
            # Analyze file content
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                # Extract classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if relative_path not in analysis["classes"]:
                            analysis["classes"][relative_path] = []
                        analysis["classes"][relative_path].append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        if relative_path not in analysis["functions"]:
                            analysis["functions"][relative_path] = []
                        analysis["functions"][relative_path].append(node.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis["imports"].add(node.module)
                            
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Find config files
        for config_file in repo_path.rglob("*.json"):
            if config_file.name in ["package.json", "config.json", "settings.json"]:
                analysis["config_files"].append(str(config_file.relative_to(repo_path)))
        
        for req_file in repo_path.rglob("requirements*.txt"):
            analysis["config_files"].append(str(req_file.relative_to(repo_path)))
            
        return analysis
    
    def _identify_target_files(self, repo_analysis: Dict[str, Any], requirements: str, target_files: Optional[List[str]] = None) -> List[str]:
        """Identify which files need to be modified or created based on requirements."""
        if target_files:
            return target_files
            
        # Use LLM to identify relevant files (both existing and new)
        prompt = f"""
Based on the repository analysis and requirements, identify which files should be modified or created.

Repository Structure:
- Python files: {repo_analysis["python_files"]}
- Main files: {repo_analysis["main_files"]}
- Classes: {repo_analysis["classes"]}
- Functions: {repo_analysis["functions"]}

Requirements: {requirements}

Instructions:
1. Identify existing files that need modification
2. Identify new files that need to be created (e.g., requirements.txt, README.md, config files)
3. Consider common missing files like requirements.txt, README.md, setup.py, etc.
4. Return a JSON list of file paths

Return only a JSON list of file paths, like: ["file1.py", "requirements.txt", "README.md"]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code analyzer. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON from the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                files = json.loads(json_match.group())
                return files
                
        except Exception as e:
            logger.warning(f"Could not identify target files with LLM: {e}")
            
        # Fallback: return main files and common missing files
        fallback_files = repo_analysis["main_files"] or repo_analysis["python_files"][:3]
        
        # Add common missing files if they don't exist
        repo_path = Path(repo_analysis.get("repo_path", ""))
        common_files = ["requirements.txt", "README.md", "setup.py"]
        
        for common_file in common_files:
            if not (repo_path / common_file).exists():
                fallback_files.append(common_file)
                
        return fallback_files
    
    def _validate_generated_content(self, file_path: str, content: str) -> Tuple[bool, str]:
        """
        Validate that generated content is appropriate for the file type.
        
        Args:
            file_path: Path to the file being modified
            content: Generated content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_extension = Path(file_path).suffix.lower()
        
        # Basic validation for different file types
        if file_extension == '.md':
            # For markdown files, check that it's not just code
            if content.strip().startswith('```python') or content.strip().startswith('```'):
                return False, "Generated content appears to be code rather than markdown documentation"
            
            # Check for basic markdown structure
            if not any(line.strip().startswith('#') for line in content.split('\n')):
                return False, "Generated content lacks markdown headers"
                
        elif file_extension == '.py':
            # For Python files, check basic syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                return False, f"Generated Python content has syntax errors: {e}"
                
        elif file_extension in ['.yaml', '.yml']:
            # For YAML files, check basic syntax
            try:
                import yaml
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                return False, f"Generated YAML content has syntax errors: {e}"
                
        elif file_extension == '.json':
            # For JSON files, check syntax
            try:
                import json
                json.loads(content)
            except json.JSONDecodeError as e:
                return False, f"Generated JSON content has syntax errors: {e}"
        
        # Check for reasonable content length
        if len(content.strip()) < 10:
            return False, "Generated content is too short"
            
        return True, ""
    
    def _generate_new_file_content(self, file_path: str, requirements: str, repo_analysis: Dict[str, Any]) -> str:
        """Generate content for a new file based on requirements."""
        file_extension = Path(file_path).suffix.lower()
        
        # Build context about the repository
        context = f"""
Repository Analysis:
- New file to create: {file_path}
- Available classes: {repo_analysis["classes"]}
- Available functions: {repo_analysis["functions"]}
- Available imports: {list(repo_analysis["imports"])[:20]}
- Other files in repo: {repo_analysis["python_files"][:10]}
"""
        
        if file_extension == '.txt':
            # Requirements.txt file
            prompt = f"""
You are a Python package manager. Create a requirements.txt file based on the project requirements.

{context}

Requirements: {requirements}

Instructions:
1. Create a requirements.txt file with all necessary Python dependencies
2. Include version numbers for stability
3. List dependencies in alphabetical order
4. Include common ML/AI libraries if the project involves machine learning
5. Ensure all dependencies are compatible

Common dependencies to consider:
- torch, transformers, numpy, pandas, scikit-learn (for ML projects)
- requests, beautifulsoup4 (for web scraping)
- pytest, unittest (for testing)
- yaml, json (for configuration)

Return only the requirements.txt content:
"""
        elif file_extension == '.md':
            # README.md or other markdown files
            prompt = f"""
You are a technical writer. Create a new markdown file based on the requirements.

{context}

Requirements: {requirements}

Instructions:
1. Create a professional, well-organized markdown document
2. Include appropriate sections like Overview, Installation, Usage, etc.
3. Make it informative and easy to follow
4. Follow markdown best practices
5. Do NOT include source code implementations - only documentation

Return only the markdown content:
"""
        elif file_extension == '.py':
            # Python file
            prompt = f"""
You are a senior software engineer. Create a new Python file based on the requirements.

{context}

Requirements: {requirements}

Instructions:
1. Create a complete, executable Python file
2. Follow Python best practices and maintain code quality
3. Add proper error handling and documentation
4. Include necessary imports
5. Ensure the code is syntactically correct and follows PEP 8

Return only the Python code:
"""
        elif file_extension in ['.yaml', '.yml']:
            # YAML configuration file
            prompt = f"""
You are a DevOps engineer. Create a new YAML configuration file based on the requirements.

{context}

Requirements: {requirements}

Instructions:
1. Create a well-structured YAML configuration
2. Use proper YAML syntax and indentation
3. Include all necessary configuration options
4. Follow YAML best practices

Return only the YAML content:
"""
        elif file_extension == '.json':
            # JSON file
            prompt = f"""
You are a data engineer. Create a new JSON file based on the requirements.

{context}

Requirements: {requirements}

Instructions:
1. Create a valid JSON structure
2. Use proper JSON formatting
3. Include all necessary data fields

Return only the JSON content:
"""
        else:
            # Generic text file
            prompt = f"""
You are a software developer. Create a new file based on the requirements.

{context}

Requirements: {requirements}

Instructions:
1. Create appropriate content for the file type
2. Ensure the content is well-structured and useful
3. Follow best practices for the file format

Return only the file content:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional software developer. Return only the complete file content without any explanations or markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split('\n')
                if len(lines) > 1:
                    content = '\n'.join(lines[1:])
                    if content.endswith("```"):
                        content = content[:-3]
                
            return content.strip()
            
        except Exception as e:
            logger.error(f"Could not generate new file content for {file_path}: {e}")
            return ""

    def _generate_file_modifications(self, repo_path: str, file_path: str, requirements: str, repo_analysis: Dict[str, Any]) -> str:
        """Generate modifications for a specific file based on requirements."""
        full_path = Path(repo_path) / file_path
        
        if not full_path.exists():
            logger.info(f"File {file_path} does not exist, creating new file")
            return self._generate_new_file_content(file_path, requirements, repo_analysis)
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return ""
        
        # Build context about the repository
        context = f"""
Repository Analysis:
- File: {file_path}
- Related classes: {repo_analysis["classes"].get(file_path, [])}
- Related functions: {repo_analysis["functions"].get(file_path, [])}
- Available imports: {list(repo_analysis["imports"])[:20]}
- Other files in repo: {repo_analysis["python_files"][:10]}
"""
        
        # Determine file type and generate appropriate prompt
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.md':
            # Markdown file (like README.md)
            prompt = f"""
You are a technical writer. Modify the following markdown file to fulfill the requirements.

{context}

Current file content:
```markdown
{current_content}
```

Requirements: {requirements}

Instructions:
1. Provide ONLY the complete modified markdown content
2. Maintain proper markdown formatting and structure
3. Create a professional, well-organized document
4. Include appropriate sections like Overview, Installation, Usage, etc.
5. Do NOT include source code implementations - only documentation
6. Ensure the content is readable and follows markdown best practices

Return the complete modified markdown content:
"""
        elif file_extension == '.py':
            # Python file
            prompt = f"""
You are a senior software engineer. Modify the following Python file to fulfill the requirements.

{context}

Current file content:
```python
{current_content}
```

Requirements: {requirements}

Instructions:
1. Provide ONLY the complete modified file content
2. Maintain existing functionality unless specifically asked to change it
3. Follow Python best practices and maintain code quality
4. Add proper error handling and documentation
5. Ensure the code is executable and syntactically correct

Return the complete modified file content:
"""
        elif file_extension in ['.yaml', '.yml']:
            # YAML configuration file
            prompt = f"""
You are a DevOps engineer. Modify the following YAML configuration file to fulfill the requirements.

{context}

Current file content:
```yaml
{current_content}
```

Requirements: {requirements}

Instructions:
1. Provide ONLY the complete modified YAML content
2. Maintain proper YAML syntax and indentation
3. Ensure the configuration is valid and well-structured
4. Follow YAML best practices

Return the complete modified YAML content:
"""
        elif file_extension == '.json':
            # JSON file
            prompt = f"""
You are a data engineer. Modify the following JSON file to fulfill the requirements.

{context}

Current file content:
```json
{current_content}
```

Requirements: {requirements}

Instructions:
1. Provide ONLY the complete modified JSON content
2. Maintain valid JSON syntax
3. Ensure proper formatting and structure

Return the complete modified JSON content:
"""
        else:
            # Generic text file
            prompt = f"""
You are a software developer. Modify the following file to fulfill the requirements.

{context}

Current file content:
```
{current_content}
```

Requirements: {requirements}

Instructions:
1. Provide ONLY the complete modified file content
2. Maintain the file's original format and structure
3. Ensure the content is appropriate for the file type

Return the complete modified file content:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional software developer. Return only the complete modified file content without any explanations or markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            modified_content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present (for all file types)
            if modified_content.startswith("```"):
                # Find the end of the code block
                lines = modified_content.split('\n')
                if len(lines) > 1:
                    # Skip the first line (```python, ```markdown, etc.)
                    modified_content = '\n'.join(lines[1:])
                    # Remove the last ``` if present
                    if modified_content.endswith("```"):
                        modified_content = modified_content[:-3]
                
            # Validate generated content
            is_valid, error_message = self._validate_generated_content(file_path, modified_content)
            if not is_valid:
                logger.warning(f"Generated content is not valid: {error_message}")
                return current_content
            
            return modified_content.strip()
            
        except Exception as e:
            logger.error(f"Could not generate modifications for {file_path}: {e}")
            return current_content
    
    def _generate_plan(self, requirements: str, repo_analysis: Dict[str, Any], target_files: List[str]) -> str:
        """Generate a plan for the modifications."""
        prompt = f"""
Create a detailed plan for implementing the following requirements in a Python codebase.

Requirements: {requirements}

Repository Information:
- Files to modify: {target_files}
- Available classes: {repo_analysis["classes"]}
- Available functions: {repo_analysis["functions"]}
- Current imports: {list(repo_analysis["imports"])[:20]}

Provide a step-by-step implementation plan (3-5 sentences).
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical architect. Provide a clear, concise implementation plan."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"Could not generate plan: {e}")
            return f"Plan: Implement the requirements '{requirements}' by modifying files: {target_files}"
    
    def _generate_file_diff(self, original_content: str, modified_content: str, file_path: str) -> str:
        """Generate a unified diff between original and modified content."""
        original_lines = original_content.splitlines(keepends=True)
        modified_lines = modified_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        )
        
        return ''.join(diff)
    
    def _analyze_changes(self, original_content: str, modified_content: str, file_path: str) -> Dict[str, Any]:
        """Analyze the changes made to a file and return detailed information."""
        original_lines = original_content.splitlines()
        modified_lines = modified_content.splitlines()
        
        # Count additions and deletions
        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        additions = 0
        deletions = 0
        modifications = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                deletions += i2 - i1
                modifications.append({
                    "type": "deletion",
                    "lines": list(range(i1 + 1, i2 + 1)),
                    "content": original_lines[i1:i2]
                })
            elif tag == 'insert':
                additions += j2 - j1
                modifications.append({
                    "type": "addition",
                    "lines": list(range(j1 + 1, j2 + 1)),
                    "content": modified_lines[j1:j2]
                })
            elif tag == 'replace':
                deletions += i2 - i1
                additions += j2 - j1
                modifications.append({
                    "type": "modification",
                    "original_lines": list(range(i1 + 1, i2 + 1)),
                    "modified_lines": list(range(j1 + 1, j2 + 1)),
                    "original_content": original_lines[i1:i2],
                    "modified_content": modified_lines[j1:j2]
                })
        
        # Try to identify what was changed using AST analysis for Python files
        change_types = []
        if file_path.endswith('.py'):
            try:
                original_tree = ast.parse(original_content)
                modified_tree = ast.parse(modified_content)
                
                # Extract function and class names
                original_functions = {node.name for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef)}
                modified_functions = {node.name for node in ast.walk(modified_tree) if isinstance(node, ast.FunctionDef)}
                
                original_classes = {node.name for node in ast.walk(original_tree) if isinstance(node, ast.ClassDef)}
                modified_classes = {node.name for node in ast.walk(modified_tree) if isinstance(node, ast.ClassDef)}
                
                # Identify new/removed functions and classes
                new_functions = modified_functions - original_functions
                removed_functions = original_functions - modified_functions
                new_classes = modified_classes - original_classes
                removed_classes = original_classes - modified_classes
                
                if new_functions:
                    change_types.append(f"Added functions: {', '.join(new_functions)}")
                if removed_functions:
                    change_types.append(f"Removed functions: {', '.join(removed_functions)}")
                if new_classes:
                    change_types.append(f"Added classes: {', '.join(new_classes)}")
                if removed_classes:
                    change_types.append(f"Removed classes: {', '.join(removed_classes)}")
                    
                # Check for modified functions/classes (same name, different content)
                common_functions = original_functions & modified_functions
                if common_functions:
                    change_types.append(f"Potentially modified functions: {', '.join(common_functions)}")
                    
            except:
                # If AST parsing fails, just note that the file was modified
                change_types.append("Python file modified (could not parse for detailed analysis)")
        
        return {
            "file_path": file_path,
            "lines_added": additions,
            "lines_deleted": deletions,
            "total_changes": additions + deletions,
            "change_types": change_types,
            "detailed_modifications": modifications[:5]  # Limit to first 5 modifications for brevity
        }
    
    def update_repo(self, update_input: RepoUpdateInput) -> RepoUpdateOutput:
        """
        Update repository based on user requirements.
        
        Args:
            update_input: The update input containing requirements and target information
            
        Returns:
            RepoUpdateOutput containing the results of the update operation
        """
        try:
            repo_path = update_input.repo_path
            if not os.path.exists(repo_path):
                return RepoUpdateOutput(
                    success=False,
                    plan="",
                    modified_files={},
                    changes_summary="",
                    error_message=f"Repository path does not exist: {repo_path}"
                )
            
            logger.info(f"Starting repo update for: {repo_path}")
            logger.info(f"Requirements: {update_input.requirements}")
            
            # Analyze repository structure
            repo_analysis = self._analyze_repo_structure(repo_path)
            
            # Identify target files
            target_files = self._identify_target_files(
                repo_analysis, 
                update_input.requirements, 
                update_input.target_files
            )
            
            logger.info(f"Target files identified: {target_files}")
            
            # Generate implementation plan
            plan = self._generate_plan(update_input.requirements, repo_analysis, target_files)
            
            # Generate modifications for each target file
            modified_files = {}
            file_diffs = {}
            detailed_changes = {}
            changes = []
            
            for file_path in target_files:
                logger.info(f"Generating modifications for: {file_path}")
                
                # Get original content
                full_path = Path(repo_path) / file_path
                original_content = ""
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read original content of {file_path}: {e}")
                
                # Generate modified content
                modified_content = self._generate_file_modifications(
                    repo_path, file_path, update_input.requirements, repo_analysis
                )
                
                if modified_content and modified_content != original_content:
                    modified_files[file_path] = modified_content
                    
                    # Generate diff
                    diff = self._generate_file_diff(original_content, modified_content, file_path)
                    file_diffs[file_path] = diff
                    
                    # Analyze changes
                    change_analysis = self._analyze_changes(original_content, modified_content, file_path)
                    detailed_changes[file_path] = change_analysis
                    
                    changes.append(f"✅ Modified {file_path} (+{change_analysis['lines_added']} -{change_analysis['lines_deleted']} lines)")
                    if change_analysis['change_types']:
                        changes.append(f"   └─ {'; '.join(change_analysis['change_types'])}")
                    else:
                        changes.append(f"❌ Could not modify {file_path}")
            
            # 简化changes_summary - 只保留关键信息
            if modified_files:
                file_list = list(modified_files.keys())
                if len(file_list) <= 3:
                    changes_summary = f"📋 Modified: {', '.join(file_list)}"
                else:
                    changes_summary = f"📋 Modified {len(file_list)} files: {', '.join(file_list[:3])}... (+{len(file_list)-3} more)"
            else:
                changes_summary = "📋 No files were modified"
            
            # 简化repo_analysis_summary - 只保留核心统计
            total_classes = sum(len(classes) for classes in repo_analysis["classes"].values())
            total_functions = sum(len(funcs) for funcs in repo_analysis["functions"].values())
            repo_analysis_summary = f"Repo: {len(repo_analysis['python_files'])} files, {total_classes} classes, {total_functions} functions"
            
            return RepoUpdateOutput(
                success=True,
                plan=plan,
                modified_files=modified_files,
                changes_summary=changes_summary,
                file_diffs=file_diffs,
                detailed_changes=detailed_changes,
                repo_analysis=repo_analysis_summary
            )
        
        except Exception as e:
            logger.error(f"Repo update task failed with error: {str(e)}")
            return RepoUpdateOutput(
                success=False,
                plan="",
                modified_files={},
                changes_summary="",
                error_message=f"Repo update task failed: {str(e)}"
            )
    
    def apply_changes(self, update_output: RepoUpdateOutput, repo_path: str, save_snapshot: bool = True) -> bool:
        """
        Apply the generated changes to the actual files.
        
        Args:
            update_output: The output from update_repo containing modifications
            repo_path: Path to the repository
            save_snapshot: Whether to save a snapshot after applying changes
            
        Returns:
            True if all changes were applied successfully
        """
        if not update_output.success:
            logger.error("Cannot apply changes: update operation was not successful")
            return False
            
        repo_path = Path(repo_path)
        applied_files = []
        failed_files = []
        
        try:
            for file_path, new_content in update_output.modified_files.items():
                full_path = repo_path / file_path
                
                # Validate content before writing
                is_valid, error_message = self._validate_generated_content(file_path, new_content)
                if not is_valid:
                    logger.error(f"Content validation failed for {file_path}: {error_message}")
                    failed_files.append((file_path, error_message))
                    continue
                
                # Write the new content
                os.makedirs(full_path.parent, exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                applied_files.append(file_path)
                logger.info(f"Applied changes to: {file_path}")
            
            if failed_files:
                logger.warning(f"Failed to apply changes to {len(failed_files)} files:")
                for file_path, error in failed_files:
                    logger.warning(f"  - {file_path}: {error}")
            
            logger.info(f"Successfully applied changes to {len(applied_files)} files")
            
            # Save snapshot after successfully applying changes
            if save_snapshot and applied_files:
                try:
                    rollback = TaskRollback()
                    snapshot_tag = rollback.save_snapshot(str(repo_path))
                    logger.info(f"💾 Saved snapshot with tag: {snapshot_tag}")
                    print(f"📸 Snapshot saved successfully: {snapshot_tag}")
                except Exception as e:
                    logger.warning(f"Failed to save snapshot: {e}")
                    print(f"⚠️  Warning: Could not save snapshot: {e}")
            
            return len(failed_files) == 0
            
        except Exception as e:
            logger.error(f"Failed to apply changes: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Advanced repo editing (feature addition)
    repo_editor = RepoUpdate(
        model="gpt-4o", 
        temperature=0.1,
        env_file=".env"
    )
    
    update_input = RepoUpdateInput(
        repo_path="/Users/suny0a/Proj/MC-Scientist/workspace/20250617_092025/debug/data/semantic-self-consistency/submission",
        requirements="The dataloader cannot sucessfully load the data. Help me fix it.",
        target_files=None,  # Auto-detect files to modify
        context="This is a machine learning project that processes datasets and needs better observability",
        preserve_structure=True
    )
    
    update_result = repo_editor.update_repo(update_input)
    print("🎯 Repo Update Result:", update_result.success)
    print("\n📋 Implementation Plan:")
    print(update_result.plan)
    print(f"\n📂 Modified Files: {list(update_result.modified_files.keys())}")
    
    # Display detailed changes
    if update_result.success and update_result.detailed_changes:
        print("\n" + "="*60)
        print("📊 DETAILED CHANGES")
        print("="*60)
        
        for file_path, changes in update_result.detailed_changes.items():
            print(f"\n📄 File: {file_path}")
            print(f"   📈 Lines added: {changes['lines_added']}")
            print(f"   📉 Lines deleted: {changes['lines_deleted']}")
            print(f"   🔄 Total changes: {changes['total_changes']}")
            
            if changes['change_types']:
                print(f"   🏷️  Change types: {', '.join(changes['change_types'])}")
            
            # Show diff if available
            if update_result.file_diffs and file_path in update_result.file_diffs:
                print(f"\n📝 Diff for {file_path}:")
                print("-" * 40)
                # Show first 20 lines of diff to avoid overwhelming output
                diff_lines = update_result.file_diffs[file_path].split('\n')
                for line in diff_lines:#[:20]:
                    print(line)
                # if len(diff_lines) > 20:
                #     print(f"... (showing first 20 lines of {len(diff_lines)} total)")
                print("-" * 40)
        
        print("\n" + update_result.changes_summary)
    
    # Apply the changes to actual files
    if update_result.success:
        applied = repo_editor.apply_changes(update_result, update_input.repo_path, save_snapshot=True)
        print(f"\n🔧 Changes applied: {applied}") 
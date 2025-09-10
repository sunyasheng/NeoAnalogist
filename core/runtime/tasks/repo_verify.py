#!/usr/bin/env python3
"""
Repository Verification Tool - Human-like Analysis with AI Enhancement

This tool analyzes repositories like a human would, enhanced with AI capabilities:
1. Read and understand the entire repo structure and call relationships
2. Configure the environment properly 
3. Run entry points in the correct order
4. Record detailed process and error information
5. Use AI for intelligent analysis and recommendations

Usage:
    python repo_verify.py <repo_path> "<requirement>" [--level <level>] [--output <format>]
"""

import os
import sys
import json
import argparse
import subprocess
import re
import ast
import importlib.util
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict, deque
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    BASIC = "basic"
    FUNCTIONAL = "functional" 
    COMPREHENSIVE = "comprehensive"


@dataclass
class CodeFile:
    """Represents a code file with its metadata and relationships"""
    path: Path
    language: str
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    local_imports: List[str] = field(default_factory=list)
    has_main: bool = False
    is_executable: bool = False
    is_entry_point: bool = False
    lines_of_code: int = 0
    complexity_score: float = 0.0


@dataclass
class ExecutionAttempt:
    """Record of an execution attempt"""
    file_path: str
    command: List[str]
    working_dir: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    error_analysis: str = ""


@dataclass
class AnalysisStep:
    """Record of an analysis step"""
    step_name: str
    description: str
    success: bool
    details: str
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AIAnalysisResult:
    """Result of AI-powered analysis"""
    analysis_type: str
    content: str
    confidence: float
    suggestions: List[str] = field(default_factory=list)
    error_details: Optional[str] = None


class RepoEvaluator:
    """Evaluates repositories to check their status and completeness after agent creation"""
    
    def __init__(self, repo_path: Path, requirement: str = "", skip_dependency_install: bool = False):
        self.repo_path = repo_path
        self.requirement = requirement
        self.skip_dependency_install = skip_dependency_install
        
        # Core analysis data
        self.code_files: Dict[str, CodeFile] = {}
        self.dependencies: Dict[str, bool] = {}
        self.call_graph: Dict[str, List[str]] = defaultdict(list)
        self.entry_points: List[CodeFile] = []
        self.analysis_steps: List[AnalysisStep] = []
        self.execution_attempts: List[ExecutionAttempt] = []
        self.environment_setup: Dict[str, Any] = {}
        
        # Enhanced analysis data (from repo-analyzer)
        self.repo_map: str = ""
        self.project_context: str = ""
        self.codebase_summary: str = ""
        self.ai_analysis_results: List[AIAnalysisResult] = []
        
        # Initialize AI client
        self._init_ai_client()
    
    def _init_ai_client(self):
        """Initialize AI client for enhanced analysis"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. AI features are required.")
        
        self.ai_client = OpenAI(api_key=api_key)
        logger.info("AI client initialized successfully")
    
    def _build_repo_map(self) -> str:
        """Build a repository structure map (from repo-analyzer)"""
        repo_map = []
        
        def add_tree_structure(path: Path, prefix: str = "", is_last: bool = True):
            """Recursively build tree structure."""
            if path.name.startswith('.'):
                return
                
            connector = "└── " if is_last else "├── "
            repo_map.append(f"{prefix}{connector}{path.name}")
            
            if path.is_dir():
                try:
                    children = [p for p in path.iterdir() if not p.name.startswith('.')]
                    # Sort: directories first, then files
                    children.sort(key=lambda p: (p.is_file(), p.name.lower()))
                    
                    for i, child in enumerate(children):
                        is_last_child = i == len(children) - 1
                        extension = "    " if is_last else "│   "
                        add_tree_structure(child, prefix + extension, is_last_child)
                except PermissionError:
                    pass
        
        repo_map.append(f"📁 Repository Structure: {self.repo_path.name}")
        repo_map.append("")
        
        try:
            children = [p for p in self.repo_path.iterdir() if not p.name.startswith('.')]
            children.sort(key=lambda p: (p.is_file(), p.name.lower()))
            
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                add_tree_structure(child, "", is_last_child)
                
        except Exception as e:
            repo_map.append(f"Error reading repository: {e}")
        
        return "\n".join(repo_map)
    
    def _get_project_context_from_repo(self) -> str:
        """Extract project context from repository (from repo-analyzer)"""
        context_parts = []
        
        # Check for README files
        readme_files = list(self.repo_path.glob("README*"))
        if readme_files:
            try:
                with open(readme_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    context_parts.append(f"README: {content[:500]}...")
            except Exception:
                pass
        
        # Check for setup files
        setup_files = ['setup.py', 'pyproject.toml', 'requirements.txt']
        for setup_file in setup_files:
            setup_path = self.repo_path / setup_file
            if setup_path.exists():
                try:
                    with open(setup_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        context_parts.append(f"{setup_file}: {content[:200]}...")
                except Exception:
                    pass
        
        # Extract main functionality from code
        main_functions = []
        for code_file in self.code_files.values():
            if code_file.has_main:
                main_functions.append(f"{code_file.path.name}: {', '.join(code_file.functions)}")
        
        if main_functions:
            context_parts.append(f"Main functions: {'; '.join(main_functions)}")
        
        return "\n".join(context_parts)
    
    def _prepare_codebase_summary(self) -> str:
        """Prepare a comprehensive summary of the codebase for AI analysis"""
        summary = []
        
        summary.append(f"## Codebase Overview:")
        summary.append(f"- Python files: {len(self.code_files)}")
        summary.append(f"- Total classes: {sum(len(cf.classes) for cf in self.code_files.values())}")
        summary.append(f"- Total functions: {sum(len(cf.functions) for cf in self.code_files.values())}")
        summary.append("")
        
        # Include repo map for structure understanding
        if self.repo_map:
            summary.append("## Repository Structure:")
            repo_map_lines = self.repo_map.split('\n')[:20]  # Limit to avoid token overflow
            for line in repo_map_lines:
                if line.strip():
                    summary.append(line)
            summary.append("")
        
        # Include file-specific structure
        summary.append("## File Structure Details:")
        for file_path, code_file in list(self.code_files.items())[:5]:  # Limit files
            if code_file.classes or code_file.functions:
                summary.append(f"### {file_path}:")
                if code_file.classes:
                    summary.append(f"  Classes: {', '.join(code_file.classes[:5])}")
                if code_file.functions:
                    summary.append(f"  Functions: {', '.join(code_file.functions[:5])}")
        summary.append("")
        
        # Include dependencies
        if self.dependencies:
            missing_deps = [dep for dep, available in self.dependencies.items() if not available]
            available_deps = [dep for dep, available in self.dependencies.items() if available]
            
            summary.append("## Dependencies:")
            summary.append(f"- Available: {', '.join(available_deps[:10])}")
            if missing_deps:
                summary.append(f"- Missing: {', '.join(missing_deps[:10])}")
        summary.append("")
        
        return "\n".join(summary)
    
    def _ai_analyze_execution_errors(self, execution_attempts: List[ExecutionAttempt]) -> AIAnalysisResult:
        """Use AI to analyze execution errors and provide intelligent solutions"""
        if not execution_attempts:
            return AIAnalysisResult(
                analysis_type="execution_error_analysis",
                content="No execution attempts to analyze",
                confidence=1.0
            )
        
        # Collect error information
        error_summary = []
        for attempt in execution_attempts:
            if not attempt.success:
                error_summary.append(f"File: {attempt.file_path}")
                error_summary.append(f"Command: {' '.join(attempt.command)}")
                error_summary.append(f"Error: {attempt.error_analysis}")
                error_summary.append(f"STDERR: {attempt.stderr[:200]}...")
                error_summary.append("---")
        
        if not error_summary:
            return AIAnalysisResult(
                analysis_type="error_analysis",
                content="No execution errors found",
                confidence=1.0
            )
        
        codebase_summary = self._prepare_codebase_summary()
        
        prompt = f"""You are an expert Python developer debugging execution errors.

## Codebase Summary:
{codebase_summary}

## Execution Errors:
{chr(10).join(error_summary)}

## Task:
Analyze these execution errors and provide:
1. Root cause analysis for each error
2. Specific solutions and fixes
3. Step-by-step debugging approach
4. Prevention strategies for future

Please provide detailed, actionable recommendations."""

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer and debugging specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract suggestions
            suggestions = []
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('-') or line.strip().startswith('*'):
                    suggestions.append(line.strip())
            
            return AIAnalysisResult(
                analysis_type="error_analysis",
                content=content,
                confidence=0.8,
                suggestions=suggestions
            )
            
        except Exception as e:
            return AIAnalysisResult(
                analysis_type="error_analysis",
                content=f"AI analysis failed: {str(e)}",
                confidence=0.0,
                error_details=str(e)
            )
    
    def _ai_analyze_requirement_compliance(self) -> AIAnalysisResult:
        """Use AI to analyze requirement compliance"""
        if not self.requirement:
            return AIAnalysisResult(
                analysis_type="requirement_compliance_analysis",
                content="No requirement specified for compliance analysis",
                confidence=1.0
            )
        
        codebase_summary = self._prepare_codebase_summary()
        
        prompt = f"""You are an expert code reviewer analyzing whether a codebase meets specific requirements.

## Requirement:
{self.requirement}

## Codebase Summary:
{codebase_summary}

## Task:
Analyze whether the codebase meets the specified requirement:
1. Identify implemented features that satisfy the requirement
2. Identify missing features or gaps
3. Provide specific recommendations for improvement
4. Rate compliance level (0-100%)

Please provide a detailed analysis with evidence from the codebase."""

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer and requirements analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract compliance percentage
            compliance_score = 0.0
            for line in content.split('\n'):
                if '%' in line and any(word in line.lower() for word in ['compliance', 'score', 'rate']):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        compliance_score = min(100.0, float(numbers[0]))
                        break
            
            return AIAnalysisResult(
                analysis_type="requirement_compliance",
                content=content,
                confidence=compliance_score / 100.0,
                suggestions=[line.strip() for line in content.split('\n') if line.strip().startswith('-')]
            )
            
        except Exception as e:
            return AIAnalysisResult(
                analysis_type="requirement_compliance",
                content=f"AI analysis failed: {str(e)}",
                confidence=0.0,
                error_details=str(e)
            )
    
    def evaluate_repository(self) -> Dict[str, Any]:
        """Simplified repository analysis focusing on reproduce.sh and core insights"""
        print("🧠 Starting simplified repository analysis...")
        print(f"📁 Repository: {self.repo_path}")
        if self.requirement:
            print(f"🎯 Requirement: {self.requirement}")
        
        # Step 1: Quick structure scan
        self._step_1_quick_structure_scan()
        
        # Step 2: Analyze reproduce.sh (main focus)
        self._step_2_analyze_reproduce_sh()
        
        # Step 3: Generate insights
        return self._step_3_generate_insights()
    
    def _step_1_quick_structure_scan(self):
        """Step 1: Quick scan of repository structure"""
        step = AnalysisStep(
            step_name="Quick Structure Scan",
            description="Quick scan of repository structure and key files",
            success=True,
            details=""
        )
        
        print("\n📖 Step 1: Quick structure scan...")
        
        # Scan for key files only
        key_files = []
        total_files = 0
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                
                # Focus on key files only
                if (file_path.name in ['reproduce.sh', 'main.py', 'requirements.txt', 'README.md', 'setup.py'] or
                    file_path.suffix in ['.py', '.sh']):
                    key_files.append(file_path.name)
                    
                    # Quick analysis of Python files
                    if file_path.suffix == '.py':
                        code_file = self._analyze_code_file(file_path)
                        self.code_files[str(file_path)] = code_file
                
        step.details = f"Scanned {total_files} files, found {len(key_files)} key files"
        step.findings.extend([
            f"Total files: {total_files}",
            f"Key files found: {', '.join(key_files)}",
            f"Python files: {len([f for f in key_files if f.endswith('.py')])}",
            f"Shell scripts: {len([f for f in key_files if f.endswith('.sh')])}"
        ])
        
        self.analysis_steps.append(step)
        print(f"   Found {len(key_files)} key files")
    
    def _step_2_analyze_reproduce_sh(self):
        """Step 2: Focus on analyzing reproduce.sh"""
        step = AnalysisStep(
            step_name="Reproduce.sh Analysis",
            description="Analyzing reproduce.sh as the main entry point",
            success=True,
            details=""
        )
        
        print("\n🎯 Step 2: Analyzing reproduce.sh...")
        
        reproduce_sh = self.repo_path / "reproduce.sh"
        
        if reproduce_sh.exists():
            print("   Found reproduce.sh - analyzing...")
            
            try:
                with open(reproduce_sh, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze reproduce.sh logic
                reproduce_analysis = self._analyze_reproduce_sh_logic(content)
                step.findings.extend(reproduce_analysis['findings'])
                step.recommendations.extend(reproduce_analysis['recommendations'])
                step.details = reproduce_analysis['summary']
                
                print(f"   ✅ reproduce.sh analysis completed")
                
            except Exception as e:
                step.findings.append(f"❌ Error analyzing reproduce.sh: {e}")
                step.success = False
                print(f"   ❌ Error analyzing reproduce.sh")
            else:
                print("   No reproduce.sh found - checking other entry points...")
            
            # Quick check for other entry points
            other_entries = []
            for file_path in self.repo_path.glob("*.py"):
                if file_path.name in ['main.py', '__main__.py'] or 'main' in file_path.name.lower():
                    other_entries.append(file_path.name)
            
            if other_entries:
                step.findings.append(f"Found alternative entry points: {', '.join(other_entries)}")
                step.recommendations.append("Consider creating reproduce.sh for complete reproduction pipeline")
            else:
                step.findings.append("No clear entry points found")
                step.recommendations.append("Add reproduce.sh or main.py as entry point")
                step.success = False
        
        self.analysis_steps.append(step)
    
    def _step_3_generate_insights(self) -> dict:
        """Step 3: Generate simplified insights"""
        print("\n💡 Step 3: Generating insights...")
        
        # Prepare summary
        summary = self._generate_simplified_summary()
        
        # AI analysis of reproduce.sh if it exists
        ai_analysis = None
        reproduce_sh = self.repo_path / "reproduce.sh"
        if reproduce_sh.exists():
            try:
                with open(reproduce_sh, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ai_analysis = self._ai_analyze_reproduce_sh(content)
            except Exception as e:
                print(f"   ⚠️ AI analysis failed: {e}")
        
        return {
            "success": True,
            "summary": summary,
            "analysis_steps": [asdict(step) for step in self.analysis_steps],
            "ai_analysis": asdict(ai_analysis) if ai_analysis else None,
            "recommendations": self._extract_recommendations(),
            "findings": self._extract_findings()
        }
    
    def _generate_simplified_summary(self) -> str:
        """Generate a simplified summary"""
        summary_parts = []
        
        # Repository overview
        summary_parts.append(f"Repository Analysis Summary")
        summary_parts.append(f"Repository: {self.repo_path.name}")
        summary_parts.append(f"Total files: {len(list(self.repo_path.rglob('*')))}")
        summary_parts.append(f"Python files: {len(self.code_files)}")
        
        # Reproduce.sh status
        reproduce_sh = self.repo_path / "reproduce.sh"
        if reproduce_sh.exists():
            summary_parts.append(f"✅ reproduce.sh found and analyzed")
        else:
            summary_parts.append(f"⚠️ reproduce.sh not found")
        
        # Key findings
        for step in self.analysis_steps:
            if step.findings:
                summary_parts.append(f"\n{step.step_name}:")
                for finding in step.findings[:3]:  # Top 3 findings
                    summary_parts.append(f"  - {finding}")
        
        return "\n".join(summary_parts)
    
    def _ai_analyze_reproduce_sh(self, content: str) -> AIAnalysisResult:
        """AI analysis of reproduce.sh content"""
        try:
            prompt = f"""Analyze this reproduce.sh script and provide insights:

{content}

Please provide:
1. Overall assessment of the reproduction pipeline
2. Key strengths and weaknesses
3. Specific recommendations for improvement
4. Potential issues or missing components

Focus on practical insights that would help improve the reproduction process."""

            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            return AIAnalysisResult(
                analysis_type="reproduce_sh_analysis",
                content=content,
                confidence=0.8,
                suggestions=[
                    "Check if all dependencies are properly handled",
                    "Ensure data download steps are included",
                    "Add proper error handling and logging"
                ]
            )
            
        except Exception as e:
            return AIAnalysisResult(
                analysis_type="reproduce_sh_analysis",
                content=f"AI analysis failed: {e}",
                confidence=0.0,
                suggestions=[]
            )
    
    def _extract_recommendations(self) -> List[str]:
        """Extract all recommendations from analysis steps"""
        recommendations = []
        for step in self.analysis_steps:
            recommendations.extend(step.recommendations)
        return list(set(recommendations))  # Remove duplicates
    
    def _extract_findings(self) -> List[str]:
        """Extract all findings from analysis steps"""
        findings = []
        for step in self.analysis_steps:
            findings.extend(step.findings)
        return findings
    
    def _analyze_code_file(self, file_path: Path) -> CodeFile:
        """Analyze a single code file"""
        code_file = CodeFile(
            path=file_path,
            language=file_path.suffix[1:] if file_path.suffix else "unknown",
            is_executable=os.access(file_path, os.X_OK)
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                code_file.lines_of_code = len([line for line in content.splitlines() if line.strip()])
            
            if file_path.suffix == '.py':
                self._analyze_python_file(code_file, content)
        except Exception:
            pass  # Skip files that can't be read
        
        return code_file
    
    def _analyze_python_file(self, code_file: CodeFile, content: str):
        """Analyze a Python file specifically"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    code_file.functions.append(node.name)
                    if node.name == "main":
                        code_file.has_main = True
                elif isinstance(node, ast.ClassDef):
                    code_file.classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        code_file.imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.level == 0:  # Absolute import
                            code_file.imports.append(node.module.split('.')[0])
                        else:  # Relative import
                            code_file.local_imports.append(node.module or ".")
            
            # Check for main guard
            if 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content:
                code_file.has_main = True
            
            # Calculate complexity (rough estimate)
            code_file.complexity_score = len(code_file.functions) + len(code_file.classes) * 2
            
        except SyntaxError:
            pass  # Skip files with syntax errors
    
    def _check_dependency(self, module_name: str) -> bool:
        """Check if a dependency is available"""
        builtin_modules = {
            'os', 'sys', 'json', 'argparse', 'subprocess', 're', 'ast',
            'pathlib', 'dataclasses', 'typing', 'enum', 'datetime',
            'collections', 'itertools', 'functools', 'math', 'random',
            'time', 'logging', 'urllib', 'http', 'email', 'xml', 'csv',
            'sqlite3', 'pickle', 'base64', 'hashlib', 'hmac', 'secrets'
        }
        
        if module_name in builtin_modules:
            return True
        
        try:
            importlib.util.find_spec(module_name)
            return True
        except:
            return False
    
    # Removed execution-related methods since we're focusing on analysis only
    
    def _detect_primary_language(self) -> str:
        """Detect the primary programming language"""
        lang_counts = defaultdict(int)
        for code_file in self.code_files.values():
            lang_counts[code_file.language] += 1
        
        if not lang_counts:
            return "unknown"
        
        return max(lang_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_repo_size(self) -> int:
        """Calculate total lines of code"""
        return sum(cf.lines_of_code for cf in self.code_files.values())
    
    def _calculate_overall_complexity(self) -> str:
        """Calculate overall complexity assessment"""
        total_complexity = sum(cf.complexity_score for cf in self.code_files.values())
        if total_complexity < 20:
            return "Simple"
        elif total_complexity < 100:
            return "Moderate"
        else:
            return "Complex"
    
    def _find_most_connected_files(self) -> List[str]:
        """Find the most connected files in the call graph"""
        connection_counts = defaultdict(int)
        for source, targets in self.call_graph.items():
            connection_counts[source] += len(targets)
            for target in targets:
                connection_counts[target] += 1
        
        sorted_files = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
        return [Path(f).name for f, _ in sorted_files[:3]]
    
    # Removed dependency management methods to simplify the tool
    
    # Removed complex AST analysis to improve performance
    
    # Removed complex AI analysis methods to simplify the tool

    def _generate_comprehensive_summary(self) -> str:
        """Generate a comprehensive LLM-based summary report"""
        # Prepare comprehensive context for LLM analysis
        context = self._prepare_summary_context()
        
        # Create LLM prompt for comprehensive analysis
        prompt = f"""
You are an expert software engineer analyzing a repository for debugging purposes. Provide a CONCISE but ACTIONABLE summary.

REPOSITORY CONTEXT:
{context}

**CRITICAL REQUIREMENT: Keep your response SHORT and FOCUSED on debugging essentials. Aim for 200-400 words maximum.**

Provide a brief analysis with:

1. **STATUS**: Repository completeness (Complete/Partial/Broken) - 1 sentence

2. **CRITICAL ISSUES** (Top 3-5 only):
   - List only the most blocking problems
   - Include specific file names and error messages
   - Focus on what prevents successful execution

3. **DEBUGGING STEPS** (3-5 actionable steps):
   - Specific commands to run
   - Files to check/fix
   - Dependencies to install
   - Keep steps concrete and executable

4. **KEY FILES**: List only the most important files (main entry points, core modules)

**IMPORTANT:**
- Be extremely concise - every word should be useful for debugging
- Include specific file names, error messages, and line numbers when available
- Focus on what an agent needs to know to fix the repository
- Skip general analysis - go straight to actionable debugging info
- If no critical issues, just state "Repository appears functional"

Format as a brief, structured list. No lengthy explanations.
"""

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a debugging specialist. Provide extremely concise, actionable analysis focused on fixing repository issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            summary = response.choices[0].message.content
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}")
            return f"Error generating summary: {e}"
    
    def _prepare_summary_context(self) -> str:
        """Prepare comprehensive context for LLM summary generation"""
        context_parts = []
        
        # Basic repository info
        context_parts.append(f"Repository Path: {self.repo_path}")
        context_parts.append(f"Requirement: {self.requirement if self.requirement else 'None specified'}")
        context_parts.append(f"Primary Language: {self._detect_primary_language()}")
        context_parts.append(f"Repository Size: {self._calculate_repo_size()} lines of code")
        context_parts.append(f"Total Code Files: {len(self.code_files)}")
        
        # Repository structure
        context_parts.append(f"\nRepository Structure:\n{self.repo_map}")
        
        # Project context
        if self.project_context:
            context_parts.append(f"\nProject Context:\n{self.project_context}")
        
        # Codebase summary
        if self.codebase_summary:
            context_parts.append(f"\nCodebase Summary:\n{self.codebase_summary}")
        
        # Entry points analysis
        context_parts.append(f"\nEntry Points Found: {len(self.entry_points)}")
        for i, ep in enumerate(self.entry_points, 1):
            context_parts.append(f"  {i}. {ep.path.name} (functions: {', '.join(ep.functions)})")
        
        # Dependencies analysis
        missing_deps = [name for name, available in self.dependencies.items() if not available]
        context_parts.append(f"\nDependencies Analysis:")
        context_parts.append(f"  Total Dependencies: {len(self.dependencies)}")
        context_parts.append(f"  Available: {len(self.dependencies) - len(missing_deps)}")
        context_parts.append(f"  Missing: {len(missing_deps)}")
        if missing_deps:
            context_parts.append(f"  Missing Dependencies: {', '.join(missing_deps)}")
        
        # Execution results
        successful_executions = sum(1 for attempt in self.execution_attempts if attempt.success)
        total_attempts = len(self.execution_attempts)
        context_parts.append(f"\nExecution Results:")
        context_parts.append(f"  Successful Executions: {successful_executions}/{total_attempts}")
        context_parts.append(f"  Success Rate: {successful_executions/max(1, total_attempts)*100:.1f}%")
        
        # Execution attempts details
        if self.execution_attempts:
            context_parts.append(f"\nExecution Attempts Details:")
            for i, attempt in enumerate(self.execution_attempts, 1):
                status = "SUCCESS" if attempt.success else "FAILED"
                context_parts.append(f"  {i}. {attempt.file_path} - {status}")
                if not attempt.success:
                    context_parts.append(f"     Error Analysis: {attempt.error_analysis}")
                    if attempt.stderr:
                        context_parts.append(f"     Full STDERR: {attempt.stderr}")
                    if attempt.stdout:
                        context_parts.append(f"     Full STDOUT: {attempt.stdout}")
                    context_parts.append(f"     Return Code: {attempt.return_code}")
                    context_parts.append(f"     Execution Time: {attempt.execution_time:.2f}s")
        
        # Static analysis results (simplified)
            context_parts.append(f"\nStatic Analysis Results:")
        context_parts.append(f"  AST Analysis: Completed")
        context_parts.append(f"  Python files analyzed: {len(self.code_files)}")
        context_parts.append(f"  Total functions found: {sum(len(cf.functions) for cf in self.code_files.values())}")
        context_parts.append(f"  Total classes found: {sum(len(cf.classes) for cf in self.code_files.values())}")
        
        # AI analysis results
        if self.ai_analysis_results:
            context_parts.append(f"\nAI Analysis Results:")
            for i, result in enumerate(self.ai_analysis_results, 1):
                context_parts.append(f"  {i}. {result.analysis_type} (confidence: {result.confidence:.2f})")
                context_parts.append(f"     Full Analysis: {result.content}")
                if result.suggestions:
                    context_parts.append(f"     All Suggestions: {', '.join(result.suggestions)}")
                if result.error_details:
                    context_parts.append(f"     Error Details: {result.error_details}")
        
        # Analysis steps summary
        context_parts.append(f"\nAnalysis Steps Summary:")
        for step in self.analysis_steps:
            status = "✓" if step.success else "✗"
            context_parts.append(f"  {status} {step.step_name}: {step.description}")
            context_parts.append(f"    Details: {step.details}")
            if step.findings:
                context_parts.append(f"    All Findings: {'; '.join(step.findings)}")
            if step.recommendations:
                context_parts.append(f"    All Recommendations: {'; '.join(step.recommendations)}")
        
        return "\n".join(context_parts)


def evaluate_repository(repo_path: str, requirement: str, level: str = "comprehensive", skip_dependency_install: bool = False) -> Dict[str, Any]:
    """
    Human-like repository verification with complete understanding and systematic execution
    """
    from datetime import datetime
    
    repo_path = Path(repo_path).resolve()
    
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    # Set default requirement if empty or None
    if not requirement or requirement.strip() == "":
        requirement = "Check the overall completeness and feasibility of the entire codebase"
    
    analyzer = RepoEvaluator(repo_path, requirement, skip_dependency_install)
    evaluation_result = analyzer.evaluate_repository()
    
    # Add metadata
    evaluation_result.update({
        "requirement": requirement,
        "repository_path": str(repo_path),
        "verification_level": level,
        "skip_dependency_install": skip_dependency_install,
        "timestamp": datetime.now().isoformat(),
        "tool_version": "human-like-v2.0"
    })
    
    return evaluation_result


def main():
    parser = argparse.ArgumentParser(
        description="Repository evaluation and status checking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool evaluates repositories to check their status and completeness:
1. Reads and understands the entire repository structure
2. Analyzes code relationships and dependencies  
3. Configures the environment properly
4. Executes entry points systematically
5. Records detailed process and error information

Examples (for agent use):
  python repo_verify.py /path/to/repo "Implement a calculator" --level comprehensive
  python repo_verify.py . "Reproduce research results" --output json
        """
    )
    
    parser.add_argument("repo_path", help="Path to the repository to verify")
    parser.add_argument("requirement", help="Description of what the repository should accomplish")
    parser.add_argument("--level", "-l", choices=["basic", "functional", "comprehensive"], 
                       default="comprehensive", help="Verification level")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text", 
                       help="Output format")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation (useful for testing or when deps are already installed)")
    
    args = parser.parse_args()
    
    result = evaluate_repository(args.repo_path, args.requirement, args.level, args.skip_deps)
    
    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print(f"\n{'='*80}")
        print(f"🧠 REPOSITORY EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"📁 Repository: {result['repository_path']}")
        print(f"📋 Requirement: {result['requirement']}")
        print(f"🏆 Status: {result['overall_status']}")
        print(f"📈 Score: {result['overall_score']}/1.0")
        print(f"📝 Description: {result['status_description']}")
        
        # Summary
        summary = result['summary']
        print(f"\n📊 REPOSITORY SUMMARY:")
        print(f"  📄 Total files analyzed: {summary['total_files']}")
        print(f"  🚀 Entry points found: {summary['entry_points_found']}")
        print(f"  ✅ Successful executions: {summary['successful_executions']}/{summary['total_execution_attempts']}")
        print(f"  🔤 Primary language: {summary['primary_language']}")
        print(f"  📏 Repository size: {summary['repository_size_loc']} lines of code")
        
        # Analysis steps
        print(f"\n🔍 EVALUATION PROCESS:")
        for i, step in enumerate(result['analysis_steps'], 1):
            status_icon = "✅" if step['success'] else "❌"
            print(f"  {status_icon} Step {i}: {step['step_name']}")
            print(f"      {step['description']}")
            print(f"      {step['details']}")
            
            if step['findings']:
                print(f"      Findings: {'; '.join(step['findings'][:3])}")
            
            if step['recommendations']:
                print(f"      Recommendations: {'; '.join(step['recommendations'][:2])}")
            print()
        
        # Execution attempts
        if result['execution_attempts']:
            print(f"🧪 EXECUTION ATTEMPTS:")
            for attempt in result['execution_attempts'][:10]:  # Show first 10
                status_icon = "✅" if attempt['success'] else "❌"
                file_name = Path(attempt['file_path']).name
                print(f"  {status_icon} {file_name}: {attempt['error_analysis'] if not attempt['success'] else 'Success'}")
                if attempt['stdout'] and len(attempt['stdout'].strip()) > 0:
                    print(f"      Output: {attempt['stdout']}")
                if not attempt['success'] and attempt['stderr']:
                    print(f"      Error: {attempt['stderr']}")
            print()
        
        # AI Analysis results
        if 'ai_analysis' in result and result['ai_analysis']['enabled']:
            print(f"🤖 AI ANALYSIS RESULTS:")
            print(f"  📊 Total analyses: {result['ai_analysis']['total_analyses']}")
            print(f"  🎯 Average confidence: {result['ai_analysis']['average_confidence']}")
            print(f"  🏆 AI bonus score: {result['enhanced_analysis']['ai_bonus_score']}")
            print()
            
            for analysis in result['ai_analysis']['analyses']:
                print(f"  🔍 {analysis['type'].replace('_', ' ').title()}:")
                print(f"      Confidence: {analysis['confidence']:.2f}")
                if analysis['suggestions']:
                    print(f"      Key suggestions:")
                    for suggestion in analysis['suggestions'][:3]:
                        print(f"        • {suggestion}")
                if analysis['content']:
                    # content_preview = analysis['content'][:200] + "..." if len(analysis['content']) > 200 else analysis['content']
                    content_preview = analysis['content']
                    print(f"      Analysis: {content_preview}")
                print()
        
        # Enhanced analysis data
        if 'enhanced_analysis' in result:
            print(f"📋 ENHANCED ANALYSIS DATA:")
            if result['enhanced_analysis']['repo_map']:
                print(f"  📁 Repository structure map: {len(result['enhanced_analysis']['repo_map'].split(chr(10)))} lines")
            if result['enhanced_analysis']['project_context']:
                print(f"  📄 Project context: {len(result['enhanced_analysis']['project_context'])} characters")
            if result['enhanced_analysis']['codebase_summary']:
                print(f"  📊 Codebase summary: {len(result['enhanced_analysis']['codebase_summary'])} characters")
            print()
        
        # Dependencies
        missing_deps = [name for name, available in result['dependencies'].items() if not available]
        if missing_deps:
            print(f"📦 MISSING DEPENDENCIES:")
            for dep in missing_deps[:10]:
                print(f"  ❌ {dep}: pip install {dep}")
            print()
        
        # Entry points
        if result['entry_points']:
            print(f"🚀 IDENTIFIED ENTRY POINTS:")
            for ep in result['entry_points'][:5]:
                exec_icon = "⚡" if ep['is_executable'] else "📄"
                main_icon = "🎯" if ep['has_main'] else ""
                print(f"  {exec_icon}{main_icon} {Path(ep['path']).name}")
                print(f"      Functions: {len(ep['functions'])}, Classes: {len(ep['classes'])}")
            print()



if __name__ == "__main__":
    main()

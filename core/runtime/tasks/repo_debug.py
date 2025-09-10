#!/usr/bin/env python3
"""
Repository Debug Tool - AI-Powered Code Fixing

This tool uses refact agent to automatically fix code issues in repositories.
It calls the refact agent CLI to analyze and fix syntax errors, bugs, and other code problems.

Usage:
    python repo_debug.py <repo_path> "<action_description>"
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)




@dataclass
class DebugResult:
    """Result of a debug operation"""
    success: bool
    output: str
    error_message: str = ""
    fixed_files: List[str] = None
    suggestions: List[str] = None
    execution_time: float = 0.0
    summary: str = ""
    
    def __post_init__(self):
        if self.fixed_files is None:
            self.fixed_files = []
        if self.suggestions is None:
            self.suggestions = []


class RepoDebugger:
    """Debugger for repositories using refact agent"""
    
    def __init__(self, repo_path: Path, action_description: str):
        self.repo_path = repo_path
        self.action_description = action_description
        
        # Refact agent configuration - use relative path
        self.refact_cli_path = self._get_refact_cli_path()
        
        # Validate paths
        self._validate_paths()
    
    def _get_refact_cli_path(self) -> str:
        """Get the path to refact agent CLI"""
        # Always use the core directory path
        return os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "thirdparty", 
            "refact_agent", 
            "engine", 
            "python_binding_and_cmdline", 
            "refact", 
            "cli_standalone.py"
        )
    
    def _validate_paths(self):
        """Validate that required paths exist"""
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        # Check if cli_standalone.py exists in the expected location
        if not Path(self.refact_cli_path).exists():
            raise ValueError(f"Refact CLI path does not exist: {self.refact_cli_path}")
            logger.error(f"Refact CLI path not found: {self.refact_cli_path}")
            logger.error(f"Current directory: {Path.cwd()}")
            logger.error(f"Current file: {__file__}")
    
    def debug_repository(self) -> DebugResult:
        """Debug the repository using refact agent"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting debug operation for repository: {self.repo_path}")
            logger.info(f"Action description: {self.action_description}")
            
            # Prepare the command
            command = self._prepare_command()
            
            # Execute the command
            result = self._execute_command(command)
            
            # Parse the result
            debug_result = self._parse_result(result)
            print('debug_result', debug_result)
            
            debug_result.execution_time = time.time() - start_time
            print(f'execution_time set: {debug_result.execution_time}')
            
            # Generate summary after execution time is set
            logger.info("Generating LLM summary for debug operation...")
            print("About to generate summary...")
            try:
                debug_result.summary = self._generate_summary(debug_result)
                print(f"Summary generated successfully, length: {len(debug_result.summary)}")
                logger.info(f"Generated summary length: {len(debug_result.summary)} characters")
            except Exception as e:
                print(f"Error generating summary: {e}")
                logger.error(f"Error generating summary: {e}")
                debug_result.summary = f"Failed to generate summary: {e}"
            
            logger.info(f"Debug operation completed in {debug_result.execution_time:.2f} seconds")
            return debug_result
            
        except Exception as e:
            logger.error(f"Error during debug operation: {str(e)}")
            debug_result = DebugResult(
                success=False,
                output="",
                error_message=f"Debug operation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
            # Generate summary for error case
            debug_result.summary = self._generate_summary(debug_result)
            return debug_result
    
    def _prepare_command(self) -> List[str]:
        """Prepare the command to execute refact agent"""
        command = [
            "python",
            self.refact_cli_path,
            str(self.repo_path),
            "--",
            "--",
            self.action_description
        ]
        
        logger.info(f"Prepared command: {' '.join(command)}")
        return command
    
    def _execute_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Execute the refact agent command"""
        logger.info("Executing refact agent command...")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd="/app_sci",  # Run from app_sci directory to avoid path issues
                timeout=300  # 5 minute timeout
            )
            
            logger.info(f"Command completed with return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error("Command timed out after 5 minutes")
            raise Exception("Debug operation timed out")
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            raise
    
    def _parse_result(self, result: subprocess.CompletedProcess) -> DebugResult:
        """Parse the result from refact agent"""
        success = result.returncode == 0
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        
        # Try to extract fixed files and suggestions from output
        fixed_files = self._extract_fixed_files(output)
        suggestions = self._extract_suggestions(output)
        
        error_message = ""
        if not success:
            error_message = f"Refact agent failed with return code {result.returncode}"
            if result.stderr:
                error_message += f": {result.stderr}"
        
        # Create debug result
        debug_result = DebugResult(
            success=success,
            output=output,
            error_message=error_message,
            fixed_files=fixed_files,
            suggestions=suggestions
        )
        
        return debug_result
    
    def _extract_fixed_files(self, output: str) -> List[str]:
        """Extract list of fixed files from output"""
        fixed_files = []
        
        # Look for patterns indicating file modifications
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            # Look for common patterns in refact agent output
            if any(pattern in line.lower() for pattern in [
                'modified', 'fixed', 'updated', 'changed', '.py', '.js', '.java'
            ]):
                # Try to extract file path
                if '/' in line or '\\' in line:
                    # Extract the file path
                    parts = line.split()
                    for part in parts:
                        if any(ext in part for ext in ['.py', '.js', '.java', '.cpp', '.c', '.h']):
                            fixed_files.append(part)
                            break
        
        return list(set(fixed_files))  # Remove duplicates
    
    def _extract_suggestions(self, output: str) -> List[str]:
        """Extract suggestions from output"""
        suggestions = []
        
        # Look for suggestion patterns
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if any(pattern in line.lower() for pattern in [
                'suggestion:', 'recommendation:', 'tip:', 'note:', 'consider:'
            ]):
                suggestions.append(line)
        
        return suggestions
    

    
    def _generate_summary(self, result: DebugResult) -> str:
        """Generate a summary of the debug operation using LLM"""
        try:
            print("Starting LLM summary generation...")
            import openai
            
            # Check if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("WARNING: OPENAI_API_KEY not found in environment variables")
                raise Exception("OPENAI_API_KEY environment variable not set")
            
            print(f"API key found, length: {len(api_key)}")
            
            # Prepare the context for LLM summarization
            context = f"""
Repository Debug Operation Summary Request:

Repository Path: {self.repo_path}
Action Description: {self.action_description}
Execution Time: {result.execution_time:.2f} seconds
Success: {result.success}

Raw Output from Refact Agent:
{result.output}

Fixed Files: {result.fixed_files}
Suggestions: {result.suggestions}
Error Message: {result.error_message}

Please provide a comprehensive summary of this debug operation including:
1. What was the main goal of the debug operation?
2. What tools were used and what did they accomplish?
3. What files were modified or fixed?
4. What suggestions were provided?
5. What was the overall outcome?
6. Any important insights or recommendations?

Format the summary in a clear, structured way with appropriate sections and bullet points.
            """
            
            print("Sending request to OpenAI API...")
            # Use OpenAI API to generate summary with GPT-4o (new API format)
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes repository debug operations. Provide clear, concise summaries that highlight the key actions taken and outcomes achieved."},
                    {"role": "user", "content": context}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            print(f"Received response from OpenAI API, summary length: {len(summary)}")
            return summary
            
        except Exception as e:
            print(f"Exception in _generate_summary: {e}")
            logger.warning(f"Failed to generate LLM summary: {e}")
            # Fallback to basic summary
            return f"Debug operation {'completed successfully' if result.success else 'failed'}. Execution time: {result.execution_time:.2f} seconds. Files fixed: {len(result.fixed_files)}. Suggestions: {len(result.suggestions)}."


def debug_repository(repo_path: str, action_description: str) -> Dict[str, Any]:
    """Debug a repository using refact agent
    
    Args:
        repo_path: Path to the repository to debug
        action_description: For errors: paste exact error message. For editing: describe what to add/modify
        
    Returns:
        Dictionary containing debug results
    """
    try:
        # Create debugger instance
        debugger = RepoDebugger(Path(repo_path), action_description)
        
        # Perform debug operation
        result = debugger.debug_repository()
        
        # Convert to dictionary
        return asdict(result)
        
    except Exception as e:
        logger.error(f"Error in debug_repository: {str(e)}")
        return {
            "success": False,
            "output": "",
            "error_message": f"Debug operation failed: {str(e)}",
            "fixed_files": [],
            "suggestions": [],
            "execution_time": 0.0,
            "summary": ""
        }


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Debug repository using refact agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python repo_debug.py /path/to/repo "Fix syntax errors in model.py"
    python repo_debug.py /path/to/repo "Fix indentation issues in all Python files"
    python repo_debug.py /path/to/repo "Optimize code performance and fix memory leaks"
        """
    )
    
    parser.add_argument(
        "repo_path",
        help="Path to the repository to debug"
    )
    
    parser.add_argument(
        "action_description",
        help="For errors: paste exact error message. For editing: describe what to add/modify"
    )
    
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Perform debug operation
    result = debug_repository(args.repo_path, args.action_description)
    
    # Output results
    if args.output == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("🔧 Repository Debug Results")
        print("=" * 50)
        print(f"Success: {result['success']}")
        print(f"Execution Time: {result['execution_time']:.2f} seconds")
        
        if result['error_message']:
            print(f"\n❌ Error: {result['error_message']}")
        
        if result['fixed_files']:
            print(f"\n📝 Fixed Files ({len(result['fixed_files'])}):")
            for file in result['fixed_files']:
                print(f"  - {file}")
        
        if result['suggestions']:
            print(f"\n💡 Suggestions ({len(result['suggestions'])}):")
            for suggestion in result['suggestions']:
                print(f"  - {suggestion}")
        
        if result['output']:
            print(f"\n📋 Output:")
            print(result['output'])


class RepoDebugTask:
    """Task class for repository debugging operations"""
    
    def __init__(self, repo_path: str, action_description: str):
        self.repo_path = repo_path
        self.action_description = action_description
    
    async def run(self) -> DebugResult:
        """Run the debug task"""
        try:
            # Create debugger instance
            debugger = RepoDebugger(Path(self.repo_path), self.action_description)
            
            # Perform debug operation
            result = debugger.debug_repository()
            
            # Generate summary using LLM
            result.summary = debugger._generate_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RepoDebugTask.run: {str(e)}")
            return DebugResult(
                success=False,
                output="",
                error_message=f"Debug operation failed: {str(e)}",
                fixed_files=[],
                suggestions=[],
                execution_time=0.0,
                summary=f"Debug operation failed: {str(e)}"
            )


if __name__ == "__main__":
    main()

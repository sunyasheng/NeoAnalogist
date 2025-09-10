# This script integrates all functionality from debug/paper2code codes into a single file
# Includes: 1_planning.py, 1.1_extract_config.py, 2_analyzing.py, and 3_coding.py
# Source: https://github.com/going-doer/Paper2Code
# License: MIT License

from litellm import completion
import json
import os
import sys
import re
import shutil
import copy
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class RepoCreate:
    # def __init__(self, paper_path: str, gpt_version: str = "gpt-4o", output_dir: str = "", output_repo_dir: str = ""):
    # def __init__(self, paper_path: str, gpt_version: str = "anthropic/claude-3-7-sonnet-20250219", output_dir: str = "", output_repo_dir: str = ""):
    def __init__(self, paper_path: str, gpt_version: str = "openai/o3-mini", output_dir: str = "", output_repo_dir: str = ""):
        self.paper_path = paper_path
        self.gpt_version = gpt_version
        
        # Handle directory structure based on user requirements
        # Now output_repo_dir is always required, so use it as the primary target
        if output_repo_dir:
            self.output_repo_dir = output_repo_dir
            self.is_temp_dir = False
            
            # If output_dir is provided, use it for intermediate results
            if output_dir:
                self.output_dir = output_dir
            else:
                # Create intermediate results directory parallel to the repository
                # If output_repo_dir ends with 'submission', create parallel 'intermediate_results'
                if output_repo_dir.rstrip('/').endswith('submission'):
                    parent_dir = os.path.dirname(output_repo_dir.rstrip('/'))
                    self.output_dir = os.path.join(parent_dir, "intermediate_results")
                else:
                    # Otherwise, create 'intermediate_results' as sibling directory
                    parent_dir = os.path.dirname(output_repo_dir.rstrip('/'))
                    self.output_dir = os.path.join(parent_dir, "intermediate_results")
        else:
            # Fallback: use temp directory if output_repo_dir is somehow empty (shouldn't happen now)
            self.base_dir = tempfile.mkdtemp(prefix="repo_create_")
            self.output_dir = os.path.join(self.base_dir, "intermediate_results")
            self.output_repo_dir = os.path.join(self.base_dir, "submission")
            self.is_temp_dir = True
            print(f"[INFO] Using temporary directory: {self.base_dir}")
        
        # LiteLLM routes by model name; ensure likely API key exists for the chosen model
        model_lower = (self.gpt_version or "").lower()
        if "claude" in model_lower:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("[WARN] ANTHROPIC_API_KEY not set; LiteLLM may fail when calling Claude models.")
        elif model_lower.startswith("gpt-") or model_lower.startswith("o3"):
            if not os.environ.get("OPENAI_API_KEY"):
                print("[WARN] OPENAI_API_KEY not set; LiteLLM may fail when calling OpenAI models.")
        # No explicit client needed with LiteLLM
        self.responses = []
        self.trajectories = []
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_repo_dir, exist_ok=True)
        
        print(f"[INFO] Intermediate results will be saved to: {self.output_dir}")
        print(f"[INFO] Generated code will be saved to: {self.output_repo_dir}")

    def _load_paper_content(self) -> str:
        """Load paper content from the specified path."""
        with open(self.paper_path) as f:
            return f.read()

    # =============================================================================
    # UTILITY FUNCTIONS (from utils.py)
    # =============================================================================
    
    def extract_planning(self, trajectories_json_file_path):
        """Extract planning content from trajectories."""
        with open(trajectories_json_file_path) as f:
            traj = json.load(f)

        context_lst = []
        for turn in traj:
            if turn['role'] == 'assistant':
                content = turn['content']
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                context_lst.append(content)

        context_lst = context_lst[:3] 
        return context_lst

    def content_to_json(self, data):
        """Convert content to JSON with multiple fallback strategies."""
        clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
        clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)
        clean_data = re.sub(r',\s*\]', ']', clean_data)
        clean_data = re.sub(r'\n\s*', '', clean_data)

        # Try multiple parsing strategies
        for attempt in range(1, 5):
            try:
                if attempt == 1:
                    json_data = json.loads(clean_data)
                elif attempt == 2:
                    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)
                    json_data = json.loads(clean_data)
                elif attempt == 3:
                    clean_data = re.sub(r'"""', '"', clean_data)
                    clean_data = re.sub(r"'''", "'", clean_data) 
                    clean_data = re.sub(r"\\", "'", clean_data)
                    json_data = json.loads(f"""{clean_data}""")
                else:
                    # Extract Logic Analysis and Task list specifically
                    pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])'
                    match = re.search(pattern, data)
                    if match:
                        logic_analysis = json.loads(match.group(1))
                        task_list = json.loads(match.group(2))
                        json_data = {
                            "Logic Analysis": logic_analysis,
                            "Task list": task_list
                        }
                    else:
                        json_data = {}
                return json_data
            except json.JSONDecodeError:
                continue
        
        return {}

    def format_json_data(self, data):
        """Format JSON data for output."""
        formatted_text = ""
        for key, value in data.items():
            formatted_text += "-" * 40 + "\n"
            formatted_text += "[" + key + "]\n"
            if isinstance(value, list):
                for item in value:
                    formatted_text += f"- {item}\n"
            else:
                formatted_text += str(value) + "\n"
            formatted_text += "\n"
        return formatted_text

    def extract_code_from_content(self, content):
        """Extract code from content."""
        pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
        code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        if len(code) == 0:
            return ""
        else:
            return code[0]

    def print_response(self, completion_json, is_llm=False):
        """Print response content."""
        if not is_llm:
            content = completion_json['choices'][0]['message']['content']
            print(content)
        else:
            print(completion_json)

    # =============================================================================
    # MESSAGE CREATION FUNCTIONS (from 1_planning.py)
    # =============================================================================

    def _create_plan_message(self, paper_content: str) -> List[Dict[str, str]]:
        """Create the initial planning message."""
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

    def _create_file_list_message(self) -> List[Dict[str, str]]:
        """Create the file list message."""
        return [
            {"role": "user", "content": """Your goal is to create a concise, usable, and complete software system design for reproducing the paper's method. Use appropriate open-source libraries and keep the overall architecture simple.
             
Based on the plan for reproducing the paper's main method, please design a concise, usable, and complete software system. 
Keep the architecture simple and make effective use of open-source libraries.

-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will ... ,
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "model.py",  
        "trainer.py",
        "evaluation.py" 
    ],
    "Data structures and interfaces": "\nclassDiagram\n    class Main {\n        +__init__()\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(config: dict)\n        +load_data() -> Any\n    }\n    class Model {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n    }\n    class Trainer {\n        +__init__(model: Model, data: Any)\n        +train() -> None\n    }\n    class Evaluation {\n        +__init__(model: Model, data: Any)\n        +evaluate() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> Trainer\n    Main --> Evaluation\n    Trainer --> Model\n",
    "Program call flow": "\nsequenceDiagram\n    participant M as Main\n    participant DL as DatasetLoader\n    participant MD as Model\n    participant TR as Trainer\n    participant EV as Evaluation\n    M->>DL: load_data()\n    DL-->>M: return dataset\n    M->>MD: initialize model()\n    M->>TR: train(model, dataset)\n    TR->>MD: forward(x)\n    MD-->>TR: predictions\n    TR-->>M: training complete\n    M->>EV: evaluate(model, dataset)\n    EV->>MD: forward(x)\n    MD-->>EV: predictions\n    EV-->>M: metrics\n",
    "Anything UNCLEAR": "Need clarification on the exact dataset format and any specialized hyperparameters."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Implementation approach: <class 'str'>  # Summarize the chosen solution strategy.
- File list: typing.List[str]  # Only need relative paths. ALWAYS write a main.py or app.py here.
- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax, including classes, method(__init__ etc.) and functions with type annotations, CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design.
- Program call flow: typing.Optional[str] # Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.
- Anything UNCLEAR: <class 'str'>  # Mention ambiguities and ask for clarifications.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the instructions for the nodes, generate the output, and ensure it follows the format example."""}
        ]

    def _create_task_list_message(self) -> List[Dict[str, str]]:
        """Create the task list message."""
        return [
            {'role': 'user', 'content': """Your goal is break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies. 
You will break down tasks, analyze dependencies.
             
You outline a clear PRD/technical design for reproducing the paper's method and experiments. 

Now, let's break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist in writing the code needed to reproduce the paper.

IMPORTANT: You must include ALL necessary files for a complete, runnable repository:
- Core implementation files (main.py, model.py, etc.)
- Environment setup files (requirements.txt, setup.py)
- Data handling files (dataset_loader.py)
- Documentation files (README.md)
- Reproduction scripts (reproduce.sh)
- Configuration files (config.yaml will be generated separately in the next step)

-----

## Format Example
[CONTENT]
{
            "Required packages": [
            "numpy>=1.26.4",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "tqdm>=4.65.0"
        ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "requirements.txt",
            "Complete dependency list with specific versions for reproducibility"
        ],
        [
            "setup.py",
            "Python package setup for proper installation"
        ],
        [
            "README.md",
            "Comprehensive documentation with installation and usage instructions"
        ],
        [
            "dataset_loader.py",
            "Handles loading and preprocessing of datasets"
        ],
        [
            "model.py",
            "Defines the model architecture and forward pass"
        ],
        [
            "trainer.py",
            "Training loop and optimization logic"
        ],
        [
            "evaluation.py",
            "Evaluation metrics and testing procedures"
        ],
        [
            "main.py",
            "Entry point that orchestrates the entire pipeline"
        ],
        [
            "reproduce.sh",
            "Complete reproduction script with environment setup, data download, and experiment execution"
        ]
    ],
    "Task list": [
        "requirements.txt",
        "setup.py", 
        "README.md",
        "dataset_loader.py", 
        "model.py",  
        "trainer.py", 
        "evaluation.py",
        "main.py",
        "reproduce.sh"
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "All modules share common configuration from config.yaml. Data preprocessing is handled by dataset_loader.py and used by trainer.py and evaluation.py.",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}

[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format.(e.g., 'numpy>=1.26.4'). Include ALL dependencies needed for the complete pipeline. Use >= for version flexibility and Python 3.12 compatibility.
- Required Other language third-party packages: typing.List[str]  # List down packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible. MUST include environment setup, data handling, and reproduction files. NOTE: config.yaml will be generated separately.
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list PLUS environment setup, data handling, and reproduction files. NOTE: config.yaml will be generated separately.
- Full API spec: <class 'str'>  # Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end and back-end communication is not required, leave it blank.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common utility functions or configuration variables.
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions or clarifications needed from the paper or project scope.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example."""}
        ]

    def _create_config_message(self) -> List[Dict[str, str]]:
        """Create the configuration message."""
        return [
            {'role': 'user', 'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, plan, design specified previously, follow the "Format Example" and generate the code. 
Extract the training details from the above paper (e.g., learning rate, batch size, epochs, etc.), follow the "Format example" and generate the code. 
DO NOT FABRICATE DETAILS — only use what the paper provides.

You must write `config.yaml`.

ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Your output format must follow the example below exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: ...
  batch_size: ...
  epochs: ...
...
```

-----

## Code: config.yaml
"""}
        ]

    # =============================================================================
    # ANALYSIS FUNCTIONS (from 2_analyzing.py)
    # =============================================================================

    def get_write_msg(self, todo_file_name, todo_file_desc, paper_content, context_lst, config_yaml):
        """Create analysis message for a specific file."""
        draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
        if len(todo_file_desc.strip()) == 0:
            draft_desc = f"Write the logic analysis in '{todo_file_name}'."

        write_msg=[{'role': 'user', "content": f"""## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml). 
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""}]
        return write_msg

    # =============================================================================
    # CODING FUNCTIONS (from 3_coding.py)
    # =============================================================================

    def get_coding_msg(self, todo_file_name, detailed_logic_analysis, done_file_lst, done_file_dict, paper_content, context_lst, config_yaml):
        """Create coding message for a specific file."""
        code_files = ""
        for done_file in done_file_lst:
            if done_file.endswith(".yaml"): 
                continue
            if done_file in done_file_dict:
                code_files += f"""
```python
{done_file_dict[done_file]}
```

"""

        # Enhanced instructions for different file types
        enhanced_instructions = ""
        if todo_file_name == "reproduce.sh":
            enhanced_instructions = """
SPECIAL INSTRUCTIONS FOR reproduce.sh:
- Do NOT use venv, virtualenv.
- Do NOT use tee or write any log file. All output should go to stdout/stderr.
- Do NOT add network checks, dependency verification, or complex logging functions.
- Only include the essential steps: install dependencies, prepare data, run main experiment.
- Example:
#!/bin/bash
set -e
python3 -m pip install -r requirements.txt
python3 main.py
echo "Experiment completed."
"""
        elif todo_file_name == "requirements.txt":
            enhanced_instructions = """
SPECIAL INSTRUCTIONS FOR requirements.txt:
1. Include ALL necessary dependencies with specific versions
2. Include both direct and indirect dependencies
3. Specify compatible versions to avoid conflicts
4. Include development dependencies if needed
5. Add comments explaining why each package is needed
6. IMPORTANT: Use >= for version flexibility and ensure Python 3.12 compatibility
7. Avoid old versions that require compilation (e.g., use numpy>=1.26.4 instead of numpy==1.21.0)
8. Prefer pre-compiled binary packages when possible
"""
        elif todo_file_name == "README.md":
            enhanced_instructions = """
SPECIAL INSTRUCTIONS FOR README.md:
1. Create a comprehensive README with:
   - Paper summary and methodology
   - Installation instructions
   - Data preparation steps
   - Usage examples
   - Expected outputs and results
   - Troubleshooting guide
   - Citation information
2. Make it clear and user-friendly
3. Include all necessary information for reproduction
"""
        elif todo_file_name == "setup.py":
            enhanced_instructions = """
SPECIAL INSTRUCTIONS FOR setup.py:
1. Create a proper Python package setup
2. Include all dependencies and metadata
3. Add proper versioning and description
4. Include entry points if needed
5. Make it installable via pip
"""
        # Note: config.yaml is handled by the existing _create_config_message() method
        # and should not be overridden here to avoid conflicts

        write_msg=[
            {'role': 'user', "content": f"""# Context
## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Code Files
{code_files}

-----

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on the paper, plan, design, task and configuration file(config.yaml) specified previously, follow "Format example", write the code. 

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
8. REFER TO CONFIGURATION: you must use configuration from "config.yaml". DO NOT FABRICATE any configuration values.

{enhanced_instructions}

{detailed_logic_analysis}

## Code: {todo_file_name}"""}]
        return write_msg

    def _api_call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call LLM via LiteLLM and return a normalized response dict.

        Normalized format:
        {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
        """
        # Strict retry-after handling with fallback exponential backoff
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                extra_kwargs = {}
                try:
                    if "o3-mini" in (self.gpt_version or "").lower():
                        # OpenAI o3-mini expects reasoning_effort, not reasoning{}
                        extra_kwargs["reasoning_effort"] = "high"
                except Exception:
                    pass
                resp = completion(
                    model=self.gpt_version,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=16384,
                    **extra_kwargs
                )
                break
            except Exception as exc:
                # Read retry-after header if provided
                wait_seconds = None
                headers = getattr(exc, "headers", None)
                if isinstance(headers, dict):
                    wait_seconds = headers.get("retry-after") or headers.get("Retry-After")
                if wait_seconds is None:
                    response = getattr(exc, "response", None)
                    h2 = getattr(response, "headers", None)
                    if isinstance(h2, dict):
                        wait_seconds = h2.get("retry-after") or h2.get("Retry-After")
                # Try Anthropic reset headers (RFC3339) to compute exact wait
                reset_wait = None
                try:
                    from datetime import datetime, timezone
                    def _parse_reset(val: str):
                        if not val:
                            return None
                        vs = str(val)
                        # normalize trailing Z
                        if vs.endswith('Z'):
                            vs = vs.replace('Z', '+00:00')
                        try:
                            dt = datetime.fromisoformat(vs)
                            now = datetime.now(timezone.utc)
                            delta = (dt - now).total_seconds()
                            return delta if delta and delta > 0 else None
                        except Exception:
                            return None
                    for cand in (headers, h2):
                        if isinstance(cand, dict):
                            reset_wait = _parse_reset(cand.get('anthropic-ratelimit-tokens-reset'))
                            if reset_wait is None:
                                reset_wait = _parse_reset(cand.get('anthropic-ratelimit-requests-reset'))
                            if reset_wait is not None:
                                break
                except Exception:
                    reset_wait = None
                try:
                    wait_seconds = float(wait_seconds) if wait_seconds is not None else None
                except Exception:
                    wait_seconds = None

                if attempt < max_retries - 1:
                    import time as _t
                    used_wait = (
                        float(wait_seconds) if wait_seconds is not None
                        else (float(reset_wait) if reset_wait is not None else float(backoff))
                    )
                    # small jitter to avoid synchronized retries
                    try:
                        import random as _rand
                        used_wait = max(0.0, used_wait * (1.0 + _rand.uniform(-0.1, 0.1)))
                    except Exception:
                        pass
                    # Print how long we will wait before retrying
                    if wait_seconds is not None:
                        print(f"[RATE-LIMIT] retry-after detected. Sleeping {used_wait:.2f}s before retry (attempt {attempt+1}/{max_retries}).")
                    elif reset_wait is not None:
                        print(f"[RATE-LIMIT] using reset header. Sleeping {used_wait:.2f}s before retry (attempt {attempt+1}/{max_retries}).")
                    else:
                        print(f"[RATE-LIMIT] no retry-after header. Backing off {used_wait:.2f}s before retry (attempt {attempt+1}/{max_retries}).")
                    _t.sleep(used_wait)
                    backoff *= 2
                    continue
                raise
        # Extract text content robustly across providers
        assistant_text = ""
        assistant_role = "assistant"
        try:
            first_choice = resp.choices[0]
            msg = getattr(first_choice, "message", None)
            if msg is None and isinstance(first_choice, dict):
                msg = first_choice.get("message")
            if isinstance(msg, dict):
                assistant_text = msg.get("content", "") or ""
                assistant_role = msg.get("role", "assistant") or "assistant"
            else:
                assistant_text = getattr(msg, "content", "") or ""
                assistant_role = getattr(msg, "role", "assistant") or "assistant"
        except Exception:
            try:
                # Fallback: OpenAI-like dict
                assistant_text = resp["choices"][0]["message"]["content"]
                assistant_role = resp["choices"][0]["message"].get("role", "assistant")
            except Exception:
                assistant_text = str(resp)
                assistant_role = "assistant"

        return {
            "choices": [
                {"message": {"role": assistant_role, "content": assistant_text}}
            ]
        }

    # =============================================================================
    # CONFIG EXTRACTION FUNCTIONS (from 1.1_extract_config.py)
    # =============================================================================

    def extract_config(self):
        """Extract configuration from planning trajectories."""
        trajectories_file = f'{self.output_dir}/planning_trajectories.json'
        if not os.path.exists(trajectories_file):
            print(f"[ERROR] Planning trajectories file not found: {trajectories_file}")
            return

        with open(trajectories_file, encoding='utf8') as f:
            traj = json.load(f)

        yaml_raw_content = ""
        for turn_idx, turn in enumerate(traj):
            if turn_idx == 8:  # Config is typically in turn 8
                yaml_raw_content = turn['content']   

        if "</think>" in yaml_raw_content:
            yaml_raw_content = yaml_raw_content.split("</think>")[-1]

        match = re.search(r"```yaml\n(.*?)\n```", yaml_raw_content, re.DOTALL)
        if match:
            yaml_content = match.group(1)
            with open(f'{self.output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
                f.write(yaml_content)
        else:
            match2 = re.search(r"```yaml\\n(.*?)\\n```", yaml_raw_content, re.DOTALL)
            if match2:
                yaml_content = match2.group(1)
                with open(f'{self.output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
                    f.write(yaml_content)
            else:
                print("No YAML content found.")

        # Create planning artifacts
        artifact_output_dir = f"{self.output_dir}/planning_artifacts"
        os.makedirs(artifact_output_dir, exist_ok=True)

        context_lst = self.extract_planning(trajectories_file)
        
        if len(context_lst) >= 2:
            arch_design = self.content_to_json(context_lst[1])
            formatted_arch_design = self.format_json_data(arch_design)

            with open(f"{artifact_output_dir}/1.2_arch_design.txt", "w", encoding="utf-8") as f:
                f.write(formatted_arch_design)

        if len(context_lst) >= 3:
            logic_design = self.content_to_json(context_lst[2])
            formatted_logic_design = self.format_json_data(logic_design)

            with open(f"{artifact_output_dir}/1.3_logic_design.txt", "w", encoding="utf-8") as f:
                f.write(formatted_logic_design)

        if len(context_lst) >= 1:
            with open(f"{artifact_output_dir}/1.1_overall_plan.txt", "w", encoding="utf-8") as f:
                f.write(context_lst[0])

        if os.path.exists(f"{self.output_dir}/planning_config.yaml"):
            shutil.copy(f"{self.output_dir}/planning_config.yaml", f"{artifact_output_dir}/1.4_config.yaml")

    # =============================================================================
    # MAIN EXECUTION FUNCTIONS
    # =============================================================================

    def execute_planning(self):
        """Execute the planning process (equivalent to 1_planning.py)."""
        print("[PLANNING] Starting planning process...")
        
        # Load paper content
        paper_content = self._load_paper_content()

        # Define planning stages
        stages = [
            ("[PLANNING] Overall plan", lambda: self._create_plan_message(paper_content)),
            ("[PLANNING] Architecture design", self._create_file_list_message),
            ("[PLANNING] Logic design", self._create_task_list_message),
            ("[PLANNING] Configuration file generation", self._create_config_message)
        ]

        # Execute each planning stage
        for stage_name, message_creator in stages:
            print(stage_name)
            
            # Get messages for current stage
            messages = message_creator()
            self.trajectories.extend(messages)

            # Make API call
            resp = self._api_call(self.trajectories)

            # Process response (normalized dict)
            self.responses.append(resp)

            # Update trajectories
            message_dict = resp['choices'][0]['message']
            self.trajectories.append({'role': message_dict.get('role', 'assistant'), 'content': message_dict.get('content', '')})

            # Print response
            self.print_response(resp)

        # Save planning trajectories
        with open(f'{self.output_dir}/planning_trajectories.json', 'w') as f:
            json.dump(self.trajectories, f, indent=2)
        
        print(f"[PLANNING] Planning completed. Results saved to {self.output_dir}")

        return self.responses

    def execute_analysis(self):
        """Execute the analysis process (equivalent to 2_analyzing.py)."""
        print("[ANALYSIS] Starting analysis process...")
        
        # Load paper content
        paper_content = self._load_paper_content()

        # Load config and planning results
        config_file = f'{self.output_dir}/planning_config.yaml'
        if not os.path.exists(config_file):
            print(f"[ERROR] Config file not found: {config_file}")
            return

        with open(config_file) as f: 
            config_yaml = f.read()

        context_lst = self.extract_planning(f'{self.output_dir}/planning_trajectories.json')

        # Load or extract task list
        task_list_file = f'{self.output_dir}/task_list.json'
        if os.path.exists(task_list_file):
            with open(task_list_file) as f:
                task_list = json.load(f)
        else:
            if len(context_lst) >= 3:
                task_list = self.content_to_json(context_lst[2])
            else:
                print("[ERROR] Unable to load task list.")
                return

        # Extract task and logic analysis lists
        todo_file_lst = None
        logic_analysis = None
        
        for key in ['Task list', 'task_list', 'task list']:
            if key in task_list:
                todo_file_lst = task_list[key]
                break
                
        for key in ['Logic Analysis', 'logic_analysis', 'logic analysis']:
            if key in task_list:
                logic_analysis = task_list[key]
                break

        if not todo_file_lst:
            print("[ERROR] 'Task list' does not exist. Please re-generate the planning.")
            return

        if not logic_analysis:
            print("[ERROR] 'Logic Analysis' does not exist. Please re-generate the planning.")
            return

        # Create logic analysis dictionary
        logic_analysis_dict = {}
        for desc in logic_analysis:
            logic_analysis_dict[desc[0]] = desc[1]

        # Create analysis system message
        analysis_msg = [
            {"role": "system", "content": """You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in markdown format, an overview of the plan, a design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 

Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies described in the research paper. 
This analysis must align precisely with the paper's methodology, experimental setup, and evaluation criteria.

1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.
3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values—only use configurations explicitly provided.
"""}]

        # Create analysis artifacts directory
        artifact_output_dir = f'{self.output_dir}/analyzing_artifacts'
        os.makedirs(artifact_output_dir, exist_ok=True)
        
        # Analyze each file
        for todo_file_name in tqdm(todo_file_lst):
            if todo_file_name == "config.yaml":
                continue
                
            current_stage = f"[ANALYSIS] {todo_file_name}"
            print(current_stage)
            
            responses = []
            trajectories = copy.deepcopy(analysis_msg)
            
            if todo_file_name not in logic_analysis_dict:
                logic_analysis_dict[todo_file_name] = ""
                
            instruction_msg = self.get_write_msg(
                todo_file_name, 
                logic_analysis_dict[todo_file_name], 
                paper_content, 
                context_lst, 
                config_yaml
            )
            trajectories.extend(instruction_msg)
                
            resp = self._api_call(trajectories)

            # Process response (normalized dict)
            responses.append(resp)
            self.responses.append(resp)  # Add to main responses list

            # Update trajectories
            message_dict = resp['choices'][0]['message']
            trajectories.append({'role': message_dict.get('role', 'assistant'), 'content': message_dict.get('content', '')})

            # Print response
            self.print_response(resp)

            # Save analysis results
            with open(f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt', 'w') as f:
                f.write(resp['choices'][0]['message']['content'])

            # Save for next stage (coding)
            safe_filename = todo_file_name.replace("/", "_") 
            with open(f'{self.output_dir}/{safe_filename}_simple_analysis_response.json', 'w') as f:
                json.dump(responses, f)

            with open(f'{self.output_dir}/{safe_filename}_simple_analysis_trajectories.json', 'w') as f:
                json.dump(trajectories, f)

        print(f"[ANALYSIS] Analysis completed. Results saved to {artifact_output_dir}")

    def execute_coding(self):
        """Execute the coding process (equivalent to 3_coding.py)."""
        print("[CODING] Starting coding process...")
        
        # Load paper content
        paper_content = self._load_paper_content()

        # Load config and planning results
        config_file = f'{self.output_dir}/planning_config.yaml'
        if not os.path.exists(config_file):
            print(f"[ERROR] Config file not found: {config_file}")
            return

        with open(config_file) as f: 
            config_yaml = f.read()

        context_lst = self.extract_planning(f'{self.output_dir}/planning_trajectories.json')
        task_list = self.content_to_json(context_lst[2])

        todo_file_lst = task_list['Task list']
        done_file_lst = ['config.yaml']
        done_file_dict = {}

        # Create coding system message
        code_msg = [
            {"role": "system", "content": """You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in markdown format, an overview of the plan, a Design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a Task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 
Your task is to write code to reproduce the experiments and methodologies described in the paper. 

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines. 
The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics. 
Write code with triple quoto."""}]

        # Load detailed logic analysis for each file
        detailed_logic_analysis_dict = {}
        for todo_file_name in todo_file_lst:
            if todo_file_name == "config.yaml":
                continue
                
            save_todo_file_name = todo_file_name.replace("/", "_")
            analysis_file = f"{self.output_dir}/{save_todo_file_name}_simple_analysis_response.json"
            
            if os.path.exists(analysis_file):
                with open(analysis_file) as f:
                    detailed_logic_analysis_response = json.load(f)
                detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']
            else:
                detailed_logic_analysis_dict[todo_file_name] = ""

        # Create coding artifacts directory
        artifact_output_dir = f'{self.output_dir}/coding_artifacts'
        os.makedirs(artifact_output_dir, exist_ok=True)

        # Generate code for each file
        for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
            if todo_file_name == "config.yaml":
                continue

            current_stage = f"[CODING] {todo_file_name}"
            print(current_stage)

            responses = []
            trajectories = copy.deepcopy(code_msg)

            instruction_msg = self.get_coding_msg(
                todo_file_name, 
                detailed_logic_analysis_dict[todo_file_name], 
                done_file_lst, 
                done_file_dict, 
                paper_content, 
                context_lst, 
                config_yaml
            )
            trajectories.extend(instruction_msg)

            resp = self._api_call(trajectories)

            # Process response (normalized dict)
            responses.append(resp)
            self.responses.append(resp)  # Add to main responses list

            # Update trajectories
            message_dict = resp['choices'][0]['message']
            trajectories.append({'role': message_dict.get('role', 'assistant'), 'content': message_dict.get('content', '')})

            done_file_lst.append(todo_file_name)

            # Print response
            self.print_response(resp)

            # Save coding artifacts
            save_todo_file_name = todo_file_name.replace("/", "_")
            with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w') as f:
                f.write(resp['choices'][0]['message']['content'])

            # Extract and save code
            code = self.extract_code_from_content(message_dict.get('content', ''))
            if len(code) == 0:
                code = message_dict.get('content', '')

            done_file_dict[todo_file_name] = code
            
            # Create subdirectories if needed
            if save_todo_file_name != todo_file_name:
                todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
                os.makedirs(f"{self.output_repo_dir}/{todo_file_dir}", exist_ok=True)

            # Save code file
            with open(f"{self.output_repo_dir}/{todo_file_name}", 'w') as f:
                f.write(code)

        print(f"[CODING] Coding completed. Generated code saved to {self.output_repo_dir}")

    def execute_full_pipeline(self):
        """Execute the complete pipeline: planning -> config extraction -> analysis -> coding."""
        print("="*60)
        print("STARTING FULL PAPER2CODE PIPELINE WITH CODING")
        print("="*60)
        
        print(f"📁 Intermediate results (planning, analysis) will be saved to: {self.output_dir}")
        print(f"🚀 Generated code will be saved to: {self.output_repo_dir}")
        if self.is_temp_dir:
            print(f"[INFO] Using temporary directory. Copy files if you want to keep them.")
        
        # Step 1: Planning
        self.execute_planning()
        
        # Step 2: Config extraction
        print("\n" + "="*60)
        print("EXTRACTING CONFIGURATION")
        print("="*60)
        self.extract_config()
        
        # Step 3: Analysis
        print("\n" + "="*60)
        print("STARTING ANALYSIS")
        print("="*60)
        self.execute_analysis()
        
        # Step 4: Coding
        print("\n" + "="*60)
        print("STARTING CODING")
        print("="*60)
        self.execute_coding()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Copy config.yaml to the generated repo directory
        config_file = f'{self.output_dir}/planning_config.yaml'
        if os.path.exists(config_file):
            shutil.copy(config_file, f"{self.output_repo_dir}/config.yaml")
            print(f"📋 Configuration file copied to: {self.output_repo_dir}/config.yaml")
        
        print(f"\n✅ 完成! 文件保存位置:")
        print(f"   📁 中间结果: {self.output_dir}")
        print(f"   🚀 生成代码: {self.output_repo_dir}")
        
        return self.responses 
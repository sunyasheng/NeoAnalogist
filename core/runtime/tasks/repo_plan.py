# This script integrates all functionality from debug/paper2code codes into a single file
# Includes: 1_planning.py, 1.1_extract_config.py, 2_analyzing.py, and utils.py
# Source: https://github.com/going-doer/Paper2Code
# License: MIT License

from openai import OpenAI
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

class RepoPlan:
    def __init__(self, paper_path: str, gpt_version: str = "gpt-4o", output_dir: str = ""):
        self.paper_path = paper_path
        self.gpt_version = gpt_version
        
        # Create output directory - use temp dir if not specified
        if output_dir:
            self.output_dir = output_dir
            self.is_temp_dir = False
        else:
            self.output_dir = tempfile.mkdtemp(prefix="repo_plan_")
            self.is_temp_dir = True
            print(f"[INFO] Using temporary directory: {self.output_dir}")
        
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.responses = []
        self.trajectories = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[INFO] Planning results will be saved to: {self.output_dir}")

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

-----

## Format Example
[CONTENT]
{
    "Required packages": [
        "numpy>=1.26.4",
        "torch>=2.0.0"  
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "data_preprocessing.py",
            "DataPreprocessing class ........"
        ],
        [
            "trainer.py",
            "Trainer ....... "
        ],
        [
            "dataset_loader.py",
            "Handles loading and ........"
        ],
        [
            "model.py",
            "Defines the model ......."
        ],
        [
            "evaluation.py",
            "Evaluation class ........ "
        ],
        [
            "main.py",
            "Entry point  ......."
        ]
    ],
    "Task list": [
        "dataset_loader.py", 
        "model.py",  
        "trainer.py", 
        "evaluation.py",
        "main.py"  
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both data_preprocessing.py and trainer.py share ........",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}

[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format.(e.g., 'numpy>=1.26.4'). Use >= for version flexibility and Python 3.12 compatibility.
- Required Other language third-party packages: typing.List[str]  # List down packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible.
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list.
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

    def _api_call(self, messages: List[Dict[str, str]]) -> Any:
        """Make API call to OpenAI."""
        if "o3-mini" in self.gpt_version:
            completion = self.client.chat.completions.create(
                model=self.gpt_version,
                reasoning_effort="high",
                messages=messages
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.gpt_version,
                messages=messages
            )
        return completion

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
        artifact_output_dir = os.path.join(self.output_dir, "planning_artifacts")
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
            shutil.copy(os.path.join(self.output_dir, "planning_config.yaml"), os.path.join(artifact_output_dir, "1.4_config.yaml"))

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
            completion = self._api_call(self.trajectories)
            
            # Process response
            completion_json = json.loads(completion.model_dump_json())
            self.responses.append(completion_json)

            # Update trajectories
            message = completion.choices[0].message
            self.trajectories.append({'role': message.role, 'content': message.content})

            # Print response
            self.print_response(completion_json)

        # Save planning trajectories
        if self.output_dir:
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
        artifact_output_dir = os.path.join(self.output_dir, 'analyzing_artifacts')
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
                
            completion = self._api_call(trajectories)
            
            # Process response
            completion_json = json.loads(completion.model_dump_json())
            responses.append(completion_json)
            self.responses.append(completion_json)  # Add to main responses list
            
            # Update trajectories
            message = completion.choices[0].message
            trajectories.append({'role': message.role, 'content': message.content})

            # Print response
            self.print_response(completion_json)

            # Save analysis results
            with open(f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt', 'w') as f:
                f.write(completion_json['choices'][0]['message']['content'])

            # Save for next stage (coding)
            safe_filename = todo_file_name.replace("/", "_") 
            with open(f'{self.output_dir}/{safe_filename}_simple_analysis_response.json', 'w') as f:
                json.dump(responses, f)

            with open(f'{self.output_dir}/{safe_filename}_simple_analysis_trajectories.json', 'w') as f:
                json.dump(trajectories, f)

        print(f"[ANALYSIS] Analysis completed. Results saved to {artifact_output_dir}")

    def execute_full_pipeline(self):
        """Execute the complete pipeline: planning -> config extraction -> analysis."""
        print("="*60)
        print("STARTING FULL PAPER2CODE PIPELINE")
        print("="*60)
        
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
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print(f"\n✅ 完成! 规划结果保存到: {self.output_dir}")
        if self.is_temp_dir:
            print(f"[INFO] Results are in temporary directory. Copy files if you want to keep them.")


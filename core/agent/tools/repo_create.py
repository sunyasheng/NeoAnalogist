from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_REPO_CREATE_DESCRIPTION = """Generate a full repository implementation with complete code generation based on a paper.

Use this tool to analyze a paper (in markdown, LaTeX, or JSON format) and generate both a comprehensive implementation plan AND complete working code including:
- Overall strategy and methodology breakdown
- Detailed architecture design with file structure
- Logic analysis for each component
- Configuration extraction
- File-by-file implementation guidance
- **Complete source code generation for all files**
- **Environment setup files (requirements.txt, setup.py)**
- **Data download and preparation scripts**
- **Comprehensive reproduction scripts (reproduce.sh)**
- **Complete documentation (README.md)**
- **Working repository with all necessary files and dependencies**

This tool will produce a fully functional repository in the specified output_repo_dir that includes:
1. **Core Implementation**: All Python code files (main.py, model.py, etc.)
2. **Environment Setup**: requirements.txt, setup.py for dependency management
3. **Data Handling**: data_download.py for automatic dataset acquisition
4. **Documentation**: README.md with installation and usage instructions
5. **Reproduction Script**: reproduce.sh with complete pipeline execution
6. **Configuration**: config.yaml with all experiment parameters

The generated repository will be immediately runnable with proper environment setup, data download, and experiment execution capabilities.

Typical use cases:
- When you want to fully implement a research paper from scratch with working code
- When you need both planning AND actual implementation files ready to run
- When you want to create a complete, executable reproduction of a paper's methodology
- When you need a working codebase that can be immediately tested and used
- When you want a complete research environment with data handling and reproduction scripts

IMPORTANT: You must specify the output_repo_dir where the generated code will be placed (e.g., "submission" directory)."""

RepoCreateTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="repo_create",
        description=_REPO_CREATE_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "paper_path": {"type": "string", "description": "Path to the paper file (markdown, LaTeX, or JSON)."},
                "output_repo_dir": {"type": "string", "description": "Directory to save the generated code repository. This is where all the implementation files will be created."},
            },
            "required": ["paper_path", "output_repo_dir"],
        },
    ),
) 
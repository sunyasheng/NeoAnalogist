from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_REPO_PLAN_DESCRIPTION = """Generate a comprehensive repository implementation plan and detailed analysis based on a paper.

Use this tool to analyze a paper (in markdown, LaTeX, or JSON format) and generate a complete implementation plan including:
- Overall strategy and methodology breakdown
- Detailed architecture design with file structure
- Logic analysis for each component
- Configuration extraction
- File-by-file implementation guidance

Typical use cases:
- When you want to break down a research paper into actionable implementation steps
- When you need detailed analysis for each file and component in the implementation
- When you want to reproduce or extend a paper's methodology with comprehensive planning

The tool will return a complete plan with detailed analysis, clear steps, modules, dependencies, and implementation guidance for each file."""

RepoPlanTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="repo_plan",
        description=_REPO_PLAN_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "paper_path": {"type": "string", "description": "Path to the paper file (markdown, LaTeX, or JSON)."},
                "output_dir": {"type": "string", "description": "Optional directory to save the planning results and artifacts. If not provided, a default location will be used."},
            },
            "required": ["paper_path"],
        },
    ),
)
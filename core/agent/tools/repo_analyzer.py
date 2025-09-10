from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_REPO_ANALYZER_DESCRIPTION = """Analyze repository implementation by comparing paper requirements with existing codebase.

Use this tool to perform detailed analysis comparing what's described in a research paper versus what's actually implemented in an existing codebase. This provides:
- Concrete requirements extraction from paper (datasets, algorithms, parameters, APIs)
- Implementation status analysis (implemented, possibly implemented, missing)
- Detailed comparison report with specific evidence
- Missing functionality identification with actionable recommendations
- Code structure analysis and dependency mapping

Typical use cases:
- When you want to evaluate how well a codebase implements a research paper
- When you need to identify missing components or functionality gaps
- When you want to understand implementation completeness and quality
- When you need specific recommendations for improving implementation coverage

The tool will return a comprehensive analysis report with concrete findings, implementation status, and actionable recommendations."""

RepoAnalyzerTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="repo_analyzer",
        description=_REPO_ANALYZER_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "paper_path": {"type": "string", "description": "Path to the paper file (markdown, LaTeX, or JSON)."},
                "codebase_path": {"type": "string", "description": "Path to the existing codebase directory to analyze."},
                "output_dir": {"type": "string", "description": "Optional directory to save the analysis results. If not provided, a default location will be used."},
            },
            "required": ["paper_path", "codebase_path"],
        },
    ),
)

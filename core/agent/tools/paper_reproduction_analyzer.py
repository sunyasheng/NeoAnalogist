"""Paper Reproduction Analyzer Tool for Agent

This tool provides agent access to paper reproduction analysis functionality,
allowing the agent to analyze research papers and extract implementation requirements.
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_PAPER_REPRODUCTION_ANALYZER_DESCRIPTION = """Analyze research paper content and extract comprehensive implementation requirements for reproduction.

Use this tool to analyze a research paper (in markdown, LaTeX, or plain text format) and generate a detailed breakdown of what needs to be implemented to reproduce the paper. This tool provides:

- Data requirements analysis (datasets, preprocessing, formats)
- Algorithm/method implementation requirements
- Experimental setup and configuration details
- Implementation architecture and component breakdown
- Evaluation framework and validation procedures
- Reproduction checklist and implementation guidance

Typical use cases:
- When you need to understand what components are required to reproduce a research paper
- When you want to break down a paper's methodology into implementable requirements
- When you need detailed implementation guidance for research reproduction
- When you want to assess the complexity and scope of paper reproduction

The tool supports three analysis levels:
- 'basic': High-level overview of main requirements
- 'detailed': Comprehensive breakdown with implementation details
- 'comprehensive': Complete reproduction plan with step-by-step guidance

This tool will return a structured analysis with clear sections for each requirement category, providing actionable implementation guidance."""

PaperReproductionAnalyzerTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="paper_reproduction_analyzer",
        description=_PAPER_REPRODUCTION_ANALYZER_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "paper_content": {
                    "type": "string", 
                    "description": "The content of the research paper to analyze (can be markdown, LaTeX, or plain text)."
                },
                "paper_path": {
                    "type": "string", 
                    "description": "Optional path to the paper file. If provided, will read the file content instead of using paper_content."
                },
                "analysis_level": {
                    "type": "string",
                    "enum": ["basic", "detailed", "comprehensive"],
                    "description": "Level of analysis detail. 'basic' for high-level overview, 'detailed' for implementation details, 'comprehensive' for full reproduction plan.",
                    "default": "detailed"
                },
            },
            "required": ["paper_content"],
        },
    ),
) 
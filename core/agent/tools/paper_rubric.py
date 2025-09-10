from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_PAPER_RUBRIC_DESCRIPTION = """Extract structured rubrics (dynamic and static) from a research paper markdown file and save them to a text file.

Use this tool to analyze a research paper (markdown format) and extract:
- Code Development rubrics: Implementation requirements, model architectures, datasets, algorithms, parameters, etc.
- Code Execution rubrics: Runtime verification requirements, execution steps, data processing, etc.
- Result Analysis rubrics: Experimental results, performance metrics, statistical tests, exact numerical values from the paper, etc.

The extracted rubrics are automatically saved to a text file in a format suitable for repo-judge evaluation.

IMPORTANT: You must specify the output_dir where the rubric file will be saved (e.g., "workspace/rubrics" directory).

WORKFLOW: This tool is typically used as the FIRST STEP in a paper evaluation workflow:
1. Use paper_rubric to extract requirements from the markdown paper and save to a rubric file
2. Use repo_judge with the generated rubric file to evaluate repository implementation

Typical use cases:
- When you want to quickly extract all requirements and results from a markdown paper for reproduction or evaluation.
- When you want to generate a checklist for code or experiment implementation based on the paper.
- When you want to create rubric files that can be directly used by repo-judge for evaluation.
- When you need to establish evaluation criteria before judging repository implementation.

Returns a structured summary of all extracted rubrics and saves them to a text file."""

PaperRubricTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="paper_rubric",
        description=_PAPER_RUBRIC_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "paper_path": {
                    "type": "string",
                    "description": "Path to the markdown paper file to analyze."
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where the rubric file will be saved (e.g., 'workspace/rubrics')."
                }
            },
            "required": ["paper_path", "output_dir"],
        },
    ),
) 
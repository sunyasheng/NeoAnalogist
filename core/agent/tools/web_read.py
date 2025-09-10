from litellm import (ChatCompletionToolParam,
                     ChatCompletionToolParamFunctionChunk)

_WEB_DESCRIPTION = """Read (convert to markdown) content from a webpage. This is a powerful tool for gathering information and solving problems that require external data or resources.

Use this tool to:
- Search for datasets, documentation, or solutions online
- Find correct URLs, download links, or API endpoints
- Research implementation details or best practices
- Verify information from reliable sources

You should prefer using the `web_read` tool over the `browser` tool for simple reading tasks, but do use the `browser` tool if you need to interact with a webpage (e.g., click a button, fill out a form, etc.) or search for datasets that require navigation and interaction. For dataset searches, ALWAYS use the browser tool first as most dataset repositories require navigation and interaction.

You may use the `web_read` tool to read content from a webpage, and even search the webpage content using a Google search query (e.g., url=`https://www.google.com/search?q=YOUR_QUERY`).
"""

WebReadTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="web_read",
        description=_WEB_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to read. You can also use a Google search query here (e.g., `https://www.google.com/search?q=YOUR_QUERY`).",
                }
            },
            "required": ["url"],
        },
    ),
)

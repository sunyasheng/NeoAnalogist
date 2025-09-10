"""PDF Query Tool for Agent

This tool provides agent access to PDF query functionality,
allowing the agent to query PDF documents using semantic search and retrieval.
"""

from litellm import ChatCompletionToolParam

PDFQueryTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "pdf_query",
        "description": (
            "Query PDF documents using semantic search and retrieval. "
            "This tool can analyze PDF content, perform semantic search to find relevant information, "
            "and generate answers based on the retrieved context. It supports various embedding models "
            "and provides source document tracking for answer attribution. "
            "IMPORTANT: This tool ONLY accepts PDF files (.pdf extension). "
            "The tool is suitable for extracting information from research papers, reports, manuals, "
            "and other PDF documents. Provide a specific question about the PDF content for best results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the PDF file to query (must have .pdf extension)"
                },
                "query": {
                    "type": "string",
                    "description": "The question or query to ask about the PDF content"
                },
                "embedding_model": {
                    "type": "string",
                    "description": "Embedding model to use for semantic search (options: 'openai', 'huggingface', 'bedrock')",
                    "default": "openai"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to retrieve from the vector database",
                    "default": 5
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Size of text chunks for processing",
                    "default": 1000
                },
                "chunk_overlap": {
                    "type": "integer",
                    "description": "Overlap between text chunks",
                    "default": 200
                },
                "cache_dir": {
                    "type": "string",
                    "description": "Directory for caching embeddings and processed documents (optional)"
                }
            },
            "required": ["pdf_path", "query"]
        }
    }
} 
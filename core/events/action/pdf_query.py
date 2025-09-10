from dataclasses import dataclass
from typing import ClassVar, Optional, Dict, Any

from core.events.event import Action, ActionType


@dataclass
class PDFQueryAction(Action):
    """Action for querying PDF documents with semantic search and retrieval.
    
    This action takes a PDF file path and a query, then performs semantic search
    to find relevant content and returns detailed answers with source citations.
    
    IMPORTANT: pdf_path must point to a PDF file (.pdf extension).
    
    Attributes:
        pdf_path (str): Path to the PDF file to query (must have .pdf extension)
        query (str): The query/question to ask about the PDF content
        embedding_model (str): Embedding model to use for semantic search
        chunk_size (int): Size of text chunks for processing
        chunk_overlap (int): Overlap between chunks
        top_k (int): Number of top results to retrieve
        cache_dir (str): Directory for caching embeddings and processed documents
        thought (str): The reasoning behind the query
        action (str): The action type, namely ActionType.PDF_QUERY
    """
    
    pdf_path: str
    query: str
    embedding_model: str = "openai"  # Options: "openai", "huggingface", "bedrock"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    cache_dir: str = ""
    thought: str = ""
    action: str = ActionType.PDF_QUERY
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"Querying PDF at: {self.pdf_path} with question: {self.query[:100]}..."

    def __str__(self) -> str:
        ret = "**PDFQueryAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"PDF_PATH: {self.pdf_path}\n"
        ret += f"QUERY: {self.query}\n"
        ret += f"EMBEDDING_MODEL: {self.embedding_model}\n"
        ret += f"CHUNK_SIZE: {self.chunk_size}\n"
        ret += f"CHUNK_OVERLAP: {self.chunk_overlap}\n"
        ret += f"TOP_K: {self.top_k}\n"
        if self.cache_dir:
            ret += f"CACHE_DIR: {self.cache_dir}"
        return ret 
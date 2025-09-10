from dataclasses import dataclass
from core.events.event import Observation, ObservationType


@dataclass
class PDFQueryObservation(Observation):
    """This data class represents the result of a PDF query action.
    
    The observation contains the query results including the answer, source documents,
    and metadata about the search and retrieval process.
    
    IMPORTANT: This observation is generated from querying PDF files (.pdf extension).
    
    Attributes:
        content (str): The main content of the observation (required by base class)
        answer (str): The detailed answer to the query
        source_documents (str): Information about source documents used
        search_results (str): Raw search results and scores
        metadata (str): Additional metadata about the query process
        success (bool): Whether the query was successful
        error_message (str): Error message if query failed
        execution_time (float): Time taken for the query operation
        observation (str): The observation type, namely ObservationType.PDF_QUERY
    """
    
    content: str = ""
    answer: str = ""
    source_documents: str = ""
    search_results: str = ""
    metadata: str = ""
    success: bool = False
    error_message: str = ""
    execution_time: float = 0.0
    observation: str = ObservationType.PDF_QUERY

    @property
    def message(self) -> str:
        if self.success:
            return f"Successfully queried PDF and found relevant information."
        else:
            return f"Failed to query PDF: {self.error_message}"

    def __str__(self) -> str:
        ret = "**PDFQueryObservation**\n"
        ret += f"SUCCESS: {self.success}\n"
        ret += f"EXECUTION_TIME: {self.execution_time:.2f} seconds\n"
        
        if self.answer:
            ret += "\n📋 ANSWER:\n"
            ret += "=" * 60 + "\n"
            ret += self.answer + "\n"
        
        if self.source_documents:
            ret += "\n📄 SOURCE DOCUMENTS:\n"
            ret += "=" * 60 + "\n"
            ret += self.source_documents + "\n"
        
        if self.search_results:
            ret += "\n🔍 SEARCH RESULTS:\n"
            ret += "=" * 60 + "\n"
            ret += self.search_results + "\n"
        
        if self.metadata:
            ret += "\n📊 METADATA:\n"
            ret += "=" * 60 + "\n"
            ret += self.metadata + "\n"
        
        if self.error_message:
            ret += f"\n❌ ERROR: {self.error_message}\n"
        
        return ret 
import time
import logging
import os
import hashlib
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json

from core.events.action import PDFQueryAction
from core.events.observation import PDFQueryObservation

# PDF processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector database - try FAISS first, fallback to Chroma
try:
    import faiss  # 这里会直接触发 numpy.distutils 的 ImportError
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except Exception:
    from langchain_community.vectorstores import Chroma
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available, using Chroma as fallback")

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_aws import BedrockEmbeddings

# QA Chain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Utilities
import boto3
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 避免循环导入
if TYPE_CHECKING:
    from core.runtime.impl.docker.docker_runtime import DockerRuntime


@dataclass
class PDFQueryInput:
    """Input data structure for PDF query task.
    
    IMPORTANT: pdf_path must point to a PDF file (.pdf extension).
    """
    pdf_path: str
    question: str
    max_source_docs: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 0.1
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    cache_dir: Optional[str] = None


@dataclass
class PDFQueryOutput:
    """Output data structure for PDF query task."""
    answer: str
    sources: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    document_info: Optional[Dict[str, Any]] = None


class PDFQueryTool:
    """
    A comprehensive PDF query and analysis tool.
    
    This tool provides intelligent PDF document analysis with advanced features:
    - Multi-platform embedding support (OpenAI, HuggingFace, AWS Bedrock)
    - Intelligent caching for performance optimization
    - Source document tracking for answer attribution
    - Comprehensive error handling and logging
    
    IMPORTANT: This tool ONLY accepts PDF files (.pdf extension).
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 env_file: Optional[str] = None,
                 default_embedding_model: str = "text-embedding-3-large",
                 default_llm_model: str = "gpt-4o"):
        """
        Initialize the PDF query tool.
        
        Args:
            cache_dir: Directory for caching embeddings and indices
            env_file: Path to .env file
            default_embedding_model: Default embedding model to use
            default_llm_model: Default LLM model to use
        """
        # Load environment variables
        self._load_environment_variables(env_file)
        
        # Initialize cache
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "pdf_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize caches
        self.document_cache = {}
        self.index_cache = {}
        self.embedding_cache = {}
        
        # Default models
        self.default_embedding_model = default_embedding_model
        self.default_llm_model = default_llm_model
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        logger.info(f"PDF Query Tool initialized with cache_dir: {self.cache_dir}")
    
    def _load_environment_variables(self, env_file: Optional[str] = None):
        """Load environment variables from .env file."""
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from: {env_file}")
        else:
            # Try common locations
            env_locations = [".env", "../.env", "../../.env", os.path.expanduser("~/.env")]
            for env_path in env_locations:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from: {env_path}")
                    break
            else:
                load_dotenv()
    
    def _initialize_embeddings(self):
        """Initialize embedding models based on available credentials."""
        self.embedding_models = {}
        
        # Try OpenAI embeddings
        if os.environ.get("OPENAI_API_KEY"):
            try:
                self.embedding_models["openai"] = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=os.environ["OPENAI_API_KEY"]
                )
                logger.info("Initialized OpenAI embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
        
        # Try HuggingFace embeddings
        try:
            self.embedding_models["huggingface"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Initialized HuggingFace embeddings")
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace embeddings: {e}")
        
        # Try AWS Bedrock embeddings
        if all(key in os.environ for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"]):
            try:
                bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=os.environ['AWS_REGION_NAME'],
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
                )
                self.embedding_models["bedrock"] = BedrockEmbeddings(
                    model_id='amazon.titan-embed-text-v2:0',
                    client=bedrock_client
                )
                logger.info("Initialized AWS Bedrock embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS Bedrock embeddings: {e}")
        
        if not self.embedding_models:
            raise ValueError("No embedding models could be initialized. Please check your credentials.")
    
    def _get_embedding_model(self, model_name: Optional[str] = None) -> Any:
        """Get embedding model by name or return default."""
        if model_name and model_name in self.embedding_models:
            return self.embedding_models[model_name]
        
        # Return first available model as default
        return next(iter(self.embedding_models.values()))
    
    def _get_cache_key(self, pdf_path: str, embedding_model: str) -> str:
        """Generate cache key for PDF and embedding model combination."""
        file_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()
        model_hash = hashlib.md5(embedding_model.encode()).hexdigest()
        return f"{file_hash}_{model_hash}"
    
    def _get_cache_path(self, cache_key: str, suffix: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}_{suffix}")
    
    def _load_pdf_document(self, pdf_path: str) -> List[Any]:
        """Load PDF document with caching."""
        cache_key = self._get_cache_key(pdf_path, "documents")
        cache_path = self._get_cache_path(cache_key, "documents.json")
        
        # Check memory cache first
        if pdf_path in self.document_cache:
            logger.info(f"Using cached documents for: {pdf_path}")
            return self.document_cache[pdf_path]
        
        # Check file cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    documents = cached_data['documents']
                    # Convert back to Document objects
                    from langchain.schema import Document
                    docs = [Document(page_content=doc['content'], metadata=doc['metadata']) 
                           for doc in documents]
                    self.document_cache[pdf_path] = docs
                    logger.info(f"Loaded cached documents for: {pdf_path}")
                    return docs
            except Exception as e:
                logger.warning(f"Failed to load cached documents: {e}")
        
        # Load from PDF
        try:
            logger.info(f"Loading PDF document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Cache documents
            self.document_cache[pdf_path] = documents
            
            # Save to file cache
            try:
                cached_data = {
                    'documents': [
                        {'content': doc.page_content, 'metadata': doc.metadata}
                        for doc in documents
                    ]
                }
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cached_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to cache documents: {e}")
            
            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load PDF document: {e}")
            raise
    
    def _split_documents(self, documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200, pdf_path: str = None) -> List[Any]:
        """Split documents into chunks, with caching by (pdf_path, chunk_size, chunk_overlap)."""
        # Use a tuple key for chunk cache
        cache_key = (pdf_path, chunk_size, chunk_overlap)
        if hasattr(self, '_chunk_cache'):
            chunk_cache = self._chunk_cache
        else:
            self._chunk_cache = {}
            chunk_cache = self._chunk_cache
        if cache_key in chunk_cache:
            logger.info(f"Using cached chunks for: {cache_key}")
            return chunk_cache[cache_key]
        logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        chunk_cache[cache_key] = chunks
        return chunks
    
    def _create_vector_index(self, chunks: List[Any], embedding_model: Any, cache_key: str) -> Any:
        """Create or load vector index."""
        cache_path = self._get_cache_path(cache_key, "faiss_index")
        
        # Check memory cache
        if cache_key in self.index_cache:
            logger.info(f"Using cached vector index for: {cache_key}")
            return self.index_cache[cache_key]
        
        # Try to load from file cache
        if os.path.exists(cache_path + ".pkl") or os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached vector index: {cache_path}")
                if FAISS_AVAILABLE:
                    vector_index = FAISS.load_local(cache_path, embedding_model)
                else:
                    vector_index = Chroma.load_local(cache_path, embedding_model)
                self.index_cache[cache_key] = vector_index
                return vector_index
            except Exception as e:
                logger.warning(f"Failed to load cached vector index: {e}")
        
        # Create new index
        try:
            logger.info("Creating new vector index")
            if FAISS_AVAILABLE:
                vector_index = FAISS.from_documents(chunks, embedding_model)
            else:
                vector_index = Chroma.from_documents(chunks, embedding_model)
            # Cache index
            self.index_cache[cache_key] = vector_index
            # Save to file cache
            try:
                if FAISS_AVAILABLE:
                    vector_index.save_local(cache_path)
                else:
                    vector_index.save_local(cache_path)
                logger.info(f"Saved vector index to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save vector index: {e}")
            return vector_index
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise
    
    def _initialize_llm(self, model: str) -> Any:
        """Initialize LLM for question answering."""
        if model.startswith("gpt-"):
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key required for GPT models")
            return ChatOpenAI(
                model=model,
                temperature=0.1,
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
        elif model.startswith("claude-"):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("Anthropic API key required for Claude models")
            return ChatAnthropic(
                model=model,
                temperature=0.1,
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def _format_sources(self, source_documents: List[Any], pdf_path: str) -> List[Dict[str, Any]]:
        """Format source documents for output."""
        sources = []
        for i, doc in enumerate(source_documents):
            source = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "page": doc.metadata.get("page", "Unknown"),
                "source": doc.metadata.get("source", pdf_path),
                "relevance_score": getattr(doc, 'metadata', {}).get('score', None),
                "chunk_id": i
            }
            sources.append(source)
        return sources
    
    def query_pdf(self, query_input: PDFQueryInput) -> PDFQueryOutput:
        """
        Query a PDF document with intelligent caching and error handling.
        
        Args:
            query_input: Input containing PDF path, question, and parameters
                        IMPORTANT: pdf_path must point to a PDF file (.pdf extension)
            
        Returns:
            PDFQueryOutput with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(query_input.pdf_path):
                return PDFQueryOutput(
                    answer="",
                    sources=[],
                    success=False,
                    error_message=f"PDF file not found: {query_input.pdf_path}"
                )
            
            # Get embedding model
            embedding_model = self._get_embedding_model(query_input.embedding_model)
            
            # Generate cache key
            cache_key = self._get_cache_key(query_input.pdf_path, str(embedding_model))
            
            # Load and process documents
            documents = self._load_pdf_document(query_input.pdf_path)
            # Use the new chunk cache
            chunks = self._split_documents(
                documents, 
                query_input.chunk_size, 
                query_input.chunk_overlap,
                pdf_path=query_input.pdf_path
            )
            
            # Create vector index
            vector_index = self._create_vector_index(chunks, embedding_model, cache_key)
            
            # Initialize LLM
            llm = self._initialize_llm(query_input.model)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_index.as_retriever(
                    search_kwargs={"k": query_input.max_source_docs}
                ),
                return_source_documents=True
            )
            
            # Run query
            logger.info(f"Running query: {query_input.question}")
            result = qa_chain({"query": query_input.question})
            
            # Format sources
            sources = self._format_sources(result["source_documents"], query_input.pdf_path)
            
            # Prepare document info
            document_info = {
                "total_pages": len(documents),
                "total_chunks": len(chunks),
                "embedding_model": str(embedding_model),
                "llm_model": query_input.model,
                "cache_key": cache_key
            }
            
            processing_time = time.time() - start_time
            
            logger.info(f"Query completed in {processing_time:.2f}s")
            
            return PDFQueryOutput(
                answer=result["result"],
                sources=sources,
                success=True,
                processing_time=processing_time,
                document_info=document_info
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in PDF query: {e}")
            
            return PDFQueryOutput(
                answer="",
                sources=[],
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def get_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get information about a PDF document."""
        try:
            documents = self._load_pdf_document(pdf_path)
            chunks = self._split_documents(documents)
            
            return {
                "total_pages": len(documents),
                "total_chunks": len(chunks),
                "file_size": os.path.getsize(pdf_path),
                "file_path": pdf_path,
                "is_cached": pdf_path in self.document_cache
            }
        except Exception as e:
            return {
                "error": str(e),
                "file_path": pdf_path
            }
    
    def clear_cache(self, pdf_path: Optional[str] = None):
        """Clear cache for specific PDF or all caches."""
        if pdf_path:
            # Clear specific PDF cache
            if pdf_path in self.document_cache:
                del self.document_cache[pdf_path]
            
            # Clear related index caches
            cache_keys_to_remove = [key for key in self.index_cache.keys() 
                                  if pdf_path in key]
            for key in cache_keys_to_remove:
                del self.index_cache[key]
            
            logger.info(f"Cleared cache for: {pdf_path}")
        else:
            # Clear all caches
            self.document_cache.clear()
            self.index_cache.clear()
            self.embedding_cache.clear()
            
            # Clear file cache
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith(('.json', '.faiss', '.pkl')):
                        os.remove(os.path.join(self.cache_dir, file))
            
            logger.info("Cleared all caches")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information."""
        return {
            "document_cache_size": len(self.document_cache),
            "index_cache_size": len(self.index_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "cache_directory": self.cache_dir,
            "available_embedding_models": list(self.embedding_models.keys())
        }


class PDFQueryTask:
    """Task for querying PDF documents with semantic search and retrieval.
    
    IMPORTANT: This task ONLY accepts PDF files (.pdf extension).
    """
    
    def __init__(self, runtime: 'DockerRuntime'):
        self.runtime = runtime
        self.logger = logging.getLogger(__name__)
        self.pdf_query_tool = PDFQueryTool()
        
    def configure(self, embedding_model: str = "openai", chunk_size: int = 1000, 
                  chunk_overlap: int = 200, top_k: int = 5, cache_dir: str = None):
        """Configure the PDF query tool parameters"""
        # The PDFQueryTool handles configuration internally
        # This method is kept for compatibility with the action interface
        pass
        
    def load_document(self, pdf_path: str):
        """Load a document for querying"""
        # The PDFQueryTool loads documents on-demand during query
        # This method is kept for compatibility with the action interface
        pass
        
    def query(self, question: str) -> dict:
        """Query the loaded document"""
        # This method is kept for compatibility but delegates to the tool
        # The actual query will be performed in the run method
        return {
            'answer': '',
            'source_documents': '',
            'search_results': '',
            'metadata': ''
        }
        
    async def run(self, action: PDFQueryAction) -> PDFQueryObservation:
        """
        Query a PDF document with semantic search and retrieval
        
        Args:
            action: PDFQueryAction containing PDF path and query parameters
                    IMPORTANT: pdf_path must point to a PDF file (.pdf extension)
            
        Returns:
            PDFQueryObservation with query results
        """
        start_time = time.time()
        
        try:
            # Validate PDF path
            pdf_path = Path(action.pdf_path)
            if not pdf_path.exists():
                return PDFQueryObservation(
                    success=False,
                    execution_time=0,
                    error_message=f"PDF file does not exist: {action.pdf_path}"
                )
            
            # Create input for the PDF query tool
            query_input = PDFQueryInput(
                pdf_path=str(pdf_path),
                question=action.query,
                max_source_docs=action.top_k,
                chunk_size=action.chunk_size,
                chunk_overlap=action.chunk_overlap,
                embedding_model=action.embedding_model,
                cache_dir=action.cache_dir if action.cache_dir else None
            )
            
            # Perform the query using the PDFQueryTool
            self.logger.info(f"Querying PDF: {action.pdf_path}")
            self.logger.info(f"Question: {action.query}")
            
            query_result = self.pdf_query_tool.query_pdf(query_input)
            
            execution_time = time.time() - start_time
            
            if query_result.success:
                # Format the results for observation
                answer = query_result.answer
                source_documents = self._format_sources(query_result.sources)
                search_results = self._format_search_results(query_result.sources)
                metadata = self._format_metadata(query_result.document_info, execution_time)
                
                return PDFQueryObservation(
                    success=True,
                    execution_time=execution_time,
                    content=answer,
                    answer=answer,
                    source_documents=source_documents,
                    search_results=search_results,
                    metadata=metadata
                )
            else:
                return PDFQueryObservation(
                    success=False,
                    execution_time=execution_time,
                    error_message=query_result.error_message
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error querying PDF: {str(e)}")
            return PDFQueryObservation(
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _format_sources(self, sources: list) -> str:
        """Format source documents for output"""
        if not sources:
            return "No source documents found."
        
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            formatted_sources.append(
                f"Source {i}:\n"
                f"  Page: {source.get('page', 'Unknown')}\n"
                f"  Content: {source.get('content', 'No content')}\n"
                f"  Relevance Score: {source.get('relevance_score', 'N/A')}\n"
            )
        
        return "\n".join(formatted_sources)
    
    def _format_search_results(self, sources: list) -> str:
        """Format search results for output"""
        if not sources:
            return "No search results found."
        
        results = []
        for i, source in enumerate(sources, 1):
            results.append(
                f"Result {i}:\n"
                f"  Score: {source.get('relevance_score', 'N/A')}\n"
                f"  Page: {source.get('page', 'Unknown')}\n"
                f"  Content: {source.get('content', 'No content')}\n"
            )
        
        return "\n".join(results)
    
    def _format_metadata(self, document_info: dict, execution_time: float) -> str:
        """Format metadata for output"""
        if not document_info:
            return f"Execution time: {execution_time:.2f} seconds"
        
        metadata = [
            f"Execution time: {execution_time:.2f} seconds",
            f"Total pages: {document_info.get('total_pages', 'Unknown')}",
            f"Total chunks: {document_info.get('total_chunks', 'Unknown')}",
            f"Embedding model: {document_info.get('embedding_model', 'Unknown')}",
            f"LLM model: {document_info.get('llm_model', 'Unknown')}",
            f"Cache key: {document_info.get('cache_key', 'Unknown')}"
        ]
        
        return "\n".join(metadata) 
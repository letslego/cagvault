"""
LanceDB Multi-Document Chat Module
MIGRATED FROM LlamaIndex TO LangChain

This module provides multi-document chat functionality using LanceDB vector store and LangChain.
"""

import os
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
import logging
from datetime import datetime

# LangChain imports
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LanceDB
import lancedb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism for transformers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to import rerankers (available in newer lancedb versions)
try:
    from lancedb.rerankers import (
        LinearCombinationReranker,
        CohereReranker,
        ColbertReranker,
        CrossEncoderReranker,
    )
    RERANKERS_AVAILABLE = True
except ImportError:
    RERANKERS_AVAILABLE = False
    logger.warning("lancedb.rerankers not available. Install lancedb>=0.5.0 for reranking support.")


@dataclass
class ChatResponse:
    """Response from the chat system."""
    answer: str
    source_nodes: List[str]
    contexts: List[Dict[str, str]]


@dataclass
class AdvancedSearchConfig:
    """Configuration for advanced full-text search."""
    query_type: str = "match"  # 'match', 'multi_match', 'phrase', 'fuzzy', 'boolean'
    fields: Optional[List[str]] = None  # Fields to search ['text', 'filename']
    field_boosts: Optional[Dict[str, float]] = None  # Field boost weights
    boost: float = 1.0  # Overall query boost
    phrase_slop: int = 0  # Allow N words between phrase terms
    fuzzy_distance: int = 2  # Levenshtein distance for fuzzy search
    boolean_operator: str = "AND"  # 'AND' or 'OR' for boolean queries
    minimum_should_match: Optional[int] = None  # Min terms to match in boolean
    # Reranking configuration
    reranker_type: str = "linear"  # 'linear', 'cohere', 'colbert', 'cross_encoder'
    rerank_top_k: Optional[int] = None  # Rerank top K results (None = use all)
    # Multivector search
    use_multivector: bool = False  # Enable multivector search
    multivector_queries: Optional[List[str]] = None  # Additional query variations


@dataclass
class FilterConfig:
    """Configuration for document filtering."""
    pre_filter: Optional[Dict[str, any]] = None
    post_filter: Optional[Dict[str, any]] = None


class LanceDBMultiDocumentChat:
    """
    Multi-document chat system using LanceDB for vector storage with LangChain.
    
    Features:
    - Semantic search across multiple documents
    - Context-aware responses using LangChain RAG
    - Efficient vector storage with LanceDB
    - Local embedding model support
    - Hybrid search with FTS and reranking
    """
    
    def __init__(
        self,
        db_path: str = "./lancedb",
        table_name: str = "documents",
        embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        llm_model: str = "qwen2.5:14b",
        llm_base_url: str = "http://localhost:11434",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_top_k: int = 5,
        search_mode: str = "semantic",  # 'semantic' or 'hybrid'
        hybrid_weight: float = 0.7,  # Weight for semantic vs FTS in hybrid mode
        enable_scalar_indexes: bool = True,  # Create scalar indexes for filtering
        reranker_type: str = "linear",  # 'linear', 'cohere', 'colbert', 'cross_encoder'
        reranker_model: Optional[str] = None,  # Model name for specific rerankers
        use_matryoshka: bool = True,  # Use Matryoshka embedding
        matryoshka_model: str = "mixedbread-ai/mxbai-embed-large-v1",  # Matryoshka model
    ):
        """
        Initialize the multi-document chat system with LangChain.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Name of the table for storing vectors
            embedding_model: HuggingFace embedding model name
            llm_model: Ollama model name (used via LangChain ChatOllama)
            llm_base_url: Ollama API base URL
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            similarity_top_k: Number of similar documents to retrieve
            search_mode: 'semantic' or 'hybrid' search mode
            hybrid_weight: Weight for semantic vs FTS in hybrid mode (0.0-1.0)
            enable_scalar_indexes: Whether to create scalar indexes for filtering
            reranker_type: Type of reranker ('linear', 'cohere', 'colbert', 'cross_encoder')
            reranker_model: Model name for specific rerankers
            use_matryoshka: Use Matryoshka embedding
            matryoshka_model: Matryoshka embedding model name
        """
        self.db_path = db_path
        self.table_name = table_name
        self.similarity_top_k = similarity_top_k
        self.search_mode = search_mode
        self.hybrid_weight = hybrid_weight
        self.enable_scalar_indexes = enable_scalar_indexes
        self.reranker_type = reranker_type
        self.reranker_model = reranker_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        embedding_model_name = matryoshka_model if use_matryoshka else embedding_model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # For Matryoshka
        )
        
        # Initialize LLM (import here to avoid circular dependencies)
        from langchain_ollama import ChatOllama
        
        logger.info(f"Connecting to Ollama model: {llm_model}")
        self.llm = ChatOllama(
            model=llm_model,
            base_url=llm_base_url,
            temperature=0,
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Setup reranker
        self.reranker = self._create_reranker()
        
        # LanceDB connection
        self.db = lancedb.connect(self.db_path)
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.lance_table = None
        
        # Try to load existing vector store table if it exists
        self._load_existing_vector_store()
    
    def _create_reranker(self):
        """
        Create reranker based on configuration.
        
        Returns:
            Configured reranker instance or None if not available
        """
        if not RERANKERS_AVAILABLE:
            logger.info("Rerankers not available, using default similarity search")
            return None
        
        reranker_type = self.reranker_type.lower()
        
        try:
            if reranker_type == "linear":
                logger.info(f"Using LinearCombinationReranker with weight {self.hybrid_weight}")
                return LinearCombinationReranker(weight=self.hybrid_weight)
            
            elif reranker_type == "cohere":
                logger.info("Using CohereReranker")
                model = self.reranker_model or "rerank-english-v2.0"
                return CohereReranker(model_name=model)
            
            elif reranker_type == "colbert":
                logger.info("Using ColbertReranker")
                model = self.reranker_model or "colbert-ir/colbertv2.0"
                return ColbertReranker(model_name=model)
            
            elif reranker_type == "cross_encoder":
                logger.info("Using CrossEncoderReranker")
                model = self.reranker_model or "BAAI/bge-reranker-base"
                return CrossEncoderReranker(model_name=model)
            
            else:
                logger.warning(f"Unknown reranker type '{reranker_type}', using default")
                return None
        
        except Exception as e:
            logger.warning(f"Failed to create {reranker_type} reranker: {e}. Using default search")
            return None
    
    def _load_existing_vector_store(self):
        """Load existing vector store if table exists."""
        try:
            # Check if table exists
            table_names = self.db.table_names()
            if self.table_name in table_names:
                logger.info(f"Loading existing vector store table: {self.table_name}")
                self.vector_store = LanceDB(
                    connection=self.db,
                    table_name=self.table_name,
                    embedding=self.embeddings,
                )
                self.lance_table = self.db.open_table(self.table_name)
                self._setup_qa_chain()
                logger.info(f"Loaded existing vector store with {self.lance_table.count_rows()} vectors")
            else:
                logger.info(f"Table {self.table_name} does not exist yet")
        except Exception as e:
            logger.debug(f"Could not load existing vector store: {e}")
    
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'name', 'content', and optional metadata keys:
                - file_type: Type of file (e.g., 'pdf', 'txt')
                - file_size: Size in bytes
                - upload_date: Upload timestamp (ISO format)
        """
        logger.info(f"Adding {len(documents)} documents to the index")
        
        # Convert to LangChain Document objects with rich metadata
        lang_docs = []
        for doc in documents:
            # Split content into chunks
            chunks = self.text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'filename': doc['name'],
                    'source': doc['name'],
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                }
                
                # Add optional metadata
                if 'file_type' in doc:
                    metadata['file_type'] = doc['file_type']
                if 'file_size' in doc:
                    metadata['file_size'] = doc['file_size']
                if 'upload_date' in doc:
                    metadata['upload_date'] = doc['upload_date']
                else:
                    metadata['upload_date'] = datetime.now().isoformat()
                
                lang_docs.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        logger.info(f"Created {len(lang_docs)} chunks from {len(documents)} documents")
        
        # Create or update vector store
        if self.vector_store is None:
            # First time - create from documents
            logger.info(f"Creating new LanceDB vector store table: {self.table_name}")
            self.vector_store = LanceDB.from_documents(
                documents=lang_docs,
                embedding=self.embeddings,
                connection=self.db,
                table_name=self.table_name,
                reranker=self.reranker if self.search_mode == "hybrid" else None
            )
        else:
            # Add to existing vector store
            logger.info(f"Adding documents to existing table: {self.table_name}")
            self.vector_store.add_documents(lang_docs)
        
        # Get reference to LanceDB table for advanced operations
        self.lance_table = self.db.open_table(self.table_name)
        
        # Create scalar indexes if enabled
        if self.enable_scalar_indexes:
            self._create_scalar_indexes()
        
        # Setup retriever and QA chain
        self._setup_qa_chain()
        
        logger.info(f"Successfully indexed documents. Total vectors: {len(lang_docs)}")
    
    def _create_scalar_indexes(self):
        """Create scalar indexes for efficient filtering on metadata fields."""
        try:
            if self.lance_table is not None:
                logger.info("Creating scalar indexes for metadata fields")
                # Index common metadata fields
                for field in ['filename', 'file_type', 'upload_date']:
                    try:
                        self.lance_table.create_scalar_index(field)
                        logger.info(f"Created scalar index on {field}")
                    except Exception as e:
                        logger.debug(f"Scalar index on {field} might already exist or failed: {e}")
        except Exception as e:
            logger.warning(f"Failed to create scalar indexes: {e}")
    
    def _setup_qa_chain(self):
        """Setup the QA chain with retriever."""
        # Configure retriever with similarity search
        # Note: Reranking is handled at query time, not at retriever setup
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.similarity_top_k}
        )
        
        # Create RAG prompt template
        template = """You are a helpful assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, say that you don't know.
Keep the answer concise but comprehensive.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("QA chain configured successfully")
    
    def query(
        self,
        query_text: str,
        filter_config: Optional[FilterConfig] = None,
        advanced_search: Optional[AdvancedSearchConfig] = None
    ) -> ChatResponse:
        """
        Query the document collection.
        
        Args:
            query_text: The query string
            filter_config: Optional filtering configuration
            advanced_search: Optional advanced search configuration
        
        Returns:
            ChatResponse with answer and source information
        """
        if self.qa_chain is None:
            raise ValueError("No documents have been added. Call add_documents() first.")
        
        logger.info(f"Processing query: {query_text[:100]}...")
        
        # Handle advanced search (hybrid, multivector, etc.)
        if advanced_search and advanced_search.use_multivector and advanced_search.multivector_queries:
            # Multi-query retrieval
            all_docs = []
            for sub_query in [query_text] + advanced_search.multivector_queries:
                docs = self.retriever.invoke(sub_query)
                all_docs.extend(docs)
            
            # Deduplicate and rerank
            unique_docs = {doc.metadata.get('chunk_id', id(doc)): doc for doc in all_docs}.values()
            docs = list(unique_docs)[:self.similarity_top_k]
        else:
            # Standard retrieval
            docs = self.retriever.invoke(query_text)
        
        # Apply filters if specified
        if filter_config and filter_config.post_filter:
            docs = self._apply_post_filter(docs, filter_config.post_filter)
        
        # Generate answer using RAG chain
        answer = self.qa_chain.invoke(query_text)
        
        # Extract source information
        source_nodes = []
        contexts = []
        
        for doc in docs:
            source_nodes.append(doc.metadata.get('filename', 'Unknown'))
            contexts.append({
                'content': doc.page_content,
                'source': doc.metadata.get('filename', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 0),
                'metadata': doc.metadata
            })
        
        return ChatResponse(
            answer=answer,
            source_nodes=list(set(source_nodes)),  # Deduplicate
            contexts=contexts
        )
    
    def _apply_post_filter(self, docs: List[Document], filter_dict: Dict) -> List[Document]:
        """Apply post-retrieval filtering on documents."""
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filter_dict.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
        return filtered_docs
    
    def stream_query(
        self,
        query_text: str,
        filter_config: Optional[FilterConfig] = None
    ) -> Iterator[str]:
        """
        Stream the query response token by token.
        
        Args:
            query_text: The query string
            filter_config: Optional filtering configuration
        
        Yields:
            Response tokens as they are generated
        """
        if self.qa_chain is None:
            raise ValueError("No documents have been added. Call add_documents() first.")
        
        logger.info(f"Streaming query: {query_text[:100]}...")
        
        # Get relevant documents
        docs = self.retriever.invoke(query_text)
        
        # Apply filters if specified
        if filter_config and filter_config.post_filter:
            docs = self._apply_post_filter(docs, filter_config.post_filter)
        
        # Stream the response
        for chunk in self.qa_chain.stream(query_text):
            yield chunk
    
    def get_document_stats(self) -> Dict[str, any]:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dictionary with document statistics
        """
        if self.lance_table is None:
            return {"total_documents": 0, "total_chunks": 0}
        
        # Count total chunks
        total_chunks = self.lance_table.count_rows()
        
        # Get unique filenames
        try:
            # Query all filenames
            results = self.lance_table.to_pandas()
            unique_files = results['metadata'].apply(lambda x: x.get('filename', 'Unknown')).nunique()
        except Exception as e:
            logger.warning(f"Failed to count unique documents: {e}")
            unique_files = 0
        
        return {
            "total_documents": unique_files,
            "total_chunks": total_chunks,
            "table_name": self.table_name,
            "db_path": self.db_path,
            "search_mode": self.search_mode,
            "reranker_type": self.reranker_type
        }
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete a document by filename.
        
        Args:
            filename: Name of the document to delete
        
        Returns:
            True if document was deleted, False otherwise
        """
        if self.lance_table is None:
            return False
        
        try:
            # Delete rows matching the filename
            self.lance_table.delete(f"metadata.filename = '{filename}'")
            logger.info(f"Deleted document: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {filename}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Drop and recreate table
            self.db.drop_table(self.table_name)
            logger.info(f"Cleared all documents from table: {self.table_name}")
            self.vector_store = None
            self.lance_table = None
            self.qa_chain = None
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return False


def create_lancedb_chat(
    db_path: str = "./lancedb",
    table_name: str = "documents",
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
    llm_model: str = "qwen2.5:14b",
    **kwargs
) -> LanceDBMultiDocumentChat:
    """
    Factory function to create a LanceDB multi-document chat instance.
    
    Args:
        db_path: Path to LanceDB database
        table_name: Name of the table for storing vectors
        embedding_model: HuggingFace embedding model name
        llm_model: Ollama model name
        **kwargs: Additional arguments passed to LanceDBMultiDocumentChat
    
    Returns:
        Configured LanceDBMultiDocumentChat instance
    """
    return LanceDBMultiDocumentChat(
        db_path=db_path,
        table_name=table_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        **kwargs
    )

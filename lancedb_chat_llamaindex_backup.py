"""
LanceDB Multi-Document Chat Module
Based on https://github.com/lancedb/vectordb-recipes/tree/main/examples/multi-document-agentic-rag

This module provides multi-document chat functionality using LanceDB vector store and LlamaIndex.
"""

import os
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
import logging
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
import lancedb
from lancedb.rerankers import (
    LinearCombinationReranker,
    CohereReranker,
    ColbertReranker,
    CrossEncoderReranker,
)
# LangChain moved embeddings to langchain_community in newer releases
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:  # Older LangChain versions
    from langchain.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism for transformers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class LangChainMatryoshkaEmbedding:
    """Adapter to use LangChain HuggingFace embeddings (Matryoshka) inside LlamaIndex/LanceDB flows."""

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        self.model_name = model_name
        # Normalize embeddings for cosine similarity per Matryoshka guidance
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # Matryoshka M3 embeddings output 1024 dimensions
        self.dim = 1024

    def get_text_embedding(self, text: str):
        return self.embedder.embed_query(text)

    def get_text_embedding_batch(self, texts: List[str]):
        return self.embedder.embed_documents(texts)

    def get_query_embedding(self, query: str):
        return self.embedder.embed_query(query)


class LanceDBMultiDocumentChat:
    """
    Multi-document chat system using LanceDB for vector storage.
    
    Features:
    - Semantic search across multiple documents
    - Context-aware responses using LlamaIndex
    - Efficient vector storage with LanceDB
    - Local embedding model support
    """
    
    def __init__(
        self,
        db_path: str = "./lancedb",
        table_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
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
        use_matryoshka: bool = True,  # Use Matryoshka embedding via LangChain
        matryoshka_model: str = "mixedbread-ai/mxbai-embed-large-v1",  # Matryoshka model
    ):
        """
        Initialize the multi-document chat system.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Name of the table for storing vectors
            embedding_model: HuggingFace embedding model name
            llm_model: Ollama model name
            llm_base_url: Ollama API base URL
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            similarity_top_k: Number of similar documents to retrieve
            search_mode: 'semantic' or 'hybrid' search mode
            hybrid_weight: Weight for semantic vs FTS in hybrid mode (0.0-1.0)
            enable_scalar_indexes: Whether to create scalar indexes for filtering
            reranker_type: Type of reranker ('linear', 'cohere', 'colbert', 'cross_encoder')
            reranker_model: Model name for specific rerankers (e.g., 'BAAI/bge-reranker-base')
            use_matryoshka: Use LangChain Matryoshka embeddings for LanceDB
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
        self.use_matryoshka = use_matryoshka
        self.matryoshka_model = matryoshka_model
        
        # Initialize embedding model
        if use_matryoshka:
            logger.info(f"Loading Matryoshka embedding model via LangChain: {matryoshka_model}")
            self.embed_model = LangChainMatryoshkaEmbedding(model_name=matryoshka_model)
        else:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # LanceDB-compatible embedding function using the loaded HF model
        self.embedding_function = self._create_lance_embedding_function()
        
        # Initialize LLM
        logger.info(f"Connecting to Ollama model: {llm_model}")
        self.llm = Ollama(
            model=llm_model,
            base_url=llm_base_url,
            request_timeout=120.0,
        )
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vector store
        logger.info(f"Initializing LanceDB vector store at {db_path}")
        self.vector_store = LanceDBVectorStore(
            uri=db_path,
            table_name=table_name,
            mode="overwrite",
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Setup reranker based on type
        self.reranker = self._create_reranker()
        
        self.index = None
        self.query_engine = None
        self.lance_table = None  # Direct LanceDB table for hybrid search

    def _create_lance_embedding_function(self):
        """
        Create a LanceDB-compatible embedding function.
        Uses LangChain Matryoshka embeddings when enabled, otherwise HuggingFaceEmbedding.
        """

        class LanceEmbeddingWrapper:
            def __init__(self, embed_model, name: str, dim: int):
                self.embed_model = embed_model
                self.name = name
                self.source_columns = ["text"]
                self.vector_column = "vector"
                self.dim = dim

            def __call__(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                # Support both adapter (get_text_embedding_batch) and LC embed_documents
                if hasattr(self.embed_model, "get_text_embedding_batch"):
                    return self.embed_model.get_text_embedding_batch(texts)
                if hasattr(self.embed_model, "embed_documents"):
                    return self.embed_model.embed_documents(texts)
                raise ValueError("Embedding model does not support batch embeddings")

        dim = 1024 if self.use_matryoshka else 384
        name = "matryoshka-mxbai" if self.use_matryoshka else "hf-all-minilm-l6-v2"
        return LanceEmbeddingWrapper(self.embed_model, name=name, dim=dim)

    def _connect_lance_table(self):
        """Open LanceDB table with the registered embedding function."""
        db = lancedb.connect(self.db_path)
        self.lance_table = db.open_table(
            self.table_name,
            embedding_function=self.embedding_function,
        )
    
    def _create_reranker(self):
        """
        Create reranker based on configuration.
        
        Returns:
            Configured reranker instance
        """
        reranker_type = self.reranker_type.lower()
        
        try:
            if reranker_type == "linear":
                # Linear combination for hybrid search
                logger.info(f"Using LinearCombinationReranker with weight {self.hybrid_weight}")
                return LinearCombinationReranker(weight=self.hybrid_weight)
            
            elif reranker_type == "cohere":
                # Cohere reranker (requires API key)
                logger.info("Using CohereReranker")
                model = self.reranker_model or "rerank-english-v2.0"
                return CohereReranker(model_name=model)
            
            elif reranker_type == "colbert":
                # ColBERT reranker for neural reranking
                logger.info("Using ColbertReranker")
                model = self.reranker_model or "colbert-ir/colbertv2.0"
                return ColbertReranker(model_name=model)
            
            elif reranker_type == "cross_encoder":
                # Cross-encoder reranker
                logger.info("Using CrossEncoderReranker")
                model = self.reranker_model or "BAAI/bge-reranker-base"
                return CrossEncoderReranker(model_name=model)
            
            else:
                logger.warning(f"Unknown reranker type '{reranker_type}', using LinearCombinationReranker")
                return LinearCombinationReranker(weight=self.hybrid_weight)
        
        except Exception as e:
            logger.warning(f"Failed to create {reranker_type} reranker: {e}. Falling back to LinearCombinationReranker")
            return LinearCombinationReranker(weight=self.hybrid_weight)
    
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
        
        # Convert to LlamaIndex Document objects with rich metadata
        docs = []
        for doc in documents:
            # Extract file extension from name
            file_type = doc.get("file_type") or doc["name"].split(".")[-1].lower()
            
            # Build metadata with filtering-friendly fields
            metadata = {
                "filename": doc["name"],
                "file_type": file_type,
                "file_size": doc.get("file_size", len(doc["content"])),
                "upload_date": doc.get("upload_date", datetime.now().isoformat()),
                "char_count": len(doc["content"]),
                # Extract first 500 chars as summary for boosted searching
                "summary": doc["content"][:500] if len(doc["content"]) > 500 else doc["content"],
            }
            
            docs.append(
                Document(
                    text=doc["content"],
                    metadata=metadata,
                )
            )
        
        # Parse documents into nodes
        nodes = self.node_parser.get_nodes_from_documents(docs)
        
        # Create or update index
        if self.index is None:
            self.index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
            )
        else:
            # Add to existing index
            for node in nodes:
                self.index.insert_nodes([node])
        
        # Get direct access to LanceDB table for advanced features (with embedding function)
        self._connect_lance_table()
        
        # Create FTS indexes for hybrid/advanced search
        if self.search_mode == "hybrid":
            try:
                logger.info("Creating multi-field FTS indexes for advanced search")
                # Create FTS index on main text field
                self.lance_table.create_fts_index("text", replace=True)
                # Create FTS index on filename for multi-field search
                self.lance_table.create_fts_index("metadata.filename", replace=True)
                # Create FTS index on summary for boosted search
                self.lance_table.create_fts_index("metadata.summary", replace=True)
                logger.info("Multi-field FTS indexes created successfully")
            except Exception as e:
                logger.warning(f"Could not create FTS indexes: {e}")
        
        # Create scalar indexes for efficient filtering
        if self.enable_scalar_indexes:
            try:
                logger.info("Creating scalar indexes for filtering")
                # Create indexes on commonly filtered fields
                self.lance_table.create_scalar_index("metadata.file_type", replace=True)
                self.lance_table.create_scalar_index("metadata.upload_date", replace=True)
                self.lance_table.create_scalar_index("metadata.filename", replace=True)
                logger.info("Scalar indexes created successfully")
            except Exception as e:
                logger.warning(f"Could not create scalar indexes: {e}")
        
        # Create query engine
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
        )
        
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        logger.info(f"Documents indexed successfully with {self.search_mode} search")
    
    def _build_fts_query(self, query: str, config: Optional[AdvancedSearchConfig] = None) -> str:
        """
        Build advanced FTS query string.
        
        Args:
            query: Base query string
            config: Advanced search configuration
            
        Returns:
            Formatted FTS query string
        """
        if config is None:
            return query
        
        query_type = config.query_type
        
        # Multi-match query with field boosting
        if query_type == "multi_match":
            if config.fields and config.field_boosts:
                # Build boosted multi-field query
                # Example: text:search^2.0 OR filename:search^1.5
                terms = []
                for field in config.fields:
                    boost = config.field_boosts.get(field, 1.0)
                    terms.append(f"{field}:{query}^{boost}")
                return " OR ".join(terms)
            elif config.fields:
                # Multi-field without boosts
                return " OR ".join([f"{field}:{query}" for field in config.fields])
        
        # Phrase query with slop
        elif query_type == "phrase":
            if config.phrase_slop > 0:
                # Phrase with slop allows N words between terms
                return f'"{query}"~{config.phrase_slop}'
            else:
                # Exact phrase match
                return f'"{query}"'
        
        # Fuzzy search with edit distance
        elif query_type == "fuzzy":
            # Apply fuzzy to each term
            terms = query.split()
            fuzzy_terms = [f"{term}~{config.fuzzy_distance}" for term in terms]
            return " ".join(fuzzy_terms)
        
        # Boolean query
        elif query_type == "boolean":
            terms = query.split()
            operator = config.boolean_operator
            if operator == "AND":
                return " AND ".join(terms)
            elif operator == "OR":
                return " OR ".join(terms)
        
        # Boost query
        if config.boost != 1.0:
            return f"({query})^{config.boost}"
        
        # Default: simple match
        return query
    
    def _multivector_search(
        self,
        question: str,
        mode: str,
        pre_filter: Optional[str],
        post_filter: Optional[str],
        config: AdvancedSearchConfig,
    ) -> ChatResponse:
        """
        Perform multivector search using multiple query variations.
        
        Args:
            question: Main query
            mode: Search mode
            pre_filter: Pre-filter SQL
            post_filter: Post-filter SQL
            config: Advanced search configuration
            
        Returns:
            ChatResponse with aggregated results
        """
        logger.info("Performing multivector search")
        
        if self.lance_table is None:
            self._connect_lance_table()
            if self.lance_table is None:
                raise ValueError("LanceDB table not initialized. Please add documents first.")
        
        # Generate query variations
        queries = [question]
        if config.multivector_queries:
            queries.extend(config.multivector_queries)
        else:
            # Auto-generate variations using LLM
            variation_prompt = f"Generate 2 alternative phrasings of this question: {question}\nProvide only the questions, one per line."
            try:
                variations_response = self.llm.complete(variation_prompt)
                variations = str(variations_response).strip().split('\n')[:2]
                queries.extend([v.strip() for v in variations if v.strip()])
            except Exception as e:
                logger.warning(f"Could not generate query variations: {e}")
        
        logger.info(f"Searching with {len(queries)} query variations: {queries}")
        
        # Collect results from all query variations
        all_results = []
        seen_texts = set()
        
        for query_variation in queries:
            try:
                # Build FTS query if needed
                fts_query = self._build_fts_query(query_variation, config)
                
                # Perform search for this variation
                if mode == "hybrid":
                    search = self.lance_table.search(fts_query, query_type="hybrid")
                else:
                    search = self.lance_table.search(query_variation, query_type="vector")
                
                # Apply pre-filter
                if pre_filter:
                    search = search.where(pre_filter, prefilter=True)
                
                # Get results
                results = search.limit(self.similarity_top_k).to_pandas()
                
                # Deduplicate by text content
                for _, row in results.iterrows():
                    text = row.get('text', '')
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        all_results.append(row)
            
            except Exception as e:
                logger.warning(f"Failed to search with variation '{query_variation}': {e}")
        
        # Convert back to DataFrame
        import pandas as pd
        if not all_results:
            raise ValueError("No results found from multivector search")
        
        combined_results = pd.DataFrame(all_results)
        
        # Apply reranking to combined results
        logger.info(f"Reranking {len(combined_results)} unique results")
        try:
            # Use LanceDB's reranking on combined results
            # Note: We need to convert back to LanceDB format
            # For now, sort by score if available
            if '_distance' in combined_results.columns:
                combined_results = combined_results.sort_values('_distance').head(self.similarity_top_k)
        except Exception as e:
            logger.warning(f"Could not rerank combined results: {e}")
        
        # Apply post-filter
        if post_filter:
            logger.info(f"Applying post-filter: {post_filter}")
            post_filter_pandas = post_filter.replace("metadata.", "")
            combined_results = combined_results.query(post_filter_pandas).head(self.similarity_top_k)
        else:
            combined_results = combined_results.head(self.similarity_top_k)
        
        # Build context from results
        contexts = [
            {
                "filename": row.get("filename", "Unknown"),
                "text": row.get("text", ""),
            }
            for _, row in combined_results.iterrows()
        ]
        context = "\n\n".join([
            f"Document: {c['filename']}\n{c['text']}" for c in contexts
        ])
        
        # Query LLM with aggregated context
        prompt = f"Context from multiple search perspectives:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = self.llm.complete(prompt)
        
        # Extract source files
        source_nodes = list(combined_results['filename'].unique()) if 'filename' in combined_results.columns else []
        
        return ChatResponse(
            answer=str(response),
            source_nodes=source_nodes,
            contexts=contexts,
        )
    
    def clear_documents(self) -> None:
        """Clear all documents from the index."""
        logger.info("Clearing document index")
        self.vector_store = LanceDBVectorStore(
            uri=self.db_path,
            table_name=self.table_name,
            mode="overwrite",
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = None
        self.query_engine = None
        self.lance_table = None
    
    def query(
        self,
        question: str,
        search_mode: Optional[str] = None,
        pre_filter: Optional[str] = None,
        post_filter: Optional[str] = None,
        advanced_search_config: Optional[AdvancedSearchConfig] = None,
    ) -> ChatResponse:
        """
        Query the indexed documents with optional filtering and advanced FTS.
        
        Args:
            question: User's question
            search_mode: Override search mode ('semantic' or 'hybrid')
            pre_filter: SQL WHERE clause for pre-filtering (applied before vector search)
                       Example: "metadata.file_type = 'pdf'" or 
                               "metadata.upload_date >= '2024-01-01'"
            post_filter: SQL WHERE clause for post-filtering (applied after vector search)
                        Example: "metadata.file_size > 1000"
            advanced_search_config: Advanced FTS configuration for multi-match, boosting, etc.
            
        Returns:
            ChatResponse with answer and source information
        """
        if self.query_engine is None:
            raise ValueError("No documents indexed. Please add documents first.")
        
        mode = search_mode or self.search_mode
        logger.info(f"Querying with {mode} search: {question}")

        # Ensure LanceDB table is available with embedding function
        if self.lance_table is None:
            try:
                self._connect_lance_table()
            except Exception as e:
                logger.warning(f"Could not open LanceDB table: {e}")
        
        # Handle multivector search if enabled
        if advanced_search_config and advanced_search_config.use_multivector:
            return self._multivector_search(
                question,
                mode,
                pre_filter,
                post_filter,
                advanced_search_config
            )
        
        # For hybrid search, use LanceDB's hybrid search directly
        if mode == "hybrid" and self.lance_table is not None:
            try:
                # Build advanced FTS query if configured
                fts_query = self._build_fts_query(question, advanced_search_config)
                logger.info(f"FTS Query: {fts_query}")
                
                # Start hybrid search with advanced FTS
                search = self.lance_table.search(fts_query, query_type="hybrid")
                
                # Apply pre-filter (before vector search - more efficient)
                if pre_filter:
                    logger.info(f"Applying pre-filter: {pre_filter}")
                    search = search.where(pre_filter, prefilter=True)
                
                # Determine rerank count
                rerank_limit = self.similarity_top_k * 2 if post_filter else self.similarity_top_k
                if advanced_search_config and advanced_search_config.rerank_top_k:
                    rerank_limit = advanced_search_config.rerank_top_k
                
                # Perform hybrid search with reranking
                hybrid_results = (
                    search
                    .limit(rerank_limit)
                    .rerank(reranker=self.reranker)
                    .to_pandas()
                )
                
                # Apply post-filter (after vector search - for refinement)
                if post_filter:
                    logger.info(f"Applying post-filter: {post_filter}")
                    # Convert SQL-like filter to pandas query
                    post_filter_pandas = post_filter.replace("metadata.", "")
                    hybrid_results = hybrid_results.query(post_filter_pandas).head(self.similarity_top_k)
                
                # Build context from hybrid results
                contexts = [
                    {
                        "filename": row.get("filename", "Unknown"),
                        "text": row.get("text", ""),
                    }
                    for _, row in hybrid_results.iterrows()
                ]
                context = "\n\n".join([
                    f"Document: {c['filename']}\n{c['text']}" for c in contexts
                ])
                
                # Query LLM with hybrid context
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                response = self.llm.complete(prompt)
                
                # Extract source files
                source_nodes = list(hybrid_results['filename'].unique()) if 'filename' in hybrid_results.columns else []
                
                return ChatResponse(
                    answer=str(response),
                    source_nodes=source_nodes,
                    contexts=contexts,
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
        
        # Semantic search with optional filtering
        if pre_filter or post_filter:
            # Use LanceDB directly for filtered semantic search
            try:
                search = self.lance_table.search(question, query_type="vector")
                
                # Apply pre-filter
                if pre_filter:
                    logger.info(f"Applying pre-filter: {pre_filter}")
                    search = search.where(pre_filter, prefilter=True)
                
                results = search.limit(self.similarity_top_k * 2 if post_filter else self.similarity_top_k).to_pandas()
                
                # Apply post-filter
                if post_filter:
                    logger.info(f"Applying post-filter: {post_filter}")
                    post_filter_pandas = post_filter.replace("metadata.", "")
                    results = results.query(post_filter_pandas).head(self.similarity_top_k)
                
                # Build context from filtered results
                contexts = [
                    {
                        "filename": row.get("filename", "Unknown"),
                        "text": row.get("text", ""),
                    }
                    for _, row in results.iterrows()
                ]
                context = "\n\n".join([
                    f"Document: {c['filename']}\n{c['text']}" for c in contexts
                ])
                
                # Query LLM with filtered context
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                response = self.llm.complete(prompt)
                
                # Extract source files
                source_nodes = list(results['filename'].unique()) if 'filename' in results.columns else []
                
                return ChatResponse(
                    answer=str(response),
                    source_nodes=source_nodes,
                    contexts=contexts,
                )
            except Exception as e:
                logger.warning(f"Filtered search failed, falling back to standard query: {e}")
        
        # Default semantic search via query engine
        response = self.query_engine.query(question)
        
        # Extract source information
        source_nodes = []
        contexts: List[Dict[str, str]] = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                if hasattr(node, 'metadata') and 'filename' in node.metadata:
                    source_nodes.append(node.metadata['filename'])
                if hasattr(node, 'text'):  # nodes hold the chunk text
                    contexts.append({
                        "filename": node.metadata.get('filename', 'Unknown') if hasattr(node, 'metadata') else 'Unknown',
                        "text": getattr(node, 'text', ''),
                    })
        
        # Remove duplicates while preserving order
        source_nodes = list(dict.fromkeys(source_nodes))
        
        return ChatResponse(
            answer=str(response),
            source_nodes=source_nodes,
            contexts=contexts,
        )
    
    def stream_query(self, question: str) -> Iterator[str]:
        """
        Stream query response.
        
        Args:
            question: User's question
            
        Yields:
            Text chunks of the response
        """
        if self.query_engine is None:
            raise ValueError("No documents indexed. Please add documents first.")
        
        logger.info(f"Streaming query: {question}")
        streaming_response = self.query_engine.query(question)
        
        # Stream the response
        for text in streaming_response.response_gen:
            yield text


def create_lancedb_chat(
    config: Optional[Dict] = None
) -> LanceDBMultiDocumentChat:
    """
    Factory function to create a LanceDB chat instance.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Configured LanceDBMultiDocumentChat instance
    """
    if config is None:
        config = {}
    
    return LanceDBMultiDocumentChat(
        db_path=config.get("db_path", "./lancedb"),
        table_name=config.get("table_name", "documents"),
        embedding_model=config.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        llm_model=config.get("llm_model", "qwen2.5:14b"),
        llm_base_url=config.get("llm_base_url", "http://localhost:11434"),
        chunk_size=config.get("chunk_size", 512),
        chunk_overlap=config.get("chunk_overlap", 50),
        similarity_top_k=config.get("similarity_top_k", 5),
        search_mode=config.get("search_mode", "semantic"),
        hybrid_weight=config.get("hybrid_weight", 0.7),
        enable_scalar_indexes=config.get("enable_scalar_indexes", True),
        reranker_type=config.get("reranker_type", "linear"),
        reranker_model=config.get("reranker_model"),
        use_matryoshka=config.get("use_matryoshka", True),
        matryoshka_model=config.get("matryoshka_model", "mixedbread-ai/mxbai-embed-large-v1"),
    )

from typing import List, Dict, Any, Optional, Union, Path
import asyncio
from datetime import datetime
import logging
import numpy as np
import torch

from .config.manager import ConfigManager, SystemConfig
from .models.embeddings import EmbeddingModelRegistry, HuggingFaceEmbedding
from .vector_stores.base import VectorStoreRegistry, QdrantVectorStore
from .llm.service import LLMRegistry, HuggingFaceLLM
from .llm.prompt_management import PromptLibrary, PromptManager, PromptTemplate, initialize_prompt_management
from .llm.default_prompts import DefaultPromptTemplates
from .active_learning.learner import ActiveLearner, RelevanceFeedback
from .document_processor.categorizer import DocumentCategorizer, Category
from .workflows.engine import WorkflowEngine, WorkflowStep, WorkflowStatus
from .enhancers.category_filter import CategoryFilter, CategoryFilterConfig
from .enhancers.feedback_analytics import FeedbackAnalytics, FeedbackAnalyticsConfig, FeedbackEntry
from sklearn.feature_extraction.text import TfidfVectorizer # Will be removed if not used elsewhere
import spacy
from dataclasses import dataclass
# Remove BM25Okapi if it was imported directly and no longer used. Assuming it was.
# from rank_bm25 import BM25Okapi
from ..integrations.elasticsearch_fallback import ElasticsearchFallback
from ..integrations.elasticsearch_fallback import ElasticsearchConfig as ESFallbackConfigInternal # Use the one from integrations
import re
from collections import defaultdict
import json
import hashlib
from datetime import datetime
import lmdb
import pickle
import tempfile
import time

from document_processor import DocumentProcessor
from embedding_generator import EmbeddingGenerator
from cache_manager import CacheManager
from text_search_helper import TextSearchHelper
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for hybrid search capabilities"""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    min_score: float = 0.1
    max_results: int = 10
    enable_spell_check: bool = True
    enable_query_expansion: bool = True
    enable_faceted_search: bool = True
    cache_results: bool = True
    cache_dir: str = "search_cache"
    
@dataclass
class CacheConfig:
    """Configuration for caching layer"""
    enabled: bool = True
    cache_dir: str = "cache"
    ttl: int = 3600  # 1 hour
    max_size: int = 1024 * 1024 * 1024  # 1GB

class CacheManager:
    """Manages caching for search results and embeddings"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        if self.config.enabled:
            self._initialize_cache()
            
    def _initialize_cache(self):
        """Initialize LMDB environment"""
        try:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            self.env = lmdb.open(
                str(cache_path),
                map_size=self.config.max_size,
                subdir=True,
                metasync=True,
                sync=True,
                map_async=False,
                mode=0o644
            )
        except Exception as e:
            logger.error(f"Failed to initialize cache: {str(e)}")
            self.config.enabled = False
            
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        if not self.config.enabled:
            return None
            
        try:
            cache_key = self._generate_key(key)
            with self.env.begin() as txn:
                data = txn.get(cache_key.encode())
                if data:
                    cache_data = pickle.loads(data)
                    if time.time() - cache_data['timestamp'] < self.config.ttl:
                        return cache_data['value']
                    # Delete expired entry
                    with self.env.begin(write=True) as wtxn:
                        wtxn.delete(cache_key.encode())
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None
            
    def put(self, key: str, value: Any) -> bool:
        """Store item in cache"""
        if not self.config.enabled:
            return False
            
        try:
            cache_key = self._generate_key(key)
            cache_data = {
                'value': value,
                'timestamp': time.time()
            }
            
            with self.env.begin(write=True) as txn:
                txn.put(cache_key.encode(), pickle.dumps(cache_data))
            return True
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
            return False
            
    def _generate_key(self, key: str) -> str:
        """Generate cache key"""
        return hashlib.md5(key.encode()).hexdigest()

# External RAG Pipeline components (assuming these are imported elsewhere or become part of this class)
from qdrant_client import QdrantClient, models as qdrant_models # Assuming 'models' is used like this
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    def __init__(self,
                 app_config: SystemConfig, # Changed to accept SystemConfig from manager.py
                 doc_processor: DocumentProcessor, # Pass instances
                 embedding_generator: EmbeddingGenerator # Pass instances
                ):
        self.app_config = app_config
        self.doc_processor = doc_processor
        self.embedding_generator = embedding_generator

        self.collection_name = app_config.vector_store.collection_name
        self.embedding_model_name = app_config.model.embedding_model # Use from config

        # Initialize Qdrant client from config
        self.client = QdrantClient(
            host=app_config.vector_store.host,
            port=app_config.vector_store.port
            # Potentially add other Qdrant client configs like api_key, https, etc.
        )
        
        # Initialize embedding model from config
        # This assumes SentenceTransformer can take model_name and device
        self.model = SentenceTransformer(self.embedding_model_name)
        if app_config.model.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        else:
            self.model = self.model.to('cpu') # Default to CPU if cuda not available or specified

        # Initialize search components - these could also be made configurable
        # For now, keeping their direct instantiation but they could take app_config parts
        self.search_config = SearchConfig(
            cache_dir=str(Path(app_config.paths.cache_dir) / "search_cache") # Use configured base cache_dir
        )
        self.cache_config = CacheConfig(
            cache_dir=str(Path(app_config.paths.cache_dir) / "rag_pipeline_cache"), # Specific cache for RAG pipeline
            ttl=app_config.cache_settings.ttl
        )
        self.cache = CacheManager(self.cache_config) # This is the LMDB cache
        self.nlp = self._initialize_nlp()
        # self.vectorizer = TfidfVectorizer(stop_words='english') # Removed as ES handles keyword search

        if self.app_config.feature_flags.enable_elasticsearch_fallback:
            es_config_data = self.app_config.elasticsearch.__dict__
            es_fallback_internal_config = ESFallbackConfigInternal(**es_config_data) # type: ignore
            self.es_fallback = ElasticsearchFallback(config=es_fallback_internal_config)
            # DO NOT call initialize here. It will be called in async_initialize_components.
        else:
            self.es_fallback = None
        
        # Initialize category filter
        self.category_filter = CategoryFilter() # This line was missing in the previous diff attempt's "REPLACE" block
        
        # Initialize feedback analytics
        self.feedback_analytics = FeedbackAnalytics()
        
        # Initialize prompt management system
        prompt_library = initialize_prompt_management()
        self.prompt_manager = PromptManager(library=prompt_library)
        
        # Make sure default templates are loaded
        for template in DefaultPromptTemplates.get_default_templates():
            if not prompt_library.get_template(template.name):
                prompt_library.add_template(template)
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
    def _initialize_nlp(self) -> Optional[Any]:
        """Initialize spaCy NLP model"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to initialize NLP: {str(e)}")
            return None

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with the correct configuration."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Dimension for all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def process_documents(self, input_dir: Union[str, Path], batch_size: int = 32) -> None:
        """Process documents from a directory and store them in Qdrant."""
        input_dir = Path(input_dir)
        
        # Process documents and get chunks
        logger.info("Processing documents...")
        chunks = self.doc_processor.process_directory(input_dir)
        
        if not chunks:
            logger.warning("No documents were processed.")
            return

        # Generate embeddings with GPU acceleration
        logger.info("Generating embeddings...")
        processed_chunks = self.embedding_generator.process_chunks(chunks, batch_size=batch_size)
        
        # Prepare points for Qdrant
        points = []
        for i, chunk in enumerate(processed_chunks):
            points.append(models.PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
            ))
        
        # Upload to Qdrant in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        # Clear cache when new documents are added
        # self.cache_manager.clear_cache_by_prefix(self.collection_name) # This CacheManager is LMDB, might not have this method
        logger.info(f"Successfully processed and stored {len(points)} chunks in Qdrant")

    async def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None, categories: Optional[Union[List[str], str]] = None) -> List[Dict[str, Any]]:
        """Enhanced hybrid search combining semantic and keyword-based approaches with category filtering. Now asynchronous.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            filters: Dictionary of metadata filters to apply
            categories: Category or list of categories to filter by
            
        Returns:
            Ranked and filtered search results
        """
        
        # Check cache first
        cache_key = f"search:{query}:{limit}:{json.dumps(filters or {})}:{json.dumps(categories or [])}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            return cached_results
            
        # Process query
        processed_query = self._preprocess_query(query)
        
        # Get semantic search results
        semantic_results = self._semantic_search(processed_query, limit * 2, filters)  # Get more results for filtering
        
        # Get keyword search results
        keyword_results = self._keyword_search(processed_query, limit * 2, filters)  # Get more results for filtering
        
        # Combine and rank results
        combined_results = self._combine_search_results(
            semantic_results,
            keyword_results,
            query
        )
        
        # Apply faceted filtering if enabled and filters provided
        if self.search_config.enable_faceted_search and filters:
            combined_results = self._apply_filters(combined_results, filters)
        
        # Apply category filtering if categories provided
        if categories:
            combined_results = self.category_filter.filter_by_categories(combined_results, categories)
            
        # Sort and limit results
        final_results = sorted(
            combined_results,
            key=lambda x: x['score'],
            reverse=True
        )[:limit]
        
        # Update category statistics
        self.category_filter.update_category_stats(final_results)
        
        # Cache results
        self.cache.put(cache_key, final_results)
        
        return final_results
        
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and expand query"""
        if not self.nlp:
            return query.lower().strip()
            
        # Basic cleaning
        query = query.lower().strip()
        
        # Spell checking
        if self.search_config.enable_spell_check:
            query = self._spell_check(query)
            
        # Query expansion
        if self.search_config.enable_query_expansion:
            query = self._expand_query(query)
            
        return query
        
    def _spell_check(self, text: str) -> str:
        """Apply spell checking to text"""
        doc = self.nlp(text)
        corrected = []
        
        for token in doc:
            if not token.is_punct and not token.is_space:
                # Use token vector similarity for suggestions
                if token.has_vector:
                    most_similar = self.nlp.vocab.vectors.most_similar(
                        token.vector.reshape(1, -1),
                        n=1
                    )
                    if most_similar[2][0] > 0.9:  # Similarity threshold
                        corrected.append(self.nlp.vocab[most_similar[0][0]].text)
                    else:
                        corrected.append(token.text)
                else:
                    corrected.append(token.text)
                    
        return ' '.join(corrected)
        
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        doc = self.nlp(query)
        expanded_terms = set(query.split())
        
        for token in doc:
            # Add lemmatized form
            if token.lemma_ != token.text:
                expanded_terms.add(token.lemma_)
                
            # Add synonyms using word vectors
            if token.has_vector:
                most_similar = self.nlp.vocab.vectors.most_similar(
                    token.vector.reshape(1, -1),
                    n=2
                )
                for word_id in most_similar[0][0]:
                    similar_word = self.nlp.vocab[word_id].text
                    if similar_word != token.text:
                        expanded_terms.add(similar_word)
                        
        return ' '.join(expanded_terms)
        
    def _semantic_search(self, query: str, limit: int, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using Qdrant"""
        query_vector = self.model.encode(query, convert_to_numpy=True)
        
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False
        )
        
        query_filter = None
        if filters:
            query_filter = self._build_qdrant_filter(filters)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            search_params=search_params,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                'text': result.payload.get('text', ''),
                'metadata': result.payload.get('metadata', {}),
                'semantic_score': float(result.score)
            }
            for result in results
        ]
        
    def _keyword_search(self, query: str, limit: int, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform keyword-based search using BM25"""
        # Get all documents from collection
        documents = self._get_documents(filters)
        
        if not documents:
            return []
            
        # Prepare texts for BM25
        texts = [doc['text'] for doc in documents]
        tokenized_texts = [text.split() for text in texts]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_texts)
        
        # Get scores
        scores = bm25.get_scores(query.split())
        
        # Combine with documents and sort
        results = [
            {
                'text': documents[i]['text'],
                'metadata': documents[i]['metadata'],
                'keyword_score': float(score)
            }
            for i, score in enumerate(scores)
        ]
        
        return sorted(results, key=lambda x: x['keyword_score'], reverse=True)[:limit]
        
    def _combine_search_results(self,
                              semantic_results: List[Dict[str, Any]],
                              keyword_results: List[Dict[str, Any]],
                              query: str) -> List[Dict[str, Any]]:
        """Combine and normalize semantic and keyword search results"""
        combined_results = defaultdict(dict)
        
        # Process semantic results
        for result in semantic_results:
            doc_id = self._get_document_id(result['text'])
            combined_results[doc_id].update(result)
            combined_results[doc_id]['semantic_score'] = result['semantic_score']
            
        # Process keyword results
        for result in keyword_results:
            doc_id = self._get_document_id(result['text'])
            if doc_id not in combined_results:
                combined_results[doc_id].update(result)
            combined_results[doc_id]['keyword_score'] = result['keyword_score']
            
        # Calculate final scores
        final_results = []
        for doc_id, result in combined_results.items():
            semantic_score = result.get('semantic_score', 0.0)
            keyword_score = result.get('keyword_score', 0.0)
            
            # Calculate weighted score
            final_score = (
                self.search_config.semantic_weight * semantic_score +
                self.search_config.keyword_weight * keyword_score
            )
            
            if final_score >= self.search_config.min_score:
                result['score'] = final_score
                final_results.append(result)
                
        return final_results
        
    def _get_document_id(self, text: str) -> str:
        """Generate consistent document ID"""
        return hashlib.md5(text.encode()).hexdigest()
        
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict) -> List[Dict[str, Any]]:
        """Apply metadata filters to results"""
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            if self._matches_filters(metadata, filters):
                filtered_results.append(result)
                
        return filtered_results
        
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        for field, value in filters.items():
            if field not in metadata:
                return False
                
            meta_value = metadata[field]
            
            # Handle different filter types
            if isinstance(value, (list, tuple)):
                if meta_value not in value:
                    return False
            elif isinstance(value, dict):
                # Handle range queries
                if 'gte' in value and meta_value < value['gte']:
                    return False
                if 'lte' in value and meta_value > value['lte']:
                    return False
                if 'gt' in value and meta_value <= value['gt']:
                    return False
                if 'lt' in value and meta_value >= value['lt']:
                    return False
            elif meta_value != value:
                return False
                
        return True

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        cache_stats = self.cache_manager.get_cache_stats(self.collection_name)
        
        return {
            "name": collection_info.name,
            "points_count": collection_info.points_count,
            "vectors_config": collection_info.config.params.vectors,
            "status": collection_info.status,
            "cache_stats": cache_stats
        }
    
    def get_popular_categories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular categories based on usage.
        
        Args:
            limit: Maximum number of categories to return
            
        Returns:
            List of popular categories with usage counts
        """
        return self.category_filter.get_popular_categories(limit)
    
    def suggest_categories(self, prefix: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Suggest categories based on prefix and usage statistics.
        
        Args:
            prefix: Category prefix to filter by
            limit: Maximum number of suggestions
            
        Returns:
            List of category suggestions
        """
        return self.category_filter.suggest_categories(prefix, limit)
    
    def update_category_hierarchy(self, hierarchy: Dict[str, str]):
        """Update category hierarchy.
        
        Args:
            hierarchy: Dictionary mapping category ID to parent category ID
        """
        self.category_filter.update_category_hierarchy(hierarchy)
    
    def update_category_synonyms(self, synonyms: Dict[str, List[str]]):
        """Update category synonyms.
        
        Args:
            synonyms: Dictionary mapping category ID to list of synonyms
        """
        self.category_filter.update_category_synonyms(synonyms)

    def add_feedback(
        self, 
        query: str, 
        document_id: str, 
        document_text: str, 
        relevance_score: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add feedback for a search result.
        
        Args:
            query: The original search query
            document_id: ID of the document for feedback
            document_text: Text content of the document
            relevance_score: Relevance score (0.0-1.0)
            user_id: Optional user ID
            session_id: Optional session ID
            metadata: Optional metadata
        """
        feedback = FeedbackEntry(
            query=query,
            document_id=document_id,
            document_text=document_text,
            relevance_score=relevance_score,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.feedback_analytics.add_feedback(feedback)
        logger.info(f"Added feedback for query '{query}', document {document_id}, score: {relevance_score}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics.
        
        Returns:
            Dictionary with feedback statistics
        """
        return self.feedback_analytics.get_feedback_stats()
    
    def generate_feedback_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a feedback analytics report.
        
        Args:
            output_path: Path to save the report visualizations
            
        Returns:
            Dictionary with report data
        """
        return self.feedback_analytics.generate_analytics_report(output_path)
    
    def trigger_model_finetune(self) -> bool:
        """Trigger model fine-tuning based on collected feedback.
        
        Returns:
            True if fine-tuning was successful, False otherwise
        """
        return self.feedback_analytics.trigger_model_finetune()

    def generate_response(self, question: str, search_results: List[Dict[str, Any]], template_name: str = "qa_prompt") -> Dict[str, Any]:
        """Generate a response to a question using the retrieved context
        
        Args:
            question: The user's question
            search_results: The search results from the RAG pipeline
            template_name: The prompt template to use
            
        Returns:
            A dictionary containing the response and metadata
        """
        # Get LLM service
        llm_service = LLMRegistry.get_default_llm()
        
        # Format context from search results
        context = self._format_context_from_results(search_results)
        
        # Get prompt from prompt management system
        try:
            template = self.prompt_manager.library.get_template(template_name)
            if not template:
                logger.warning(f"Prompt template '{template_name}' not found. Using default template.")
                template = self.prompt_manager.library.get_template("qa_prompt")
                
            prompt = template.format(context=context, question=question)
        except Exception as e:
            # Fallback to manual prompt construction if template not found or error
            logger.warning(f"Error using prompt template: {e}. Using default format.")
            prompt = f"Answer the question based on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        
        # Generate response
        response = llm_service.generate(prompt)
        
        # Track prompt performance
        try:
            metrics = {
                "query_length": len(question),
                "context_length": len(context),
                "response_length": len(response),
                "results_count": len(search_results),
                "top_score": search_results[0]["score"] if search_results else 0
            }
            
            # Record as a version to track performance
            version = template.version if template else "1.0"
            self.prompt_manager.update_version_metrics(template_name, version, metrics)
        except Exception as e:
            logger.warning(f"Failed to track prompt performance: {e}")
        
        # Return response with metadata
        return {
            "question": question,
            "answer": response,
            "sources": [result.get("metadata", {}).get("source", f"Document {i}") for i, result in enumerate(search_results[:3])],
            "confidence": self._calculate_response_confidence(search_results),
            "metadata": {
                "prompt_template": template_name,
                "result_count": len(search_results),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _format_context_from_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a context string for the LLM
        
        Args:
            results: Search results from the RAG pipeline
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results[:5]):  # Use top 5 results
            content = result.get("content", "")
            source = result.get("metadata", {}).get("source", f"Document {i+1}")
            score = result.get("score", 0)
            
            # Format each result with source and relevance info
            context_parts.append(f"[Source: {source}] (Relevance: {score:.2f})\n{content}\n")
            
        return "\n".join(context_parts)
    
    def _calculate_response_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the generated response
        
        Args:
            results: Search results used for the response
            
        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0
            
        # Use a weighted average of top result scores
        weights = [0.5, 0.3, 0.1, 0.05, 0.05]  # Weights for top 5 results
        scores = [result.get("score", 0) for result in results[:5]]
        
        # Pad scores if less than 5 results
        while len(scores) < 5:
            scores.append(0)
            
        # Calculate weighted average
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        
        return min(1.0, max(0.0, weighted_score))

if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline(collection_name="documents")
    
    # Process documents from a directory
    pipeline.process_documents("path/to/documents")
    
    # Search for documents without category filter
    print("Search without category filter:")
    results = pipeline.search("your query here")
    for result in results:
        print(f"Combined Score: {result['score']:.4f}")
        print(f"Semantic Score: {result['semantic_score']:.4f}")
        print(f"Keyword Score: {result['keyword_score']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Source: {result['metadata']['source']}")
        print("-" * 80)
        
    # Search with category filter
    print("\nSearch with category filter:")
    results = pipeline.search("your query here", categories=["technical", "documentation"])
    for result in results:
        print(f"Combined Score: {result['score']:.4f}")
        print(f"Semantic Score: {result['semantic_score']:.4f}")
        print(f"Keyword Score: {result['keyword_score']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Source: {result['metadata']['source']}")
        print(f"Categories: {result['metadata'].get('categories', [])}")
        print("-" * 80)
        
    # Get popular categories
    print("\nPopular categories:")
    popular_categories = pipeline.get_popular_categories(5)
    for category in popular_categories:
        print(f"Category: {category['id']}, Count: {category['count']}")

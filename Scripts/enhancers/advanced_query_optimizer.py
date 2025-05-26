"""
Advanced Query Optimization System for RAG Pipeline

This module provides comprehensive query optimization including:
- Multi-modal query processing
- Semantic query expansion
- Intelligent query routing
- Query performance optimization
- Dynamic embedding model selection
- Query intent analysis
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from enum import Enum
import json
import re
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

# Advanced NLP and ML imports
try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class QueryType(Enum):
    """Query type classification."""
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    COMPARATIVE = "comparative"

@dataclass
class QueryAnalysis:
    """Comprehensive query analysis results."""
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    query_type: QueryType
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    semantic_variations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    estimated_processing_time: float = 0.0
    recommended_embedding_model: str = "default"
    query_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    """Configuration for advanced query optimization."""
    enable_semantic_expansion: bool = True
    enable_entity_extraction: bool = True
    enable_intent_analysis: bool = True
    enable_query_rewriting: bool = True
    enable_multi_embedding: bool = True
    enable_query_caching: bool = True
    enable_performance_optimization: bool = True
    
    # Expansion settings
    max_expanded_terms: int = 10
    semantic_similarity_threshold: float = 0.7
    expansion_diversity_threshold: float = 0.3
    
    # Performance settings
    max_processing_time: float = 30.0
    enable_early_stopping: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Caching settings
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Model settings
    spacy_model: str = "en_core_web_sm"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    intent_classification_model: str = "facebook/bart-large-mnli"

class AdvancedQueryOptimizer:
    """Advanced query optimization system for RAG pipeline."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize the advanced query optimizer."""
        self.config = config or OptimizationConfig()
        self.nlp = None
        self.sentence_transformer = None
        self.intent_classifier = None
        self.tfidf_vectorizer = None
        self.query_cache: Dict[str, Tuple[QueryAnalysis, datetime]] = {}
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        self.entity_cache: Dict[str, Set[str]] = {}
        self.expansion_cache: Dict[str, List[str]] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.lock = threading.RLock()
        
        # Query patterns for intent classification
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what|when|where|who|which)\b',
                r'\b(is|are|was|were)\b.*\?',
                r'\bdefine\b',
                r'\btell me about\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\b(why|how|analyze|explain)\b',
                r'\bcause\b',
                r'\breason\b',
                r'\bimpact\b'
            ],
            QueryIntent.COMPARISON: [
                r'\b(compare|contrast|difference|similar|versus|vs)\b',
                r'\bbetter|worse|best|worst\b',
                r'\bmore.*than|less.*than\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(how to|steps|process|procedure|guide)\b',
                r'\binstructions\b',
                r'\btutorial\b'
            ]
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and components."""
        if not ADVANCED_NLP_AVAILABLE:
            logger.warning("Advanced NLP libraries not available. Some features will be limited.")
            return
        
        try:
            # Initialize spaCy for entity extraction
            if self.config.enable_entity_extraction:
                try:
                    self.nlp = spacy.load(self.config.spacy_model)
                except OSError:
                    logger.warning(f"spaCy model '{self.config.spacy_model}' not found. Install with: python -m spacy download {self.config.spacy_model}")
                    self.nlp = None
            
            # Initialize sentence transformer for semantic similarity
            if self.config.enable_semantic_expansion:
                try:
                    self.sentence_transformer = SentenceTransformer(self.config.sentence_transformer_model)
                except Exception as e:
                    logger.warning(f"Failed to load sentence transformer: {e}")
                    self.sentence_transformer = None
            
            # Initialize intent classifier
            if self.config.enable_intent_analysis:
                try:
                    self.intent_classifier = pipeline(
                        "zero-shot-classification",
                        model=self.config.intent_classification_model
                    )
                except Exception as e:
                    logger.warning(f"Failed to load intent classifier: {e}")
                    self.intent_classifier = None
            
            # Initialize TF-IDF vectorizer for keyword extraction
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            logger.info("Advanced query optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def optimize_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """
        Perform comprehensive query optimization.
        
        Args:
            query: Input query string
            context: Additional context for optimization
            
        Returns:
            QueryAnalysis object with optimization results
        """
        start_time = time.time()
        
        # Check cache first
        if self.config.enable_query_caching:
            cached_result = self._get_cached_result(query)
            if cached_result:
                logger.info(f"Retrieved query analysis from cache for: {query[:50]}...")
                return cached_result
        
        try:
            # Parallel processing of different analysis components
            tasks = []
            
            if self.config.parallel_processing:
                # Run analysis components in parallel
                tasks = [
                    self._analyze_intent(query),
                    self._extract_entities(query),
                    self._extract_keywords(query),
                    self._classify_complexity(query),
                    self._classify_query_type(query)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                intent, entities, keywords, complexity, query_type = results
            else:
                # Sequential processing
                intent = await self._analyze_intent(query)
                entities = await self._extract_entities(query)
                keywords = await self._extract_keywords(query)
                complexity = await self._classify_complexity(query)
                query_type = await self._classify_query_type(query)
            
            # Handle any exceptions from parallel processing
            if isinstance(intent, Exception):
                intent = QueryIntent.UNKNOWN
            if isinstance(entities, Exception):
                entities = []
            if isinstance(keywords, Exception):
                keywords = []
            if isinstance(complexity, Exception):
                complexity = QueryComplexity.MODERATE
            if isinstance(query_type, Exception):
                query_type = QueryType.SINGLE_HOP
            
            # Semantic expansion
            expanded_terms = []
            semantic_variations = []
            if self.config.enable_semantic_expansion:
                expanded_terms = await self._expand_query_semantically(query, keywords)
                semantic_variations = await self._generate_semantic_variations(query)
            
            # Query rewriting if needed
            if self.config.enable_query_rewriting:
                query = await self._rewrite_query_if_needed(query, intent, complexity)
            
            # Recommend optimal embedding model
            recommended_model = self._recommend_embedding_model(intent, complexity, query_type)
            
            # Generate query vector
            query_vector = await self._generate_query_vector(query)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                intent, entities, keywords, complexity
            )
            
            # Estimate processing time
            processing_time = time.time() - start_time
            estimated_time = self._estimate_processing_time(complexity, query_type)
            
            # Create analysis result
            analysis = QueryAnalysis(
                original_query=query,
                intent=intent,
                complexity=complexity,
                query_type=query_type,
                entities=entities,
                keywords=keywords,
                expanded_terms=expanded_terms,
                semantic_variations=semantic_variations,
                confidence_score=confidence_score,
                estimated_processing_time=estimated_time,
                recommended_embedding_model=recommended_model,
                query_vector=query_vector,
                metadata={
                    'processing_time': processing_time,
                    'context': context or {},
                    'timestamp': datetime.now().isoformat(),
                    'optimizer_version': '1.0'
                }
            )
            
            # Cache the result
            if self.config.enable_query_caching:
                self._cache_result(query, analysis)
            
            # Update performance stats
            self._update_performance_stats(processing_time, complexity)
            
            logger.info(f"Query optimization completed in {processing_time:.3f}s for: {query[:50]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Error optimizing query '{query}': {e}")
            # Return basic analysis on error
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.UNKNOWN,
                complexity=QueryComplexity.MODERATE,
                query_type=QueryType.SINGLE_HOP,
                confidence_score=0.0,
                estimated_processing_time=1.0,
                recommended_embedding_model="default"
            )
    
    async def _analyze_intent(self, query: str) -> QueryIntent:
        """Analyze query intent using pattern matching and ML classification."""
        try:
            # Pattern-based classification first (fast)
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query.lower()):
                        return intent
            
            # ML-based classification if available
            if self.intent_classifier and ADVANCED_NLP_AVAILABLE:
                candidate_labels = [intent.value for intent in QueryIntent if intent != QueryIntent.UNKNOWN]
                result = self.intent_classifier(query, candidate_labels)
                if result['scores'][0] > 0.6:  # Confidence threshold
                    return QueryIntent(result['labels'][0])
            
            return QueryIntent.FACTUAL  # Default
            
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            return QueryIntent.UNKNOWN
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        try:
            # Check cache first
            if query in self.entity_cache:
                return list(self.entity_cache[query])
            
            entities = []
            
            if self.nlp:
                doc = self.nlp(query)
                entities = [ent.text for ent in doc.ents if ent.label_ in [
                    'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                    'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                    'QUANTITY', 'ORDINAL', 'CARDINAL'
                ]]
            
            # Fallback: simple pattern matching for common entities
            if not entities:
                # Simple patterns for dates, numbers, etc.
                date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
                number_pattern = r'\b\d+(?:\.\d+)?\b'
                
                entities.extend(re.findall(date_pattern, query))
                entities.extend(re.findall(number_pattern, query))
            
            # Cache the result
            self.entity_cache[query] = set(entities)
            
            return entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        try:
            # Use spaCy for advanced keyword extraction if available
            if self.nlp:
                doc = self.nlp(query)
                keywords = [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_stop and not token.is_punct and len(token.text) > 2
                    and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']
                ]
            else:
                # Fallback: simple tokenization and filtering
                import string
                query_clean = query.translate(str.maketrans('', '', string.punctuation))
                tokens = query_clean.lower().split()
                
                # Simple stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
                keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for keyword in keywords:
                if keyword not in seen:
                    unique_keywords.append(keyword)
                    seen.add(keyword)
            
            return unique_keywords[:15]  # Limit to top 15 keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return query.lower().split()[:10]  # Simple fallback
    
    async def _classify_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity based on various factors."""
        try:
            complexity_score = 0
            
            # Length factor
            if len(query.split()) > 20:
                complexity_score += 2
            elif len(query.split()) > 10:
                complexity_score += 1
            
            # Question words and complexity indicators
            complex_indicators = [
                'compare', 'analyze', 'evaluate', 'synthesize', 'relationship',
                'correlation', 'impact', 'consequence', 'multiple', 'various',
                'different', 'several', 'why', 'how', 'explain'
            ]
            
            for indicator in complex_indicators:
                if indicator in query.lower():
                    complexity_score += 1
            
            # Multiple questions
            if query.count('?') > 1:
                complexity_score += 1
            
            # Conditional statements
            if any(word in query.lower() for word in ['if', 'when', 'unless', 'provided']):
                complexity_score += 1
            
            # Classify based on score
            if complexity_score >= 5:
                return QueryComplexity.VERY_COMPLEX
            elif complexity_score >= 3:
                return QueryComplexity.COMPLEX
            elif complexity_score >= 1:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.SIMPLE
                
        except Exception as e:
            logger.warning(f"Complexity classification failed: {e}")
            return QueryComplexity.MODERATE
    
    async def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query for optimal processing."""
        try:
            query_lower = query.lower()
            
            # Temporal queries
            temporal_indicators = ['when', 'time', 'date', 'year', 'month', 'day', 'ago', 'future', 'past', 'recent']
            if any(indicator in query_lower for indicator in temporal_indicators):
                return QueryType.TEMPORAL
            
            # Spatial queries
            spatial_indicators = ['where', 'location', 'place', 'near', 'distance', 'map', 'country', 'city']
            if any(indicator in query_lower for indicator in spatial_indicators):
                return QueryType.SPATIAL
            
            # Comparative queries
            comparative_indicators = ['compare', 'versus', 'vs', 'difference', 'better', 'worse', 'similar']
            if any(indicator in query_lower for indicator in comparative_indicators):
                return QueryType.COMPARATIVE
            
            # Aggregation queries
            aggregation_indicators = ['total', 'sum', 'average', 'count', 'number', 'how many', 'statistics']
            if any(indicator in query_lower for indicator in aggregation_indicators):
                return QueryType.AGGREGATION
            
            # Multi-hop indicators
            multi_hop_indicators = ['and then', 'followed by', 'after that', 'multiple', 'several']
            if any(indicator in query_lower for indicator in multi_hop_indicators) or query.count('?') > 1:
                return QueryType.MULTI_HOP
            
            return QueryType.SINGLE_HOP
            
        except Exception as e:
            logger.warning(f"Query type classification failed: {e}")
            return QueryType.SINGLE_HOP
    
    async def _expand_query_semantically(self, query: str, keywords: List[str]) -> List[str]:
        """Generate semantic expansions of the query."""
        try:
            if not self.sentence_transformer or not keywords:
                return []
            
            # Check cache first
            cache_key = f"{query}:{','.join(keywords)}"
            if cache_key in self.expansion_cache:
                return self.expansion_cache[cache_key]
            
            # Generate embeddings for keywords
            keyword_embeddings = self.sentence_transformer.encode(keywords)
            
            # Predefined expansion terms (domain-specific)
            expansion_candidates = [
                # Synonyms and related terms
                'information', 'data', 'details', 'facts', 'evidence',
                'research', 'study', 'analysis', 'report', 'documentation',
                'explanation', 'description', 'overview', 'summary',
                'example', 'instance', 'case', 'scenario'
            ]
            
            # Filter candidates based on semantic similarity
            if len(expansion_candidates) > 0:
                candidate_embeddings = self.sentence_transformer.encode(expansion_candidates)
                
                expanded_terms = []
                for i, keyword in enumerate(keywords):
                    similarities = cosine_similarity(
                        [keyword_embeddings[i]], candidate_embeddings
                    )[0]
                    
                    for j, sim in enumerate(similarities):
                        if sim > self.config.semantic_similarity_threshold:
                            expanded_terms.append(expansion_candidates[j])
                
                # Remove duplicates and limit results
                expanded_terms = list(set(expanded_terms))[:self.config.max_expanded_terms]
                
                # Cache the result
                self.expansion_cache[cache_key] = expanded_terms
                
                return expanded_terms
            
            return []
            
        except Exception as e:
            logger.warning(f"Semantic expansion failed: {e}")
            return []
    
    async def _generate_semantic_variations(self, query: str) -> List[str]:
        """Generate semantic variations of the query."""
        try:
            variations = []
            
            # Simple rule-based variations
            # Question to statement conversion
            if query.endswith('?'):
                variations.append(query[:-1])
            
            # Add question forms
            if not query.endswith('?'):
                variations.append(f"What is {query}?")
                variations.append(f"Tell me about {query}")
            
            # Rephrase with different question words
            question_words = ['what', 'how', 'why', 'when', 'where']
            for word in question_words:
                if word not in query.lower():
                    variations.append(f"{word.capitalize()} {query.lower().lstrip('what how why when where ')}")
            
            return variations[:5]  # Limit variations
            
        except Exception as e:
            logger.warning(f"Semantic variation generation failed: {e}")
            return []
    
    async def _rewrite_query_if_needed(self, query: str, intent: QueryIntent, complexity: QueryComplexity) -> str:
        """Rewrite query for better processing if needed."""
        try:
            # Only rewrite very complex queries or those with unclear intent
            if complexity != QueryComplexity.VERY_COMPLEX and intent != QueryIntent.UNKNOWN:
                return query
            
            # Simple query cleaning and normalization
            # Remove redundant words
            redundant_phrases = [
                'can you tell me', 'i want to know', 'please explain',
                'i would like to understand', 'could you help me with'
            ]
            
            cleaned_query = query.lower()
            for phrase in redundant_phrases:
                cleaned_query = cleaned_query.replace(phrase, '')
            
            # Normalize whitespace
            cleaned_query = ' '.join(cleaned_query.split())
            
            # Capitalize first letter
            if cleaned_query:
                cleaned_query = cleaned_query[0].upper() + cleaned_query[1:]
            
            return cleaned_query if cleaned_query else query
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query
    
    def _recommend_embedding_model(self, intent: QueryIntent, complexity: QueryComplexity, query_type: QueryType) -> str:
        """Recommend optimal embedding model based on query characteristics."""
        try:
            # Model recommendations based on query characteristics
            if complexity == QueryComplexity.VERY_COMPLEX:
                return "text-embedding-3-large"  # Most capable model
            elif intent in [QueryIntent.ANALYTICAL, QueryIntent.COMPARISON]:
                return "text-embedding-3-small"  # Good for reasoning
            elif query_type in [QueryType.TEMPORAL, QueryType.SPATIAL]:
                return "arctic-embed-2"  # Good for factual queries
            elif complexity == QueryComplexity.SIMPLE:
                return "all-MiniLM-L6-v2"  # Fast and efficient
            else:
                return "default"  # Use system default
                
        except Exception as e:
            logger.warning(f"Model recommendation failed: {e}")
            return "default"
    
    async def _generate_query_vector(self, query: str) -> Optional[List[float]]:
        """Generate vector representation of the query."""
        try:
            if self.sentence_transformer:
                embedding = self.sentence_transformer.encode(query)
                return embedding.tolist()
            return None
            
        except Exception as e:
            logger.warning(f"Query vector generation failed: {e}")
            return None
    
    def _calculate_confidence_score(self, intent: QueryIntent, entities: List[str], 
                                  keywords: List[str], complexity: QueryComplexity) -> float:
        """Calculate confidence score for the analysis."""
        try:
            score = 0.5  # Base score
            
            # Intent confidence
            if intent != QueryIntent.UNKNOWN:
                score += 0.2
            
            # Entity extraction success
            if entities:
                score += min(0.2, len(entities) * 0.05)
            
            # Keyword extraction success
            if keywords:
                score += min(0.1, len(keywords) * 0.02)
            
            # Complexity assessment confidence
            if complexity != QueryComplexity.MODERATE:  # Default fallback
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _estimate_processing_time(self, complexity: QueryComplexity, query_type: QueryType) -> float:
        """Estimate processing time based on query characteristics."""
        try:
            base_time = 1.0  # Base processing time in seconds
            
            # Complexity multiplier
            complexity_multipliers = {
                QueryComplexity.SIMPLE: 0.5,
                QueryComplexity.MODERATE: 1.0,
                QueryComplexity.COMPLEX: 2.0,
                QueryComplexity.VERY_COMPLEX: 3.0
            }
            
            # Query type multiplier
            type_multipliers = {
                QueryType.SINGLE_HOP: 1.0,
                QueryType.MULTI_HOP: 2.5,
                QueryType.AGGREGATION: 1.5,
                QueryType.TEMPORAL: 1.2,
                QueryType.SPATIAL: 1.2,
                QueryType.COMPARATIVE: 2.0
            }
            
            estimated_time = base_time * complexity_multipliers.get(complexity, 1.0) * type_multipliers.get(query_type, 1.0)
            
            return min(estimated_time, self.config.max_processing_time)
            
        except Exception as e:
            logger.warning(f"Processing time estimation failed: {e}")
            return 1.0
    
    def _get_cached_result(self, query: str) -> Optional[QueryAnalysis]:
        """Retrieve cached query analysis if available and not expired."""
        try:
            with self.lock:
                if query in self.query_cache:
                    result, timestamp = self.query_cache[query]
                    if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl_seconds):
                        return result
                    else:
                        # Remove expired entry
                        del self.query_cache[query]
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_result(self, query: str, analysis: QueryAnalysis):
        """Cache query analysis result."""
        try:
            with self.lock:
                # Implement LRU-like cache management
                if len(self.query_cache) >= self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.query_cache.keys(), 
                                   key=lambda k: self.query_cache[k][1])
                    del self.query_cache[oldest_key]
                
                self.query_cache[query] = (analysis, datetime.now())
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _update_performance_stats(self, processing_time: float, complexity: QueryComplexity):
        """Update performance statistics."""
        try:
            with self.lock:
                self.performance_stats['total_queries'].append(processing_time)
                self.performance_stats[f'{complexity.value}_queries'].append(processing_time)
                
                # Keep only recent stats (last 1000 queries)
                for key in self.performance_stats:
                    if len(self.performance_stats[key]) > 1000:
                        self.performance_stats[key] = self.performance_stats[key][-1000:]
                        
        except Exception as e:
            logger.warning(f"Performance stats update failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            with self.lock:
                metrics = {}
                
                for category, times in self.performance_stats.items():
                    if times:
                        metrics[category] = {
                            'count': len(times),
                            'avg_time': np.mean(times),
                            'min_time': np.min(times),
                            'max_time': np.max(times),
                            'p95_time': np.percentile(times, 95) if len(times) > 1 else times[0]
                        }
                
                metrics['cache_hit_rate'] = len(self.query_cache) / max(1, len(self.performance_stats.get('total_queries', [1])))
                metrics['cache_size'] = len(self.query_cache)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all caches."""
        try:
            with self.lock:
                self.query_cache.clear()
                self.entity_cache.clear()
                self.expansion_cache.clear()
                logger.info("All caches cleared")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass

# Global optimizer instance
_global_optimizer: Optional[AdvancedQueryOptimizer] = None

def get_query_optimizer(config: OptimizationConfig = None) -> AdvancedQueryOptimizer:
    """Get or create global query optimizer instance."""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = AdvancedQueryOptimizer(config)
    
    return _global_optimizer

# Example usage and testing
async def test_query_optimizer():
    """Test the advanced query optimizer."""
    config = OptimizationConfig(
        enable_semantic_expansion=True,
        enable_entity_extraction=True,
        enable_intent_analysis=True,
        max_expanded_terms=5
    )
    
    optimizer = AdvancedQueryOptimizer(config)
    
    test_queries = [
        "What is machine learning?",
        "Compare the performance of different neural network architectures for computer vision tasks",
        "How do I implement a transformer model in PyTorch?",
        "What are the latest developments in quantum computing and how do they impact AI?",
        "Who invented the transformer architecture and when?"
    ]
    
    for query in test_queries:
        print(f"\nAnalyzing: {query}")
        analysis = await optimizer.optimize_query(query)
        
        print(f"Intent: {analysis.intent.value}")
        print(f"Complexity: {analysis.complexity.value}")
        print(f"Type: {analysis.query_type.value}")
        print(f"Entities: {analysis.entities}")
        print(f"Keywords: {analysis.keywords}")
        print(f"Expanded terms: {analysis.expanded_terms}")
        print(f"Confidence: {analysis.confidence_score:.2f}")
        print(f"Recommended model: {analysis.recommended_embedding_model}")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = optimizer.get_performance_metrics()
    for category, stats in metrics.items():
        if isinstance(stats, dict):
            print(f"{category}: {stats}")

if __name__ == "__main__":
    asyncio.run(test_query_optimizer())

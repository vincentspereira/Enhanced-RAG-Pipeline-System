from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import spacy
import re
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import faiss
from enum import Enum

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

@dataclass
class SearchConfig:
    mode: SearchMode = SearchMode.HYBRID
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    use_gpu: bool = True
    top_k: int = 10
    min_score: float = 0.5
    use_bm25: bool = True
    enable_spellcheck: bool = True
    enable_query_expansion: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    use_faiss: bool = True
    distance_metric: str = "cosine"  # or "euclidean", "dot_product"

class HybridSearch:
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self._init_components()

    def _init_components(self):
        """Initialize search components"""
        # Initialize semantic search model
        try:
            self.semantic_model = SentenceTransformer(self.config.model_name)
            if self.config.use_gpu and torch.cuda.is_available():
                self.semantic_model = self.semantic_model.to('cuda')
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.semantic_model = None

        # Initialize keyword search components
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english'
        )
        
        # Initialize spaCy for query expansion
        if self.config.enable_query_expansion:
            try:
                self.nlp = spacy.load('en_core_web_md')
            except:
                logger.warning("Downloading spaCy model...")
                spacy.cli.download('en_core_web_md')
                self.nlp = spacy.load('en_core_web_md')

        # Initialize FAISS index if configured
        if self.config.use_faiss:
            self.index = None  # Will be initialized when vectors are added

    def build_faiss_index(self, vectors: np.ndarray):
        """Build FAISS index for fast similarity search"""
        if not self.config.use_faiss:
            return

        dimension = vectors.shape[1]
        if self.config.distance_metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(vectors)  # Normalize vectors for cosine similarity
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance for euclidean

        if self.config.use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(vectors.astype('float32'))

    def semantic_search(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Perform semantic search using sentence transformers"""
        if not self.semantic_model:
            return [(i, 0.0) for i in range(len(documents))]

        try:
            # Encode query and documents
            query_embedding = self.semantic_model.encode(
                query,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            doc_embeddings = self.semantic_model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size
            )

            # Calculate similarities
            if self.config.use_faiss and self.index:
                # Use FAISS for fast similarity search
                query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
                if self.config.distance_metric == "cosine":
                    faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding, len(documents))
                results = list(zip(indices[0], scores[0]))
            else:
                # Calculate similarities using dot product
                similarities = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    doc_embeddings
                )
                results = [(i, score.item()) for i, score in enumerate(similarities)]

            # Sort by similarity score
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:self.config.top_k]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [(i, 0.0) for i in range(len(documents))]

    def keyword_search(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Perform keyword-based search using BM25 or TF-IDF"""
        try:
            if self.config.use_bm25:
                # Tokenize documents
                tokenized_docs = [doc.lower().split() for doc in documents]
                bm25 = BM25Okapi(tokenized_docs)
                
                # Get BM25 scores
                scores = bm25.get_scores(query.lower().split())
                results = [(i, score) for i, score in enumerate(scores)]
            else:
                # Use TF-IDF for keyword search
                tfidf_matrix = self.vectorizer.fit_transform(documents)
                query_vector = self.vectorizer.transform([query])
                
                # Calculate similarities
                similarities = (tfidf_matrix @ query_vector.T).toarray().flatten()
                results = [(i, score) for i, score in enumerate(similarities)]

            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:self.config.top_k]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return [(i, 0.0) for i in range(len(documents))]

    def expand_query(self, query: str) -> str:
        """Expand query using synonyms and related terms"""
        if not self.config.enable_query_expansion:
            return query

        try:
            doc = self.nlp(query)
            expanded_terms = set()

            # Add original query terms
            expanded_terms.update(query.split())

            # Add synonyms and similar words
            for token in doc:
                # Add lemmatized form
                expanded_terms.add(token.lemma_)
                
                # Add synonyms from WordNet
                if not token.is_stop and not token.is_punct:
                    # Get similar words based on word vectors
                    similar_words = [
                        w for w, _ in token.vocab.vectors.most_similar(
                            token.vector, n=3
                        ) if w in self.nlp.vocab
                    ]
                    expanded_terms.update(similar_words)

            return " ".join(expanded_terms)
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query

    def spell_check(self, query: str) -> str:
        """Apply spell checking to the query"""
        if not self.config.enable_spellcheck:
            return query

        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            
            words = query.split()
            corrected_words = []
            
            for word in words:
                # Skip special characters and numbers
                if not re.match(r'^[a-zA-Z]+$', word):
                    corrected_words.append(word)
                    continue
                    
                correction = spell.correction(word)
                if correction and correction != word:
                    logger.info(f"Corrected '{word}' to '{correction}'")
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
                    
            return " ".join(corrected_words)
        except Exception as e:
            logger.error(f"Spell checking failed: {e}")
            return query

    def search(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword-based approaches"""
        if not documents:
            return []

        # Preprocess query
        if self.config.enable_spellcheck:
            query = self.spell_check(query)
        if self.config.enable_query_expansion:
            query = self.expand_query(query)

        results = []
        
        if self.config.mode in [SearchMode.SEMANTIC, SearchMode.HYBRID]:
            semantic_results = self.semantic_search(query, documents)
            
        if self.config.mode in [SearchMode.KEYWORD, SearchMode.HYBRID]:
            keyword_results = self.keyword_search(query, documents)

        if self.config.mode == SearchMode.HYBRID:
            # Combine scores
            combined_scores = {}
            
            # Normalize and weight semantic scores
            max_semantic = max(score for _, score in semantic_results) if semantic_results else 1
            for idx, score in semantic_results:
                combined_scores[idx] = (score / max_semantic) * self.config.semantic_weight

            # Normalize and weight keyword scores
            max_keyword = max(score for _, score in keyword_results) if keyword_results else 1
            for idx, score in keyword_results:
                if idx in combined_scores:
                    combined_scores[idx] += (score / max_keyword) * self.config.keyword_weight
                else:
                    combined_scores[idx] = (score / max_keyword) * self.config.keyword_weight

            # Sort by combined score
            results = [
                {
                    "index": idx,
                    "score": score,
                    "content": documents[idx]
                }
                for idx, score in sorted(
                    combined_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                if score >= self.config.min_score
            ][:self.config.top_k]

        elif self.config.mode == SearchMode.SEMANTIC:
            results = [
                {
                    "index": idx,
                    "score": score,
                    "content": documents[idx]
                }
                for idx, score in semantic_results
                if score >= self.config.min_score
            ]

        else:  # KEYWORD mode
            results = [
                {
                    "index": idx,
                    "score": score,
                    "content": documents[idx]
                }
                for idx, score in keyword_results
                if score >= self.config.min_score
            ]

        return results

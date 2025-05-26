from typing import List, Dict, Any, Optional, Set, Tuple
import spacy
from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict
import logging
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class QueryExpansionConfig:
    use_wordnet: bool = True
    use_word2vec: bool = True
    use_bert: bool = True
    max_expansions: int = 3
    min_similarity: float = 0.7
    enable_cache: bool = True
    cache_dir: str = "query_expansion_cache"
    use_gpu: bool = True
    word2vec_path: Optional[str] = None
    enable_context_aware: bool = True
    enable_query_rewriting: bool = True
    confidence_threshold: float = 0.6

class QueryExpansion:
    def __init__(self, config: QueryExpansionConfig = None):
        self.config = config or QueryExpansionConfig()
        self._init_components()
        self.cache = {}
        
        if self.config.enable_cache:
            self._load_cache()

    def _init_components(self):
        """Initialize NLP components"""
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_md')
        except:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download('en_core_web_md')
            self.nlp = spacy.load('en_core_web_md')

        # Initialize Word2Vec
        if self.config.use_word2vec:
            try:
                if self.config.word2vec_path:
                    self.word2vec = KeyedVectors.load(self.config.word2vec_path)
                else:
                    # Download a small pre-trained model
                    import gensim.downloader as api
                    self.word2vec = api.load('glove-wiki-gigaword-100')
            except Exception as e:
                logger.error(f"Failed to load Word2Vec model: {e}")
                self.config.use_word2vec = False

        # Initialize BERT
        if self.config.use_bert:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
                if self.config.use_gpu and torch.cuda.is_available():
                    self.bert_model = self.bert_model.to('cuda')
            except Exception as e:
                logger.error(f"Failed to load BERT model: {e}")
                self.config.use_bert = False

        # Initialize NLTK
        if self.config.use_wordnet:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK data: {e}")
                self.config.use_wordnet = False

    def _load_cache(self):
        """Load expansion cache from disk"""
        if not self.config.enable_cache:
            return

        try:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.config.cache_dir, "expansion_cache.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save expansion cache to disk"""
        if not self.config.enable_cache:
            return

        try:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.config.cache_dir, "expansion_cache.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def expand_query(
        self,
        query: str,
        context: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Expand query using multiple techniques
        Returns expanded query and confidence scores
        """
        # Check cache first
        cache_key = f"{query}:{str(context)}"
        if self.config.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Process query
        doc = self.nlp(query)
        expansion_terms = defaultdict(float)
        
        # Get expansion candidates from different sources
        wordnet_terms = (
            self._get_wordnet_expansions(doc)
            if self.config.use_wordnet else {}
        )
        word2vec_terms = (
            self._get_word2vec_expansions(doc)
            if self.config.use_word2vec else {}
        )
        bert_terms = (
            self._get_bert_expansions(query, context)
            if self.config.use_bert else {}
        )
        
        # Combine expansions with weights
        sources = [
            (wordnet_terms, 0.3),
            (word2vec_terms, 0.3),
            (bert_terms, 0.4)
        ]
        
        for terms, weight in sources:
            for term, score in terms.items():
                expansion_terms[term] += score * weight

        # Filter and sort expansions
        filtered_terms = {
            term: score
            for term, score in expansion_terms.items()
            if score >= self.config.min_similarity
        }
        
        sorted_terms = sorted(
            filtered_terms.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.config.max_expansions]

        # Build expanded query
        expansion_text = " ".join(term for term, _ in sorted_terms)
        expanded_query = query
        if expansion_text:
            expanded_query = f"{query} {expansion_text}"

        # Store in cache
        if self.config.enable_cache:
            self.cache[cache_key] = (expanded_query, dict(sorted_terms))
            self._save_cache()

        return expanded_query, dict(sorted_terms)

    def _get_wordnet_expansions(self, doc) -> Dict[str, float]:
        """Get expansion terms from WordNet"""
        expansions = {}
        
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
                
            # Get word synsets
            synsets = wordnet.synsets(token.text)
            
            for synset in synsets:
                # Add synonyms
                for lemma in synset.lemmas():
                    if lemma.name() != token.text:
                        expansions[lemma.name()] = 0.8
                        
                # Add hypernyms (more general terms)
                for hypernym in synset.hypernyms():
                    expansions[hypernym.lemmas()[0].name()] = 0.6
                    
                # Add hyponyms (more specific terms)
                for hyponym in synset.hyponyms():
                    expansions[hyponym.lemmas()[0].name()] = 0.7
                    
        return expansions

    def _get_word2vec_expansions(self, doc) -> Dict[str, float]:
        """Get expansion terms from Word2Vec"""
        if not self.config.use_word2vec:
            return {}

        expansions = {}
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
                
            try:
                # Find similar words
                similar_words = self.word2vec.most_similar(
                    token.text.lower(),
                    topn=self.config.max_expansions
                )
                for word, score in similar_words:
                    if score >= self.config.min_similarity:
                        expansions[word] = score
            except KeyError:
                continue
                
        return expansions

    def _get_bert_expansions(
        self,
        query: str,
        context: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get expansion terms using BERT masked language model"""
        if not self.config.use_bert:
            return {}

        expansions = {}
        try:
            # Tokenize query
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if self.config.use_gpu and torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Get predictions for each token
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = outputs.logits

            # Get token predictions
            for i in range(inputs['input_ids'].shape[1]):
                # Skip special tokens
                token_id = inputs['input_ids'][0, i].item()
                if token_id in [self.tokenizer.cls_token_id,
                              self.tokenizer.sep_token_id,
                              self.tokenizer.pad_token_id]:
                    continue

                # Get top predictions for this position
                token_predictions = predictions[0, i]
                top_k = torch.topk(token_predictions, k=5)
                
                for score, idx in zip(top_k.values, top_k.indices):
                    token = self.tokenizer.decode([idx])
                    if token.strip() and token not in query:
                        expansions[token] = score.item()

            # Normalize scores
            if expansions:
                max_score = max(expansions.values())
                expansions = {
                    k: v/max_score
                    for k, v in expansions.items()
                }

        except Exception as e:
            logger.error(f"BERT expansion failed: {e}")

        return expansions

    def rewrite_query(
        self,
        query: str,
        context: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """Rewrite query to improve clarity and effectiveness"""
        if not self.config.enable_query_rewriting:
            return query, 0.0

        try:
            # Use BERT to generate alternative phrasings
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if self.config.use_gpu and torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
            # Get alternative tokens for each position
            rewritten_tokens = []
            confidence_scores = []
            
            for i in range(inputs['input_ids'].shape[1]):
                token_id = inputs['input_ids'][0, i].item()
                if token_id in [self.tokenizer.cls_token_id,
                              self.tokenizer.sep_token_id,
                              self.tokenizer.pad_token_id]:
                    continue
                    
                token_predictions = outputs.logits[0, i]
                top_pred = torch.topk(token_predictions, k=1)
                
                pred_token = self.tokenizer.decode([top_pred.indices[0]])
                orig_token = self.tokenizer.decode([token_id])
                
                if pred_token != orig_token and top_pred.values[0] > self.config.confidence_threshold:
                    rewritten_tokens.append(pred_token)
                    confidence_scores.append(top_pred.values[0].item())
                else:
                    rewritten_tokens.append(orig_token)
                    confidence_scores.append(1.0)

            rewritten_query = " ".join(rewritten_tokens)
            confidence = np.mean(confidence_scores)

            return rewritten_query.strip(), float(confidence)

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query, 0.0

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and provide insights"""
        doc = self.nlp(query)
        
        analysis = {
            "length": len(query),
            "token_count": len(doc),
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "main_verbs": [token.text for token in doc if token.pos_ == "VERB"],
            "keywords": []
        }
        
        # Extract keywords using TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=5,
                stop_words='english'
            )
            vectorizer.fit_transform([query])
            analysis["keywords"] = vectorizer.get_feature_names_out().tolist()
        except:
            pass
        
        return analysis

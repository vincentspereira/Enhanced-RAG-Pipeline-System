from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    target_chunk_size: int = 1000
    overlap_size: int = 200
    similarity_threshold: float = 0.7
    use_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    balance_chunks: bool = True
    
    # Advanced optimization parameters
    semantic_weight: float = 0.4
    syntactic_weight: float = 0.3
    length_weight: float = 0.3
    min_sentences_per_chunk: int = 2
    max_sentences_per_chunk: int = 20
    adaptive_chunking: bool = True
    content_density_threshold: float = 0.6
    use_dynamic_overlap: bool = True

class ChunkOptimizer:
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Installing en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def calculate_optimal_chunk_size(self, text: str) -> int:
        """Calculate optimal chunk size based on document characteristics"""
        # Analyze document structure
        doc = self.nlp(text[:min(len(text), 10000)])  # Analyze first 10k chars for efficiency
        
        # Get average sentence length
        sentences = list(doc.sents)
        if not sentences:
            return self.config.target_chunk_size
            
        avg_sentence_length = np.mean([len(str(sent)) for sent in sentences])
        
        # Get average paragraph length
        paragraphs = text.split('\n\n')
        avg_paragraph_length = np.mean([len(p) for p in paragraphs if p.strip()])
        
        # Calculate optimal size based on document characteristics
        if avg_sentence_length > 100:  # Long, complex sentences
            optimal_size = min(max(int(avg_sentence_length * 5), self.config.min_chunk_size), 
                             self.config.max_chunk_size)
        elif avg_paragraph_length < 200:  # Short paragraphs
            optimal_size = min(max(int(avg_paragraph_length * 3), self.config.min_chunk_size), 
                             self.config.max_chunk_size)
        else:
            optimal_size = self.config.target_chunk_size
            
        return optimal_size

    def optimize_chunks(self, chunks: List[str]) -> List[str]:
        """Optimize existing chunks based on content similarity and coherence"""
        if not chunks:
            return chunks

        # Calculate similarity matrix
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            logger.warning("Failed to calculate similarity matrix")
            return chunks

        # Merge similar consecutive chunks
        optimized_chunks = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            while (i + 1 < len(chunks) and 
                   similarity_matrix[i][i + 1] > self.config.similarity_threshold and 
                   len(current_chunk) + len(chunks[i + 1]) <= self.config.max_chunk_size):
                current_chunk += " " + chunks[i + 1]
                i += 1
            optimized_chunks.append(current_chunk)
            i += 1

        return optimized_chunks

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into optimally sized chunks"""
        if not text:
            return []

        # Calculate optimal chunk size
        chunk_size = self.calculate_optimal_chunk_size(text)
        
        # Prepare text
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sentence_text = str(sent).strip()
            sentence_length = len(sentence_text)
            
            # Handle very long sentences
            if sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence at logical points
                sub_sentences = self._split_long_sentence(sentence_text)
                chunks.extend(sub_sentences)
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence_text)
            current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Add overlap between chunks
        if self.config.overlap_size > 0:
            chunks = self._add_overlap(chunks)
        
        # Optimize chunks
        return self.optimize_chunks(chunks)

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller, meaningful parts"""
        # Try splitting at punctuation first
        parts = re.split(r'[,;:]', sentence)
        if len(parts) > 1:
            return [part.strip() for part in parts if part.strip()]
        
        # If no punctuation, split by phrase length while keeping words intact
        words = sentence.split()
        sub_sentences = []
        current_part = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > self.config.target_chunk_size and current_part:
                sub_sentences.append(" ".join(current_part))
                current_part = []
                current_length = 0
            current_part.append(word)
            current_length += len(word) + 1  # +1 for space
            
        if current_part:
            sub_sentences.append(" ".join(current_part))
            
        return sub_sentences

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks while respecting sentence boundaries"""
        if len(chunks) < 2:
            return chunks
            
        result = []
        for i in range(len(chunks)):
            if i == 0:
                result.append(chunks[i])
                continue
                
            # Get overlap from previous chunk
            prev_chunk = chunks[i-1]
            words = prev_chunk.split()[-self.config.overlap_size:]
            overlap = " ".join(words)
            
            # Add overlap to current chunk
            result.append(f"{overlap} {chunks[i]}")
            
        return result

    def analyze_chunk_quality(self, chunks: List[str]) -> Dict[str, Any]:
        """Analyze the quality of text chunks"""
        if not chunks:
            return {}
            
        analysis = {
            "num_chunks": len(chunks),
            "avg_chunk_size": np.mean([len(chunk) for chunk in chunks]),
            "std_chunk_size": np.std([len(chunk) for chunk in chunks]),
            "max_chunk_size": max(len(chunk) for chunk in chunks),
            "min_chunk_size": min(len(chunk) for chunk in chunks),
        }
        
        # Analyze content overlap
        if len(chunks) > 1:
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform(chunks)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                analysis["avg_similarity"] = np.mean([
                    similarity_matrix[i][i+1] 
                    for i in range(len(chunks)-1)
                ])
            except:
                analysis["avg_similarity"] = None
        
        return analysis

    def rebalance_chunks(self, chunks: List[str]) -> List[str]:
        """Rebalance chunks to make them more uniform in size"""
        if not chunks or not self.config.balance_chunks:
            return chunks
            
        # Calculate target size
        total_length = sum(len(chunk) for chunk in chunks)
        target_size = total_length // len(chunks)
        
        # Rebalance only if there's significant variation
        sizes = [len(chunk) for chunk in chunks]
        if np.std(sizes) < 0.2 * np.mean(sizes):
            return chunks
            
        # Split text back into sentences
        all_sentences = []
        for chunk in chunks:
            doc = self.nlp(chunk)
            all_sentences.extend([str(sent).strip() for sent in doc.sents])
        
        # Redistribute sentences
        balanced_chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in all_sentences:
            if current_length + len(sent) > target_size and current_chunk:
                balanced_chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sent)
            current_length += len(sent)
        
        if current_chunk:
            balanced_chunks.append(" ".join(current_chunk))
        
        return balanced_chunks

    def _estimate_memory_usage(self, text: str) -> Dict[str, int]:
        """Estimate memory usage for text processing."""
        char_size = len(text.encode('utf-8'))
        word_count = len(text.split())
        
        # Estimate embeddings size (assuming 768-dimensional embeddings)
        embedding_size = word_count * 768 * 4  # 4 bytes per float
        
        # Estimate working memory (temp variables, etc.)
        working_memory = char_size * 4  # Conservative estimate
        
        return {
            'text_size': char_size,
            'embedding_size': embedding_size,
            'working_memory': working_memory,
            'total': char_size + embedding_size + working_memory
        }

    def optimize_chunk_size_with_memory(self, text: str, max_memory_mb: int = 1024) -> int:
        """Optimize chunk size while respecting memory constraints."""
        base_size = self.optimize_chunk_size(text)
        memory_usage = self._estimate_memory_usage(text[:base_size])
        
        # Convert max_memory_mb to bytes
        max_memory = max_memory_mb * 1024 * 1024
        
        if memory_usage['total'] > max_memory:
            # Adjust chunk size to fit within memory constraints
            ratio = max_memory / memory_usage['total']
            adjusted_size = int(base_size * ratio * 0.9)  # 10% safety margin
            
            # Ensure size stays within bounds
            return max(
                self.config.min_chunk_size,
                min(adjusted_size, self.config.max_chunk_size)
            )
        
        return base_size

    def create_memory_efficient_chunks(
        self,
        text: str,
        max_memory_mb: int = 1024,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Create chunks efficiently while managing memory usage."""
        chunk_size = self.optimize_chunk_size_with_memory(text, max_memory_mb)
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        batch = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(self.tokenizer.encode(sent_text))
            
            if current_length + sent_length > chunk_size and current_chunk:
                # Process the current chunk
                chunk_text = " ".join(current_chunk)
                batch.append({
                    "text": chunk_text,
                    "size": current_length,
                })
                
                # Reset current chunk
                current_chunk = [sent_text]
                current_length = sent_length
                
                # Process batch if it's full
                if len(batch) >= batch_size:
                    chunks.extend(self._process_batch(batch))
                    batch = []
            else:
                current_chunk.append(sent_text)
                current_length += sent_length
        
        # Handle remaining text
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            batch.append({
                "text": chunk_text,
                "size": current_length,
            })
        
        # Process remaining batch
        if batch:
            chunks.extend(self._process_batch(batch))
        
        return chunks

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of chunks with metadata."""
        processed_chunks = []
        
        for chunk in batch:
            chunk_text = chunk["text"]
            processed_chunks.append({
                "text": chunk_text,
                "size": chunk["size"],
                "metadata": {
                    "complexity": self._calculate_complexity(chunk_text),
                    "semantic_density": self._calculate_semantic_density(chunk_text),
                    "memory_usage": self._estimate_memory_usage(chunk_text)
                }
            })
        
        return processed_chunks

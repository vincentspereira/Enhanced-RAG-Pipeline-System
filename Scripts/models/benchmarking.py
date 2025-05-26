"""
Model benchmarking and performance tracking module.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import uuid
from datetime import datetime
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """Benchmark and track performance metrics for embedding models and LLMs."""
    
    def __init__(self, output_dir: str = "data/benchmarks"):
        """Initialize the benchmark tracker.
        
        Args:
            output_dir: Directory to store benchmark results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics history
        self.metrics_history = {}
        
        # Load existing history if available
        self.history_path = os.path.join(output_dir, "benchmark_history.json")
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.metrics_history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load benchmark history: {e}")
    
    def benchmark_embedding_model(self, model_name: str, texts: List[str], 
                                 batch_sizes: List[int] = None,
                                 runs_per_batch: int = 3) -> Dict[str, Any]:
        """Benchmark embedding model performance.
        
        Args:
            model_name: Name of the embedding model to benchmark
            texts: List of sample texts to embed
            batch_sizes: List of batch sizes to test (default: [1, 8, 32, 128])
            runs_per_batch: Number of runs for each batch size
            
        Returns:
            Benchmark results
        """
        try:
            from Scripts.models.embeddings import get_embedding_model
            
            logger.info(f"Benchmarking embedding model: {model_name}")
            
            # Default batch sizes if not provided
            if batch_sizes is None:
                batch_sizes = [1, 8, 32, 128]
                
            # Filter batch sizes that are too large
            max_texts = len(texts)
            batch_sizes = [b for b in batch_sizes if b <= max_texts]
            
            # Prepare results structure
            results = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(texts),
                "batch_metrics": {},
                "total_time": 0,
                "avg_tokens_per_second": 0
            }
            
            # Load model
            start_time = time.time()
            model = get_embedding_model(model_name)
            load_time = time.time() - start_time
            results["model_load_time"] = load_time
            
            # Calculate total tokens
            total_tokens = sum(len(text.split()) for text in texts)
            results["total_tokens"] = total_tokens
            
            # Test each batch size
            for batch_size in batch_sizes:
                logger.info(f"Testing batch size: {batch_size}")
                batch_times = []
                
                # Multiple runs for stability
                for run in range(runs_per_batch):
                    # Process in batches
                    start_time = time.time()
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        _ = model.encode(batch)
                    
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                
                # Calculate statistics
                avg_time = np.mean(batch_times)
                std_time = np.std(batch_times)
                tokens_per_second = total_tokens / avg_time
                
                # Store batch results
                results["batch_metrics"][str(batch_size)] = {
                    "avg_time_seconds": float(avg_time),
                    "std_time_seconds": float(std_time),
                    "tokens_per_second": float(tokens_per_second)
                }
            
            # Find optimal batch size
            optimal_batch = max(results["batch_metrics"].items(), 
                               key=lambda x: x[1]["tokens_per_second"])
            
            results["optimal_batch_size"] = int(optimal_batch[0])
            results["peak_tokens_per_second"] = float(optimal_batch[1]["tokens_per_second"])
            
            # Update history
            self._update_history("embedding", model_name, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking embedding model {model_name}: {e}")
            return {"error": str(e)}
    
    def benchmark_llm(self, model_name: str, prompts: List[str],
                     max_tokens: int = 100,
                     temperature: float = 0.7) -> Dict[str, Any]:
        """Benchmark LLM generation performance.
        
        Args:
            model_name: Name of the LLM to benchmark
            prompts: List of prompts to test
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Benchmark results
        """
        try:
            from Scripts.llm import get_llm_service
            
            logger.info(f"Benchmarking LLM model: {model_name}")
            
            # Prepare results structure
            results = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "num_prompts": len(prompts),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "prompt_metrics": [],
                "avg_generation_time": 0,
                "avg_tokens_per_second": 0
            }
            
            # Load model
            start_time = time.time()
            llm = get_llm_service(model_name)
            load_time = time.time() - start_time
            results["model_load_time"] = load_time
            
            # Test each prompt
            total_time = 0
            total_output_tokens = 0
            
            for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
                prompt_metrics = {"prompt_id": i, "prompt_tokens": len(prompt.split())}
                
                # Time generation
                start_time = time.time()
                output = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
                generation_time = time.time() - start_time
                
                # Calculate metrics
                output_tokens = len(output.split())
                tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
                
                # Update prompt metrics
                prompt_metrics.update({
                    "generation_time": generation_time,
                    "output_tokens": output_tokens,
                    "tokens_per_second": tokens_per_second
                })
                
                results["prompt_metrics"].append(prompt_metrics)
                total_time += generation_time
                total_output_tokens += output_tokens
            
            # Calculate aggregated statistics
            results["avg_generation_time"] = total_time / len(prompts)
            results["total_tokens_generated"] = total_output_tokens
            results["avg_tokens_per_second"] = total_output_tokens / total_time if total_time > 0 else 0
            
            # Update history
            self._update_history("llm", model_name, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking LLM {model_name}: {e}")
            return {"error": str(e)}
            
    def benchmark_vector_store(self, store_name: str, num_vectors: int = 10000,
                              vector_dim: int = 768, batch_sizes: List[int] = None,
                              search_queries: int = 100) -> Dict[str, Any]:
        """Benchmark vector store performance.
        
        Args:
            store_name: Name of the vector store to benchmark
            num_vectors: Number of random vectors to generate
            vector_dim: Dimension of vectors
            batch_sizes: List of batch sizes to test
            search_queries: Number of search queries to perform
            
        Returns:
            Benchmark results
        """
        try:
            from Scripts.vector_stores import get_vector_store
            
            logger.info(f"Benchmarking vector store: {store_name}")
            
            # Default batch sizes if not provided
            if batch_sizes is None:
                batch_sizes = [100, 500, 1000, 5000]
            
            # Prepare results structure
            results = {
                "store_name": store_name,
                "timestamp": datetime.now().isoformat(),
                "num_vectors": num_vectors,
                "vector_dim": vector_dim,
                "search_queries": search_queries,
                "batch_metrics": {},
                "search_metrics": {}
            }
            
            # Load vector store
            start_time = time.time()
            store = get_vector_store(store_name, collection_name=f"benchmark_{int(time.time())}")
            load_time = time.time() - start_time
            results["store_load_time"] = load_time
            
            # Generate random vectors and metadata
            vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)
            metadata = [{"id": str(i), "test": f"document_{i}"} for i in range(num_vectors)]
            
            # Test batch insertion
            for batch_size in batch_sizes:
                if batch_size > num_vectors:
                    continue
                    
                batch_times = []
                # Run multiple times for stability
                for run in range(3):
                    start_time = time.time()
                    for i in range(0, num_vectors, batch_size):
                        end_idx = min(i + batch_size, num_vectors)
                        batch_vectors = vectors[i:end_idx]
                        batch_metadata = metadata[i:end_idx]
                        store.add_vectors(batch_vectors, batch_metadata)
                    
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                
                # Calculate statistics
                avg_time = np.mean(batch_times)
                std_time = np.std(batch_times)
                vectors_per_second = num_vectors / avg_time
                
                # Store batch results
                results["batch_metrics"][str(batch_size)] = {
                    "avg_time_seconds": float(avg_time),
                    "std_time_seconds": float(std_time),
                    "vectors_per_second": float(vectors_per_second)
                }
            
            # Test search performance
            top_k_values = [1, 10, 50, 100]
            for top_k in top_k_values:
                search_times = []
                
                # Generate random query vectors
                query_vectors = np.random.randn(search_queries, vector_dim).astype(np.float32)
                
                # Run searches
                for i in range(search_queries):
                    start_time = time.time()
                    _ = store.search(query_vectors[i], top_k=top_k)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                
                # Calculate statistics
                avg_time = np.mean(search_times)
                std_time = np.std(search_times)
                queries_per_second = 1.0 / avg_time
                
                # Store search results
                results["search_metrics"][str(top_k)] = {
                    "avg_time_seconds": float(avg_time),
                    "std_time_seconds": float(std_time),
                    "queries_per_second": float(queries_per_second)
                }
            
            # Find optimal batch size for insertion
            optimal_batch = max(results["batch_metrics"].items(), 
                               key=lambda x: x[1]["vectors_per_second"])
            
            results["optimal_batch_size"] = int(optimal_batch[0])
            results["peak_vectors_per_second"] = float(optimal_batch[1]["vectors_per_second"])
            
            # Update history
            self._update_history("vector_store", store_name, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking vector store {store_name}: {e}")
            return {"error": str(e)}
            
    def _update_history(self, component_type: str, name: str, results: Dict[str, Any]):
        """Update metrics history.
        
        Args:
            component_type: Type of component (embedding, llm, vector_store)
            name: Name of the component
            results: Benchmark results
        """
        # Initialize component type if not exists
        if component_type not in self.metrics_history:
            self.metrics_history[component_type] = {}
            
        # Initialize component if not exists
        if name not in self.metrics_history[component_type]:
            self.metrics_history[component_type][name] = []
            
        # Add new results
        self.metrics_history[component_type][name].append({
            "timestamp": results["timestamp"],
            "summary": self._extract_summary(component_type, results)
        })
        
        # Save updated history
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save benchmark history: {e}")
    
    def _extract_summary(self, component_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary metrics from detailed results.
        
        Args:
            component_type: Type of component
            results: Detailed benchmark results
            
        Returns:
            Summary of key metrics
        """
        summary = {}
        
        if component_type == "embedding":
            summary = {
                "optimal_batch_size": results.get("optimal_batch_size", 0),
                "peak_tokens_per_second": results.get("peak_tokens_per_second", 0),
                "model_load_time": results.get("model_load_time", 0),
                "total_tokens": results.get("total_tokens", 0)
            }
        elif component_type == "llm":
            summary = {
                "avg_generation_time": results.get("avg_generation_time", 0),
                "avg_tokens_per_second": results.get("avg_tokens_per_second", 0),
                "model_load_time": results.get("model_load_time", 0),
                "total_tokens_generated": results.get("total_tokens_generated", 0)
            }
        elif component_type == "vector_store":
            summary = {
                "optimal_batch_size": results.get("optimal_batch_size", 0),
                "peak_vectors_per_second": results.get("peak_vectors_per_second", 0),
                "store_load_time": results.get("store_load_time", 0)
            }
            
            # Add search metrics if available
            if "search_metrics" in results and "10" in results["search_metrics"]:
                summary["search_queries_per_second"] = results["search_metrics"]["10"].get("queries_per_second", 0)
        
        return summary
    
    def compare_models(self, component_type: str, names: List[str] = None) -> Dict[str, Any]:
        """Compare benchmark results across models.
        
        Args:
            component_type: Type of component to compare (embedding, llm, vector_store)
            names: List of model names to compare (if None, compare all)
            
        Returns:
            Comparison results
        """
        if component_type not in self.metrics_history:
            return {"error": f"No benchmark data for component type: {component_type}"}
            
        # Get models to compare
        models = names or list(self.metrics_history[component_type].keys())
        if not models:
            return {"error": f"No models to compare for component type: {component_type}"}
            
        # Prepare comparison data
        comparison = {
            "component_type": component_type,
            "models": models,
            "comparison_time": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Extract metrics to compare
        for model in models:
            if model not in self.metrics_history[component_type]:
                continue
                
            # Get most recent benchmark
            if not self.metrics_history[component_type][model]:
                continue
                
            latest = self.metrics_history[component_type][model][-1]["summary"]
            comparison["metrics"][model] = latest
        
        # Generate visualizations
        chart_path = self._visualize_comparison(component_type, comparison)
        if chart_path:
            comparison["visualization"] = chart_path
        
        return comparison
    
    def _visualize_comparison(self, component_type: str, comparison: Dict[str, Any]) -> str:
        """Create visualization for model comparison.
        
        Args:
            component_type: Type of component
            comparison: Comparison data
            
        Returns:
            Path to the visualization file
        """
        try:
            # Extract data for visualization
            models = list(comparison["metrics"].keys())
            if not models:
                return None
                
            # Define metrics to visualize based on component type
            if component_type == "embedding":
                metrics = [
                    {"name": "peak_tokens_per_second", "label": "Peak Tokens/Second"},
                    {"name": "model_load_time", "label": "Model Load Time (s)"}
                ]
            elif component_type == "llm":
                metrics = [
                    {"name": "avg_tokens_per_second", "label": "Avg Tokens/Second"},
                    {"name": "avg_generation_time", "label": "Avg Generation Time (s)"}
                ]
            elif component_type == "vector_store":
                metrics = [
                    {"name": "peak_vectors_per_second", "label": "Peak Vectors/Second"},
                    {"name": "search_queries_per_second", "label": "Search Queries/Second"}
                ]
            else:
                return None
                
            # Create multi-bar chart
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
                
            # Plot each metric
            for i, metric in enumerate(metrics):
                metric_name = metric["name"]
                metric_values = []
                
                for model in models:
                    if metric_name in comparison["metrics"][model]:
                        metric_values.append(comparison["metrics"][model][metric_name])
                    else:
                        metric_values.append(0)
                
                # Create bar chart
                axes[i].bar(models, metric_values)
                axes[i].set_title(metric["label"])
                axes[i].set_ylabel(metric["label"])
                axes[i].grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels
                for j, value in enumerate(metric_values):
                    axes[i].text(j, value, f"{value:.2f}", ha='center', va='bottom')
                
                # Rotate x-axis labels for readability
                plt.setp(axes[i].get_xticklabels(), rotation=30, ha='right')
            
            plt.tight_layout()
            
            # Save the visualization
            filename = f"{component_type}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None

class ModelVersionTracker:
    """Track model versions and performance over time."""
    
    def __init__(self, storage_dir: str = "data/model_versions"):
        """Initialize the model version tracker.
        
        Args:
            storage_dir: Directory to store model version information
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing registry if available
        self.registry_path = os.path.join(storage_dir, "model_registry.json")
        self.model_registry = {}
        
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model registry: {e}")
    
    def register_model(self, model_type: str, model_name: str, 
                      model_details: Dict[str, Any]) -> str:
        """Register a new model version.
        
        Args:
            model_type: Type of model (embedding, llm, etc.)
            model_name: Name of the model
            model_details: Details about the model
            
        Returns:
            Version ID for the registered model
        """
        # Generate version ID
        version_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Initialize model type if not exists
        if model_type not in self.model_registry:
            self.model_registry[model_type] = {}
            
        # Initialize model if not exists
        if model_name not in self.model_registry[model_type]:
            self.model_registry[model_type][model_name] = {
                "versions": [],
                "active_version": None
            }
        
        # Add version information
        model_version = {
            "version_id": version_id,
            "registered_at": timestamp,
            "details": model_details,
            "metrics": {},
            "status": "registered"
        }
        
        self.model_registry[model_type][model_name]["versions"].append(model_version)
        
        # If first version, set as active
        if len(self.model_registry[model_type][model_name]["versions"]) == 1:
            self.set_active_version(model_type, model_name, version_id)
        
        # Save registry
        self._save_registry()
        
        return version_id
    
    def set_active_version(self, model_type: str, model_name: str, version_id: str) -> bool:
        """Set a model version as active.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            version_id: Version ID to set as active
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists
        if (model_type not in self.model_registry or 
            model_name not in self.model_registry[model_type]):
            return False
        
        # Check if version exists
        versions = self.model_registry[model_type][model_name]["versions"]
        version_exists = any(v["version_id"] == version_id for v in versions)
        
        if not version_exists:
            return False
        
        # Set active version
        self.model_registry[model_type][model_name]["active_version"] = version_id
        
        # Save registry
        self._save_registry()
        
        return True
    
    def add_metrics(self, model_type: str, model_name: str, version_id: str,
                   metrics: Dict[str, Any]) -> bool:
        """Add performance metrics to a model version.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            version_id: Version ID
            metrics: Performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists
        if (model_type not in self.model_registry or 
            model_name not in self.model_registry[model_type]):
            return False
        
        # Find version
        for version in self.model_registry[model_type][model_name]["versions"]:
            if version["version_id"] == version_id:
                # Update metrics with timestamp
                timestamp = datetime.now().isoformat()
                
                if "metrics" not in version:
                    version["metrics"] = {}
                    
                version["metrics"][timestamp] = metrics
                
                # Save registry
                self._save_registry()
                
                return True
        
        return False
    
    def get_model_versions(self, model_type: str, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            List of model versions
        """
        if (model_type not in self.model_registry or 
            model_name not in self.model_registry[model_type]):
            return []
            
        return self.model_registry[model_type][model_name]["versions"]
    
    def get_active_version(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the active version of a model.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            Active version or None if no active version
        """
        if (model_type not in self.model_registry or 
            model_name not in self.model_registry[model_type]):
            return None
            
        active_id = self.model_registry[model_type][model_name]["active_version"]
        if not active_id:
            return None
            
        # Find active version
        for version in self.model_registry[model_type][model_name]["versions"]:
            if version["version_id"] == active_id:
                return version
                
        return None
    
    def archive_version(self, model_type: str, model_name: str, version_id: str) -> bool:
        """Archive a model version.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            version_id: Version ID to archive
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists
        if (model_type not in self.model_registry or 
            model_name not in self.model_registry[model_type]):
            return False
        
        # Find version
        for version in self.model_registry[model_type][model_name]["versions"]:
            if version["version_id"] == version_id:
                # Update status
                version["status"] = "archived"
                version["archived_at"] = datetime.now().isoformat()
                
                # If archiving active version, unset active
                if version_id == self.model_registry[model_type][model_name]["active_version"]:
                    self.model_registry[model_type][model_name]["active_version"] = None
                
                # Save registry
                self._save_registry()
                
                return True
        
        return False
    
    def delete_version(self, model_type: str, model_name: str, version_id: str) -> bool:
        """Delete a model version.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            version_id: Version ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model exists
        if (model_type not in self.model_registry or 
            model_name not in self.model_registry[model_type]):
            return False
        
        # Check if active
        if version_id == self.model_registry[model_type][model_name]["active_version"]:
            return False  # Cannot delete active version
        
        # Find and remove version
        versions = self.model_registry[model_type][model_name]["versions"]
        new_versions = [v for v in versions if v["version_id"] != version_id]
        
        if len(new_versions) < len(versions):
            self.model_registry[model_type][model_name]["versions"] = new_versions
            
            # Save registry
            self._save_registry()
            
            return True
        
        return False
    
    def _save_registry(self):
        """Save the model registry to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save model registry: {e}")

# Initialize benchmark and version tracker
model_benchmark = ModelBenchmark()
model_tracker = ModelVersionTracker()

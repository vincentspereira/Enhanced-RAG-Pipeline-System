"""
Model performance benchmarking CLI tool.
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from Scripts.models.benchmarking import ModelBenchmark, ModelVersionTracker

def run_embedding_benchmark(args):
    """Run embedding model benchmark."""
    # Load sample data
    sample_texts = load_sample_texts(args.sample_file, args.num_samples)
    if not sample_texts:
        logger.error("No sample texts available for benchmarking")
        return
    
    logger.info(f"Running benchmark for embedding model: {args.model_name}")
    logger.info(f"Using {len(sample_texts)} text samples")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    # Run benchmark
    results = benchmark.benchmark_embedding_model(
        model_name=args.model_name,
        texts=sample_texts,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        runs_per_batch=args.runs
    )
    
    # Save detailed results
    if results:
        save_results("embedding", args.model_name, results, args.output_dir)
        print_summary(results)
    
def run_llm_benchmark(args):
    """Run LLM benchmark."""
    # Load sample prompts
    sample_prompts = load_sample_texts(args.sample_file, args.num_samples)
    if not sample_prompts:
        logger.error("No sample prompts available for benchmarking")
        return
    
    logger.info(f"Running benchmark for LLM: {args.model_name}")
    logger.info(f"Using {len(sample_prompts)} sample prompts")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    # Run benchmark
    results = benchmark.benchmark_llm(
        model_name=args.model_name,
        prompts=sample_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Save detailed results
    if results:
        save_results("llm", args.model_name, results, args.output_dir)
        print_summary(results)

def run_vector_store_benchmark(args):
    """Run vector store benchmark."""
    logger.info(f"Running benchmark for vector store: {args.store_name}")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    # Run benchmark
    results = benchmark.benchmark_vector_store(
        store_name=args.store_name,
        num_vectors=args.num_vectors,
        vector_dim=args.vector_dim,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        search_queries=args.search_queries
    )
    
    # Save detailed results
    if results:
        save_results("vector_store", args.store_name, results, args.output_dir)
        print_summary(results)

def run_comparison(args):
    """Run model comparison."""
    logger.info(f"Comparing {args.component_type} models")
    
    # Parse component names
    component_names = args.names.split(",") if args.names else None
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    # Run comparison
    results = benchmark.compare_models(
        component_type=args.component_type,
        names=component_names
    )
    
    # Display comparison results
    if "error" in results:
        logger.error(f"Comparison error: {results['error']}")
        return
        
    print(f"\n=== {args.component_type.upper()} COMPARISON ===")
    print(f"Models compared: {', '.join(results['models'])}")
    
    # Format metrics as a table
    metrics_data = {}
    for model, metrics in results["metrics"].items():
        metrics_data[model] = metrics
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        print("\nMetrics Comparison:")
        print(df.T)
    
    # Show visualization path if available
    if "visualization" in results:
        print(f"\nVisualization saved to: {results['visualization']}")

def register_model_version(args):
    """Register a new model version."""
    logger.info(f"Registering new {args.model_type} model: {args.model_name}")
    
    # Load model details
    if args.details_file:
        try:
            with open(args.details_file, 'r') as f:
                details = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model details: {e}")
            return
    else:
        details = {
            "description": args.description or f"Model {args.model_name}",
            "source": args.source or "unknown",
            "created_by": args.created_by or os.environ.get("USERNAME", "unknown")
        }
    
    # Initialize tracker
    tracker = ModelVersionTracker(storage_dir=args.storage_dir)
    
    # Register model
    version_id = tracker.register_model(
        model_type=args.model_type,
        model_name=args.model_name,
        model_details=details
    )
    
    logger.info(f"Model registered with version ID: {version_id}")
    
    # Set as active if requested
    if args.set_active:
        success = tracker.set_active_version(args.model_type, args.model_name, version_id)
        if success:
            logger.info(f"Version {version_id} set as active")
        else:
            logger.error("Failed to set version as active")

def display_model_versions(args):
    """Display model versions."""
    # Initialize tracker
    tracker = ModelVersionTracker(storage_dir=args.storage_dir)
    
    # Get versions
    versions = tracker.get_model_versions(args.model_type, args.model_name)
    active_version = tracker.get_active_version(args.model_type, args.model_name)
    
    if not versions:
        logger.info(f"No versions found for {args.model_type} model {args.model_name}")
        return
    
    print(f"\n=== VERSIONS FOR {args.model_type.upper()} MODEL: {args.model_name} ===")
    print(f"Total versions: {len(versions)}")
    print(f"Active version: {active_version['version_id'] if active_version else 'None'}")
    
    # Display version details
    for i, version in enumerate(versions):
        status = "ACTIVE" if active_version and version["version_id"] == active_version["version_id"] else version["status"]
        print(f"\nVersion {i+1}:")
        print(f"  ID: {version['version_id']} ({status})")
        print(f"  Registered: {version['registered_at']}")
        
        if "details" in version and version["details"]:
            print("  Details:")
            for key, value in version["details"].items():
                print(f"    {key}: {value}")
        
        if "metrics" in version and version["metrics"]:
            print("  Latest metrics:")
            latest_timestamp = max(version["metrics"].keys())
            latest_metrics = version["metrics"][latest_timestamp]
            for key, value in latest_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

# Helper functions
def load_sample_texts(filename: Optional[str], num_samples: int = 100) -> List[str]:
    """Load sample texts for benchmarking."""
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                texts = f.readlines()
            return [text.strip() for text in texts[:num_samples] if text.strip()]
        except Exception as e:
            logger.error(f"Error loading sample texts: {e}")
    
    # Generate synthetic texts
    return [f"This is a synthetic text sample number {i} for benchmarking purposes." for i in range(num_samples)]

def parse_batch_sizes(batch_sizes_str: Optional[str]) -> Optional[List[int]]:
    """Parse batch sizes from string."""
    if not batch_sizes_str:
        return None
        
    try:
        return [int(size.strip()) for size in batch_sizes_str.split(",") if size.strip()]
    except:
        logger.warning(f"Invalid batch sizes format: {batch_sizes_str}. Using defaults.")
        return None

def save_results(component_type: str, name: str, results: Dict[str, Any], output_dir: str):
    """Save benchmark results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = int(time.time())
    filename = f"{component_type}_{name.replace('/', '_')}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def print_summary(results: Dict[str, Any]):
    """Print a summary of benchmark results."""
    print("\n=== BENCHMARK SUMMARY ===")
    
    if "model_name" in results:
        print(f"Model: {results['model_name']}")
        print(f"Optimal batch size: {results.get('optimal_batch_size', 'N/A')}")
        print(f"Peak tokens/second: {results.get('peak_tokens_per_second', 'N/A')}")
        
        if "batch_metrics" in results:
            print("\nBatch performance:")
            for batch_size, metrics in results["batch_metrics"].items():
                print(f"  Batch {batch_size}: {metrics['tokens_per_second']:.2f} tokens/s")
    
    if "store_name" in results:
        print(f"Vector store: {results['store_name']}")
        print(f"Vectors: {results.get('num_vectors', 'N/A')}")
        print(f"Dimension: {results.get('vector_dim', 'N/A')}")
        print(f"Optimal batch size: {results.get('optimal_batch_size', 'N/A')}")
        print(f"Peak vectors/second: {results.get('peak_vectors_per_second', 'N/A')}")
        
        if "search_metrics" in results:
            print("\nSearch performance:")
            for top_k, metrics in results["search_metrics"].items():
                print(f"  Top-{top_k}: {metrics['queries_per_second']:.2f} queries/s")

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Model Performance Benchmarking Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Embedding benchmark parser
    embedding_parser = subparsers.add_parser("embedding", help="Benchmark embedding model")
    embedding_parser.add_argument("model_name", help="Name of the embedding model")
    embedding_parser.add_argument("--sample-file", help="File with sample texts")
    embedding_parser.add_argument("--num-samples", type=int, default=100, help="Number of sample texts")
    embedding_parser.add_argument("--batch-sizes", help="Comma-separated batch sizes")
    embedding_parser.add_argument("--runs", type=int, default=3, help="Runs per batch size")
    embedding_parser.add_argument("--output-dir", default="data/benchmarks", help="Output directory")
    
    # LLM benchmark parser
    llm_parser = subparsers.add_parser("llm", help="Benchmark LLM")
    llm_parser.add_argument("model_name", help="Name of the LLM")
    llm_parser.add_argument("--sample-file", help="File with sample prompts")
    llm_parser.add_argument("--num-samples", type=int, default=10, help="Number of sample prompts")
    llm_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    llm_parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    llm_parser.add_argument("--output-dir", default="data/benchmarks", help="Output directory")
    
    # Vector store benchmark parser
    vector_parser = subparsers.add_parser("vector_store", help="Benchmark vector store")
    vector_parser.add_argument("store_name", help="Name of the vector store")
    vector_parser.add_argument("--num-vectors", type=int, default=10000, help="Number of vectors")
    vector_parser.add_argument("--vector-dim", type=int, default=768, help="Vector dimension")
    vector_parser.add_argument("--batch-sizes", help="Comma-separated batch sizes")
    vector_parser.add_argument("--search-queries", type=int, default=100, help="Number of search queries")
    vector_parser.add_argument("--output-dir", default="data/benchmarks", help="Output directory")
    
    # Comparison parser
    compare_parser = subparsers.add_parser("compare", help="Compare models/stores")
    compare_parser.add_argument("component_type", choices=["embedding", "llm", "vector_store"], 
                              help="Component type to compare")
    compare_parser.add_argument("--names", help="Comma-separated component names")
    compare_parser.add_argument("--output-dir", default="data/benchmarks", help="Output directory")
    
    # Model version registration parser
    register_parser = subparsers.add_parser("register", help="Register model version")
    register_parser.add_argument("model_type", choices=["embedding", "llm", "vector_store"], 
                               help="Type of model")
    register_parser.add_argument("model_name", help="Name of the model")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--source", help="Model source")
    register_parser.add_argument("--created-by", help="Creator name")
    register_parser.add_argument("--details-file", help="JSON file with model details")
    register_parser.add_argument("--set-active", action="store_true", help="Set as active version")
    register_parser.add_argument("--storage-dir", default="data/model_versions", 
                               help="Model version storage directory")
    
    # List versions parser
    versions_parser = subparsers.add_parser("versions", help="List model versions")
    versions_parser.add_argument("model_type", choices=["embedding", "llm", "vector_store"], 
                               help="Type of model")
    versions_parser.add_argument("model_name", help="Name of the model")
    versions_parser.add_argument("--storage-dir", default="data/model_versions", 
                               help="Model version storage directory")
    
    # Parse args
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "embedding":
        run_embedding_benchmark(args)
    elif args.command == "llm":
        run_llm_benchmark(args)
    elif args.command == "vector_store":
        run_vector_store_benchmark(args)
    elif args.command == "compare":
        run_comparison(args)
    elif args.command == "register":
        register_model_version(args)
    elif args.command == "versions":
        display_model_versions(args)
    else:
        parser.print_help()

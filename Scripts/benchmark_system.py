"""
Benchmark multiple components in the RAG pipeline.
"""
import argparse
import os
import sys
import json
import logging
import time
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the Scripts directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Scripts.models.benchmarking import ModelBenchmark, ModelVersionTracker

def load_config(config_file="config.yaml"):
    """Load configuration from yaml file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def benchmark_all_embedding_models(config, output_dir="data/benchmarks"):
    """Run benchmarks for all configured embedding models."""
    logger.info("Benchmarking all embedding models...")
    
    # Get sample texts
    sample_texts = generate_sample_texts(100)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=output_dir)
    
    # Get embedding model names
    model_configs = config.get("model", {})
    default_model = model_configs.get("embedding_model")
    
    # Add other model names if available
    model_names = [default_model] if default_model else []
    
    # Run benchmark for each model
    results = {}
    for model_name in model_names:
        logger.info(f"Benchmarking embedding model: {model_name}")
        try:
            result = benchmark.benchmark_embedding_model(
                model_name=model_name,
                texts=sample_texts,
                batch_sizes=[1, 8, 32, 64],
                runs_per_batch=3
            )
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
    
    # Save overall results
    output_file = os.path.join(output_dir, f"embedding_benchmarks_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All embedding benchmarks saved to {output_file}")
    return results

def benchmark_all_llm_models(config, output_dir="data/benchmarks"):
    """Run benchmarks for all configured LLM models."""
    logger.info("Benchmarking all LLM models...")
    
    # Get sample prompts
    sample_prompts = generate_sample_prompts(10)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=output_dir)
    
    # Get LLM names from config
    llm_configs = config.get("llm", {}).get("providers", {})
    model_names = []
    
    # Collect model names from all providers
    for provider, provider_config in llm_configs.items():
        if "models" in provider_config:
            model_names.extend(provider_config["models"])
    
    # Run benchmark for each model
    results = {}
    for model_name in model_names:
        logger.info(f"Benchmarking LLM: {model_name}")
        try:
            result = benchmark.benchmark_llm(
                model_name=model_name,
                prompts=sample_prompts,
                max_tokens=100,
                temperature=0.7
            )
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
    
    # Save overall results
    output_file = os.path.join(output_dir, f"llm_benchmarks_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All LLM benchmarks saved to {output_file}")
    return results

def benchmark_all_vector_stores(config, output_dir="data/benchmarks"):
    """Run benchmarks for all configured vector stores."""
    logger.info("Benchmarking all vector stores...")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=output_dir)
    
    # Get vector store names from config
    vector_config = config.get("vector_store", {})
    default_store = vector_config.get("provider")
    
    # All available store types
    store_types = [key for key in vector_config.keys() if key != "provider"]
    
    # Run benchmark for each store
    results = {}
    for store_type in store_types:
        logger.info(f"Benchmarking vector store: {store_type}")
        try:
            result = benchmark.benchmark_vector_store(
                store_name=store_type,
                num_vectors=10000,
                vector_dim=768,
                batch_sizes=[100, 500, 1000, 5000],
                search_queries=100
            )
            results[store_type] = result
        except Exception as e:
            logger.error(f"Error benchmarking {store_type}: {e}")
    
    # Save overall results
    output_file = os.path.join(output_dir, f"vector_store_benchmarks_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All vector store benchmarks saved to {output_file}")
    return results

def compare_all_components(config, output_dir="data/benchmarks"):
    """Compare all benchmarks and create reports."""
    logger.info("Comparing all benchmarked components...")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=output_dir)
    
    # Compare each component type
    component_types = ["embedding", "llm", "vector_store"]
    
    for component_type in component_types:
        logger.info(f"Comparing {component_type} components...")
        try:
            result = benchmark.compare_models(component_type)
            
            if "error" not in result:
                logger.info(f"Successfully compared {len(result.get('models', []))} {component_type} components")
                
                # Save results
                output_file = os.path.join(output_dir, f"{component_type}_comparison_{int(time.time())}.json")
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Comparison saved to {output_file}")
                
                if "visualization" in result:
                    logger.info(f"Visualization saved to {result['visualization']}")
            else:
                logger.warning(f"Error comparing {component_type} components: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error comparing {component_type} components: {e}")

def benchmark_all_components(config):
    """Run benchmarks for all components in the system."""
    # Get benchmark config
    benchmark_config = config.get("model", {}).get("benchmarking", {})
    output_dir = benchmark_config.get("output_dir", "data/benchmarks")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Starting full system benchmark at {timestamp}")
    
    # Run all benchmarks
    embedding_results = benchmark_all_embedding_models(config, output_dir)
    llm_results = benchmark_all_llm_models(config, output_dir)
    vector_store_results = benchmark_all_vector_stores(config, output_dir)
    
    # Compare all components
    compare_all_components(config, output_dir)
    
    logger.info("All benchmarks completed successfully")
    
    # Return combined results
    return {
        "timestamp": timestamp,
        "embedding_models": list(embedding_results.keys()),
        "llm_models": list(llm_results.keys()),
        "vector_stores": list(vector_store_results.keys()),
        "benchmark_dir": output_dir
    }

def generate_sample_texts(count=100, min_words=10, max_words=50):
    """Generate sample texts for benchmarking."""
    sample_phrases = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models can process natural language with remarkable accuracy",
        "Vector databases store high-dimensional representations of data points",
        "Semantic search allows for finding conceptually similar information",
        "Knowledge graphs represent relationships between different entities",
        "Information retrieval systems help find relevant documents in large corpora",
        "Document processing pipelines extract structured data from unstructured text",
        "Embedding models convert text into numerical vectors for machine learning",
        "Neural networks can learn complex patterns in data through training",
        "Large language models generate human-like text based on prompts"
    ]
    
    import random
    
    texts = []
    for i in range(count):
        # Choose random number of phrases to combine
        num_phrases = random.randint(min_words // 10, max_words // 10)
        phrase_selection = [random.choice(sample_phrases) for _ in range(num_phrases)]
        texts.append(" ".join(phrase_selection))
    
    return texts

def generate_sample_prompts(count=10):
    """Generate sample prompts for LLM benchmarking."""
    prompt_templates = [
        "Summarize the following information: {text}",
        "Answer the following question: {question}",
        "Extract key entities from this text: {text}",
        "Generate a response to this email: {email}",
        "Explain the concept of {concept} in simple terms",
        "Write a short paragraph about {topic}",
        "Translate the following from English to French: {text}",
        "List the main points from this document: {text}",
        "Generate a creative story about {theme}",
        "What are the advantages and disadvantages of {subject}?"
    ]
    
    fill_values = {
        "text": generate_sample_texts(count, 50, 100),
        "question": [
            "What is the difference between supervised and unsupervised learning?",
            "How do vector databases store and retrieve information?",
            "What are the key components of a RAG system?",
            "How does semantic search differ from keyword search?",
            "What is the role of embedding models in modern NLP?",
            "How can knowledge graphs enhance search capabilities?",
            "What are the challenges in implementing effective document processing?",
            "How do large language models generate text?",
            "What is the importance of context in natural language understanding?",
            "How can AI systems be made more explainable?"
        ],
        "email": [
            "I'm writing to inquire about your services for implementing a RAG system.",
            "Could you provide more information about your vector database integration?",
            "I'd like to schedule a meeting to discuss potential collaboration.",
            "We're interested in your knowledge graph solutions for our project.",
            "Please send me the documentation for your embedding API."
        ],
        "concept": [
            "vector embeddings", "semantic search", "knowledge graphs", 
            "document processing", "neural networks", "transformer models",
            "retrieval augmented generation", "active learning"
        ],
        "topic": [
            "artificial intelligence advancements", "modern search technologies",
            "natural language processing", "information retrieval systems",
            "vector databases", "machine learning applications"
        ],
        "theme": [
            "a world where AI helps solve global challenges",
            "future information retrieval systems",
            "a day in the life of a knowledge graph",
            "how search engines might work in 2050"
        ],
        "subject": [
            "vector databases", "large language models", "knowledge graphs",
            "semantic search", "neural embeddings", "transformer architectures"
        ]
    }
    
    import random
    
    prompts = []
    for i in range(count):
        template = random.choice(prompt_templates)
        
        # Find all placeholders
        import re
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        # Replace each placeholder with a random value
        prompt = template
        for placeholder in placeholders:
            if placeholder in fill_values:
                value = random.choice(fill_values[placeholder])
                prompt = prompt.replace(f"{{{placeholder}}}", value)
        
        prompts.append(prompt)
    
    return prompts

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark all RAG pipeline components")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        sys.exit(1)
    
    # Run all benchmarks
    try:
        results = benchmark_all_components(config)
        logger.info(f"Benchmarking completed. Results saved to {results['benchmark_dir']}")
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        sys.exit(1)

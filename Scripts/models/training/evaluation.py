"""
Model evaluation tools for embedding models.

This module provides tools for evaluating embedding models on various tasks
and metrics relevant to RAG applications.
"""
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    """Evaluator for embedding models."""
    
    def __init__(
        self, 
        model_path: str,
        device: str = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """Initialize the evaluator.
        
        Args:
            model_path: Path to the model
            device: Device to use (cuda or cpu)
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text documents
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def evaluate_similarity(self, text_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate similarity between text pairs.
        
        Args:
            text_pairs: List of (text1, text2) pairs
            
        Returns:
            Dictionary of similarity metrics
        """
        texts1, texts2 = zip(*text_pairs)
        
        # Generate embeddings
        embeddings1 = self.encode_texts(list(texts1))
        embeddings2 = self.encode_texts(list(texts2))
        
        # Compute cosine similarities
        similarities = np.sum(embeddings1 * embeddings2, axis=1) / (
            np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        )
        
        return {
            "mean_similarity": float(np.mean(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "std_similarity": float(np.std(similarities))
        }
    
    def evaluate_retrieval(
        self, 
        queries: List[str],
        corpus: List[str],
        relevant_docs: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            queries: List of queries
            corpus: List of documents
            relevant_docs: List of lists of relevant document indices for each query
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            Dictionary of retrieval metrics
        """
        # Generate embeddings
        query_embeddings = self.encode_texts(queries)
        corpus_embeddings = self.encode_texts(corpus)
        
        # Compute similarities
        similarities = cosine_similarity(query_embeddings, corpus_embeddings)
        
        # Compute metrics
        metrics = {}
        
        # For each query, find top-k documents
        for k in k_values:
            precision_at_k = []
            recall_at_k = []
            
            for i, similarity in enumerate(similarities):
                # Get top-k document indices
                top_k_indices = np.argsort(similarity)[::-1][:k]
                
                # Get relevant documents for this query
                relevant = set(relevant_docs[i])
                
                # Compute precision and recall
                retrieved_relevant = len([idx for idx in top_k_indices if idx in relevant])
                precision = retrieved_relevant / k if k > 0 else 0
                recall = retrieved_relevant / len(relevant) if len(relevant) > 0 else 0
                
                precision_at_k.append(precision)
                recall_at_k.append(recall)
            
            metrics[f"precision@{k}"] = float(np.mean(precision_at_k))
            metrics[f"recall@{k}"] = float(np.mean(recall_at_k))
        
        # Compute Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for i, similarity in enumerate(similarities):
            # Get ranking of relevant documents
            relevant = relevant_docs[i]
            if not relevant:
                continue
                
            # Get ranks of relevant documents
            rankings = np.argsort(similarity)[::-1]
            ranks = [np.where(rankings == rel)[0][0] + 1 for rel in relevant if rel in rankings]
            
            if ranks:
                # MRR uses the first relevant document
                mrr_scores.append(1.0 / min(ranks))
        
        metrics["mrr"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        
        return metrics
    
    def evaluate_classification(
        self, 
        texts: List[str],
        labels: List[int],
        method: str = "knn",
        n_neighbors: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """Evaluate classification performance using embeddings.
        
        Args:
            texts: List of text documents
            labels: List of labels
            method: Classification method (knn, svm)
            n_neighbors: Number of neighbors for KNN
            test_size: Test size for train/test split
            
        Returns:
            Dictionary of classification metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate classifier
        if method == "knn":
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train_scaled, y_train)
        elif method == "svm":
            clf = SVC(probability=True)
            clf.fit(X_train_scaled, y_train)
        else:
            raise ValueError(f"Unknown classification method: {method}")
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        
        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_precision": float(precision_score(y_test, y_pred, average="macro")),
            "macro_recall": float(recall_score(y_test, y_pred, average="macro")),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro"))
        }
        
        return metrics
    
    def visualize_embeddings(
        self, 
        texts: List[str],
        labels: Optional[List[int]] = None,
        n_components: int = 2,
        output_path: Optional[str] = None
    ):
        """Visualize embeddings using dimensionality reduction.
        
        Args:
            texts: List of text documents
            labels: Optional list of labels for coloring
            n_components: Number of components for dimensionality reduction
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the saved visualization if output_path is provided
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        # Apply dimensionality reduction
        if n_components == 2:
            # First use PCA to reduce to 50 dimensions, then t-SNE
            if embeddings.shape[1] > 50:
                pca = PCA(n_components=50)
                embeddings_reduced = pca.fit_transform(embeddings)
            else:
                embeddings_reduced = embeddings
                
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings_reduced)
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            if labels is not None:
                # Use labels for coloring
                unique_labels = list(set(labels))
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    indices = [idx for idx, l in enumerate(labels) if l == label]
                    plt.scatter(
                        embeddings_2d[indices, 0],
                        embeddings_2d[indices, 1],
                        c=[colors[i]],
                        label=f"Class {label}",
                        alpha=0.7
                    )
                plt.legend()
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
            
            plt.title("t-SNE Visualization of Embeddings")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return output_path
            else:
                plt.show()
        
        else:
            # 3D visualization
            if embeddings.shape[1] > 50:
                pca = PCA(n_components=50)
                embeddings_reduced = pca.fit_transform(embeddings)
            else:
                embeddings_reduced = embeddings
                
            tsne = TSNE(n_components=3, random_state=42)
            embeddings_3d = tsne.fit_transform(embeddings_reduced)
            
            # 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if labels is not None:
                # Use labels for coloring
                unique_labels = list(set(labels))
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    indices = [idx for idx, l in enumerate(labels) if l == label]
                    ax.scatter(
                        embeddings_3d[indices, 0],
                        embeddings_3d[indices, 1],
                        embeddings_3d[indices, 2],
                        c=[colors[i]],
                        label=f"Class {label}",
                        alpha=0.7
                    )
                ax.legend()
            else:
                ax.scatter(
                    embeddings_3d[:, 0],
                    embeddings_3d[:, 1],
                    embeddings_3d[:, 2],
                    alpha=0.7
                )
            
            ax.set_title("t-SNE Visualization of Embeddings (3D)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return output_path
            else:
                plt.show()


class EvaluationResultManager:
    """Manager for handling evaluation results."""
    
    def __init__(self, results_dir: str = "data/evaluation_results"):
        """Initialize the result manager.
        
        Args:
            results_dir: Directory to store evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(
        self, 
        model_name: str,
        task: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save evaluation results.
        
        Args:
            model_name: Name of the evaluated model
            task: Evaluation task (similarity, retrieval, classification)
            metrics: Dictionary of metrics
            metadata: Optional metadata about the evaluation
            
        Returns:
            Path to the saved results
        """
        # Create result object
        result = {
            "model_name": model_name,
            "task": task,
            "metrics": metrics,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Create result file name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_file = self.results_dir / f"{model_name}_{task}_{timestamp}.json"
        
        # Save result
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved evaluation results to {result_file}")
        return str(result_file)
    
    def load_results(self, result_file: str) -> Dict[str, Any]:
        """Load evaluation results.
        
        Args:
            result_file: Path to the result file
            
        Returns:
            Loaded results
        """
        with open(result_file, "r") as f:
            result = json.load(f)
        
        return result
    
    def list_results(self, model_name: Optional[str] = None, task: Optional[str] = None) -> List[str]:
        """List available evaluation results.
        
        Args:
            model_name: Optional filter by model name
            task: Optional filter by task
            
        Returns:
            List of result file paths
        """
        pattern = ""
        if model_name and task:
            pattern = f"{model_name}_{task}_*.json"
        elif model_name:
            pattern = f"{model_name}_*.json"
        elif task:
            pattern = f"*_{task}_*.json"
        else:
            pattern = "*.json"
        
        return [str(p) for p in self.results_dir.glob(pattern)]
    
    def compare_models(
        self,
        model_names: List[str],
        task: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare models on a specific task.
        
        Args:
            model_names: List of model names to compare
            task: Task to compare on
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary of model metrics
        """
        comparison = {}
        
        for model_name in model_names:
            # Find latest result for this model and task
            results = sorted(self.list_results(model_name, task))
            if not results:
                logger.warning(f"No results found for model {model_name} on task {task}")
                continue
            
            # Load the latest result
            result = self.load_results(results[-1])
            comparison[model_name] = result["metrics"]
        
        # Visualize comparison if requested
        if output_path and comparison:
            self._visualize_comparison(comparison, task, output_path)
        
        return comparison
    
    def _visualize_comparison(
        self,
        comparison: Dict[str, Dict[str, float]],
        task: str,
        output_path: str
    ):
        """Visualize model comparison.
        
        Args:
            comparison: Model comparison results
            task: Task being compared
            output_path: Path to save visualization
        """
        # Check if we have metrics to visualize
        if not comparison:
            return
        
        # Get all metrics
        all_metrics = set()
        for metrics in comparison.values():
            all_metrics.update(metrics.keys())
        
        # Create DataFrame for visualization
        data = []
        for model_name, metrics in comparison.items():
            row = {"Model": model_name}
            for metric in all_metrics:
                row[metric] = metrics.get(metric, 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        df.set_index("Model").plot(kind="bar", figsize=(12, 8))
        plt.title(f"Model Comparison on {task.capitalize()} Task")
        plt.ylabel("Metric Value")
        plt.legend(title="Metrics")
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path)
        plt.close()


def evaluate_model_cli():
    """CLI for model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate embedding models")
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--task", required=True, choices=["similarity", "retrieval", "classification", "visualization"],
                        help="Evaluation task")
    parser.add_argument("--data", required=True, help="Path to evaluation data")
    parser.add_argument("--output", help="Path to save output")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.data, "r") as f:
        data = json.load(f)
    
    # Create evaluator
    evaluator = EmbeddingEvaluator(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    metrics = None
    
    if args.task == "similarity":
        if "text_pairs" not in data:
            raise ValueError("Data must contain 'text_pairs' field for similarity evaluation")
        
        metrics = evaluator.evaluate_similarity(data["text_pairs"])
    
    elif args.task == "retrieval":
        if not all(k in data for k in ["queries", "corpus", "relevant_docs"]):
            raise ValueError("Data must contain 'queries', 'corpus', and 'relevant_docs' fields for retrieval evaluation")
        
        metrics = evaluator.evaluate_retrieval(
            queries=data["queries"],
            corpus=data["corpus"],
            relevant_docs=data["relevant_docs"],
            k_values=data.get("k_values", [1, 3, 5, 10])
        )
    
    elif args.task == "classification":
        if not all(k in data for k in ["texts", "labels"]):
            raise ValueError("Data must contain 'texts' and 'labels' fields for classification evaluation")
        
        metrics = evaluator.evaluate_classification(
            texts=data["texts"],
            labels=data["labels"],
            method=data.get("method", "knn"),
            n_neighbors=data.get("n_neighbors", 5),
            test_size=data.get("test_size", 0.2)
        )
    
    elif args.task == "visualization":
        if "texts" not in data:
            raise ValueError("Data must contain 'texts' field for visualization")
        
        output_path = args.output or "embedding_visualization.png"
        evaluator.visualize_embeddings(
            texts=data["texts"],
            labels=data.get("labels"),
            n_components=data.get("n_components", 2),
            output_path=output_path
        )
        print(f"Visualization saved to {output_path}")
        return
    
    # Save results
    if metrics:
        result_manager = EvaluationResultManager()
        model_name = Path(args.model).name
        result_file = result_manager.save_results(
            model_name=model_name,
            task=args.task,
            metrics=metrics,
            metadata={
                "data_file": args.data,
                "batch_size": args.batch_size,
                "device": args.device
            }
        )
        
        print(f"Evaluation results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    evaluate_model_cli()

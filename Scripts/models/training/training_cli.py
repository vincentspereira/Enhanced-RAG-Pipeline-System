"""
Command-line interface for custom embedding model training and deployment.

This module provides a CLI for training, evaluating, and deploying custom
embedding models.
"""
import os
import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import argparse
from datetime import datetime

from .pipeline import EmbeddingTrainer, ModelDeployment
from .data_management import TrainingDataManager
from .evaluation import EmbeddingEvaluator, EvaluationResultManager
from ..registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Custom Embedding Training and Deployment CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data management commands
    data_parser = subparsers.add_parser("data", help="Training data management")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data command to run")
    
    # Import text files
    import_text_parser = data_subparsers.add_parser("import-text", help="Import text files as training data")
    import_text_parser.add_argument("--source", required=True, help="Source directory with text files")
    import_text_parser.add_argument("--name", required=True, help="Dataset name")
    import_text_parser.add_argument("--description", help="Dataset description")
    import_text_parser.add_argument("--pattern", default="*.txt", help="File pattern (default: *.txt)")
    
    # Import CSV
    import_csv_parser = data_subparsers.add_parser("import-csv", help="Import CSV data for training")
    import_csv_parser.add_argument("--file", required=True, help="CSV file path")
    import_csv_parser.add_argument("--name", required=True, help="Dataset name")
    import_csv_parser.add_argument("--text-column", required=True, help="Column containing text data")
    import_csv_parser.add_argument("--label-column", help="Column containing labels")
    import_csv_parser.add_argument("--description", help="Dataset description")
    
    # Create contrastive pairs
    contrastive_parser = data_subparsers.add_parser("create-pairs", help="Create contrastive pairs for training")
    contrastive_parser.add_argument("--source", required=True, help="Source dataset name")
    contrastive_parser.add_argument("--name", required=True, help="Output dataset name")
    contrastive_parser.add_argument("--pairs", type=int, default=10000, help="Number of pairs to generate")
    contrastive_parser.add_argument("--description", help="Dataset description")
    
    # Split dataset
    split_parser = data_subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--name", required=True, help="Dataset name")
    split_parser.add_argument("--train", type=float, default=0.8, help="Training set ratio")
    split_parser.add_argument("--val", type=float, default=0.1, help="Validation set ratio")
    split_parser.add_argument("--test", type=float, default=0.1, help="Test set ratio")
    
    # List datasets
    list_parser = data_subparsers.add_parser("list", help="List available datasets")
    
    # Show dataset info
    info_parser = data_subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("--name", required=True, help="Dataset name")
    
    # Training commands
    train_parser = subparsers.add_parser("train", help="Train embedding model")
    train_parser.add_argument("--dataset", required=True, help="Training dataset name")
    train_parser.add_argument("--name", required=True, help="Model name")
    train_parser.add_argument("--base-model", default="Snowflake-Labs/arctic-embed2", help="Base model to fine-tune")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--output-dir", default="data/custom_embeddings", help="Output directory")
    train_parser.add_argument("--description", help="Model description")
    train_parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for tracking")
    
    # Evaluation commands
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate embedding model")
    eval_parser.add_argument("--model", required=True, help="Model path or name")
    eval_parser.add_argument("--dataset", required=True, help="Evaluation dataset name")
    eval_parser.add_argument("--task", required=True, choices=["similarity", "retrieval", "classification", "visualization"],
                            help="Evaluation task")
    eval_parser.add_argument("--output", help="Output path for visualization")
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare embedding models")
    compare_parser.add_argument("--models", required=True, nargs="+", help="Models to compare")
    compare_parser.add_argument("--task", required=True, help="Task to compare on")
    compare_parser.add_argument("--output", default="model_comparison.png", help="Output visualization path")
    
    # Deployment commands
    deploy_parser = subparsers.add_parser("deploy", help="Deploy embedding model")
    deploy_parser.add_argument("--model", required=True, help="Model name to deploy")
    deploy_parser.add_argument("--version", default="latest", help="Model version to deploy")
    
    # List models
    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Initialize common components
    model_registry = ModelRegistry()
    data_manager = TrainingDataManager()
    result_manager = EvaluationResultManager()
    
    # Handle data management commands
    if args.command == "data":
        if args.data_command == "import-text":
            dataset = data_manager.import_text_files(
                source_dir=args.source,
                dataset_name=args.name,
                description=args.description,
                file_pattern=args.pattern
            )
            print(f"Imported dataset: {dataset.name}")
            print(f"  Description: {dataset.description}")
            print(f"  Path: {dataset.data_path}")
            print(f"  Files: {dataset.metadata['file_count']}")
        
        elif args.data_command == "import-csv":
            dataset = data_manager.import_csv(
                csv_path=args.file,
                dataset_name=args.name,
                text_column=args.text_column,
                label_column=args.label_column,
                description=args.description
            )
            print(f"Imported CSV dataset: {dataset.name}")
            print(f"  Description: {dataset.description}")
            print(f"  Path: {dataset.data_path}")
            print(f"  Records: {dataset.metadata['record_count']}")
        
        elif args.data_command == "create-pairs":
            dataset = data_manager.create_contrastive_pairs(
                source_dataset=args.source,
                output_dataset=args.name,
                n_pairs=args.pairs,
                description=args.description
            )
            print(f"Created contrastive pairs dataset: {dataset.name}")
            print(f"  Description: {dataset.description}")
            print(f"  Path: {dataset.data_path}")
            print(f"  Positive pairs: {dataset.metadata['positive_pair_count']}")
            print(f"  Negative pairs: {dataset.metadata['negative_pair_count']}")
        
        elif args.data_command == "split":
            split_data = data_manager.split_dataset(
                dataset_name=args.name,
                train_ratio=args.train,
                val_ratio=args.val,
                test_ratio=args.test
            )
            print(f"Split dataset: {args.name}")
            print(f"  Train: {len(split_data['train']['texts']) if 'texts' in split_data['train'] else len(split_data['train']['positive_pairs'])} samples")
            print(f"  Validation: {len(split_data['validation']['texts']) if 'texts' in split_data['validation'] else len(split_data['validation']['positive_pairs'])} samples")
            print(f"  Test: {len(split_data['test']['texts']) if 'texts' in split_data['test'] else len(split_data['test']['positive_pairs'])} samples")
        
        elif args.data_command == "list":
            datasets = data_manager.list_datasets()
            print(f"Available datasets ({len(datasets)}):")
            for dataset_name in datasets:
                dataset = data_manager.get_dataset(dataset_name)
                print(f"  {dataset.name}: {dataset.description}")
        
        elif args.data_command == "info":
            dataset = data_manager.get_dataset(args.name)
            if not dataset:
                print(f"Dataset {args.name} not found")
                return
            
            print(f"Dataset: {dataset.name}")
            print(f"  Description: {dataset.description}")
            print(f"  Path: {dataset.data_path}")
            print(f"  Created: {dataset.created_at}")
            print("  Metadata:")
            for key, value in dataset.metadata.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {value}")
    
    # Handle training command
    elif args.command == "train":
        # Get dataset
        dataset = data_manager.get_dataset(args.dataset)
        if not dataset:
            print(f"Dataset {args.dataset} not found")
            return
        
        # Check if dataset has been split
        if "splits" not in dataset.metadata:
            print(f"Dataset {args.dataset} has not been split. Run 'data split' command first.")
            return
        
        # Load dataset splits
        split_path = Path(dataset.data_path) / "splits.json"
        with open(split_path, "r") as f:
            splits = json.load(f)
        
        # Create trainer
        trainer = EmbeddingTrainer(
            base_model=args.base_model,
            output_dir=args.output_dir,
            model_registry=model_registry,
            use_wandb=args.wandb
        )
        
        # Prepare data
        if "texts" in splits["train"]:
            # Simple text dataset
            train_texts = splits["train"]["texts"]
            train_labels = splits["train"].get("labels")
            
            val_texts = splits["validation"]["texts"]
            val_labels = splits["validation"].get("labels")
            
            # Prepare training data
            trainer.prepare_data(
                texts=train_texts + val_texts,  # Use both train and validation for training
                labels=train_labels + val_labels if train_labels and val_labels else None,
                test_size=len(val_texts) / (len(train_texts) + len(val_texts))
            )
        
        elif "positive_pairs" in splits["train"]:
            # Contrastive pairs dataset
            # For contrastive learning, we would need to implement custom training logic
            print("Contrastive pair training not implemented yet")
            return
        
        # Configure training
        trainer.configure_training(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs
        )
        
        # Train the model
        print(f"Training model {args.name} based on {args.base_model}")
        print(f"Using dataset {args.dataset}")
        print(f"Training configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Epochs: {args.epochs}")
        
        model_path = trainer.train(
            model_name=args.name,
            model_description=args.description or f"Fine-tuned {args.base_model} on {args.dataset}"
        )
        
        print(f"Training completed. Model saved to {model_path}")
    
    # Handle evaluation command
    elif args.command == "evaluate":
        # Get dataset
        dataset = data_manager.get_dataset(args.dataset)
        if not dataset:
            print(f"Dataset {args.dataset} not found")
            return
        
        # Load dataset
        if "splits" in dataset.metadata:
            # Use test split if available
            split_path = Path(dataset.data_path) / "splits.json"
            with open(split_path, "r") as f:
                splits = json.load(f)
            
            # Use test split for evaluation
            if "texts" in splits["test"]:
                eval_data = {
                    "texts": splits["test"]["texts"],
                    "labels": splits["test"].get("labels")
                }
            elif "positive_pairs" in splits["test"]:
                eval_data = {
                    "positive_pairs": splits["test"]["positive_pairs"],
                    "negative_pairs": splits["test"]["negative_pairs"]
                }
            else:
                print(f"Unknown data format in dataset {args.dataset}")
                return
        else:
            # Use entire dataset
            processed_path = Path(dataset.data_path) / "processed.json"
            with open(processed_path, "r") as f:
                eval_data = json.load(f)
        
        # Create evaluator
        evaluator = EmbeddingEvaluator(model_path=args.model)
        
        # Run evaluation based on task
        if args.task == "similarity":
            if "positive_pairs" in eval_data and "negative_pairs" in eval_data:
                # Evaluate on both positive and negative pairs
                pos_metrics = evaluator.evaluate_similarity(eval_data["positive_pairs"])
                neg_metrics = evaluator.evaluate_similarity(eval_data["negative_pairs"])
                
                print("Positive pair similarity:")
                for metric, value in pos_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                print("\nNegative pair similarity:")
                for metric, value in neg_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                # Save results
                model_name = Path(args.model).name
                result_manager.save_results(
                    model_name=model_name,
                    task="similarity_positive",
                    metrics=pos_metrics,
                    metadata={"dataset": args.dataset, "pair_type": "positive"}
                )
                
                result_manager.save_results(
                    model_name=model_name,
                    task="similarity_negative",
                    metrics=neg_metrics,
                    metadata={"dataset": args.dataset, "pair_type": "negative"}
                )
            else:
                print("Dataset does not contain text pairs for similarity evaluation")
                return
        
        elif args.task == "classification":
            if "texts" in eval_data and "labels" in eval_data:
                # Evaluate classification performance
                metrics = evaluator.evaluate_classification(
                    texts=eval_data["texts"],
                    labels=eval_data["labels"]
                )
                
                print("Classification metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                # Save results
                model_name = Path(args.model).name
                result_manager.save_results(
                    model_name=model_name,
                    task="classification",
                    metrics=metrics,
                    metadata={"dataset": args.dataset}
                )
            else:
                print("Dataset does not contain text and labels for classification evaluation")
                return
        
        elif args.task == "visualization":
            if "texts" in eval_data:
                # Create visualization
                output_path = args.output or f"{Path(args.model).name}_visualization.png"
                evaluator.visualize_embeddings(
                    texts=eval_data["texts"],
                    labels=eval_data.get("labels"),
                    output_path=output_path
                )
                print(f"Visualization saved to {output_path}")
            else:
                print("Dataset does not contain texts for visualization")
                return
    
    # Handle compare command
    elif args.command == "compare":
        comparison = result_manager.compare_models(
            model_names=args.models,
            task=args.task,
            output_path=args.output
        )
        
        if not comparison:
            print(f"No results found for models on task {args.task}")
            return
        
        print(f"Model comparison on task {args.task}:")
        for model_name, metrics in comparison.items():
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        print(f"\nComparison visualization saved to {args.output}")
    
    # Handle deployment command
    elif args.command == "deploy":
        deployment = ModelDeployment(registry=model_registry)
        
        # Deploy the model
        try:
            deploy_path = deployment.deploy_model(
                model_name=args.model,
                version=args.version
            )
            
            print(f"Deployed model {args.model} (version {args.version}) to {deploy_path}")
            print("Updated config.yaml to use the deployed model")
        except ValueError as e:
            print(f"Deployment error: {e}")
    
    # Handle list-models command
    elif args.command == "list-models":
        models = model_registry.list_models()
        
        print(f"Available models ({len(models)}):")
        for model_name in models:
            # Get latest version
            version = model_registry.get_model_version(model_name)
            print(f"  {model_name} (latest: v{version.version})")
            print(f"    {version.description}")
            print(f"    Path: {version.path}")
    
    # No command or invalid command
    else:
        print("Please specify a valid command")
        print("Available commands: data, train, evaluate, compare, deploy, list-models")


if __name__ == "__main__":
    main()

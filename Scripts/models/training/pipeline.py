"""
Custom embedding model training pipeline for domain-specific optimization.

This module provides functionality for training domain-specific embedding models
to enhance the performance of the RAG pipeline on specialized content.
"""
import os
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer,
    AdamW, get_linear_schedule_with_warmup
)
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.model_selection import train_test_split
from datetime import datetime
import wandb

from ..registry import ModelRegistry, ModelVersion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    """Dataset for training embedding models."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length=512):
        """Initialize the dataset.
        
        Args:
            texts: List of text documents/chunks
            labels: Optional list of labels for supervised training scenarios
            tokenizer: The tokenizer to use for encoding texts
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        item = {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

class EmbeddingTrainer:
    """Trainer for custom embedding models."""
    
    def __init__(
        self, 
        base_model: str = "Snowflake-Labs/arctic-embed2",
        output_dir: str = "data/custom_embeddings",
        model_registry: Optional[ModelRegistry] = None,
        use_wandb: bool = False,
        wandb_project: str = "rag-embedding-training"
    ):
        """Initialize the embedding trainer.
        
        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save model outputs
            model_registry: Optional model registry for version tracking
            use_wandb: Whether to use Weights & Biases for experiment tracking
            wandb_project: WandB project name
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry = model_registry
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Initialize model components
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModel.from_pretrained(base_model)
        
        # Training attributes
        self.best_model_path = None
        self.training_args = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def prepare_data(
        self, 
        texts: List[str], 
        labels: Optional[List[int]] = None,
        test_size: float = 0.2,
        max_length: int = 512
    ):
        """Prepare datasets for training.
        
        Args:
            texts: List of text documents/chunks
            labels: Optional list of labels for supervised training
            test_size: Fraction of data to use for evaluation
            max_length: Maximum sequence length for tokenization
        """
        train_texts, eval_texts = train_test_split(texts, test_size=test_size, random_state=42)
        
        train_labels, eval_labels = None, None
        if labels is not None:
            train_labels, eval_labels = train_test_split(labels, test_size=test_size, random_state=42)
        
        self.train_dataset = EmbeddingDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        self.eval_dataset = EmbeddingDataset(
            texts=eval_texts,
            labels=eval_labels,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        logger.info(f"Prepared training dataset with {len(self.train_dataset)} samples")
        logger.info(f"Prepared evaluation dataset with {len(self.eval_dataset)} samples")
    
    def configure_training(
        self,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        fp16: bool = True,
        save_steps: int = 1000,
        eval_steps: int = 1000
    ):
        """Configure training parameters.
        
        Args:
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            fp16: Whether to use mixed precision training
            save_steps: Steps between model checkpoints
            eval_steps: Steps between evaluations
        """
        run_name = f"embedding-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            save_steps=save_steps,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb" if self.use_wandb else "none",
            run_name=run_name
        )
        
        logger.info(f"Configured training with {num_epochs} epochs and {batch_size} batch size")
        
    def train(self, model_name: str, model_description: str = None):
        """Train the embedding model.
        
        Args:
            model_name: Name for the trained model
            model_description: Optional description for the model
        
        Returns:
            Path to the best model checkpoint
        """
        if self.train_dataset is None or self.eval_dataset is None:
            raise ValueError("Data must be prepared before training. Call prepare_data() first.")
            
        if self.training_args is None:
            raise ValueError("Training must be configured before starting. Call configure_training() first.")
        
        if self.use_wandb:
            wandb.init(project=self.wandb_project, name=model_name)
        
        # Create custom trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        
        # Train the model
        logger.info(f"Starting training for model: {model_name}")
        train_result = trainer.train()
        
        # Evaluate the model
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
          # Save the best model
        final_model_dir = self.output_dir / model_name
        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))
        
        # Save training arguments and metrics
        with open(final_model_dir / "training_args.json", "w") as f:
            json.dump(self.training_args.to_dict(), f, indent=2)
            
        with open(final_model_dir / "eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)
        
        self.best_model_path = final_model_dir
        
        # Register the model if registry is available
        if self.model_registry is not None:
            version_info = {
                "base_model": self.base_model,
                "training_date": datetime.now().isoformat(),
                "eval_metrics": eval_metrics
            }
            
            model_version = ModelVersion(
                name=model_name,
                version="1.0",
                path=str(final_model_dir),
                metadata=version_info,
                description=model_description or f"Fine-tuned embedding model based on {self.base_model}"
            )
            
            self.model_registry.register_model(model_version)
            logger.info(f"Registered model {model_name} in the model registry")
        
        if self.use_wandb:
            wandb.finish()
            
        logger.info(f"Training completed. Model saved to {self.best_model_path}")
        return self.best_model_path

    def evaluate_model(self, test_texts: List[str], test_labels: Optional[List[int]] = None):
        """Evaluate the trained model on a test set.
        
        Args:
            test_texts: List of text documents for evaluation
            test_labels: Optional list of labels for supervised evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model_path is None:
            raise ValueError("No trained model available. Run train() first.")
        
        # Load the best model
        model = AutoModel.from_pretrained(str(self.best_model_path))
        model.eval()
        
        # Create test dataset
        test_dataset = EmbeddingDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        test_dataloader = DataLoader(test_dataset, batch_size=16)
        
        # Compute embeddings and evaluate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        eval_metrics = {}
        # Compute embeddings and metrics
        # This would involve domain-specific evaluation
        
        return eval_metrics


class DataManager:
    """Manager for training data collection and preparation."""
    
    def __init__(self, data_dir: str = "data/training_data"):
        """Initialize the data manager.
        
        Args:
            data_dir: Directory to store training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_data(self, source_path: str, output_name: str):
        """Collect data from a source and prepare it for training.
        
        Args:
            source_path: Path to source data (file or directory)
            output_name: Name for the processed dataset
        
        Returns:
            Path to the processed dataset
        """
        # Implementation depends on data format
        # This would handle various data sources and formats
        pass
    
    def create_training_pairs(self, texts: List[str], method: str = "contrastive"):
        """Create training pairs for different embedding training methods.
        
        Args:
            texts: List of text documents
            method: Training method (contrastive, triplet, etc.)
            
        Returns:
            Training data pairs/triplets
        """
        # Implementation for different training approaches
        pass
    
    def save_dataset(self, data: Dict[str, Any], name: str):
        """Save a prepared dataset to disk.
        
        Args:
            data: Dataset to save
            name: Name for the dataset
            
        Returns:
            Path to the saved dataset
        """
        output_path = self.data_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved dataset to {output_path}")
        return output_path
    
    def load_dataset(self, name: str):
        """Load a prepared dataset from disk.
        
        Args:
            name: Name of the dataset to load
            
        Returns:
            Loaded dataset
        """
        dataset_path = self.data_dir / f"{name}.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {name} not found at {dataset_path}")
        
        with open(dataset_path, "r") as f:
            data = json.load(f)
        
        return data


class ModelDeployment:
    """System for deploying trained embedding models."""
    
    def __init__(
        self, 
        registry: ModelRegistry,
        deployment_dir: str = "data/deployed_models",
        config_path: str = "config.yaml"
    ):
        """Initialize the model deployment system.
        
        Args:
            registry: Model registry instance
            deployment_dir: Directory for deployed models
            config_path: Path to the config file
        """
        self.registry = registry
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path
        
    def deploy_model(self, model_name: str, version: str = "latest"):
        """Deploy a model from the registry.
        
        Args:
            model_name: Name of the model to deploy
            version: Version to deploy (defaults to latest)
            
        Returns:
            Path to the deployed model
        """
        # Get model version from registry
        model_version = self.registry.get_model_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Model {model_name} version {version} not found in registry")
        
        # Copy model to deployment directory
        deploy_path = self.deployment_dir / f"{model_name}-{model_version.version}"
        
        # If the model is not already deployed, copy it
        if not deploy_path.exists():
            source_path = Path(model_version.path)
            deploy_path.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            for file in source_path.glob("*"):
                if file.is_file():
                    os.system(f'copy "{file}" "{deploy_path}"')
        
        # Update config to point to the new model
        self._update_config(model_name, str(deploy_path))
        
        logger.info(f"Deployed model {model_name} version {model_version.version} to {deploy_path}")
        return deploy_path
    
    def _update_config(self, model_name: str, model_path: str):
        """Update the configuration to use the deployed model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the deployed model
        """
        import yaml
        
        # Load current config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Update embedding model path
        config["model"]["embedding_model"] = model_path
        
        # Save updated config
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Updated config to use deployed model at {model_path}")

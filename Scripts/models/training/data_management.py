"""
Training data management for custom embedding models.

This module provides functionality for collecting, preparing, and managing
training data for custom embedding models.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import shutil
import hashlib
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDataSet:
    """Class representing a training dataset."""
    
    def __init__(
        self, 
        name: str,
        description: str = None,
        data_path: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize the training dataset.
        
        Args:
            name: Dataset name
            description: Optional dataset description
            data_path: Path to the dataset files
            metadata: Optional metadata about the dataset
        """
        self.name = name
        self.description = description
        self.data_path = data_path
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "data_path": self.data_path,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary representation."""
        dataset = cls(
            name=data["name"],
            description=data.get("description"),
            data_path=data.get("data_path"),
            metadata=data.get("metadata", {})
        )
        dataset.created_at = data.get("created_at", datetime.now().isoformat())
        return dataset


class TrainingDataManager:
    """Manager for training data collection and preparation."""
    
    def __init__(self, data_dir: str = "data/training_data"):
        """Initialize the data manager.
        
        Args:
            data_dir: Directory to store training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure dataset catalog directory exists
        self.catalog_path = self.data_dir / "catalog.json"
        if not self.catalog_path.exists():
            with open(self.catalog_path, "w") as f:
                json.dump({"datasets": []}, f)
        
        # Load catalog
        self._load_catalog()
    
    def _load_catalog(self):
        """Load the dataset catalog."""
        with open(self.catalog_path, "r") as f:
            self.catalog = json.load(f)
    
    def _save_catalog(self):
        """Save the dataset catalog."""
        with open(self.catalog_path, "w") as f:
            json.dump(self.catalog, f, indent=2)
    
    def register_dataset(self, dataset: TrainingDataSet):
        """Register a dataset in the catalog.
        
        Args:
            dataset: Dataset to register
        """
        # Check if dataset already exists
        existing = [d for d in self.catalog["datasets"] if d["name"] == dataset.name]
        if existing:
            logger.warning(f"Dataset {dataset.name} already exists in catalog. Updating.")
            idx = self.catalog["datasets"].index(existing[0])
            self.catalog["datasets"][idx] = dataset.to_dict()
        else:
            self.catalog["datasets"].append(dataset.to_dict())
        
        self._save_catalog()
        logger.info(f"Registered dataset {dataset.name} in catalog")
    
    def get_dataset(self, name: str) -> Optional[TrainingDataSet]:
        """Get a dataset from the catalog.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset if found, None otherwise
        """
        for dataset_dict in self.catalog["datasets"]:
            if dataset_dict["name"] == name:
                return TrainingDataSet.from_dict(dataset_dict)
        return None
    
    def list_datasets(self) -> List[str]:
        """List all available datasets.
        
        Returns:
            List of dataset names
        """
        return [d["name"] for d in self.catalog["datasets"]]
    
    def import_text_files(
        self, 
        source_dir: str, 
        dataset_name: str,
        description: str = None,
        file_pattern: str = "*.txt"
    ) -> TrainingDataSet:
        """Import text files as a training dataset.
        
        Args:
            source_dir: Directory containing text files
            dataset_name: Name for the new dataset
            description: Optional dataset description
            file_pattern: Glob pattern for finding text files
            
        Returns:
            Created dataset
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} not found")
        
        # Create dataset directory
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Find and copy text files
        files = list(source_path.glob(file_pattern))
        if not files:
            raise ValueError(f"No files matching {file_pattern} found in {source_dir}")
        
        # Copy files to dataset directory
        file_count = 0
        for file in files:
            dest_file = dataset_dir / file.name
            shutil.copy(file, dest_file)
            file_count += 1
        
        # Extract texts for processing
        texts = []
        for file in dataset_dir.glob(file_pattern):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    text = f.read()
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Error reading {file}: {e}")
        
        # Create a processed version as JSON
        processed_path = dataset_dir / "processed.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump({"texts": texts}, f)
        
        # Create dataset and register it
        metadata = {
            "file_count": file_count,
            "text_count": len(texts),
            "source_dir": str(source_dir),
            "import_date": datetime.now().isoformat()
        }
        
        dataset = TrainingDataSet(
            name=dataset_name,
            description=description or f"Imported from {source_dir}",
            data_path=str(dataset_dir),
            metadata=metadata
        )
        
        self.register_dataset(dataset)
        logger.info(f"Imported {file_count} files into dataset {dataset_name}")
        return dataset
    
    def import_csv(
        self, 
        csv_path: str, 
        dataset_name: str,
        text_column: str,
        label_column: Optional[str] = None,
        description: str = None
    ) -> TrainingDataSet:
        """Import data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            dataset_name: Name for the new dataset
            text_column: Column containing text data
            label_column: Optional column containing labels
            description: Optional dataset description
            
        Returns:
            Created dataset
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file {csv_path} not found")
        
        # Create dataset directory
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Copy original file
        dest_file = dataset_dir / csv_file.name
        shutil.copy(csv_file, dest_file)
        
        # Read and process CSV
        df = pd.read_csv(csv_file)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
        
        if label_column and label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV")
        
        # Extract texts and labels
        texts = df[text_column].tolist()
        labels = None
        if label_column:
            labels = df[label_column].tolist()
        
        # Create a processed version as JSON
        processed_data = {"texts": texts}
        if labels:
            processed_data["labels"] = labels
            
        processed_path = dataset_dir / "processed.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f)
        
        # Create dataset and register it
        metadata = {
            "source_file": str(csv_path),
            "text_column": text_column,
            "label_column": label_column,
            "record_count": len(texts),
            "import_date": datetime.now().isoformat()
        }
        
        dataset = TrainingDataSet(
            name=dataset_name,
            description=description or f"Imported from {csv_path}",
            data_path=str(dataset_dir),
            metadata=metadata
        )
        
        self.register_dataset(dataset)
        logger.info(f"Imported {len(texts)} records into dataset {dataset_name}")
        return dataset
    
    def create_contrastive_pairs(
        self, 
        source_dataset: str, 
        output_dataset: str,
        n_pairs: int = 10000,
        description: str = None
    ) -> TrainingDataSet:
        """Create contrastive pairs for training.
        
        Args:
            source_dataset: Source dataset name
            output_dataset: Name for the new dataset
            n_pairs: Number of pairs to generate
            description: Optional dataset description
            
        Returns:
            Created dataset
        """
        # Get source dataset
        dataset = self.get_dataset(source_dataset)
        if not dataset:
            raise ValueError(f"Source dataset {source_dataset} not found")
        
        # Load processed data
        processed_path = Path(dataset.data_path) / "processed.json"
        with open(processed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = data["texts"]
        if len(texts) < 2:
            raise ValueError(f"Source dataset {source_dataset} has insufficient data")
        
        # Create dataset directory
        dataset_dir = self.data_dir / output_dataset
        dataset_dir.mkdir(exist_ok=True)
        
        # Generate contrastive pairs
        import random
        random.seed(42)
        
        positive_pairs = []
        negative_pairs = []
        
        # Create positive pairs (similar texts)
        # This is just a simple example - in practice, use more sophisticated methods
        for i in range(min(n_pairs // 2, len(texts))):
            text = texts[i]
            # Create a positive pair by slightly modifying the text
            # In practice, use semantically similar texts
            tokens = text.split()
            if len(tokens) > 10:
                # Remove or shuffle some tokens to create a similar variant
                variant_tokens = tokens.copy()
                remove_idx = random.sample(range(len(tokens)), min(3, len(tokens) // 10))
                for idx in sorted(remove_idx, reverse=True):
                    del variant_tokens[idx]
                positive_pairs.append((text, " ".join(variant_tokens)))
        
        # Create negative pairs (dissimilar texts)
        for i in range(min(n_pairs // 2, len(texts))):
            text1 = texts[i]
            # Find a dissimilar text
            j = random.randint(0, len(texts) - 1)
            while i == j:
                j = random.randint(0, len(texts) - 1)
            text2 = texts[j]
            negative_pairs.append((text1, text2))
        
        # Create a processed version as JSON
        processed_data = {
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs
        }
            
        processed_path = dataset_dir / "processed.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f)
        
        # Create dataset and register it
        metadata = {
            "source_dataset": source_dataset,
            "positive_pair_count": len(positive_pairs),
            "negative_pair_count": len(negative_pairs),
            "creation_date": datetime.now().isoformat()
        }
        
        dataset = TrainingDataSet(
            name=output_dataset,
            description=description or f"Contrastive pairs from {source_dataset}",
            data_path=str(dataset_dir),
            metadata=metadata
        )
        
        self.register_dataset(dataset)
        logger.info(f"Created contrastive dataset {output_dataset} with {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        return dataset
    
    def split_dataset(
        self, 
        dataset_name: str, 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Split a dataset into train/validation/test sets.
        
        Args:
            dataset_name: Dataset name
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split datasets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")
        
        # Get source dataset
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        # Load processed data
        processed_path = Path(dataset.data_path) / "processed.json"
        with open(processed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "texts" in data:
            # Simple text dataset
            texts = data["texts"]
            labels = data.get("labels")
            
            if labels:
                # Split with stratification
                train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                    texts, labels, train_size=train_ratio, random_state=seed, stratify=labels
                )
                
                # Adjust val_ratio for the remaining data
                val_size = val_ratio / (val_ratio + test_ratio)
                val_texts, test_texts, val_labels, test_labels = train_test_split(
                    temp_texts, temp_labels, train_size=val_size, random_state=seed, stratify=temp_labels
                )
                
                split_data = {
                    "train": {"texts": train_texts, "labels": train_labels},
                    "validation": {"texts": val_texts, "labels": val_labels},
                    "test": {"texts": test_texts, "labels": test_labels}
                }
            else:
                # Split without stratification
                train_texts, temp_texts = train_test_split(
                    texts, train_size=train_ratio, random_state=seed
                )
                
                # Adjust val_ratio for the remaining data
                val_size = val_ratio / (val_ratio + test_ratio)
                val_texts, test_texts = train_test_split(
                    temp_texts, train_size=val_size, random_state=seed
                )
                
                split_data = {
                    "train": {"texts": train_texts},
                    "validation": {"texts": val_texts},
                    "test": {"texts": test_texts}
                }
        
        elif "positive_pairs" in data and "negative_pairs" in data:
            # Contrastive pairs dataset
            pos_pairs = data["positive_pairs"]
            neg_pairs = data["negative_pairs"]
            
            # Split positive pairs
            train_pos, temp_pos = train_test_split(
                pos_pairs, train_size=train_ratio, random_state=seed
            )
            
            val_size = val_ratio / (val_ratio + test_ratio)
            val_pos, test_pos = train_test_split(
                temp_pos, train_size=val_size, random_state=seed
            )
            
            # Split negative pairs
            train_neg, temp_neg = train_test_split(
                neg_pairs, train_size=train_ratio, random_state=seed
            )
            
            val_neg, test_neg = train_test_split(
                temp_neg, train_size=val_size, random_state=seed
            )
            
            split_data = {
                "train": {"positive_pairs": train_pos, "negative_pairs": train_neg},
                "validation": {"positive_pairs": val_pos, "negative_pairs": val_neg},
                "test": {"positive_pairs": test_pos, "negative_pairs": test_neg}
            }
        
        else:
            raise ValueError(f"Unknown dataset format for {dataset_name}")
        
        # Save split data
        split_path = Path(dataset.data_path) / "splits.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f)
        
        # Update dataset metadata
        dataset.metadata["splits"] = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "train_size": len(split_data["train"]["texts"]) if "texts" in split_data["train"] else len(split_data["train"]["positive_pairs"]),
            "val_size": len(split_data["validation"]["texts"]) if "texts" in split_data["validation"] else len(split_data["validation"]["positive_pairs"]),
            "test_size": len(split_data["test"]["texts"]) if "texts" in split_data["test"] else len(split_data["test"]["positive_pairs"]),
            "split_date": datetime.now().isoformat()
        }
        
        self.register_dataset(dataset)
        logger.info(f"Split dataset {dataset_name} into train/validation/test sets")
        
        return split_data

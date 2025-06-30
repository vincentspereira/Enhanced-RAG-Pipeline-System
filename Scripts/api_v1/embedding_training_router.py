"""
API router for custom embedding model training.

This module provides API endpoints for managing datasets, training, 
evaluating and deploying custom embedding models optimized for 
domain-specific content.
"""
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Body, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import os
import json
import logging
from datetime import datetime
import uuid
import shutil
from pathlib import Path
import asyncio

from Scripts.models.training.pipeline import EmbeddingTrainer
from Scripts.models.training.data_management import DatasetManager
from Scripts.models.training.evaluation import ModelEvaluator
from Scripts.models.registry import ModelRegistry, ModelVersion
from Scripts.models.benchmarking import benchmark_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API router
router = APIRouter(prefix="/embedding-training", tags=["Embedding Training"])

# Auth imports
from ...auth.dependencies import get_auth_manager_dependency
from ...auth.auth_manager import AuthManager, AuthUser, Permission


# Pydantic models for API
class DatasetCreateRequest(BaseModel):
    """Model for dataset creation request."""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    source_type: str = Field(..., description="Dataset source type (file_upload, directory, existing_collection)")
    source_path: Optional[str] = Field(None, description="Path to source data (for directory or collection)")
    tags: Optional[List[str]] = Field(None, description="Tags for the dataset")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "financial_documents",
                "description": "Financial documents and reports for domain-specific training",
                "source_type": "directory",
                "source_path": "data/financial_docs",
                "tags": ["financial", "reports", "banking"]
            }
        }


class DatasetSplitRequest(BaseModel):
    """Model for dataset splitting request."""
    dataset_id: str = Field(..., description="Dataset ID to split")
    train_ratio: float = Field(0.8, description="Ratio of data for training set (0.0-1.0)")
    validation_ratio: float = Field(0.1, description="Ratio of data for validation set (0.0-1.0)")
    test_ratio: float = Field(0.1, description="Ratio of data for test set (0.0-1.0)")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    @validator('train_ratio', 'validation_ratio', 'test_ratio')
    def validate_ratios(cls, v, values):
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {v}")
        
        # Check sum of ratios if we have all three
        if 'train_ratio' in values and 'validation_ratio' in values:
            total = values['train_ratio'] + values['validation_ratio'] + v
            if abs(total - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "ds_12345",
                "train_ratio": 0.8,
                "validation_ratio": 0.1,
                "test_ratio": 0.1,
                "random_seed": 42
            }
        }


class ModelTrainingRequest(BaseModel):
    """Model for embedding model training request."""
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    base_model: str = Field("Snowflake-Labs/arctic-embed2", description="Base model to fine-tune")
    dataset_id: str = Field(..., description="Dataset ID to use for training")
    training_params: Optional[Dict[str, Any]] = Field(None, description="Training parameters")
    tags: Optional[List[str]] = Field(None, description="Tags for the model")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "financial_embed",
                "description": "Financial domain optimized embedding model",
                "base_model": "Snowflake-Labs/arctic-embed2",
                "dataset_id": "ds_12345",
                "training_params": {
                    "epochs": 3,
                    "learning_rate": 2e-5,
                    "batch_size": 32,
                    "max_seq_length": 512
                },
                "tags": ["financial", "embeddings"]
            }
        }


class ModelEvaluationRequest(BaseModel):
    """Model for embedding model evaluation request."""
    model_id: str = Field(..., description="Model ID to evaluate")
    dataset_id: Optional[str] = Field(None, description="Dataset ID to use for evaluation")
    evaluation_type: str = Field("similarity", description="Evaluation type (similarity, classification, retrieval)")
    metrics: Optional[List[str]] = Field(None, description="Metrics to compute")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model_12345",
                "dataset_id": "ds_12345",
                "evaluation_type": "similarity",
                "metrics": ["cosine_similarity", "euclidean_distance", "manhattan_distance"]
            }
        }


class ModelComparisonRequest(BaseModel):
    """Model for embedding model comparison request."""
    model_ids: List[str] = Field(..., description="List of model IDs to compare")
    dataset_id: str = Field(..., description="Dataset ID to use for comparison")
    metrics: Optional[List[str]] = Field(None, description="Metrics to compute")
    
    class Config:
        schema_extra = {
            "example": {
                "model_ids": ["model_12345", "model_67890"],
                "dataset_id": "ds_12345",
                "metrics": ["cosine_similarity", "precision", "recall", "f1_score"]
            }
        }


class ModelDeploymentRequest(BaseModel):
    """Model for embedding model deployment request."""
    model_id: str = Field(..., description="Model ID to deploy")
    deployment_name: str = Field(..., description="Deployment name")
    description: Optional[str] = Field(None, description="Deployment description")
    production: bool = Field(False, description="Whether to deploy as production model")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model_12345",
                "deployment_name": "financial_embeddings_v1",
                "description": "Financial domain optimized embeddings for production use",
                "production": True
            }
        }


# Initialize managers
dataset_manager = DatasetManager(data_dir="data/embedding_datasets")
model_registry = ModelRegistry(registry_dir="data/models/embeddings")

# Ongoing training jobs
training_jobs = {}
evaluation_jobs = {}


# Helper function to get background tasks status
def get_job_status(job_id: str, job_type: str = "training"):
    """Get status of a background job."""
    jobs = training_jobs if job_type == "training" else evaluation_jobs
    
    if job_id not in jobs:
        return None
    
    return jobs[job_id]


@router.post("/datasets", summary="Create a new dataset")
async def create_dataset(request: DatasetCreateRequest):
    """Create a new dataset for embedding model training."""
    try:
        # Create dataset
        dataset_id = dataset_manager.create_dataset(
            name=request.name,
            description=request.description,
            source_type=request.source_type,
            source_path=request.source_path,
            tags=request.tags
        )
        
        # Get dataset info
        dataset = dataset_manager.get_dataset(dataset_id)
        
        return {
            "message": f"Dataset '{request.name}' created successfully",
            "dataset_id": dataset_id,
            "dataset": dataset
        }
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/upload", summary="Upload files for a dataset")
async def upload_dataset_files(
    dataset_id: str = Form(..., description="Dataset ID"),
    files: List[UploadFile] = File(..., description="Files to upload")
):
    """Upload files to add to a dataset."""
    try:
        # Check if dataset exists
        dataset = dataset_manager.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' not found")
        
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(dataset_manager.data_dir, dataset_id, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            
            # Save file
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            saved_files.append(file.filename)
        
        # Process uploaded files
        processed = dataset_manager.process_uploads(dataset_id)
        
        return {
            "message": f"Uploaded {len(saved_files)} files to dataset '{dataset['name']}'",
            "dataset_id": dataset_id,
            "files": saved_files,
            "processed": processed
        }
    except Exception as e:
        logger.error(f"Error uploading dataset files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", summary="List all datasets")
async def list_datasets(
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    limit: int = Query(100, description="Maximum number of datasets to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """List all available datasets for embedding model training."""
    try:
        datasets = dataset_manager.list_datasets(tags=tags, limit=limit, offset=offset)
        return {
            "datasets": datasets,
            "count": len(datasets),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}", summary="Get dataset details")
async def get_dataset(dataset_id: str = Path(..., description="Dataset ID")):
    """Get details of a specific dataset."""
    try:
        dataset = dataset_manager.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' not found")
        
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/split", summary="Split dataset into train/validation/test")
async def split_dataset(
    request: DatasetSplitRequest,
    dataset_id: str = Path(..., description="Dataset ID")
):
    """Split a dataset into training, validation, and test sets."""
    if dataset_id != request.dataset_id:
        raise HTTPException(status_code=400, detail="Dataset ID in path and request body must match")
    
    try:
        # Check if dataset exists
        dataset = dataset_manager.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{dataset_id}' not found")
        
        # Split dataset
        split_info = dataset_manager.split_dataset(
            dataset_id=dataset_id,
            train_ratio=request.train_ratio,
            validation_ratio=request.validation_ratio,
            test_ratio=request.test_ratio,
            random_seed=request.random_seed
        )
        
        return {
            "message": f"Dataset '{dataset['name']}' split successfully",
            "dataset_id": dataset_id,
            "split_info": split_info
        }
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/train", summary="Train a new embedding model")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train a new embedding model on a dataset."""
    try:
        # Check if dataset exists
        dataset = dataset_manager.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{request.dataset_id}' not found")
        
        # Check if dataset is split
        if not dataset.get("split_info"):
            raise HTTPException(status_code=400, detail=f"Dataset '{request.dataset_id}' has not been split. Please split the dataset first.")
        
        # Create job ID
        job_id = f"train_{uuid.uuid4().hex}"
        
        # Create model data
        model_data = {
            "name": request.name,
            "description": request.description,
            "base_model": request.base_model,
            "dataset_id": request.dataset_id,
            "training_params": request.training_params or {},
            "tags": request.tags or [],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "job_id": job_id
        }
        
        # Save model metadata
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        model_dir = os.path.join(model_registry.registry_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(model_data, f, indent=2)
        
        # Initialize trainer
        trainer = EmbeddingTrainer(
            base_model=request.base_model,
            output_dir=model_dir,
            model_registry=model_registry
        )
        
        # Store job
        training_jobs[job_id] = {
            "model_id": model_id,
            "status": "pending",
            "message": "Job created, waiting to start",
            "progress": 0,
            "started_at": None,
            "completed_at": None
        }
        
        # Start training in background
        def _train_model_task():
            try:
                # Update job status
                training_jobs[job_id]["status"] = "running"
                training_jobs[job_id]["message"] = "Training started"
                training_jobs[job_id]["started_at"] = datetime.now().isoformat()
                
                # Get dataset paths
                train_path = os.path.join(dataset_manager.data_dir, request.dataset_id, "splits", "train.jsonl")
                val_path = os.path.join(dataset_manager.data_dir, request.dataset_id, "splits", "validation.jsonl")
                
                # Train model
                model_version = trainer.train(
                    train_data_path=train_path,
                    val_data_path=val_path,
                    **request.training_params or {}
                )
                
                # Register model
                if model_version:
                    model_registry.register_model(
                        model_id=model_id,
                        model_path=model_version.path,
                        metadata={
                            "name": request.name,
                            "description": request.description,
                            "base_model": request.base_model,
                            "dataset_id": request.dataset_id,
                            "training_params": request.training_params or {},
                            "tags": request.tags or []
                        }
                    )
                
                # Update job status
                training_jobs[job_id]["status"] = "completed"
                training_jobs[job_id]["message"] = "Training completed successfully"
                training_jobs[job_id]["progress"] = 100
                training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                training_jobs[job_id]["model_version"] = model_version.to_dict() if model_version else None
                
            except Exception as e:
                logger.error(f"Error in training job {job_id}: {e}")
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
                training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Start background task
        background_tasks.add_task(_train_model_task)
        
        return {
            "message": f"Training job started for model '{request.name}'",
            "model_id": model_id,
            "job_id": job_id,
            "status": "pending"
        }
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/train/{job_id}", summary="Get training job status")
async def get_training_status(job_id: str = Path(..., description="Training job ID")):
    """Get the status of a model training job."""
    try:
        job_status = get_job_status(job_id, "training")
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Training job with ID '{job_id}' not found")
        
        return job_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", summary="List all models")
async def list_models(
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of models to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """List all available embedding models."""
    try:
        # Get models from registry
        models = model_registry.list_models(model_type="embedding")
        
        # Filter by tags if provided
        if tags:
            models = [
                model for model in models 
                if any(tag in model.get("metadata", {}).get("tags", []) for tag in tags)
            ]
        
        # Filter by status if provided
        if status:
            models = [
                model for model in models 
                if model.get("metadata", {}).get("status") == status
            ]
        
        # Apply pagination
        paginated_models = models[offset:offset+limit]
        
        return {
            "models": paginated_models,
            "count": len(paginated_models),
            "total": len(models),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", summary="Get model details")
async def get_model(model_id: str = Path(..., description="Model ID")):
    """Get details of a specific embedding model."""
    try:
        model = model_registry.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found")
        
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/evaluate", summary="Evaluate an embedding model")
async def evaluate_model(request: ModelEvaluationRequest, background_tasks: BackgroundTasks):
    """Evaluate an embedding model on a dataset."""
    try:
        # Check if model exists
        model = model_registry.get_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID '{request.model_id}' not found")
        
        # Check if dataset exists if provided
        dataset = None
        if request.dataset_id:
            dataset = dataset_manager.get_dataset(request.dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"Dataset with ID '{request.dataset_id}' not found")
        
        # Create job ID
        job_id = f"eval_{uuid.uuid4().hex}"
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_registry=model_registry)
        
        # Store job
        evaluation_jobs[job_id] = {
            "model_id": request.model_id,
            "dataset_id": request.dataset_id,
            "status": "pending",
            "message": "Evaluation job created, waiting to start",
            "progress": 0,
            "started_at": None,
            "completed_at": None
        }
        
        # Start evaluation in background
        def _evaluate_model_task():
            try:
                # Update job status
                evaluation_jobs[job_id]["status"] = "running"
                evaluation_jobs[job_id]["message"] = "Evaluation started"
                evaluation_jobs[job_id]["started_at"] = datetime.now().isoformat()
                
                # Get dataset paths
                test_path = None
                if request.dataset_id:
                    test_path = os.path.join(dataset_manager.data_dir, request.dataset_id, "splits", "test.jsonl")
                
                # Evaluate model
                evaluation_results = evaluator.evaluate_model(
                    model_id=request.model_id,
                    test_data_path=test_path,
                    evaluation_type=request.evaluation_type,
                    metrics=request.metrics
                )
                
                # Update job status
                evaluation_jobs[job_id]["status"] = "completed"
                evaluation_jobs[job_id]["message"] = "Evaluation completed successfully"
                evaluation_jobs[job_id]["progress"] = 100
                evaluation_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                evaluation_jobs[job_id]["results"] = evaluation_results
                
            except Exception as e:
                logger.error(f"Error in evaluation job {job_id}: {e}")
                evaluation_jobs[job_id]["status"] = "failed"
                evaluation_jobs[job_id]["message"] = f"Evaluation failed: {str(e)}"
                evaluation_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Start background task
        background_tasks.add_task(_evaluate_model_task)
        
        return {
            "message": f"Evaluation job started for model '{request.model_id}'",
            "job_id": job_id,
            "status": "pending"
        }
    except Exception as e:
        logger.error(f"Error creating evaluation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/evaluate/{job_id}", summary="Get evaluation job status")
async def get_evaluation_status(job_id: str = Path(..., description="Evaluation job ID")):
    """Get the status of a model evaluation job."""
    try:
        job_status = get_job_status(job_id, "evaluation")
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Evaluation job with ID '{job_id}' not found")
        
        return job_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/compare", summary="Compare multiple embedding models")
async def compare_models(request: ModelComparisonRequest):
    """Compare multiple embedding models on a dataset."""
    try:
        # Check if models exist
        for model_id in request.model_ids:
            model = model_registry.get_model(model_id)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found")
        
        # Check if dataset exists
        dataset = dataset_manager.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID '{request.dataset_id}' not found")
        
        # Get test data path
        test_path = os.path.join(dataset_manager.data_dir, request.dataset_id, "splits", "test.jsonl")
        if not os.path.exists(test_path):
            raise HTTPException(status_code=400, detail=f"Test split not found for dataset '{request.dataset_id}'. Please split the dataset first.")
        
        # Run benchmarking
        comparison_results = benchmark_models(
            model_ids=request.model_ids,
            test_data_path=test_path,
            metrics=request.metrics,
            model_registry=model_registry
        )
        
        return {
            "message": f"Compared {len(request.model_ids)} models on dataset '{dataset['name']}'",
            "model_ids": request.model_ids,
            "dataset_id": request.dataset_id,
            "results": comparison_results
        }
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/deploy", summary="Deploy an embedding model")
async def deploy_model(request: ModelDeploymentRequest):
    """Deploy an embedding model for use in the RAG pipeline."""
    try:
        # Check if model exists
        model = model_registry.get_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID '{request.model_id}' not found")
        
        # Deploy model
        deployment_id = model_registry.deploy_model(
            model_id=request.model_id,
            deployment_name=request.deployment_name,
            description=request.description,
            production=request.production
        )
        
        return {
            "message": f"Model '{model['name']}' deployed successfully as '{request.deployment_name}'",
            "deployment_id": deployment_id,
            "model_id": request.model_id,
            "deployment_name": request.deployment_name,
            "production": request.production
        }
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

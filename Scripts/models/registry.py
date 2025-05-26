"""
Model registry and versioning system.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from functools import wraps
import json
import os
import logging
from datetime import datetime
from pathlib import Path

from Scripts.models.benchmarking import ModelVersionTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelVersion:
    """Class representing a model version."""
    
    def __init__(
        self, 
        name: str,
        version: str,
        path: str,
        metadata: Dict[str, Any] = None,
        description: str = None
    ):
        """Initialize a model version.
        
        Args:
            name: Model name
            version: Version string
            path: Path to the model files
            metadata: Optional metadata about the model
            description: Optional description of the model
        """
        self.name = name
        self.version = version
        self.path = path
        self.metadata = metadata or {}
        self.description = description or f"{name} version {version}"
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "metadata": self.metadata,
            "description": self.description,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary representation."""
        model_version = cls(
            name=data["name"],
            version=data["version"],
            path=data["path"],
            metadata=data.get("metadata", {}),
            description=data.get("description")
        )
        model_version.created_at = data.get("created_at", datetime.now().isoformat())
        return model_version

# Global instances
_model_trackers = {}

def get_model_tracker(model_type: str, storage_dir: str = "data/model_versions") -> ModelVersionTracker:
    """Get a model version tracker for a specific model type."""
    global _model_trackers
    
    key = f"{model_type}:{storage_dir}"
    if key not in _model_trackers:
        _model_trackers[key] = ModelVersionTracker(storage_dir)
    
    return _model_trackers[key]

class VersionedModel:
    """Wrapper for model objects with version tracking."""
    
    def __init__(self, model_type: str, model_name: str, model_instance: Any,
                model_details: Optional[Dict[str, Any]] = None,
                version_id: Optional[str] = None):
        """Initialize a versioned model.
        
        Args:
            model_type: Type of model (embedding, llm, etc.)
            model_name: Name of the model
            model_instance: The actual model instance
            model_details: Details about the model
            version_id: Explicit version ID to use (otherwise, will get active or create new)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = model_instance
        
        # Get model tracker
        self.tracker = get_model_tracker(model_type)
        
        # Get or create version ID
        if version_id:
            self.version_id = version_id
        else:
            # Try to get active version
            active_version = self.tracker.get_active_version(model_type, model_name)
            if active_version:
                self.version_id = active_version["version_id"]
            else:
                # Register new version
                details = model_details or {
                    "description": f"Auto-registered {model_name}",
                    "created_at": datetime.now().isoformat()
                }
                self.version_id = self.tracker.register_model(model_type, model_name, details)
                self.tracker.set_active_version(model_type, model_name, self.version_id)
    
    def add_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Add metrics to the model version.
        
        Args:
            metrics: Performance metrics to add
            
        Returns:
            True if successful, False otherwise
        """
        return self.tracker.add_metrics(
            self.model_type, 
            self.model_name, 
            self.version_id, 
            metrics
        )
    
    def get_details(self) -> Dict[str, Any]:
        """Get model version details.
        
        Returns:
            Details for this model version
        """
        versions = self.tracker.get_model_versions(self.model_type, self.model_name)
        for version in versions:
            if version["version_id"] == self.version_id:
                return {
                    "model_type": self.model_type,
                    "model_name": self.model_name,
                    "version_id": self.version_id,
                    "is_active": self.is_active(),
                    **version
                }
        
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "version_id": self.version_id,
            "is_active": self.is_active(),
            "error": "Version details not found"
        }
    
    def is_active(self) -> bool:
        """Check if this is the active version.
        
        Returns:
            True if this is the active version, False otherwise
        """
        active = self.tracker.get_active_version(self.model_type, self.model_name)
        return active and active["version_id"] == self.version_id
    
    def set_as_active(self) -> bool:
        """Set this model version as active.
        
        Returns:
            True if successful, False otherwise
        """
        return self.tracker.set_active_version(self.model_type, self.model_name, self.version_id)
    
    def __getattr__(self, name):
        """Delegate attribute access to the model instance."""
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        """Call the model instance."""
        if callable(self.model):
            return self.model(*args, **kwargs)
        raise TypeError(f"'{self.model.__class__.__name__}' object is not callable")

def track_model_version(model_type: str):
    """Decorator to track model versions.
    
    This decorator can be applied to model factory functions to automatically
    track model versions and performance.
    
    Args:
        model_type: Type of model (embedding, llm, etc.)
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract model name from args or kwargs
            model_name = kwargs.get('model_name')
            if model_name is None and args:
                model_name = args[0]
            
            if model_name is None:
                raise ValueError("Model name must be provided as first argument or model_name kwarg")
            
            # Version ID can be provided or will be auto-determined
            version_id = kwargs.pop('version_id', None)
            
            # Create a model instance
            model_instance = func(*args, **kwargs)
            
            # Extract model details if provided
            model_details = kwargs.get('model_details')
            
            # Create and return a versioned model
            return VersionedModel(
                model_type=model_type,
                model_name=model_name,
                model_instance=model_instance,
                model_details=model_details,
                version_id=version_id
            )
        
        return wrapper
    
    return decorator

class ModelRegistry:
    """Registry for managing model providers."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._factories = {}
        self._models = {}
    
    def register_model(self, model_version: ModelVersion) -> bool:
        """Register a model version in the registry.
        
        Args:
            model_version: Model version to register
            
        Returns:
            True if successful, False otherwise
        """
        model_type = "embedding"  # Default to embedding type for custom models
        model_name = model_version.name
        
        # Get model tracker
        tracker = get_model_tracker(model_type)
        
        # Register with the tracker
        version_id = tracker.register_model(
            model_type=model_type,
            model_name=model_name,
            model_details={
                "path": model_version.path,
                "version": model_version.version,
                "description": model_version.description,
                "metadata": model_version.metadata
            }
        )
        
        logger.info(f"Registered model {model_name} with version ID {version_id}")
        return True
    
    def get_model_version(self, model_name: str, version: str = "latest") -> Optional[ModelVersion]:
        """Get a model version from the registry.
        
        Args:
            model_name: Model name
            version: Version string or "latest"
            
        Returns:
            ModelVersion object if found, None otherwise
        """
        model_type = "embedding"  # Default to embedding type for custom models
        
        # Get model tracker
        tracker = get_model_tracker(model_type)
        
        if version == "latest":
            # Get active version
            version_data = tracker.get_active_version(model_type, model_name)
            if not version_data:
                logger.warning(f"No active version found for model {model_name}")
                
                # Try to get any version
                versions = tracker.get_model_versions(model_type, model_name)
                if versions:
                    version_data = versions[-1]  # Most recent
                else:
                    return None
        else:
            # Find specific version
            versions = tracker.get_model_versions(model_type, model_name)
            version_data = None
            
            for v in versions:
                if v.get("details", {}).get("version") == version:
                    version_data = v
                    break
            
            if not version_data:
                logger.warning(f"Version {version} not found for model {model_name}")
                return None
        
        # Convert tracker version to ModelVersion
        details = version_data.get("details", {})
        
        model_version = ModelVersion(
            name=model_name,
            version=details.get("version", "1.0"),
            path=details.get("path", ""),
            metadata=details.get("metadata", {}),
            description=details.get("description", "")
        )
        
        return model_version
    
    def register_factory(self, model_type: str, factory_func):
        """Register a model factory function.
        
        Args:
            model_type: Type of model (embedding, llm, etc.)
            factory_func: Factory function to create models
        """
        # Apply tracking decorator if not already applied
        if not hasattr(factory_func, '_tracked'):
            factory_func = track_model_version(model_type)(factory_func)
            factory_func._tracked = True
            
        self._factories[model_type] = factory_func
    
    def create_model(self, model_type: str, model_name: str, **kwargs) -> VersionedModel:
        """Create a new model instance.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            **kwargs: Additional arguments for the factory function
            
        Returns:
            VersionedModel instance
        """
        if model_type not in self._factories:
            raise ValueError(f"No factory registered for model type: {model_type}")
            
        # Create model using factory
        model = self._factories[model_type](model_name, **kwargs)
        
        # Cache model instance
        cache_key = f"{model_type}:{model_name}"
        self._models[cache_key] = model
        
        return model
    
    def get_model(self, model_type: str, model_name: str, **kwargs) -> VersionedModel:
        """Get a model instance, creating it if necessary.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            **kwargs: Additional arguments for the factory function
            
        Returns:
            VersionedModel instance
        """
        cache_key = f"{model_type}:{model_name}"
        if cache_key in self._models:
            return self._models[cache_key]
            
        return self.create_model(model_type, model_name, **kwargs)
    
    def get_version(self, model_type: str, model_name: str, 
                   version_id: str, **kwargs) -> VersionedModel:
        """Get a specific version of a model.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            version_id: Version ID
            **kwargs: Additional arguments for the factory function
            
        Returns:
            VersionedModel instance
        """
        # Force create with specific version
        return self.create_model(
            model_type=model_type,
            model_name=model_name,
            version_id=version_id,
            **kwargs
        )
      def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """List available models.
        
        Args:
            model_type: Optional type to filter by
            
        Returns:
            List of model names
        """
        if model_type == "embedding" or model_type is None:
            # For embedding models, check custom models in the tracker
            tracker = get_model_tracker("embedding")
            if "embedding" in tracker.model_registry:
                return list(tracker.model_registry["embedding"].keys())
            
        return []
    
    def get_active_versions(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """Get active versions for all models.
        
        Args:
            model_type: Optional type to filter by
            
        Returns:
            Dictionary mapping model types to dictionaries of model name -> version ID
        """
        active_versions = {}
        
        # Get all trackers
        for key, tracker in _model_trackers.items():
            tracker_type = key.split(':')[0]
            
            if model_type and tracker_type != model_type:
                continue
                
            # Get model registry from tracker
            registry = tracker.model_registry.get(tracker_type, {})
            
            active_versions[tracker_type] = {}
            for model_name, model_data in registry.items():
                if model_data.get("active_version"):
                    active_versions[tracker_type][model_name] = model_data["active_version"]
        
        return active_versions

# Create global registry instance
model_registry = ModelRegistry()

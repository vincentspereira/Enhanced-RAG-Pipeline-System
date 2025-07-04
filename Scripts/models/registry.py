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
        metadata: Optional[Dict[str, Any]] = None, # Made metadata Optional for clarity
        description: Optional[str] = None,
        model_type: Optional[str] = None # Added model_type
    ):
        """Initialize a model version.
        
        Args:
            name: Model name
            version: Version string
            path: Path to the model files
            metadata: Optional metadata about the model
            description: Optional description of the model
            model_type: Type of model (e.g., "embedding", "llm")
        """
        self.name = name
        self.version = version
        self.path = path
        self.metadata = metadata or {}
        self.description = description or f"{name} version {version}"
        self.created_at = datetime.now().isoformat()

        # Ensure model_type is part of metadata if provided
        if model_type:
            self.metadata["model_type"] = model_type
        elif "model_type" not in self.metadata: # Default if not in metadata either
             self.metadata["model_type"] = "unknown"


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Ensure model_type from metadata is included at top level for clarity if needed,
        # but it's primarily stored within self.metadata.
        data_dict = {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "metadata": self.metadata, # Contains model_type
            "description": self.description,
            "created_at": self.created_at
        }
        return data_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary representation."""
        # model_type can be directly in metadata or passed if the dict structure changes
        metadata = data.get("metadata", {})
        model_version = cls(
            name=data["name"],
            version=data["version"],
            path=data["path"],
            metadata=metadata,
            description=data.get("description"),
            # model_type=metadata.get("model_type") # Redundant if it's in metadata
        )
        model_version.created_at = data.get("created_at", datetime.now().isoformat())
        return model_version

# Global instances
_model_trackers: Dict[str, ModelVersionTracker] = {} # Added type hint

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
        model_type = model_version.metadata.get("model_type", "unknown")

        if model_type == "unknown":
            # Default to 'embedding' if called by old code (like EmbeddingTrainer without model_type in meta),
            # but prefer explicit type for new registrations.
            model_type = "embedding"
            logger.warning(
                f"Registering model '{model_version.name}' with inferred model_type='{model_type}'. "
                f"It's recommended to set 'model_type' in ModelVersion.metadata for future registrations."
            )

        model_name = model_version.name
        tracker = get_model_tracker(model_type) # Uses the determined model_type
        
        # These are the details stored *by the tracker* for its internal version_id
        tracker_model_details = {
            "path": model_version.path,
            "version_str": model_version.version,  # User-defined version string from ModelVersion
            "description": model_version.description,
            "created_at_iso": model_version.created_at, # Store ModelVersion's own creation time
            "full_metadata": model_version.metadata # Store the entire original metadata from ModelVersion
        }
        
        version_id_from_tracker = tracker.register_model(
            model_type=model_type,
            model_name=model_name,
            model_details=tracker_model_details
        )
        
        logger.info(
            f"Registered model '{model_name}' (type: {model_type}, version_str: {model_version.version}) "
            f"with internal tracker version_id '{version_id_from_tracker}'"
        )

        # Set as active if it's the first version for this model_name/model_type combination
        # or if explicitly requested via metadata (e.g., model_version.metadata.get("set_active", False))
        if not tracker.get_active_version(model_type, model_name) or model_version.metadata.get("set_active_on_register"):
            tracker.set_active_version(model_type, model_name, version_id_from_tracker)
            logger.info(
                f"Set version_id '{version_id_from_tracker}' as active for '{model_name}' (type: {model_type})."
            )
        return True
    

    def get_model_version(self, model_name: str, model_type: str, version_str: str = "latest") -> Optional[ModelVersion]:
        """
        Get a model version from the registry by its user-defined version string or 'latest'.
        
        Args:
            model_name: Model name
            model_type: Type of model (e.g., "embedding", "llm")
            version_str: User-defined version string (e.g., "1.0.0", "v2-finetuned") or "latest"
            
        Returns:
            ModelVersion object if found, None otherwise
        """
        tracker = get_model_tracker(model_type)
        
        version_info_from_tracker = None # This is the dict stored by the tracker for a specific version_id

        if version_str == "latest":
            version_info_from_tracker = tracker.get_active_version(model_type, model_name)
            if not version_info_from_tracker:
                logger.warning(f"No active version found for model '{model_name}' of type '{model_type}'. Trying most recent.")
                # get_model_versions returns a list of dicts, each dict is { 'version_id': ..., 'details': ... }
                versions_list = tracker.get_model_versions(model_type, model_name)
                if versions_list:
                    version_info_from_tracker = versions_list[-1] # Get the last one registered with tracker
                else:
                    logger.warning(f"No versions found at all for model '{model_name}' of type '{model_type}'.")
                    return None
        else:
            versions_list = tracker.get_model_versions(model_type, model_name)
            for v_info_dict in versions_list: # Iterate through the list of version dicts from tracker
                if v_info_dict.get("details", {}).get("version_str") == version_str:
                    version_info_from_tracker = v_info_dict
                    break
            if not version_info_from_tracker:
                logger.warning(f"Version string '{version_str}' not found for model '{model_name}' of type '{model_type}'.")
                return None

        # Reconstruct ModelVersion from the details stored by the tracker
        details = version_info_from_tracker.get("details", {})
        full_meta = details.get("full_metadata", {}) # This contains the original model_type and other meta

        # Create ModelVersion instance. The model_type within full_meta should be correct.
        mv = ModelVersion(
            name=model_name,
            version=details.get("version_str", "unknown_tracker_version"), # Use the stored user-defined version string
            path=details.get("path", ""),
            metadata=full_meta,
            description=details.get("description", "")
            # model_type is already handled by ModelVersion.__init__ via full_meta
        )
        # Restore original ModelVersion instance creation timestamp if it was stored by the tracker
        mv.created_at = details.get("created_at_iso", mv.created_at)
        return mv

    def list_model_types(self) -> List[str]:
        """Lists all unique model types known to the registry (based on tracker storage)."""
        # This relies on ModelVersionTracker's internal storage structure or how it exposes types.
        # A simple way is to scan the base storage_dir if trackers save files like 'llm_models.json'.
        # Or, if _model_trackers is populated, derive from its keys.
        # For now, let's assume trackers are created on demand, so we might not know all types unless they've been accessed.
        # A more robust way would be for ModelVersionTracker to have a static method to list types from its storage_dir.
        # Let's refine this: if trackers save to files like `data/model_versions/{model_type}_models.json`
        storage_dir = Path("data/model_versions") # Default storage_dir for get_model_tracker
        if storage_dir.exists():
            return list(set(f.stem.replace("_models", "") for f in storage_dir.glob("*_models.json")))
        return list(set(key.split(':')[0] for key in _model_trackers.keys())) # Fallback to active trackers


    def list_models(self, model_type: str) -> List[str]:
        """List available model names for a specific model type."""
        tracker = get_model_tracker(model_type) # This will create a tracker if one doesn't exist for the type yet
        # ModelVersionTracker.model_registry is {model_type: {model_name: {'versions': [], 'active_version': id}}}
        if model_type in tracker.model_registry and isinstance(tracker.model_registry[model_type], dict):
            return list(tracker.model_registry[model_type].keys())
        return []

    def set_active_model_version(self, model_name: str, model_type: str, version_str: str) -> bool:
        """Sets a model version (by its user-defined string) as active."""
        tracker = get_model_tracker(model_type)
        versions_list = tracker.get_model_versions(model_type, model_name) # List of dicts from tracker
        target_tracker_version_id = None
        for v_info_dict in versions_list:
            if v_info_dict.get("details", {}).get("version_str") == version_str:
                target_tracker_version_id = v_info_dict.get("version_id")
                break

        if target_tracker_version_id:
            is_set = tracker.set_active_version(model_type, model_name, target_tracker_version_id)
            if is_set:
                logger.info(f"Successfully set version '{version_str}' (tracker_id: {target_tracker_version_id}) as active for model '{model_name}' (type: {model_type}).")
            else:
                logger.error(f"Failed to set version '{version_str}' as active for model '{model_name}' (type: {model_type}) using tracker.")
            return is_set
        else:
            logger.error(f"Cannot set active: Version string '{version_str}' for model '{model_name}' (type '{model_type}') not found.")
            return False

    def get_active_model_version_details(self, model_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Gets the raw details dictionary of the active version for a model from the tracker."""
        tracker = get_model_tracker(model_type)
        return tracker.get_active_version(model_type, model_name) # This returns the tracker's version dict


    def get_all_versions_for_model(self, model_name: str, model_type: str) -> List[ModelVersion]:
        """Gets all registered ModelVersion objects for a specific model and type."""
        tracker = get_model_tracker(model_type)
        versions_data_list = tracker.get_model_versions(model_type, model_name) # List of dicts from tracker
        model_versions_reconstructed = []
        for v_info_dict in versions_data_list:
            details = v_info_dict.get("details", {})
            full_meta = details.get("full_metadata", {})
            mv = ModelVersion(
                name=model_name, # The key under which it's stored in tracker
                version=details.get("version_str", "unknown_tracker_version"),
                path=details.get("path", ""),
                metadata=full_meta,
                description=details.get("description", "")
            )
            mv.created_at = details.get("created_at_iso", mv.created_at) # Restore original creation time
            model_versions_reconstructed.append(mv)
        return model_versions_reconstructed


    # --- Methods related to factories and VersionedModel wrapper ---
    # These seem less relevant if ModelRegistry is primarily for storing metadata of trained models.
    # Keeping them for now if they are used elsewhere, but they might be simplified or removed
    # if the focus is purely on registering and retrieving ModelVersion metadata.

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
            version_id: Version ID (tracker's internal ID)
            **kwargs: Additional arguments for the factory function

        Returns:
            VersionedModel instance
        """
        # Force create with specific version
        return self.create_model(
            model_type=model_type,
            model_name=model_name,
            version_id=version_id, # This uses tracker's version_id
            **kwargs
        )

    def get_active_versions(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, str]]:
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

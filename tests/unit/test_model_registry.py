import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from datetime import datetime

# Ensure Scripts directory is in path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from models.registry import ModelVersion, ModelRegistry, get_model_tracker
from models.benchmarking import ModelVersionTracker


# --- Fixtures ---

@pytest.fixture
def model_version_data():
    return {
        "name": "test_model",
        "version": "1.0.0",
        "path": "/path/to/model/v1",
        "metadata": {"accuracy": 0.95, "model_type": "embedding"},
        "description": "Test embedding model v1.0.0"
    }

@pytest.fixture
def model_version_obj(model_version_data):
    return ModelVersion(**model_version_data)

@pytest.fixture
@patch('models.benchmarking.ModelVersionTracker._save_registry') # Mock saving to file
@patch('models.benchmarking.ModelVersionTracker._load_registry', return_value={}) # Mock loading empty
def model_registry_with_mocked_tracker(mock_load, mock_save):
    # Reset global _model_trackers for isolation if get_model_tracker uses a global
    # This is important if tests run in parallel or affect each other.
    # For now, assuming get_model_tracker handles its state or we test its effect.

    # Clear the global _model_trackers dict in models.registry before each test run using this fixture
    # This prevents state leakage between tests using the global get_model_tracker cache.
    # Accessing it directly is a bit of a hack for testing, ideally there'd be a reset mechanism.
    if hasattr(sys.modules['models.registry'], '_model_trackers'):
         sys.modules['models.registry']._model_trackers.clear()

    registry = ModelRegistry()
    return registry


# --- Tests for ModelVersion ---

def test_model_version_creation(model_version_data):
    mv = ModelVersion(
        name=model_version_data["name"],
        version=model_version_data["version"],
        path=model_version_data["path"],
        metadata=model_version_data["metadata"],
        description=model_version_data["description"]
    )
    assert mv.name == model_version_data["name"]
    assert mv.version == model_version_data["version"]
    assert mv.path == model_version_data["path"]
    assert mv.metadata == model_version_data["metadata"]
    assert mv.metadata["model_type"] == "embedding"
    assert mv.description == model_version_data["description"]
    assert isinstance(mv.created_at, str)

def test_model_version_to_dict(model_version_obj: ModelVersion, model_version_data):
    d = model_version_obj.to_dict()
    assert d["name"] == model_version_data["name"]
    assert d["version"] == model_version_data["version"]
    assert d["path"] == model_version_data["path"]
    assert d["metadata"] == model_version_data["metadata"]
    assert d["description"] == model_version_data["description"]
    assert "created_at" in d

def test_model_version_from_dict(model_version_data):
    # Add created_at to the dict as from_dict expects it
    data_with_created_at = model_version_data.copy()
    data_with_created_at["created_at"] = datetime.now().isoformat()

    mv = ModelVersion.from_dict(data_with_created_at)
    assert mv.name == model_version_data["name"]
    assert mv.version == model_version_data["version"]
    assert mv.metadata.get("model_type") == "embedding"


# --- Tests for ModelRegistry ---

def test_model_registry_register_model(model_registry_with_mocked_tracker: ModelRegistry, model_version_obj: ModelVersion):
    registry = model_registry_with_mocked_tracker
    assert registry.register_model(model_version_obj) == True

    # Verify that the underlying tracker was used
    tracker = get_model_tracker(model_version_obj.metadata["model_type"])

    # Tracker's register_model would have been called.
    # We can check if the model appears in the tracker's internal representation.
    # This part is a bit of an integration test with ModelVersionTracker.
    registered_versions = tracker.get_model_versions(
        model_version_obj.metadata["model_type"],
        model_version_obj.name
    )
    assert len(registered_versions) > 0
    # The details stored by tracker include the user-defined version string
    assert any(v['details']['version_str'] == model_version_obj.version for v in registered_versions)


def test_model_registry_get_model_version(model_registry_with_mocked_tracker: ModelRegistry, model_version_obj: ModelVersion):
    registry = model_registry_with_mocked_tracker
    registry.register_model(model_version_obj)

    retrieved_mv = registry.get_model_version(
        model_name=model_version_obj.name,
        model_type=model_version_obj.metadata["model_type"],
        version_str=model_version_obj.version
    )
    assert retrieved_mv is not None
    assert retrieved_mv.name == model_version_obj.name
    assert retrieved_mv.version == model_version_obj.version
    assert retrieved_mv.path == model_version_obj.path
    assert retrieved_mv.metadata == model_version_obj.metadata

def test_model_registry_get_latest_model_version(model_registry_with_mocked_tracker: ModelRegistry, model_version_data):
    registry = model_registry_with_mocked_tracker

    mv1_data = model_version_data.copy()
    mv1 = ModelVersion(**mv1_data)
    registry.register_model(mv1)

    # Simulate time passing for a new version
    with patch('models.registry.datetime') as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 2)
        mv2_data = model_version_data.copy()
        mv2_data["version"] = "1.0.1"
        mv2_data["path"] = "/path/to/model/v1.0.1"
        mv2 = ModelVersion(**mv2_data)
        registry.register_model(mv2)
        # Manually set this one as active for "latest" to pick it up
        registry.set_active_model_version(mv2.name, mv2.metadata["model_type"], mv2.version)


    retrieved_latest = registry.get_model_version(
        model_name=mv1.name,
        model_type=mv1.metadata["model_type"],
        version_str="latest" # This should get mv2 because we set it active
    )
    assert retrieved_latest is not None
    assert retrieved_latest.version == "1.0.1"
    assert retrieved_latest.path == "/path/to/model/v1.0.1"

def test_model_registry_list_models(model_registry_with_mocked_tracker: ModelRegistry, model_version_data):
    registry = model_registry_with_mocked_tracker
    model_type_embedding = "embedding"
    model_type_llm = "llm"

    mv_emb_data = model_version_data.copy()
    mv_emb_data["name"] = "embedding_model_1"
    mv_emb_data["metadata"]["model_type"] = model_type_embedding
    mv_emb = ModelVersion(**mv_emb_data)
    registry.register_model(mv_emb)

    mv_llm_data = model_version_data.copy()
    mv_llm_data["name"] = "llm_model_1"
    mv_llm_data["metadata"]["model_type"] = model_type_llm
    mv_llm = ModelVersion(**mv_llm_data)
    registry.register_model(mv_llm)

    embedding_models = registry.list_models(model_type=model_type_embedding)
    assert "embedding_model_1" in embedding_models
    assert "llm_model_1" not in embedding_models

    llm_models = registry.list_models(model_type=model_type_llm)
    assert "llm_model_1" in llm_models
    assert "embedding_model_1" not in llm_models

@patch('models.benchmarking.ModelVersionTracker._get_storage_path')
def test_model_registry_list_model_types(mock_get_storage_path, model_registry_with_mocked_tracker: ModelRegistry, model_version_data):
    registry = model_registry_with_mocked_tracker

    # Mock the storage directory and create dummy tracker files
    mock_storage_dir = MagicMock(spec=os.PathLike)
    mock_get_storage_path.return_value = mock_storage_dir

    # Simulate what ModelVersionTracker does (it creates files like embedding_models.json)
    # We need to mock the behavior of ModelVersionTracker._get_storage_path or how list_model_types finds types.
    # The current list_model_types scans a directory.

    # Let's patch Path from models.registry to control its behavior for list_model_types
    with patch('models.registry.Path') as mock_path_module:
        mock_storage_dir_instance = MagicMock()
        mock_storage_dir_instance.exists.return_value = True

        # Simulate files found by glob
        mock_file_emb = MagicMock()
        mock_file_emb.stem = "embedding_models"
        mock_file_llm = MagicMock()
        mock_file_llm.stem = "llm_models"
        mock_storage_dir_instance.glob.return_value = [mock_file_emb, mock_file_llm]

        mock_path_module.return_value = mock_storage_dir_instance # Path("data/model_versions") -> mock_storage_dir_instance

        types = registry.list_model_types()
        assert "embedding" in types
        assert "llm" in types
        assert len(types) == 2


def test_model_registry_get_non_existent_version(model_registry_with_mocked_tracker: ModelRegistry):
    registry = model_registry_with_mocked_tracker
    retrieved = registry.get_model_version("test_model", "embedding", "non_existent_version")
    assert retrieved is None

# Note: The factory-related methods (register_factory, create_model, get_model, get_version for VersionedModel)
# in ModelRegistry seem to be from an older design or for a different purpose than just storing ModelVersion metadata.
# The core functionality tested above (register_model with ModelVersion, get_model_version)
# relies on ModelVersionTracker. If the factory methods are indeed used, they'd need separate tests,
# likely involving mocking the factory functions and the VersionedModel wrapper.
# For now, focusing on the ModelVersion metadata registry aspect.

```

"""
Test model benchmarking and versioning.
"""
import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Scripts.models.benchmarking import ModelBenchmark, ModelVersionTracker
from Scripts.models.registry import model_registry, VersionedModel

class TestModelBenchmarking(unittest.TestCase):
    """Test model benchmarking functionality."""
    
    def setUp(self):
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create benchmark instance
        self.benchmark = ModelBenchmark(output_dir=self.temp_dir)
        
    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('Scripts.models.benchmarking.get_embedding_model')
    def test_embedding_benchmarks(self, mock_get_model):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768] * 10
        mock_get_model.return_value = mock_model
        
        # Run benchmark
        results = self.benchmark.benchmark_embedding_model(
            model_name="test-model",
            texts=["Test text"] * 10,
            batch_sizes=[1, 5, 10],
            runs_per_batch=1
        )
        
        # Verify results
        self.assertEqual(results["model_name"], "test-model")
        self.assertIn("batch_metrics", results)
        self.assertIn("1", results["batch_metrics"])
        self.assertIn("5", results["batch_metrics"])
        self.assertIn("10", results["batch_metrics"])
        self.assertIn("optimal_batch_size", results)
        self.assertIn("peak_tokens_per_second", results)
    
    @patch('Scripts.models.benchmarking.get_llm_service')
    def test_llm_benchmarks(self, mock_get_llm):
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated text response"
        mock_get_llm.return_value = mock_llm
        
        # Run benchmark
        results = self.benchmark.benchmark_llm(
            model_name="test-llm",
            prompts=["Generate something"] * 3,
            max_tokens=10,
            temperature=0.5
        )
        
        # Verify results
        self.assertEqual(results["model_name"], "test-llm")
        self.assertEqual(results["num_prompts"], 3)
        self.assertEqual(results["max_tokens"], 10)
        self.assertEqual(results["temperature"], 0.5)
        self.assertIn("prompt_metrics", results)
        self.assertEqual(len(results["prompt_metrics"]), 3)
        self.assertIn("avg_generation_time", results)
        self.assertIn("avg_tokens_per_second", results)

class TestModelVersioning(unittest.TestCase):
    """Test model versioning functionality."""
    
    def setUp(self):
        # Create temporary directory for tracking data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create tracker instance
        self.tracker = ModelVersionTracker(storage_dir=self.temp_dir)
        
    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_model_registration(self):
        # Register a test model
        version_id = self.tracker.register_model(
            model_type="embedding",
            model_name="test-model",
            model_details={"description": "Test model for unit tests"}
        )
        
        # Verify registration
        self.assertIsNotNone(version_id)
        versions = self.tracker.get_model_versions("embedding", "test-model")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["version_id"], version_id)
        self.assertEqual(versions[0]["details"]["description"], "Test model for unit tests")
    
    def test_active_version(self):
        # Register models
        version_id1 = self.tracker.register_model(
            model_type="embedding",
            model_name="test-model",
            model_details={"version": "1.0"}
        )
        version_id2 = self.tracker.register_model(
            model_type="embedding",
            model_name="test-model",
            model_details={"version": "2.0"}
        )
        
        # Set active version
        result = self.tracker.set_active_version("embedding", "test-model", version_id2)
        self.assertTrue(result)
        
        # Get active version
        active = self.tracker.get_active_version("embedding", "test-model")
        self.assertIsNotNone(active)
        self.assertEqual(active["version_id"], version_id2)
        self.assertEqual(active["details"]["version"], "2.0")
    
    def test_add_metrics(self):
        # Register model
        version_id = self.tracker.register_model(
            model_type="embedding",
            model_name="test-model",
            model_details={"description": "Test model"}
        )
        
        # Add metrics
        metrics = {
            "accuracy": 0.95,
            "latency": 0.02,
            "throughput": 150.5
        }
        result = self.tracker.add_metrics("embedding", "test-model", version_id, metrics)
        self.assertTrue(result)
        
        # Verify metrics were added
        versions = self.tracker.get_model_versions("embedding", "test-model")
        self.assertEqual(len(versions), 1)
        
        version = versions[0]
        self.assertIn("metrics", version)
        self.assertEqual(len(version["metrics"]), 1)
        
        # Get the timestamp key
        timestamp = list(version["metrics"].keys())[0]
        added_metrics = version["metrics"][timestamp]
        
        self.assertEqual(added_metrics["accuracy"], 0.95)
        self.assertEqual(added_metrics["latency"], 0.02)
        self.assertEqual(added_metrics["throughput"], 150.5)

class TestVersionedModel(unittest.TestCase):
    """Test VersionedModel functionality."""
    
    def setUp(self):
        # Create temporary directory for tracking data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = "prediction"
        
        # Setup patching
        self.patch_tracker = patch('Scripts.models.registry.get_model_tracker')
        self.mock_tracker = self.patch_tracker.start()
        
        # Set up model tracker
        self.model_tracker = ModelVersionTracker(storage_dir=self.temp_dir)
        self.mock_tracker.return_value = self.model_tracker
        
    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        self.patch_tracker.stop()
    
    def test_versioned_model_creation(self):
        # Create a versioned model
        model = VersionedModel(
            model_type="embedding",
            model_name="test-model",
            model_instance=self.mock_model,
            model_details={"description": "Test model wrapper"}
        )
        
        # Verify model attributes
        self.assertEqual(model.model_type, "embedding")
        self.assertEqual(model.model_name, "test-model")
        self.assertEqual(model.model, self.mock_model)
        
        # Verify version was created
        versions = self.model_tracker.get_model_versions("embedding", "test-model")
        self.assertEqual(len(versions), 1)
        self.assertEqual(model.version_id, versions[0]["version_id"])
    
    def test_method_delegation(self):
        # Create a versioned model
        model = VersionedModel(
            model_type="embedding",
            model_name="test-model",
            model_instance=self.mock_model
        )
        
        # Test method delegation
        result = model.predict("input")
        self.assertEqual(result, "prediction")
        self.mock_model.predict.assert_called_once_with("input")
    
    def test_versioned_model_metrics(self):
        # Create a versioned model
        model = VersionedModel(
            model_type="embedding",
            model_name="test-model",
            model_instance=self.mock_model
        )
        
        # Add metrics
        result = model.add_metrics({"accuracy": 0.98, "f1_score": 0.97})
        self.assertTrue(result)
        
        # Get model details
        details = model.get_details()
        self.assertIn("metrics", details)
        
        # Check active status
        self.assertTrue(model.is_active())

if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, mock_open
import yaml
import os
from pathlib import Path

from Scripts.config.manager import ConfigManager, SystemConfig, ElasticsearchConfig, ModelConfig, VectorStoreConfig, ProcessingConfig, APIConfig, PathsConfig, CacheSettingsConfig, FeatureFlagsConfig

class TestConfigManagerElasticsearch(unittest.TestCase):

    def create_dummy_config_file(self, content: dict, path: str = "test_config.yaml"):
        with open(path, 'w') as f:
            yaml.dump(content, f)
        return path

    def tearDown(self):
        # Clean up any dummy config files created
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")
        if os.path.exists("custom_test_config.yaml"):
            os.remove("custom_test_config.yaml")

    def test_load_elasticsearch_config_defaults(self):
        """Test ConfigManager loads default ElasticsearchConfig when section is missing."""
        dummy_config_content = {
            "model": {"embedding_model": "test_embed"},
            # No elasticsearch section
        }
        config_path = self.create_dummy_config_file(dummy_config_content)

        manager = ConfigManager(config_path=config_path)
        self.assertIsInstance(manager.config.elasticsearch, ElasticsearchConfig)
        self.assertEqual(manager.config.elasticsearch.hosts, ["http://localhost:9200"])
        self.assertEqual(manager.config.elasticsearch.index_name, "rag_elasticsearch_index")
        self.assertFalse(manager.config.elasticsearch.enable_hybrid_search_in_es) # Default value

    def test_load_elasticsearch_config_custom_values(self):
        """Test ConfigManager loads custom ElasticsearchConfig values."""
        custom_es_settings = {
            "hosts": ["http://my-es-host:9201"],
            "index_name": "my_custom_es_index",
            "username": "es_user",
            "password": "es_password",
            "enable_hybrid_search_in_es": True,
            "semantic_weight": 0.6,
            "keyword_weight": 0.4
        }
        dummy_config_content = {
            "elasticsearch": custom_es_settings
        }
        config_path = self.create_dummy_config_file(dummy_config_content)

        manager = ConfigManager(config_path=config_path)
        es_config = manager.config.elasticsearch

        self.assertEqual(es_config.hosts, custom_es_settings["hosts"])
        self.assertEqual(es_config.index_name, custom_es_settings["index_name"])
        self.assertEqual(es_config.username, custom_es_settings["username"])
        self.assertEqual(es_config.password, custom_es_settings["password"])
        self.assertTrue(es_config.enable_hybrid_search_in_es)
        self.assertEqual(es_config.semantic_weight, 0.6)
        self.assertEqual(es_config.keyword_weight, 0.4)

    def test_system_config_composes_elasticsearch_config(self):
        """Test that SystemConfig object correctly contains the ElasticsearchConfig."""
        dummy_config_content = {
            "elasticsearch": {"index_name": "composed_index"}
        }
        config_path = self.create_dummy_config_file(dummy_config_content)
        manager = ConfigManager(config_path=config_path)

        self.assertIsInstance(manager.config, SystemConfig)
        self.assertTrue(hasattr(manager.config, 'elasticsearch'))
        self.assertIsInstance(manager.config.elasticsearch, ElasticsearchConfig)
        self.assertEqual(manager.config.elasticsearch.index_name, "composed_index")

    def test_save_and_load_persists_elasticsearch_config(self):
        """Test saving and then loading configuration preserves ElasticsearchConfig."""
        initial_es_settings = {
            "hosts": ["http://another-es:9200"],
            "index_name": "persistent_index",
            "enable_hybrid_search_in_es": True
        }
        initial_config_dict = {
            "model": {"device": "cpu"},
            "elasticsearch": initial_es_settings
        }

        config_path = "custom_test_config.yaml"

        # Initial save
        manager1 = ConfigManager(config_path=config_path) # Loads defaults or empty
        # Manually update parts of its config for the test
        manager1.config.model.device = "cpu"
        manager1.config.elasticsearch = ElasticsearchConfig(**initial_es_settings)
        manager1.save_config()

        # New manager loads from the saved file
        manager2 = ConfigManager(config_path=config_path)

        self.assertEqual(manager2.config.model.device, "cpu")
        self.assertIsInstance(manager2.config.elasticsearch, ElasticsearchConfig)
        self.assertEqual(manager2.config.elasticsearch.hosts, initial_es_settings["hosts"])
        self.assertEqual(manager2.config.elasticsearch.index_name, initial_es_settings["index_name"])
        self.assertTrue(manager2.config.elasticsearch.enable_hybrid_search_in_es)

    def test_feature_flag_for_elasticsearch_fallback_default(self):
        """Test default value of enable_elasticsearch_fallback feature flag."""
        config_path = self.create_dummy_config_file({}) # Empty config
        manager = ConfigManager(config_path=config_path)
        self.assertTrue(manager.config.feature_flags.enable_elasticsearch_fallback) # Default is True

    def test_feature_flag_for_elasticsearch_fallback_custom(self):
        """Test custom value of enable_elasticsearch_fallback feature flag."""
        dummy_config_content = {
            "feature_flags": {"enable_elasticsearch_fallback": False}
        }
        config_path = self.create_dummy_config_file(dummy_config_content)
        manager = ConfigManager(config_path=config_path)
        self.assertFalse(manager.config.feature_flags.enable_elasticsearch_fallback)

if __name__ == '__main__':
    unittest.main()

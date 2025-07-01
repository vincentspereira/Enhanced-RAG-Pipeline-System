import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Determine project root assuming this script is in Scripts/utils/
# Adjust if your structure is different or pass an absolute path for config_file_path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')

def _get_nested_value(config_dict, keys):
    """
    Helper function to navigate a nested dictionary using a list of keys.
    """
    for key in keys:
        if isinstance(config_dict, dict) and key in config_dict:
            config_dict = config_dict[key]
        else:
            return None
    return config_dict

def get_config_value(env_var_name, default=None, yaml_path=None, config_file_path=None):
    """
    Retrieves a configuration value with the following priority:
    1. Environment Variable (env_var_name)
    2. Value from YAML file (config_file_path) using yaml_path (e.g., "vector_store.qdrant.host")
    3. Provided default value

    Args:
        env_var_name (str): The name of the environment variable (e.g., "QDRANT_HOST").
                            The function will also try with common prefixes like "APP_" if not found directly.
        default: The default value to return if not found elsewhere.
        yaml_path (str, optional): The dot-separated path to the value in the YAML file
                                   (e.g., "vector_store.qdrant.host"). If None, YAML lookup is skipped.
        config_file_path (str, optional): Path to the YAML configuration file.
                                          Defaults to PROJECT_ROOT/config.yaml.
    Returns:
        The configuration value.
    """
    # Use provided config_file_path or the default
    actual_config_path = config_file_path if config_file_path is not None else DEFAULT_CONFIG_PATH

    # 1. Check environment variable first
    env_value = os.getenv(env_var_name)
    if env_value is not None:
        logger.debug(f"Config: Found value for '{env_var_name}' in environment variables.")
        return env_value

    # Try with a common prefix if direct lookup failed (e.g. if services prefix their env vars)
    # Example: if env_var_name is "DB_HOST", also try "MYAPP_DB_HOST" if relevant.
    # For now, this is simple; more sophisticated prefix handling could be added.
    # prefixed_env_var_name = "APP_" + env_var_name
    # env_value_prefixed = os.getenv(prefixed_env_var_name)
    # if env_value_prefixed is not None:
    #     logger.debug(f"Config: Found value for '{prefixed_env_var_name}' in environment variables.")
    #     return env_value_prefixed


    # 2. Check config.yaml (or specified file) if yaml_path is provided
    if yaml_path:
        try:
            with open(actual_config_path, 'r') as f:
                config_from_file = yaml.safe_load(f)

            if config_from_file:
                nested_keys = yaml_path.split('.')
                yaml_value = _get_nested_value(config_from_file, nested_keys)
                if yaml_value is not None:
                    logger.debug(f"Config: Found value for '{yaml_path}' in '{actual_config_path}'.")
                    return yaml_value
                else:
                    logger.debug(f"Config: Key '{yaml_path}' not found in '{actual_config_path}'.")
            else:
                logger.debug(f"Config: YAML file '{actual_config_path}' is empty or not valid YAML.")

        except FileNotFoundError:
            logger.debug(f"Config: File '{actual_config_path}' not found when looking for '{yaml_path}'.")
        except yaml.YAMLError as e:
            logger.warning(f"Config: YAML error parsing '{actual_config_path}': {e}")
        except Exception as e:
            logger.warning(f"Config: Error reading config file '{actual_config_path}': {e}")


    # 3. Return provided default
    logger.debug(f"Config: Using default value for '{env_var_name}' (YAML path: '{yaml_path}').")
    return default

if __name__ == '__main__':
    # Example Usage and Testing
    logging.basicConfig(level=logging.DEBUG) # Enable debug logs for this test

    print(f"Default config path being used: {DEFAULT_CONFIG_PATH}")
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        print(f"WARNING: Default config file {DEFAULT_CONFIG_PATH} does not exist. YAML lookups will fail unless a different file is specified.")

    # Test 1: Value from environment variable
    os.environ["TEST_ENV_VAR"] = "env_value"
    val = get_config_value("TEST_ENV_VAR", yaml_path="a.b.c", default="default_value")
    print(f"Test 1 (env var): Expected 'env_value', Got '{val}'")
    assert val == "env_value"
    del os.environ["TEST_ENV_VAR"]

    # Test 2: Value from YAML (assuming config.yaml has model.device = "cuda")
    # Ensure your config.yaml has this path or change the yaml_path for the test
    # For example, if config.yaml has:
    # model:
    #   device: "cuda"
    val = get_config_value("MODEL_DEVICE_TEST", yaml_path="model.device", default="cpu")
    print(f"Test 2 (YAML): Expected 'cuda' (if in config.yaml), Got '{val}'")
    # No assert here as it depends on actual config.yaml content

    # Test 3: Default value (env var and YAML path don't exist)
    val = get_config_value("NON_EXISTENT_VAR", yaml_path="non.existent.path", default="default_value_for_non_existent")
    print(f"Test 3 (default): Expected 'default_value_for_non_existent', Got '{val}'")
    assert val == "default_value_for_non_existent"

    # Test 4: Env var overrides YAML
    os.environ["MODEL_DEVICE_TEST_OVERRIDE"] = "env_override_cpu"
    # Assuming model.device = "cuda" in YAML
    val = get_config_value("MODEL_DEVICE_TEST_OVERRIDE", yaml_path="model.device", default="default_dev")
    print(f"Test 4 (env overrides YAML): Expected 'env_override_cpu', Got '{val}'")
    assert val == "env_override_cpu"
    del os.environ["MODEL_DEVICE_TEST_OVERRIDE"]

    # Test 5: YAML path not found, use default
    val = get_config_value("ANOTHER_VAR", yaml_path="this.path.surely.does.not.exist", default="default_for_missing_yaml_path")
    print(f"Test 5 (YAML path not found): Expected 'default_for_missing_yaml_path', Got '{val}'")
    assert val == "default_for_missing_yaml_path"

    # Test 6: No YAML path provided, only env var or default
    os.environ["NO_YAML_PATH_TEST"] = "env_val_no_yaml"
    val = get_config_value("NO_YAML_PATH_TEST", default="default_no_yaml")
    print(f"Test 6 (no YAML path, env): Expected 'env_val_no_yaml', Got '{val}'")
    assert val == "env_val_no_yaml"
    del os.environ["NO_YAML_PATH_TEST"]

    val = get_config_value("NO_YAML_PATH_TEST_DEFAULT", default="default_no_yaml_2")
    print(f"Test 6 (no YAML path, default): Expected 'default_no_yaml_2', Got '{val}'")
    assert val == "default_no_yaml_2"

    print("Config loader tests completed.")

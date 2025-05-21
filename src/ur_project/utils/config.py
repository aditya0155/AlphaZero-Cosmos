# src/ur_project/utils/config.py

import yaml
import os
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "base_config.yaml")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (Optional[str]): Path to the YAML configuration file. 
                                     If None, tries to load from DEFAULT_CONFIG_PATH.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration.
    """
    path_to_load = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    
    if not os.path.exists(path_to_load):
        print(f"Warning: Config file not found at {path_to_load}. Returning empty config.")
        return {}

    try:
        with open(path_to_load, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None: # Handle empty YAML file
            print(f"Warning: Config file {path_to_load} is empty. Returning empty config.")
            return {}
        return config_data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file {path_to_load}: {e}")
        # Potentially raise the error or return a default/empty config
        raise # Or return {} if a fallback is preferred
    except Exception as e:
        print(f"An unexpected error occurred while loading config {path_to_load}: {e}")
        raise # Or return {}

def get_config_value(config: Dict[str, Any], key_path: str, default: Optional[Any] = None) -> Any:
    """
    Retrieves a value from a nested dictionary using a dot-separated key path.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        key_path (str): Dot-separated path to the desired key (e.g., "llm.model.name").
        default (Optional[Any]): Default value to return if the key is not found.

    Returns:
        Any: The value found at the key path, or the default value.
    """
    keys = key_path.split('.')
    current_level = config
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return default
    return current_level

# Example Usage (for illustration):
# if __name__ == '__main__':
#     # Create a dummy base_config.yaml in a temporary configs directory for testing
#     dummy_config_dir = os.path.join(os.path.dirname(__file__), "..", "configs_temp")
#     dummy_config_file = os.path.join(dummy_config_dir, "test_config.yaml")
#     os.makedirs(dummy_config_dir, exist_ok=True)
#     with open(dummy_config_file, "w") as f:
#         yaml.dump({
#             "project_name": "UniversalReasoner",
#             "version": "0.1.0",
#             "llm": {
#                 "model_path": "google/gemma-2b-it",
#                 "parameters": {
#                     "temperature": 0.7,
#                     "max_new_tokens": 512
#                 }
#             },
#             "az_loop": {
#                 "num_steps": 100
#             }
#         }, f)

#     print(f"--- Testing load_config with {dummy_config_file} ---")
#     cfg = load_config(dummy_config_file)
#     print(f"Loaded config: {cfg}")

#     print("\n--- Testing get_config_value ---")
#     print(f"Project Name: {get_config_value(cfg, 'project_name', 'DefaultProject')}")
#     print(f"LLM Temp: {get_config_value(cfg, 'llm.parameters.temperature', 0.5)}")
#     print(f"LLM Top_K (missing): {get_config_value(cfg, 'llm.parameters.top_k', 50)}")
#     print(f"Nonexistent Key: {get_config_value(cfg, 'foo.bar.baz')}")
#     print(f"Nonexistent Key with default: {get_config_value(cfg, 'foo.bar.baz', 'default_baz')}")

#     # Clean up
#     os.remove(dummy_config_file)
#     os.rmdir(dummy_config_dir)

#     # Test with default path (will likely warn or error if base_config.yaml doesn't exist or is empty)
#     print("\n--- Testing load_config with default path (may warn/error) ---")
#     try:
#        default_cfg = load_config()
#        print(f"Default config loaded: {default_cfg}")
#     except Exception as e:
#        print(f"Error loading default config: {e}") 
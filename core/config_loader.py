"""
Configuration loader module for the adaptive trading bot.

This module provides functionality to load configuration from YAML files
with proper error handling and validation.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Union


def load_config(path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to the configuration file. Can be a string or Path object.
              Defaults to "config/config.yaml".
    
    Returns:
        Dictionary containing the configuration data.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid or cannot be parsed.
        Exception: For other unexpected errors during file reading.
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path.absolute()}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path.absolute()}")
            
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format in {config_path.absolute()}: {e}")
    except Exception as e:
        raise Exception(f"Error reading configuration file {config_path.absolute()}: {e}")
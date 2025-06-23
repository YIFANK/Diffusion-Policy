"""Configuration management for diffusion policy."""

import os
import yaml
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

def load_config(config_path: str = None) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
        
    Returns:
        OmegaConf configuration object.
    """
    if config_path is None:
        # Load default config from package
        config_dir = os.path.dirname(__file__)
        config_path = os.path.join(config_dir, "config.yaml")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)

def get_default_config() -> DictConfig:
    """Get the default configuration."""
    return load_config()

__all__ = ["load_config", "get_default_config"] 
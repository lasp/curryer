"""Mission-agnostic configuration utilities for Monte Carlo geolocation analysis.

This module provides general-purpose functions for reading and validating
configuration files. It contains NO mission-specific logic.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_config_schema() -> dict[str, Any]:
    """
    Return the expected configuration schema.

    Returns
    -------
    dict[str, Any]
        Dictionary describing required and optional config sections.
    """

    return {
        "mission_config": {
            "required": ["mission_name", "kernel_mappings"],
            "optional": ["instrument_name"],
            "kernel_mappings": {
                "constant_kernel": "Dict[str, str] - Frame names to kernel files",
                "offset_kernel": "Dict[str, str] - Sensor names to kernel files",
            },
        },
        "monte_carlo": {"required": ["parameters"], "optional": ["seed", "n_iterations"]},
        "geolocation": {
            "required": ["meta_kernel_file", "generic_kernel_dir", "instrument_name", "time_field"],
            "optional": ["dynamic_kernels", "minimum_correlation"],
        },
    }


def validate_config_file(config_path: Path) -> bool:
    """Validate that a config file exists and is valid JSON.

    Args:
        config_path: Path to configuration file

    Returns:
        True if valid

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


def extract_mission_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Extract mission configuration from config dictionary.

    Args:
        config_data: Full configuration dictionary from JSON

    Returns:
        mission_config section

    Raises:
        KeyError: If mission_config section is missing
    """
    if "mission_config" not in config_data:
        raise KeyError("Missing required 'mission_config' section in config file")

    mission_config = config_data["mission_config"]

    if "mission_name" not in mission_config:
        raise KeyError("Missing required 'mission_name' in mission_config")

    if "kernel_mappings" not in mission_config:
        raise KeyError("Missing required 'kernel_mappings' in mission_config")

    logger.info(f"Loaded mission config for: {mission_config.get('mission_name', 'UNKNOWN')}")
    return mission_config


def get_kernel_mapping(config_data: dict[str, Any], kernel_type: str) -> dict[str, str]:
    """Get kernel mappings for a specific kernel type.

    Args:
        config_data: Full configuration dictionary
        kernel_type: Type of kernel ('constant_kernel' or 'offset_kernel')

    Returns:
        Dict mapping names to kernel files (e.g., {'hysics': 'cprs_hysics_v01.attitude.ck.json'})
    """
    mission_config = config_data.get("mission_config", {})
    kernel_mappings = mission_config.get("kernel_mappings", {})
    return kernel_mappings.get(kernel_type, {})


def find_kernel_file(name: str, kernel_mapping: dict[str, str]) -> str | None:
    """Find kernel file for a given name using substring matching.

    Performs case-insensitive matching against kernel mapping keys.

    Args:
        name: Parameter or sensor name to match
        kernel_mapping: Dict of key patterns to kernel files

    Returns:
        Kernel file name if found, None otherwise

    Example:
        >>> mapping = {'hysics': 'cprs_hysics_v01.attitude.ck.json'}
        >>> find_kernel_file('hysics_roll', mapping)
        'cprs_hysics_v01.attitude.ck.json'
    """
    name_lower = name.lower()
    for key, kernel_file in kernel_mapping.items():
        if key.lower() in name_lower:
            logger.debug(f"Found kernel mapping for '{name}': {key} → {kernel_file}")
            return kernel_file

    return None

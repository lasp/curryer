"""Kernel utilities for path handling and temp directory management.

This module provides utility functions used across the kernels package,
particularly for handling SPICE's 80-character path limitation.
"""

import copy
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def get_short_temp_dir() -> Path:
    """Get a short base directory for temporary files.

    Returns a platform-appropriate short path that respects
    environment variables for customization. This avoids SPICE's
    80-character path limit by using short base directories.

    Returns
    -------
    Path
        A short base path for temporary directories.

    Raises
    ------
    ValueError
        If CURRYER_TEMP_DIR is set to an invalid path (too long).

    Notes
    -----
    Priority order:
    1. CURRYER_TEMP_DIR environment variable (if set and validated)
    2. /tmp on Unix (bypasses macOS TMPDIR which can be longer by default)
    3. C:\\Temp on Windows (shorter than default AppData path)
    4. Python's tempfile.gettempdir() (fallback, could be 40 characters)

    Why not just use tempfile.gettempdir()?
    - On macOS, TMPDIR is set to /var/folders/.../T (~49 chars)
    - This leaves only ~30 chars for filenames
    - We prefer /tmp (4 chars) to maximize filename space (75 chars)
    - Users can still override with CURRYER_TEMP_DIR if needed

    Examples
    --------
    >>> import os
    >>> os.environ["CURRYER_TEMP_DIR"] = "/custom/path"
    >>> get_short_temp_dir()
    PosixPath('/custom/path')
    """
    # Allow user override via environment variable with validation
    if "CURRYER_TEMP_DIR" in os.environ:
        custom_path = Path(os.environ["CURRYER_TEMP_DIR"])

        # Validate the path isn't too long (defeats the purpose)
        if len(str(custom_path)) > 50:
            raise ValueError(
                f"CURRYER_TEMP_DIR too long ({len(str(custom_path))} chars). Must be ≤50 characters: {custom_path}"
            )

        custom_path.mkdir(parents=True, exist_ok=True)
        return custom_path

    # Try to use the shortest available temp directory
    # This bypasses macOS TMPDIR env var which points to long user-specific paths

    if os.name != "nt":
        # Unix/Linux/macOS: Try /tmp first (only 4 chars, leaves 75 chars for filename)
        # Note: macOS sets TMPDIR=/var/folders/.../T leaving only ~30 chars for filenames
        tmp_path = Path("/tmp")  # noqa: S108
        if tmp_path.exists() and os.access(str(tmp_path), os.W_OK):
            return tmp_path
    else:
        # Windows: Try C:\Temp first (7 chars, leaves 72 chars for filename)
        c_temp = Path("C:\\Temp")
        try:
            c_temp.mkdir(parents=True, exist_ok=True)
            if os.access(str(c_temp), os.W_OK):
                return c_temp
        except (OSError, PermissionError):
            pass  # Fall back to system temp

    # Fallback: Use Python's standard temp directory (cross-platform)
    # This respects TMPDIR, TEMP, TMP env vars and platform defaults
    temp_dir = Path(tempfile.gettempdir())

    # Warn if system temp directory is too long (leaves less room for filenames)
    if len(str(temp_dir)) > 35:
        logger.warning(
            f"System temp directory is long ({len(str(temp_dir))} chars): {temp_dir}. "
            f"This leaves only ~{80 - len(str(temp_dir)) - 1} chars for filenames. "
            f"Consider setting CURRYER_TEMP_DIR=/tmp for shorter paths."
        )

    return temp_dir


def create_short_symlink(source_path: Path, temp_dir: Path) -> Path | None:
    """Create a symlink in a short temp directory.

    Uses a simple naming scheme since symlinks are cheap to create/replace.
    If a symlink already exists, it's replaced (idempotent operation).

    Parameters
    ----------
    source_path : Path
        Original file path
    temp_dir : Path
        Short base directory for symlink (e.g., /tmp)

    Returns
    -------
    Path | None
        Path to symlink if successful, None if creation failed

    Notes
    -----
    - Works on Linux/macOS by default
    - May fail on Windows or restricted environments
    - Failures are logged but don't raise exceptions
    - Idempotent: reusing the same symlink name is safe
    """
    try:
        # Simple unique name: prefix + original name
        # No hash needed - symlinks are free to create/overwrite
        symlink_name = f"curryer_{source_path.name}"
        symlink_path = temp_dir / symlink_name

        # Remove existing symlink/file if present (idempotent)
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        os.symlink(source_path, symlink_path)
        logger.debug(f"Created symlink: {symlink_path}")
        return symlink_path

    except OSError as e:
        logger.debug(f"Symlink creation failed: {e}")
        return None


def get_path_strategy_config() -> dict:
    """Read path shortening configuration from environment variables.

    Supported variables:
    - CURRYER_DISABLE_COPY: "true" to disable file copying (default: "false")
    - CURRYER_TEMP_DIR: Custom short temp directory (handled by get_short_temp_dir())

    Returns
    -------
    dict
        Configuration with keys: disable_copy, try_symlink, try_copy

    Examples
    --------
    >>> import os
    >>> os.environ["CURRYER_DISABLE_COPY"] = "true"
    >>> config = get_path_strategy_config()
    >>> config["try_copy"]
    False
    """
    disable_copy = os.getenv("CURRYER_DISABLE_COPY", "false").lower() == "true"

    return {
        "disable_copy": disable_copy,
        "try_symlink": True,  # Always try symlink first (zero cost)
        "try_copy": not disable_copy,  # Copy fallback unless disabled
    }


def _is_file_property(key: str) -> bool:
    """Check if a property key likely contains file paths.

    Parameters
    ----------
    key : str
        Property key name

    Returns
    -------
    bool
        True if the property likely contains file paths

    Notes
    -----
    Two matching strategies (either one triggers processing):
    1. Regex: Matches keys ending in _FILE or _FILE_NAME (e.g., LEAPSECONDS_FILE, INPUT_DATA_FILE)
       This handles most SPICE tool config files which follow this naming convention
    2. Explicit list: Known meta-kernel properties that don't follow the _FILE convention
       (clock_kernel, frame_kernel, etc.) - these are standard SPICE meta-kernel keys
    This dual approach covers both user configs and meta-kernel properties
    """
    return re.search(r"_FILE(?:_NAME|)$", key) is not None or key in [
        "clock_kernel",
        "frame_kernel",
        "leapsecond_kernel",
        "meta_kernel",
        "planet_kernels",
    ]


def _convert_paths_to_strings(obj):
    """Recursively convert all Path objects to strings in a data structure.

    Parameters
    ----------
    obj : any
        Object to convert (dict, list, Path, or other)

    Returns
    -------
    any
        Object with all Path instances converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    else:
        return obj


def copy_to_short_path(source_path: Path, temp_dir: Path, max_len: int) -> Path | None:
    """Copy file to temp directory with short path.

    Returns Path object for consistency with create_short_symlink().

    Parameters
    ----------
    source_path : Path
        Original file path
    temp_dir : Path
        Temporary directory base
    max_len : int
        Maximum path length (default: 80 for SPICE)

    Returns
    -------
    Path | None
        Path to copied file if successful, None if copy failed

    Notes
    -----
    Uses tempfile.mkstemp() for unique naming. Warns if temp directory
    base path is too long.
    """
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=source_path.suffix, prefix="curryer_", dir=str(temp_dir))

        if len(temp_path) > max_len:
            logger.warning(
                f"Temp directory base path too long for SPICE ({len(str(temp_dir))} chars). "
                f"Consider setting CURRYER_TEMP_DIR to a shorter path."
            )
            os.close(temp_fd)
            os.remove(temp_path)
            return None

        # Copy file
        with os.fdopen(temp_fd, "wb") as dst:
            with open(source_path, "rb") as src:
                shutil.copyfileobj(src, dst)

        shutil.copystat(source_path, temp_path)
        logger.debug(f"Copied to short path: {temp_path}")
        return Path(temp_path)

    except OSError as e:
        logger.warning(f"Copy failed: {e}")
        return None


# pylint: disable=too-many-branches,too-many-nested-blocks
def update_invalid_paths(
    configs,
    max_len=80,
    try_copy=True,
    parent_dir=None,
    temp_dir=None,
):
    """Update invalid paths (too long) by creating symlinks or copying to short temp paths.

    Attempts to fix paths that exceed the maximum length by trying strategies in order:
    1. Symlink to temp directory with short path (always tried first - zero cost)
    2. Copy file to temp directory with short path (fallback if symlink fails)

    Parameters
    ----------
    configs : dict
        Configuration dictionary to update
    max_len : int
        Maximum path length (default: 80 for SPICE string values)
    try_copy : bool
        Try to copy file to temp directory with short path (default: True)
    parent_dir : Path
        Parent directory for resolving relative paths
    temp_dir : Path
        Base directory for temp copies (default: uses tempfile.gettempdir())

    Returns
    -------
    tuple
        (updated_config_dict, list_of_temp_files)
        - updated_config_dict: Configuration with shortened paths
        - list_of_temp_files: List of temp file paths created (for cleanup)

    See Also
    --------
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html
    """
    # Read environment variable configuration
    env_config = get_path_strategy_config()

    # Override parameters based on env vars
    if env_config["disable_copy"]:
        try_copy = False

    if parent_dir is not None:
        parent_dir = Path(parent_dir)
        if parent_dir.is_file():
            parent_dir = parent_dir.parent

    # Determine temp directory base - use helper for consistency
    if temp_dir is None:
        temp_dir = get_short_temp_dir()

    # Track temporary files created for cleanup
    temp_files_created = []

    # Use deepcopy to avoid mutating the caller's input dictionary
    updated_configs = copy.deepcopy(configs)

    # Check if we need to work on nested 'properties' dict
    if "properties" in updated_configs and isinstance(updated_configs["properties"], dict):
        # Work on the properties dict
        properties = updated_configs["properties"]
    else:
        # Work on the top-level config
        properties = updated_configs

    for key, value in properties.items():
        # Skip non-string/Path/list values (like integers, bools, etc.)
        if not isinstance(value, str | Path | list):
            continue

        # Check if this property likely contains file paths that need shortening
        is_file_property = _is_file_property(key)

        # Track if original value was a list or single string
        was_list = isinstance(value, list)

        if isinstance(value, str | Path):
            value = [value]

        new_vals = []
        modified_value = False
        for item in value:
            if not isinstance(item, str | Path):
                new_vals.append(item)
                continue

            # Only apply path shortening strategies to file properties
            # Skip early to avoid unnecessary Path processing for non-file properties
            if not is_file_property:
                # Still need to convert Path to string for template compatibility
                if isinstance(item, Path):
                    new_vals.append(str(item))
                    modified_value = True
                else:
                    new_vals.append(item)
                continue

            modified_item = False

            # Work with Path object directly
            if isinstance(item, Path):
                fn = item
            else:
                fn = Path(item)

            if parent_dir and not (fn.is_file() or fn.is_dir()) and not fn.is_absolute():
                abs_fn = parent_dir / fn
                if abs_fn.is_file() or abs_fn.is_dir():
                    fn = abs_fn.absolute().resolve()
                    item = fn
                    modified_item = True

            if (fn.is_file() or fn.is_dir()) and len(str(item)) > max_len:
                # Log that we detected a long path that needs shortening
                logger.info(f"Path exceeds {max_len} chars ({len(str(item))} chars): {fn.name}")

                # Strategy 1: Symlink (always try first - zero cost)
                symlink_path = create_short_symlink(fn, temp_dir)
                if symlink_path and len(str(symlink_path)) <= max_len:
                    item = str(symlink_path)  # Convert Path to str for config dict
                    temp_files_created.append(str(symlink_path))
                    modified_item = True
                    logger.info(f"  → Using symlink: {symlink_path}")

                # Strategy 2: Copy (if symlink failed and enabled)
                elif try_copy:
                    temp_path = copy_to_short_path(fn, temp_dir, max_len)
                    if temp_path:
                        item = str(temp_path)  # Convert Path to str for config dict
                        temp_files_created.append(str(temp_path))
                        modified_item = True
                        logger.info(f"  → Using copy: {temp_path}")
                    else:
                        logger.warning(f"  ✗ Failed to shorten path: {fn.name} ({len(str(fn))} chars)")

                # If both strategies failed
                else:
                    logger.warning(f"  ✗ Failed to shorten path: {fn.name} ({len(str(fn))} chars)")

                new_vals.append(item)

                modified_value |= modified_item
            else:
                # Path doesn't need shortening, add it as-is
                # Ensure item is always a string (could be Path)
                if isinstance(item, Path):
                    item = str(item)
                    modified_item = True

                new_vals.append(item)

                modified_value |= modified_item

        if modified_value:
            # Unwrap if original was a single string/path
            if not was_list and len(new_vals) == 1:
                properties[key] = new_vals[0]
            elif not was_list and len(new_vals) == 0:
                # All items were filtered out - preserve empty list or skip
                # This shouldn't normally happen but prevents IndexError
                logger.warning(f"All items filtered from property '{key}', setting to empty list")
                properties[key] = []
            else:
                properties[key] = new_vals

    # If we worked on nested properties, put them back in updated_configs
    if "properties" in configs and isinstance(configs["properties"], dict):
        updated_configs["properties"] = properties

    return updated_configs, temp_files_created

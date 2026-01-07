"""Kernel utilities for path handling and temp directory management.

This module provides utility functions used across the kernels package,
particularly for handling SPICE's 80-character path limitation.
"""

import hashlib
import logging
import os
import platform
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
        If CURRYER_TEMP_DIR is set to an invalid path (too long, not writable,
        or pointing to a sensitive system directory).

    Notes
    -----
    Priority order:
    1. CURRYER_TEMP_DIR environment variable (if set and validated)
    2. Platform-specific short defaults (/tmp on Unix, C:/Temp on Windows)

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
                f"CURRYER_TEMP_DIR path is too long ({len(str(custom_path))} chars): {custom_path}. "
                "Must be ≤50 characters to ensure temp file paths stay under SPICE's 80-char limit."
            )

        # Validate it's not pointing to sensitive system directories (or subdirectories)
        # Exclude /tmp and /var/tmp since those are meant for temporary files
        sensitive_dirs = ["/bin", "/boot", "/dev", "/etc", "/lib", "/proc", "/root", "/sbin", "/sys", "/usr"]
        if platform.system() == "Windows":
            sensitive_dirs = ["C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"]

        resolved_path = custom_path.resolve(strict=False)
        for sensitive in sensitive_dirs:
            sensitive_path = Path(sensitive).resolve(strict=False)
            # Check if custom path is under a sensitive directory
            try:
                resolved_path.relative_to(sensitive_path)
            except ValueError:
                # relative_to raises ValueError if paths are unrelated - this is what we want
                # Not under this sensitive directory; check the next one
                continue
            else:
                # If we get here, resolved_path is under sensitive_path
                raise ValueError(
                    f"CURRYER_TEMP_DIR cannot point to or be under sensitive system directory. "
                    f"Attempted: {custom_path}, resolves to: {resolved_path}, sensitive parent: {sensitive_path}"
                )
        # Try to create and verify it's writable
        try:
            custom_path.mkdir(parents=True, exist_ok=True)
            # Test writability with a temporary file
            test_file = custom_path / ".curryer_write_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise ValueError(f"CURRYER_TEMP_DIR is not writable: {custom_path}. Error: {e}") from e

        return custom_path

    # Platform-specific short defaults
    system = platform.system()

    if system == "Windows":
        # Use C:\Temp on Windows
        # Path constructor handles platform-specific separators
        short_base = Path("C:", "Temp")
    else:
        # Unix-like systems use /tmp directly
        short_base = Path("/tmp")  # noqa: S108

    # Create if it doesn't exist
    short_base.mkdir(parents=True, exist_ok=True)

    return short_base


def create_short_symlink(source_path: Path, temp_dir: Path) -> Path | None:
    """
    Attempt to create a symlink in a short temp directory.

    This is the preferred strategy because it requires no file copying
    (zero storage overhead, zero I/O).

    Parameters
    ----------
    source_path : Path
        Original file path that's too long
    temp_dir : Path
        Short base directory for symlink (e.g., /tmp/spice)

    Returns
    -------
    Path | None
        Path to symlink if successful, None if creation failed

    Notes
    -----
    - Works on Linux/macOS by default
    - May fail on Windows (requires admin/developer mode)
    - May fail in restricted containers (seccomp policies)
    - Failures should be logged but not raise exceptions
    """
    try:
        # Generate unique short filename using hash
        path_hash = hashlib.md5(str(source_path).encode()).hexdigest()[:8]  # noqa: S324
        symlink_name = f"{source_path.stem[:10]}_{path_hash}{source_path.suffix}"
        symlink_path = temp_dir / symlink_name

        # Create symlink
        os.symlink(source_path, symlink_path)

        return symlink_path
    except OSError as e:
        # Expected failures: Windows permissions, container restrictions
        logger.debug(f"Symlink creation failed: {e}")
        return None


def get_path_strategy_config() -> dict:
    """
    Read path shortening configuration from environment variables.

    Supported variables:
    - CURRYER_PATH_STRATEGY: Comma-separated priority list (default: "symlink,wrap,relative,copy")
    - CURRYER_DISABLE_SYMLINKS: "true" to disable symlinks (default: "false")
    - CURRYER_TEMP_DIR: Custom short temp directory (already implemented)
    - CURRYER_WARN_ON_COPY: "true" to warn on large file copies (default: "true")
    - CURRYER_WARN_COPY_THRESHOLD: File size in MB to trigger warning (default: "10")

    Returns
    -------
    dict
        Configuration with keys: strategy_order, disable_symlinks, temp_dir, warn_on_copy, warn_copy_threshold_mb

    Examples
    --------
    >>> import os
    >>> os.environ["CURRYER_WARN_ON_COPY"] = "true"
    >>> os.environ["CURRYER_WARN_COPY_THRESHOLD"] = "50"  # Warn on files > 50 MB
    >>> config = get_path_strategy_config()
    >>> config["warn_on_copy"]
    True
    >>> config["warn_copy_threshold_mb"]
    50
    """
    strategy_str = os.getenv("CURRYER_PATH_STRATEGY", "symlink,wrap,relative,copy")
    strategy_order = [s.strip() for s in strategy_str.split(",")]

    disable_symlinks = os.getenv("CURRYER_DISABLE_SYMLINKS", "false").lower() == "true"
    warn_on_copy = os.getenv("CURRYER_WARN_ON_COPY", "true").lower() == "true"
    temp_dir = os.getenv("CURRYER_TEMP_DIR", None)

    # Parse threshold from env var, default to 10 MB
    warn_copy_threshold_str = os.getenv("CURRYER_WARN_COPY_THRESHOLD", "10")
    try:
        warn_copy_threshold_mb = int(warn_copy_threshold_str)
    except ValueError:
        logger.warning(
            f"Invalid CURRYER_WARN_COPY_THRESHOLD value '{warn_copy_threshold_str}', using default of 10 MB"
        )
        warn_copy_threshold_mb = 10

    return {
        "strategy_order": strategy_order,
        "disable_symlinks": disable_symlinks,
        "temp_dir": temp_dir,
        "warn_on_copy": warn_on_copy,
        "warn_copy_threshold_mb": warn_copy_threshold_mb,
    }

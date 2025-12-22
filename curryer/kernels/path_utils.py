"""Kernel utilities for path handling and temp directory management.

This module provides utility functions used across the kernels package,
particularly for handling SPICE's 80-character path limitation.
"""

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
            sensitive_dirs = ["C:\\", "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"]

        resolved_path = custom_path.resolve()
        for sensitive in sensitive_dirs:
            sensitive_path = Path(sensitive).resolve()
            # Check if custom path is under a sensitive directory
            try:
                resolved_path.relative_to(sensitive_path)
                # If we get here, resolved_path is under sensitive_path
                raise ValueError(
                    f"CURRYER_TEMP_DIR cannot point to or be under sensitive system directory. "
                    f"Attempted: {custom_path}, resolves to: {resolved_path}, sensitive parent: {sensitive_path}"
                )
            except ValueError as e:
                # relative_to raises ValueError if paths are unrelated - this is what we want
                if "cannot point to" in str(e):
                    raise  # Re-raise our custom error
                # Otherwise it's the "not relative" ValueError - continue checking other sensitive dirs
                continue

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
        # Use C:\Temp instead of the deep AppData path
        # Path constructor handles platform-specific separators
        short_base = Path("C:", "Temp")
    else:
        # Unix-like systems use /tmp directly
        short_base = Path("/tmp")  # noqa: S108

    # Create if it doesn't exist
    short_base.mkdir(parents=True, exist_ok=True)

    return short_base

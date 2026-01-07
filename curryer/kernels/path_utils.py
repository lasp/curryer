"""Kernel utilities for path handling and temp directory management.

This module provides utility functions used across the kernels package,
particularly for handling SPICE's 80-character path limitation.

This module is standalone and has no dependencies on other kernel modules,
making it reusable and testable in isolation.
"""

import copy
import hashlib
import logging
import os
import platform
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
    # Generate base short filename using hash
    path_hash = hashlib.md5(str(source_path).encode()).hexdigest()[:8]  # noqa: S324
    stem_part = source_path.stem[:10]
    suffix = source_path.suffix

    # Resolve source to an absolute, normalized path for comparison
    source_real = os.path.realpath(str(source_path))

    last_error: OSError | None = None

    # Try primary name first, then fall back to suffixed variants on collision
    for i in range(10):
        if i == 0:
            symlink_name = f"{stem_part}_{path_hash}{suffix}"
        else:
            symlink_name = f"{stem_part}_{path_hash}_{i}{suffix}"

        symlink_path = temp_dir / symlink_name

        # If something already exists at this path, see if we can reuse it
        if symlink_path.exists() or symlink_path.is_symlink():
            if symlink_path.is_symlink():
                try:
                    target = os.readlink(str(symlink_path))
                    # Resolve relative symlink targets relative to symlink's directory
                    target_path = Path(target)
                    if not target_path.is_absolute():
                        target_path = (symlink_path.parent / target_path)
                    target_real = os.path.realpath(str(target_path))
                except OSError as e:
                    # Can't read target; treat as collision and try another name
                    last_error = e
                    continue

                # Reuse existing symlink if it already points to the same source
                if target_real == source_real:
                    return symlink_path

            # Existing path points elsewhere (or is not a symlink): try another name
            continue

        # No existing entry: try to create the symlink
        try:
            os.symlink(source_path, symlink_path)
            return symlink_path
        except OSError as e:
            # Record the last error and break if it's not a simple collision
            last_error = e
            # For permissions or platform issues, further attempts are unlikely to succeed
            break

    if last_error is not None:
        # Expected failures: Windows permissions, container restrictions, etc.
        logger.debug(f"Symlink creation failed: {last_error}")
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
        logger.warning(f"Invalid CURRYER_WARN_COPY_THRESHOLD value '{warn_copy_threshold_str}', using default of 10 MB")
        warn_copy_threshold_mb = 10

    return {
        "strategy_order": strategy_order,
        "disable_symlinks": disable_symlinks,
        "temp_dir": temp_dir,
        "warn_on_copy": warn_on_copy,
        "warn_copy_threshold_mb": warn_copy_threshold_mb,
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


def attempt_symlink_strategy(fn, item, temp_dir, max_len, modified_item):
    """Attempt to create a symlink in a short temp directory.

    Parameters
    ----------
    fn : Path
        Original file path
    item : str or Path
        Current item value
    temp_dir : Path
        Temporary directory base
    max_len : int
        Maximum path length
    modified_item : bool
        Whether item has been modified

    Returns
    -------
    tuple
        (success: bool, new_item: str|Path, new_modified_item: bool, fn_section: str|None)
    """
    if not fn.is_file() or modified_item:
        return False, item, modified_item, None

    logger.info("  Attempting symlink strategy...")
    symlink_path = create_short_symlink(fn, temp_dir)

    if symlink_path and len(str(symlink_path)) <= max_len:
        logger.info(f"  ✓ Created symlink: {symlink_path} ({len(str(symlink_path))} chars)")
        return True, symlink_path, True, None
    else:
        logger.warning("  ✗ Symlink creation failed or path still too long")
        return False, item, modified_item, None


def attempt_wrap_strategy(fn, item, max_len, wrap_char, modified_item):
    """Attempt to wrap path across multiple lines using continuation character.

    Parameters
    ----------
    fn : Path
        Original file path
    item : str or Path
        Current item value
    max_len : int
        Maximum path length
    wrap_char : str
        Continuation character
    modified_item : bool
        Whether item has been modified

    Returns
    -------
    tuple
        (success: bool, new_item: str|Path, new_modified_item: bool, fn_section: str|None, wrapped_lines: list)
    """
    if modified_item:
        return False, item, modified_item, None, []

    logger.info("  Attempting continuation character (+) wrapping...")

    # Split path into components
    parts = fn.parts

    # Validate: each component must be short enough when wrapped.
    # We subtract len(wrap_char) for the continuation marker and 1 for the path separator.
    # Can't split in middle of directory names.
    max_segment_len = max_len - len(wrap_char) - 1
    valid_wrap = True
    for part in parts:
        if len(part) > max_segment_len:
            logger.warning(
                f"  ✗ Wrapping not possible: segment exceeds {max_segment_len} chars: {part}"
            )
            valid_wrap = False
            break

    if not valid_wrap:
        return False, item, modified_item, None, []

    # Build wrapped string
    logger.debug("Wrapping path [%s]", fn)
    fn_section = None
    wrapped_lines = []

    line_count = 0
    for fn_part in parts:
        if fn_section is None:
            fn_section = fn_part
            continue

        next_section = str(Path(fn_section) / fn_part)
        if len(next_section) >= max_len:
            fn_section += wrap_char
            wrapped_lines.append(fn_section)
            line_count += 1
            fn_section = os.path.sep + fn_part
        else:
            fn_section = next_section

    if line_count > 0:
        logger.info(f"  ✓ Wrapped path across {line_count + 1} lines")
        return True, item, True, fn_section, wrapped_lines
    else:
        # Wrapping didn't actually split the path
        return False, item, modified_item, None, []


def attempt_relative_strategy(fn, item, max_len, relative_dir, modified_item):
    """Attempt to use relative path if shorter than absolute.

    Parameters
    ----------
    fn : Path
        Original file path
    item : str or Path
        Current item value
    max_len : int
        Maximum path length
    relative_dir : Path
        Base directory for relative paths
    modified_item : bool
        Whether item has been modified

    Returns
    -------
    tuple
        (success: bool, new_item: str|Path, new_modified_item: bool, fn_section: str|None)
    """
    if modified_item:
        return False, item, modified_item, None

    logger.info("  Attempting relative path optimization...")
    rel_fn = os.path.relpath(fn, start=relative_dir)
    if len(rel_fn) <= max_len:
        logger.info(f"  ✓ Using relative path ({len(rel_fn)} chars)")
        return True, item, True, rel_fn
    else:
        logger.debug(f"  ✗ Relative path still too long ({len(rel_fn)} chars)")
        return False, item, modified_item, None


def attempt_copy_strategy(fn, item, temp_dir, max_len, modified_item, warn_on_copy, warn_copy_threshold_mb):
    """Attempt to copy file to temp directory with short path.

    Parameters
    ----------
    fn : Path
        Original file path
    item : str or Path
        Current item value
    temp_dir : Path
        Temporary directory base
    max_len : int
        Maximum path length
    modified_item : bool
        Whether item has been modified
    warn_on_copy : bool
        Whether to warn on large file copies
    warn_copy_threshold_mb : int
        File size threshold for warnings (MB)

    Returns
    -------
    tuple
        (success: bool, new_item: str|Path, new_modified_item: bool, fn_section: str|None, temp_file_path: str|None)
    """
    if not fn.is_file() or modified_item:
        return False, item, modified_item, None, None

    logger.info("  Attempting file copy strategy...")
    temp_path = None
    try:
        # Use hash of full path to ensure uniqueness (not for security)
        path_hash = hashlib.md5(str(fn).encode()).hexdigest()[:6]  # noqa: S324
        prefix = f"{fn.stem[:10]}_{path_hash}_"

        # Pre-calculate expected temp path length to avoid unnecessary copy
        # mkstemp adds random chars (conservative estimate: 12 to handle platform variations)
        # Estimated: temp_dir + / + prefix + random_chars + suffix
        estimated_length = len(str(temp_dir)) + 1 + len(prefix) + 12 + len(fn.suffix)
        if estimated_length > max_len:
            logger.warning(
                f"  ✗ Temp path would be too long ({estimated_length} chars estimated), "
                "skipping copy strategy"
            )
            return False, item, modified_item, None, None
        else:
            # Check file size and warn if large (if configured)
            if warn_on_copy:
                try:
                    file_size_bytes = fn.stat().st_size
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    if file_size_mb > warn_copy_threshold_mb:
                        logger.warning(
                            f"  ⚠ Copying large file: {fn.name} ({file_size_mb:.1f} MB) - "
                            f"consider using symlinks or optimizing path structure"
                        )
                except OSError:
                    # If we can't get file size, continue without warning
                    pass

            temp_fd, temp_path = tempfile.mkstemp(suffix=fn.suffix, prefix=prefix, dir=str(temp_dir))

            # Verify actual path is short enough (should match estimate)
            if len(temp_path) <= max_len:
                # Keep fd open during copy to prevent race condition
                # Copy the file - if this fails, temp file will be cleaned up in except block
                try:
                    with os.fdopen(temp_fd, "wb") as temp_file:
                        with open(fn, "rb") as source_file:
                            shutil.copyfileobj(source_file, temp_file)
                    # Copy metadata after closing the file
                    shutil.copystat(fn, temp_path)
                except Exception:
                    # If copy fails, close fd if still open
                    try:
                        os.close(temp_fd)
                    except OSError:
                        pass
                    raise

                logger.info(f"  ✓ Copied to short path: {temp_path} ({len(temp_path)} chars)")
                logger.debug(f"   Successfully shortened from {len(str(fn))} to {len(temp_path)} chars")

                return True, temp_path, True, None, temp_path
            else:
                # Temp path still too long, clean up and try other strategies
                # Close the file descriptor first
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                actual_length = len(temp_path)
                os.remove(temp_path)
                temp_path = None  # Mark as cleaned up
                logger.warning(f"  ✗ Temp path still exceeds limit ({actual_length} chars)")
                return False, item, modified_item, None, None
    except (OSError, PermissionError, shutil.SameFileError, shutil.Error) as e:
        # Clean up temp file if it was created but copy/checks failed
        # Catches: OSError, PermissionError (file system issues)
        #          shutil.SameFileError and other shutil.Error subclasses (shutil-specific issues)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Best effort cleanup
        logger.warning(f"  ✗ Failed to copy to temp directory: {e}")
        return False, item, modified_item, None, None


def _try_strategies(
    fn,
    item,
    temp_dir,
    max_len,
    wrap_char,
    relative_dir,
    modified_item,
    try_symlink,
    try_wrap,
    try_relative,
    try_copy,
    strategy_order,
    use_custom_order,
    warn_on_copy,
    warn_copy_threshold_mb,
):
    """Try path shortening strategies in order until one succeeds.

    Parameters
    ----------
    fn : Path
        Original file path
    item : str or Path
        Current item value
    temp_dir : Path
        Temporary directory base
    max_len : int
        Maximum path length
    wrap_char : str
        Continuation character
    relative_dir : Path
        Base directory for relative paths
    modified_item : bool
        Whether item has been modified
    try_symlink : bool
        Whether to try symlink strategy
    try_wrap : bool
        Whether to try wrap strategy
    try_relative : bool
        Whether to try relative strategy
    try_copy : bool
        Whether to try copy strategy
    strategy_order : list
        Custom strategy order
    use_custom_order : bool
        Whether to use custom strategy order
    warn_on_copy : bool
        Whether to warn on large file copies
    warn_copy_threshold_mb : int
        File size threshold for warnings (MB)

    Returns
    -------
    tuple
        (item: str|Path, modified_item: bool, fn_section: str|None, temp_file_created: str|None, wrapped_lines: list)
    """
    fn_section = None
    temp_file_created = None
    wrapped_lines = []

    # Map strategy names to functions
    def try_symlink_func():
        nonlocal item, modified_item, fn_section, temp_file_created
        if not try_symlink:
            return False
        success, new_item, new_modified, _ = attempt_symlink_strategy(fn, item, temp_dir, max_len, modified_item)
        if success:
            item = new_item
            modified_item = new_modified
            temp_file_created = str(new_item)
            return True
        return False

    def try_wrap_func():
        nonlocal item, modified_item, fn_section, wrapped_lines
        if not try_wrap:
            return False
        success, new_item, new_modified, new_fn_section, new_wrapped_lines = attempt_wrap_strategy(
            fn, item, max_len, wrap_char, modified_item
        )
        if success:
            item = new_item
            modified_item = new_modified
            fn_section = new_fn_section
            wrapped_lines = new_wrapped_lines
            return True
        return False

    def try_relative_func():
        nonlocal item, modified_item, fn_section
        if not try_relative:
            return False
        success, new_item, new_modified, new_fn_section = attempt_relative_strategy(
            fn, item, max_len, relative_dir, modified_item
        )
        if success:
            item = new_item
            modified_item = new_modified
            fn_section = new_fn_section
            return True
        return False

    def try_copy_func():
        nonlocal item, modified_item, fn_section, temp_file_created
        if not try_copy:
            return False
        success, new_item, new_modified, _, new_temp_file = attempt_copy_strategy(
            fn, item, temp_dir, max_len, modified_item, warn_on_copy, warn_copy_threshold_mb
        )
        if success:
            item = new_item
            modified_item = new_modified
            temp_file_created = new_temp_file
            return True
        return False

    strategy_functions = {
        "symlink": try_symlink_func,
        "wrap": try_wrap_func,
        "relative": try_relative_func,
        "copy": try_copy_func,
    }

    # Execute strategies in order
    if use_custom_order:
        # Use custom strategy order from environment variable
        for strategy_name in strategy_order:
            if strategy_name in strategy_functions:
                if strategy_functions[strategy_name]():
                    # Strategy succeeded - break to prevent other strategies
                    break
            else:
                logger.warning(f"  Unknown strategy '{strategy_name}' in CURRYER_PATH_STRATEGY, skipping")
    else:
        # Use default order: symlink → wrap → relative → copy
        for strategy_func in [try_symlink_func, try_wrap_func, try_relative_func, try_copy_func]:
            if strategy_func():
                # Strategy succeeded - break to prevent other strategies
                break

    return item, modified_item, fn_section, temp_file_created, wrapped_lines


# pylint: disable=too-many-branches,too-many-nested-blocks
def update_invalid_paths(
    configs,
    max_len=80,
    try_symlink=True,
    try_relative=False,
    try_copy=True,
    try_wrap=True,
    wrap_char="+",
    relative_dir=None,
    parent_dir=None,
    temp_dir=None,
):
    """Update invalid paths (too long) by copying, relativizing, or wrapping.

    Attempts to fix paths that exceed the maximum length by trying strategies in order:
    1. Symlink to temp directory with short path (if try_symlink=True)
    2. Wrap path across multiple lines (if try_wrap=True)
    3. Relativize path (if try_relative=True)
    4. Copy file to temp directory with short path (if try_copy=True)

    Parameters
    ----------
    configs : dict
        Configuration dictionary to update
    max_len : int
        Maximum path length (default: 80 for SPICE string values)
    try_symlink : bool
        Try to create symlink in temp directory with short path (default: True)
    try_relative : bool
        Try to use relative paths if shorter (default: False)
    try_copy : bool
        Try to copy file to temp directory with short path (default: True)
    try_wrap : bool
        Try to wrap long paths across multiple lines (default: True)
    wrap_char : str
        Character for line continuation (default: "+")
    relative_dir : Path
        Base directory for relative paths
    parent_dir : Path
        Parent directory for resolving relative paths
    temp_dir : Path
        Base directory for temp copies (default: /tmp on Unix, C:/Temp on Windows)

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
    if env_config["disable_symlinks"]:
        try_symlink = False

    # Use custom strategy order if CURRYER_PATH_STRATEGY is set
    strategy_order = env_config["strategy_order"]
    use_custom_order = os.getenv("CURRYER_PATH_STRATEGY") is not None

    # Get warn_on_copy configuration
    warn_on_copy = env_config["warn_on_copy"]
    warn_copy_threshold_mb = env_config["warn_copy_threshold_mb"]

    relative_dir = Path.cwd() if relative_dir is None else Path(relative_dir)
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
            fn_section = None

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
                logger.debug(f"  Full path: {item}")

                # Try strategies until one succeeds
                item, modified_item, fn_section, temp_file_created, wrapped_lines = _try_strategies(
                    fn,
                    item,
                    temp_dir,
                    max_len,
                    wrap_char,
                    relative_dir,
                    modified_item,
                    try_symlink,
                    try_wrap,
                    try_relative,
                    try_copy,
                    strategy_order,
                    use_custom_order,
                    warn_on_copy,
                    warn_copy_threshold_mb,
                )

                # Track temp file if created
                if temp_file_created:
                    temp_files_created.append(temp_file_created)

                # Handle wrapped lines (add them to new_vals)
                if wrapped_lines:
                    new_vals.extend(wrapped_lines)
                    modified_value = True

                # Process the final item
                if fn_section is None:
                    item = fn if not modified_item else item
                else:
                    item = fn_section

                # Ensure item is always a string (could be Path if fn_section was None)
                if isinstance(item, Path):
                    item = str(item)
                    modified_item = True

                # Only append if not already added by wrapped lines
                if not wrapped_lines:
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

"""Kernel writer functions.

@author: Brandon Stone
"""

import copy
import hashlib
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import jinja2

from .path_utils import get_short_temp_dir

logger = logging.getLogger(__name__)

TEMPLATE_NAMES = {
    "spk": "mkspk_setup_template.txt",
    "ck": "msopck_setup_template.txt",
}

env = jinja2.Environment(
    loader=jinja2.PackageLoader(__package__, "templates"),
    keep_trailing_newline=True,
    autoescape=jinja2.select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
)


# pylint: disable=too-many-branches,too-many-nested-blocks
def update_invalid_paths(
    configs,
    max_len=80,
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
    1. Copy file to temp directory with short path (if try_copy=True)
    2. Relativize path (if try_relative=True)
    3. Wrap path across multiple lines (if try_wrap=True)

    Parameters
    ----------
    configs : dict
        Configuration dictionary to update
    max_len : int
        Maximum path length (default: 80 for SPICE string values)
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

    wrap_configs = copy.deepcopy(configs)

    # Check if we need to work on nested 'properties' dict
    if "properties" in wrap_configs and isinstance(wrap_configs["properties"], dict):
        # Work on the properties dict
        properties = wrap_configs["properties"]
    else:
        # Work on the top-level config
        properties = wrap_configs

    for key, value in properties.items():
        # Skip non-string/Path/list values (like integers, bools, etc.)
        if not isinstance(value, str | Path | list):
            continue

        # Check if this property likely contains file paths that need shortening
        # Two matching strategies (either one triggers processing):
        # 1. Regex: Matches keys ending in _FILE or _FILE_NAME (e.g., LEAPSECONDS_FILE, INPUT_DATA_FILE)
        #    This handles most SPICE tool config files which follow this naming convention
        # 2. Explicit list: Known meta-kernel properties that don't follow the _FILE convention
        #    (clock_kernel, frame_kernel, etc.) - these are standard SPICE meta-kernel keys
        # This dual approach covers both user configs and meta-kernel properties
        is_file_property = re.search(r"_FILE(?:_NAME|)$", key) is not None or key in [
            "clock_kernel",
            "frame_kernel",
            "leapsecond_kernel",
            "meta_kernel",
            "planet_kernels",
        ]

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

            # Convert Path objects to strings early to avoid template issues
            if isinstance(item, Path):
                item = str(item)
                modified_value = True

            modified_item = False
            fn_section = None

            # Contains a path that needs to be processed
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

                # Strategy 1: Copy to temp directory with short path
                if try_copy and fn.is_file():
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
                                f"   Temp path would be too long ({estimated_length} chars estimated), "
                                "skipping copy strategy"
                            )
                        else:
                            temp_fd, temp_path = tempfile.mkstemp(suffix=fn.suffix, prefix=prefix, dir=str(temp_dir))
                            os.close(temp_fd)

                            # Verify actual path is short enough (should match estimate)
                            if len(temp_path) <= max_len:
                                # Copy the file - if this fails, temp file will be cleaned up in except block
                                shutil.copy2(fn, temp_path)

                                logger.info(f"   Copied to short path: {temp_path} ({len(temp_path)} chars)")
                                logger.debug(f"   Successfully shortened from {len(str(fn))} to {len(temp_path)} chars")

                                # Track temp file for cleanup
                                temp_files_created.append(temp_path)

                                item = temp_path
                                modified_item = True
                                # Success! Skip other strategies
                                new_vals.append(str(item) if isinstance(item, Path) else item)
                                modified_value = True
                                continue
                            else:
                                # Temp path still too long, clean up and try other strategies
                                actual_length = len(temp_path)
                                os.remove(temp_path)
                                temp_path = None  # Mark as cleaned up
                                logger.warning(f"   Temp path still exceeds limit ({actual_length} chars)")
                    except (OSError, PermissionError, shutil.SameFileError, shutil.Error) as e:
                        # Clean up temp file if it was created but copy/checks failed
                        # Catches: OSError, PermissionError (file system issues)
                        #          shutil.SameFileError and other shutil.Error subclasses (shutil-specific issues)
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass  # Best effort cleanup
                        logger.warning(f"   Failed to copy to temp directory: {e}")

                # Strategy 2: Check if a relative path would be shorter
                if try_relative and not modified_item:
                    rel_fn = os.path.relpath(fn, start=relative_dir)
                    if len(rel_fn) <= max_len:
                        logger.debug("Updated path [%s] to be relative to [%s]", rel_fn, relative_dir)
                        modified_item = True
                        fn_section = rel_fn

                # Strategy 3: Build strings up to the limit (wrapping)
                if try_wrap and not modified_item:
                    logger.debug("Wrapping path [%s]", fn)
                    modified_item = True

                    for fn_part in fn.parts:
                        if len(fn_part) > (max_len - len(wrap_char)):
                            raise ValueError(
                                f"File part [{fn_part}] is too long! Each part must be < {max_len - len(wrap_char)} char!"
                            )
                        if fn_section is None:
                            fn_section = fn_part
                            continue

                        next_section = str(Path(fn_section) / fn_part)
                        if len(next_section) >= max_len:
                            fn_section += wrap_char
                            new_vals.append(fn_section)
                            fn_section = os.path.sep + fn_part
                        else:
                            fn_section = next_section

                item = fn if fn_section is None else fn_section

            # Ensure item is always a string (could be Path if fn_section was None)
            if isinstance(item, Path):
                item = str(item)
                modified_item = True

            new_vals.append(item)

            modified_value |= modified_item

        if modified_value:
            # Unwrap if original was a single string/path
            if not was_list and len(new_vals) == 1:
                properties[key] = new_vals[0]
            else:
                properties[key] = new_vals

    # If we worked on nested properties, put them back in wrap_configs
    if "properties" in configs and isinstance(configs["properties"], dict):
        wrap_configs["properties"] = properties

    return wrap_configs, temp_files_created


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


def write_setup(setup_file, template, configs, mappings=None, overwrite=False, validate=True, parent_dir=None):
    """Write the kernel setup file.

    Parameters
    ----------
    setup_file : str or file object or None
        File or object to write. If None, return the raw text.
    template : str
        Path to a template file for creating the setup_file, or a SPICE kernel
        type. If a SPICE kernel type, the default template will be used.
    configs : dict or dict-like
        Values to write to the setup file.
        Item "spice_body": Dict-like structure with "name" (str) and "code"
        (str or int) items. Used to define the "body" in SPICE.
    mappings : dict, optional
        Dict-like structure with "name" (str or list of str) and "code"
        (int or list of int) items. Used to define the "body" in SPICE ID
        system.
    overwrite : bool, optional
        Option to allow overwriting existing files.
    validate : bool, optional
        Validate that the created text follows the kernel file requirements.
        Default=True. Validation failure will result in a `ValueError`.
    parent_dir : str or Path, optional
        Parent directory to use for non-absolute paths. Default is to use the
        current working directory.

    Returns
    -------
    None or str
        Raw setup text if setup_file is None.

    """
    if template in TEMPLATE_NAMES:
        template = TEMPLATE_NAMES[template]
    else:
        raise ValueError("`template_name` or `ktype` must be specified.")  # E.g., 'mkspk_setup_template.txt'

    if mappings is None:
        mappings = {}

    # TODO: Only works in meta-kernels?
    #   Nope, just not supported by mkspk (etc). They don't respect the rules!
    # Wrap paths that are too long.
    # Explicitly use try_copy=False since we don't want to create temp files that need cleanup tracking
    configs, _ = update_invalid_paths(configs, try_relative=True, try_copy=False, try_wrap=False, parent_dir=parent_dir)

    # Ensure all Path objects are converted to strings before template rendering
    configs = _convert_paths_to_strings(configs)

    # Generate the text.
    logger.debug("Using template: %s", template)
    template = env.get_template(template)
    setup_txt = template.render(
        name=mappings.get("name"),
        code=mappings.get("code"),
        version=configs.pop("version", 0),
        created=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        configs=configs,
    )  # TODO(stone): Add key 'APPEND_TO_OUTPUT' with self.append value? Or does cmd line flag override it anyway?

    issues = []
    if not validate:
        logger.warning("Skipping validation of setup file. NOT RECOMMENDED!")
    elif not validate_text_kernel(StringIO(setup_txt), issues=issues):
        msg = "\n\t".join(f"{msg}:\n\t\t{ln}" for msg, ln in issues)
        raise ValueError(f"Final text failed validation. Issues[{len(issues)}]:\n\t{msg}")

    logger.debug("Created setup:\n%s", setup_txt)

    if isinstance(setup_file, str):
        if os.path.isfile(setup_file) and not overwrite:
            raise OSError(f"File already exists and `overwrite` is not set. File: {setup_file}")
        with open(setup_file, mode="w") as setup_f:
            setup_f.write(setup_txt)

    elif hasattr(setup_file, "write"):
        setup_file.write(setup_txt)
    elif setup_file is None:
        return setup_txt
    else:
        raise ValueError(f"Invalid `setup_file` value: {setup_file!r}")
    return None


# pylint: disable=too-many-branches
def validate_text_kernel(kernel, issues=None):
    """Validate major formatting elements of a text kernel.

    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html

    Parameters
    ----------
    kernel : str or obj with read attr
        Kernel filename or readable object to validate.
    issues : list, optional
        List to append the discovered issues to. Each issue is a tuple string
        pairs of the reason and the source line.

    Returns
    -------
    bool
        True if validation passed, otherwise False. See warnings for reasons.

    """
    if isinstance(kernel, str):
        kernel = open(kernel).read()
    elif hasattr(kernel, "read"):
        kernel = kernel.read()
    else:
        raise ValueError(kernel)

    # Track issues
    if issues is None:
        issues = []

    # Max line length is 132
    for ln in kernel.splitlines():
        if len(ln) > 132:
            issues.append(("Exceeds max line length (132)", ln))

    # Values - Strings must use single quotes
    for section_reg in re.finditer(r"\\begindata((?:.|\n)*)\\begintext", kernel, re.MULTILINE):
        for ln in section_reg.group(1).splitlines():
            if '"' in ln:
                issues.append(("Contains 1+ double quotes", ln))

    # Values - Max string value length is 80
    #   Single quotes can be escaped with another quote (i.e., 'Don''t split me!')
    for reg in re.finditer(r"'((?:(?:'')|[^'])+)'", kernel):
        if len(reg.group(1)) > 80:
            issues.append(("Exceeds max string length (80)", reg.group(1)))

    # File must end in a newline
    if not kernel.endswith(os.linesep):
        issues.append(("File does not end in a newline", repr(kernel[-10:])))

    # Warn about Tab. It's allowed, but not recommended.
    if "\t" in kernel:
        logger.warning("Kernel contains tab characters. Not required to use spaces, but highly urged to.")

    # Only supports ASCII 32-126 (and Tab*)
    bad_val = []
    for d in (ord(c) for c in kernel.replace("\n", " ")):
        if (d < 32 or d > 126) and d not in bad_val:
            bad_val.append(d)
            issues.append(("Unsupported character", repr(chr(d))))

    # Log the issues and return
    if len(issues) == 0:
        logger.debug("Text kernel validation passed.")
        return True

    for msg, ln in issues:
        logger.warning("%s:\n%s", msg, ln)
    return False

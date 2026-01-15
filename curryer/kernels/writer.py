"""Kernel writer functions.

@author: Brandon Stone
"""

import logging
import os
import re
from datetime import datetime, timezone
from io import StringIO

import jinja2

from .path_utils import _convert_paths_to_strings, update_invalid_paths

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

    # Try to shorten paths, but don't use copy strategy since we don't want temp files that need cleanup tracking
    # Only symlinks are attempted here - if that fails, paths remain unchanged
    configs, _ = update_invalid_paths(configs, try_copy=False, parent_dir=parent_dir)

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

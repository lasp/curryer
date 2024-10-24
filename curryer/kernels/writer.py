"""Kernel writer functions.

@author: Brandon Stone
"""
import logging
import os
import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import jinja2


logger = logging.getLogger(__name__)

TEMPLATE_NAMES = {
    'spk': 'mkspk_setup_template.txt',
    'ck': 'msopck_setup_template.txt',
}

env = jinja2.Environment(
    loader=jinja2.PackageLoader(__package__, 'templates'),
    keep_trailing_newline=True,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True
)


# pylint: disable=too-many-branches,too-many-nested-blocks
def update_invalid_paths(configs, max_len=80, try_relative=False, try_wrap=True, wrap_char='+', relative_dir=None,
                         parent_dir=None):
    """Update invalid paths (too long) by relativizing or wrapping.
    See: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html
    """
    # TODO: Do this for all strings, not just filenames?
    # TODO: Only works in meta-kernels!

    relative_dir = Path.cwd() if relative_dir is None else Path(relative_dir)
    if parent_dir is not None:
        parent_dir = Path(parent_dir)
        if parent_dir.is_file():
            parent_dir = parent_dir.parent

    wrap_configs = configs.copy()
    for key, value in configs.items():
        if re.search(r'_FILE(?:_NAME|)$', key) is None:
            continue
        if isinstance(value, (str, Path)):
            value = [value]

        new_vals = []
        modified_value = False
        for item in value:
            if not isinstance(item, (str, Path)):
                new_vals.append(item)
                continue

            modified_item = False
            fn_section = None

            # Contains a path that needs to be wrapped.
            fn = Path(item)

            if parent_dir and not (fn.is_file() or fn.is_dir()) and not fn.is_absolute():
                abs_fn = parent_dir / fn
                if abs_fn.is_file() or abs_fn.is_dir():
                    fn = abs_fn.absolute().resolve()
                    item = fn
                    modified_item = True

            if (fn.is_file() or fn.is_dir()) and len(str(item)) > max_len:
                # Check if a relative path would be shorter.
                #   Note: Requires that the CWD doesn't change!!!
                if try_relative:
                    rel_fn = os.path.relpath(fn, start=relative_dir)
                    if len(rel_fn) <= max_len:
                        logger.debug('Updated path [%s] to be relative to [%s]', rel_fn, relative_dir)
                        modified_item = True
                        fn_section = rel_fn

                # Build strings up to the limit.
                if try_wrap and not modified_item:
                    logger.debug('Wrapping path [%s]', fn)
                    modified_item = True

                    for fn_part in fn.parts:
                        if len(fn_part) > (max_len - len(wrap_char)):
                            raise ValueError('File part [{}] is too long! Each part must be < {} char!'
                                             ''.format(fn_part, max_len - len(wrap_char)))
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

            if isinstance(item, Path):
                item = str(item)
                modified_item = True

            new_vals.append(item)

            modified_value |= modified_item

        if modified_value:
            wrap_configs[key] = new_vals

    return wrap_configs


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
        raise ValueError('`template_name` or `ktype` must be specified.')  # E.g., 'mkspk_setup_template.txt'

    if mappings is None:
        mappings = {}

    # TODO: Only works in meta-kernels?
    #   Nope, just not supported by mkspk (etc). They don't respect the rules!
    # Wrap paths that are too long.
    configs = update_invalid_paths(configs, try_relative=True, try_wrap=False, parent_dir=parent_dir)

    # Generate the text.
    logger.debug('Using template: %s', template)
    template = env.get_template(template)
    setup_txt = template.render(
        name=mappings.get('name'),
        code=mappings.get('code'),
        version=configs.pop('version', 0),
        created=datetime.utcnow().strftime('%Y-%m-%d'),
        configs=configs
    )  # TODO(stone): Add key 'APPEND_TO_OUTPUT' with self.append value? Or does cmd line flag override it anyway?

    issues = []
    if not validate:
        logger.warning('Skipping validation of setup file. NOT RECOMMENDED!')
    elif not validate_text_kernel(StringIO(setup_txt), issues=issues):
        msg = '\n\t'.join(f'{msg}:\n\t\t{ln}' for msg, ln in issues)
        raise ValueError(f'Final text failed validation. Issues[{len(issues)}]:\n\t{msg}')

    logger.debug('Created setup:\n%s', setup_txt)

    if isinstance(setup_file, str):
        if os.path.isfile(setup_file) and not overwrite:
            raise IOError('File already exists and `overwrite` is not set. File: {}'.format(setup_file))
        with open(setup_file, mode='w') as setup_f:
            setup_f.write(setup_txt)

    elif hasattr(setup_file, 'write'):
        setup_file.write(setup_txt)
    elif setup_file is None:
        return setup_txt
    else:
        raise ValueError('Invalid `setup_file` value: {!r}'.format(setup_file))
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
    elif hasattr(kernel, 'read'):
        kernel = kernel.read()
    else:
        raise ValueError(kernel)

    # Track issues
    if issues is None:
        issues = []

    # Max line length is 132
    for ln in kernel.splitlines():
        if len(ln) > 132:
            issues.append(('Exceeds max line length (132)', ln))

    # Values - Strings must use single quotes
    for section_reg in re.finditer(r"\\begindata((?:.|\n)*)\\begintext", kernel, re.MULTILINE):
        for ln in section_reg.group(1).splitlines():
            if '"' in ln:
                issues.append(('Contains 1+ double quotes', ln))

    # Values - Max string value length is 80
    #   Single quotes can be escaped with another quote (i.e., 'Don''t split me!')
    for reg in re.finditer(r"'((?:(?:'')|[^'])+)'", kernel):
        if len(reg.group(1)) > 80:
            issues.append(('Exceeds max string length (80)', reg.group(1)))

    # File must end in a newline
    if not kernel.endswith(os.linesep):
        issues.append(('File does not end in a newline', repr(kernel[-10:])))

    # Warn about Tab. It's allowed, but not recommended.
    if '\t' in kernel:
        logger.warning('Kernel contains tab characters. Not required to use spaces, but highly urged to.')

    # Only supports ASCII 32-126 (and Tab*)
    bad_val = []
    for d in (ord(c) for c in kernel.replace('\n', ' ')):
        if (d < 32 or d > 126) and d not in bad_val:
            bad_val.append(d)
            issues.append(('Unsupported character', repr(chr(d))))

    # Log the issues and return
    if len(issues) == 0:
        logger.debug('Text kernel validation passed.')
        return True

    for msg, ln in issues:
        logger.warning('%s:\n%s', msg, ln)
    return False

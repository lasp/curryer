"""classes

@author: Brandon Stone
"""
import logging
import os
import shutil
import tempfile
import typing
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from .writer import write_setup
from .. import spicetime, utils


logger = logging.getLogger(__name__)


class TypedDataDescriptor:
    """Data descriptor that enforces typing through casting.
    """

    def __init__(self, *, default, dtype):  # , required=False):
        self._default = default
        self._dtype = dtype
        # self._required = required

    def __set_name__(self, owner, name):
        self._name = '_' + name

    def __get__(self, instance, owner):
        if instance is None:
            # if self._required:
            #     raise AttributeError(f'Missing required value for {self._name}')
            return self._default
        return getattr(instance, self._name, self._default)

    def __set__(self, instance, value):
        if value is None:
            value = self._default
        elif not isinstance(value, self._dtype) and self._dtype is not None:
            if issubclass(self._dtype, Enum) and isinstance(value, str) and value in self._dtype.__members__:
                value = self._dtype[value]
            elif is_dataclass(self._dtype) and isinstance(value, dict):
                value = self._dtype(**value)
            else:
                value = self._dtype(value)
        setattr(instance, self._name, value)


@dataclass  # kw_only=True (added in 3.10)
class AbstractKernelProperties:
    """Kernel properties abstraction.
    """
    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = None

    # Input properties.
    input_data_type: Enum = field()  # Required
    input_gap_threshold: str = None

    # Existing kernels that are required to build this one.
    leapsecond_kernel: str = None

    # Misc.
    version: int = 1
    author: str = 'LASP'
    mappings: dict = None
    relative_dir: Path = None

    def __post_init__(self):
        if self.SUPPORTED_INPUT_DATA_TYPES is not None and self.input_data_type not in self.SUPPORTED_INPUT_DATA_TYPES:
            raise ValueError(f'Invalid properties class for input type {self.input_data_type}')

        if self.leapsecond_kernel is None:
            self.leapsecond_kernel = str(spicetime.leapsecond.find_default_file())

        if self.relative_dir:
            self.relative_dir = Path(self.relative_dir)
        self._update_paths(['leapsecond_kernel'])

    def _update_paths(self, attr_names):
        if not self.relative_dir:
            return

        for key in attr_names:
            files = getattr(self, key)
            if not files:
                continue

            was_list = True
            if not isinstance(files, list):
                files = [files]
                was_list = False

            out_files = []
            for fn in files:
                if fn and not os.path.isfile(fn) and not os.path.isabs(fn):
                    fn = str((self.relative_dir / fn).resolve())
                out_files.append(fn)

            setattr(self, key, out_files if was_list else out_files[0])

    def to_dict(self):
        """Convert properties to a dict.
        """
        raise NotImplementedError


class AbstractKernelWriter(metaclass=ABCMeta):
    """Write to a SPICE kernel.
    """
    KTYPE = None
    FILE_EXT = None

    def __init__(self, properties: AbstractKernelProperties, bin_dir: str = None, parent_dir: str = None):
        """Define the kernel data source.

        Parameters
        ----------
        properties : AbstractKernelProperties
            Kernel properties.
        bin_dir : str, optional
            Directory containing SPICE utilities, otherwise assumed to be in the
            PATH environment variable.
        parent_dir : str, optional
            Directory to use for non-absolute file paths, default is CWD.

        """
        # Kernel configuration.
        self.properties = properties
        self.bin_dir = bin_dir
        self.parent_dir = parent_dir

        # If these are None at write time, it'll use temporary files.
        self.input_file = None
        self.setup_file = None

    def __call__(self, input_data: pd.DataFrame, filename, overwrite=False, append=False):
        """Prepare and then create a kernel file.
        """
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f'input_data must be a Pandas DataFrame, not: {type(input_data)}')

        # Prevent accidental overwrites.
        if not overwrite:
            if isinstance(self.input_file, str) and os.path.isfile(self.input_file):
                raise FileExistsError(self.input_file)
            if isinstance(self.setup_file, str) and os.path.isfile(self.setup_file):
                raise FileExistsError(self.setup_file)

        # Prepare the input data (transformed chunks) and setup configuration (text).
        input_data_chunks = self.prepare_input_data(input_data)
        setup_txt = self.prepare_setup_config()

        # Write the kernel.
        return self.write_kernel(filename, input_data_chunks, setup_txt, overwrite=overwrite, append=append)

    @abstractmethod
    def prepare_input_data(self, input_data: pd.DataFrame) -> typing.List[pd.DataFrame]:
        """Prepare an accessor that will provide input data for the kernel.
        """
        raise NotImplementedError

    def _chunk_table_by_gaps(self, table: pd.DataFrame, index_dt_type='utc'):
        """Chunk a table by the `input_gap_threshold` (if any).
        """
        chunks = []
        if table.shape[0] <= 1 or self.properties.input_gap_threshold is None:
            return chunks

        gap_size_sec = pd.to_timedelta(self.properties.input_gap_threshold).seconds
        ugps = spicetime.adapt(table.index, index_dt_type, 'ugps')
        idx_gaps, = np.where((ugps[1:] - ugps[:-1]) / 1e6 >= gap_size_sec)

        if idx_gaps.size:
            logger.info('Kernel input data has [%d] gaps larger than [%s], splitting write into [%d] segments',
                        idx_gaps.size, self.properties.input_gap_threshold, idx_gaps.size + 1)
            idx_gaps += 1

            from_idx = 0
            ith = 0
            while ith <= idx_gaps.size:
                to_idx = idx_gaps[ith] if ith < idx_gaps.size else table.index.size
                chunks.append(table.iloc[from_idx: to_idx])
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Input chunk [%d/%d] with [%d] values spanning [%s, %s]', ith + 1,
                                 idx_gaps.size + 1, chunks[-1].shape[0], chunks[-1].index[0], chunks[-1].index[-1])
                from_idx = to_idx
                ith += 1

        return chunks

    @abstractmethod
    def write_input_data(self, fobj_or_str, input_data: pd.DataFrame):
        """Write input data.
        """
        raise NotImplementedError

    @contextmanager
    def _safe_write_input_data(self, input_data: pd.DataFrame):
        """Context to write the input data to a CSV file.

        A temporary file is used if one wasn't defined; it's deleted on exit.

        Parameters
        ----------
        input_data : pd.DataFrame
            Data frame to use as the input data source.

        Yields
        ------
        str
            Path to the input data file (CSV format). Either `self.input_file`,
            if it was a str, or a temporary file if it was None.

        """
        # Write to a temporary file if one isn't defined.
        if self.input_file is None:
            try:
                fd, self.input_file = tempfile.mkstemp(text=True)
                logger.debug('Created temporary input file: %s', self.input_file)

                with os.fdopen(fd, mode='w') as fobj:
                    self.write_input_data(fobj, input_data)
                yield self.input_file

            finally:
                if isinstance(self.input_file, str):
                    os.remove(self.input_file)
                    logger.debug('Deleted temporary input file: %s', self.input_file)
                    self.input_file = None

        # Write to the defined file. It won't be deleted on exit.
        elif isinstance(self.input_file, str):
            self.write_input_data(self.input_file, input_data)
            yield self.input_file

        # Defined file is an invalid type.
        else:
            raise AttributeError('`input_file` must be a str or None, not: {!r}'.format(self.input_file))

    def prepare_setup_config(self):
        """Prepare the kernel file's setup configuration (pseudo kernel).
        """
        config = self.properties.to_dict()

        # Format the setup file text. Doesn't actually write to a file since
        # none was provided.
        return write_setup(None, self.KTYPE, config, mappings=self.properties.mappings, validate=True,
                           parent_dir=self.parent_dir)

    @contextmanager
    def _safe_write_setup_config(self, setup_txt):
        """Context to write the setup configuration to a text file.

        A temporary file is used if one wasn't defined; it's deleted on exit.

        Parameters
        ----------
        setup_txt : str
            Setup configuration text.

        Yields
        ------
        str
            Path to the setup file. Either `self.setup_file`, if it was a str,
            or a temporary file if it was None.

        """
        # Write to a temporary file if one isn't defined.
        if self.setup_file is None:
            try:
                fd, self.setup_file = tempfile.mkstemp(text=True)
                logger.debug('Created temporary setup file: %s', self.setup_file)

                with os.fdopen(fd, mode='w') as fobj:
                    fobj.write(setup_txt)
                yield self.setup_file

            finally:
                if isinstance(self.setup_file, str):
                    os.remove(self.setup_file)
                    logger.debug('Deleted temporary setup file: %s', self.setup_file)
                    self.setup_file = None

        # Write to the defined file. It won't be deleted on exit.
        elif isinstance(self.setup_file, str):
            with open(self.setup_file, mode='w') as fobj:
                fobj.write(setup_txt)
                yield self.setup_file

        # Defined file is an invalid type.
        else:
            raise AttributeError('`setup_file` must be a str or None, not: {!r}'.format(self.setup_file))

    @contextmanager
    def _backup_file(self, filename):
        """Context to create a temporary backup of a file (if it exists).
        The original file is restored if an exception occurs. The temporary
        backup is deleted when the context exits.
        """
        # No-op if the file doesn't exist.
        if not os.path.isfile(filename):
            logger.debug('Files does not exist, skipping backup step for: %s', filename)
            yield None
            return

        # Create a backup copy with the same permissions and metadata.
        backup_filename = filename + '.BACKUP'
        logger.debug('Creating a temporary backup: %s -> %s', filename, backup_filename)
        shutil.copy2(filename, backup_filename)

        # Begin the context.
        try:
            yield backup_filename

        # If an exception occurs, attempt to restore the original file.
        except:
            logger.warning('An exception occurred during backup context. Restoring the original file: %s', filename)
            try:
                shutil.copy2(backup_filename, filename)
            except:
                logger.error('Failed to restore original file: %s from %s', filename, backup_filename)
                logger.exception('Exception during failed restore:')
                raise
            raise

        # On context exit or after an exception is handled, delete the backup
        #   (if it still exists).
        finally:
            if os.path.isfile(backup_filename):
                os.remove(backup_filename)
                logger.debug('Deleted temporary file backup: %s', backup_filename)
            else:
                logger.error('Temporary file backup does not exist. An unknown source deleted: %s', backup_filename)

    @staticmethod
    def _check_write_behavior(kernel_file, overwrite=False, append=False):
        """Check if file exists and if writer should overwrite vs. append.
        Required because the SPICE utils have different behavior.
        """
        # Ignore overwrite & append if no file exists.
        if not os.path.isfile(kernel_file):
            pass

        # Don't allow overwrite and append.
        elif overwrite and append:
            raise ValueError('`overwrite` and `append` can not be used together.')

        # Delete the file to ensure we "overwrite" it.
        #   MSOCK appends if the file exists.
        elif overwrite:
            logger.info('Overwrite is set, deleting: %r', kernel_file)
            os.remove(kernel_file)

        # Allow appending.
        elif append:
            pass

        # Error if file exists and neither kw is set.
        else:
            raise FileExistsError(f'Either remove file, use `overwrite` or `append` keywords. File: {kernel_file}')

    def _get_bin(self, name):
        """Get the path to an executable.
        """
        if self.bin_dir is not None:
            bin_ = os.path.join(self.bin_dir, name)
            if not os.path.isfile(bin_):
                raise FileNotFoundError(
                    f'Unable to find executable "{name}" in user defined directory "{self.bin_dir}"')

        else:
            bin_ = shutil.which(name)
            if bin_ is None:
                raise FileNotFoundError(f'Unable to find executable "{name}" in system PATH.')
        return bin_

    @abstractmethod
    def _write_kernel(self, setup_file, input_file, kernel_file, append=False):
        """Write to the kernel.
        """
        raise NotImplementedError

    def write_kernel(self, filename, input_data, setup_txt, overwrite=False, append=False):
        """Create a new kernel or append to an existing one (config setting).
        """
        if isinstance(input_data, list):
            n_segments = len(input_data)
        else:
            n_segments = 1
            input_data = [input_data]

        if n_segments == 0:
            logger.warning('Skipping kernel creation due to missing input data: %s (n_segments=%d)',
                           filename, n_segments)
            return str(filename), None

        logger.info('Creating kernel: %s (n_segments=%d, overwrite=%s, append=%s)', filename,
                    n_segments, overwrite, append)
        filename = str(filename)

        # Create a temporary backup of the kernel (if it exists).
        #   This is necessary because failures during appends may corrupt the
        #   file. See the `mkspk` documentation for details.
        with self._backup_file(filename):
            # Check how the file should be handled (overwrite vs. append).
            self._check_write_behavior(filename, overwrite=overwrite, append=append)

            # Write the kernel!
            #   Context will create temporary files on enter, and delete them
            #   on exit (e.g., on error or after writing the kernel).
            with self._safe_write_setup_config(setup_txt) as setup_file:
                for ith, accessor in enumerate(input_data, 1):
                    if n_segments > 1:
                        logger.info('Processing input for kernel segment [%d/%d]', ith, n_segments)
                    with self._safe_write_input_data(accessor) as input_file:
                        cmd = self._write_kernel(setup_file, input_file, filename, append=append or ith > 1)
                        utils.capture_subprocess(cmd)

        return filename, None

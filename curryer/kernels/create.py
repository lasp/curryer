"""SPICE kernel creation wrapper.

@author: Brandon Stone
"""
import datetime
import importlib
import json
import logging
import shutil
import sys
from itertools import chain
from pathlib import Path

import pandas as pd
import xarray as xr

from . import attitude, ephemeris, classes
from .. import spicetime


logger = logging.getLogger(__name__)


class KernelCreator:
    """Class used to create kernels.
    """

    def __init__(self, overwrite=False, append=False, bin_dir: Path = None, parent_dir: Path = None):
        self._overwrite = overwrite
        self._append = append
        self._bin_dir = bin_dir
        self._parent_dir = parent_dir

    def find_bin_dir(self, exe: str = 'mkspk'):
        if self._bin_dir is not None:
            return self._bin_dir
        if shutil.which(exe):
            return Path(shutil.which(exe)).parent
        return Path(__file__).parents[2] / 'bin' / 'spice' / ('macintel' if sys.platform == 'darwin' else 'linux')

    @staticmethod
    def find_class(class_path):
        """Convert a class path string to an imported object.
        """
        if '.' not in class_path:
            return importlib.import_module(class_path)

        module_name, cls_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)

    def load_input_data(self, input_obj, parent_dir: Path = None, input_columns: dict = None):
        if parent_dir is None:
            parent_dir = self._parent_dir

        # File-based inputs.
        if isinstance(input_obj, (str, Path)):
            filename = Path(input_obj)
            if parent_dir and not filename.is_file() and not filename.is_absolute():
                filename = (Path(parent_dir) / filename).resolve()
            if not filename.is_file():
                raise FileNotFoundError(filename)

            if filename.suffix == '.nc':
                input_data = xr.load_dataset(filename)
                input_data = input_data.to_pandas()

            elif filename.suffix == '.csv':
                input_data = pd.read_csv(filename)

            else:
                raise ValueError(f'Unsupported file type: {filename}')

        # Static pre-defined input.
        elif isinstance(input_obj, dict):
            input_data = pd.DataFrame(input_obj)

        # Loaded netcdf representation.
        elif isinstance(input_obj, xr.Dataset):
            input_data = input_obj.to_dataframe()

        # No-op.
        elif isinstance(input_obj, pd.DataFrame):
            input_data = input_obj

        else:
            raise ValueError(f'Invalid input data object type [{type(input_obj)}]')

        # Optionally remap the columns.
        if input_columns is not None and len(input_columns):
            if isinstance(input_columns, dict):
                if input_data.index.name in input_columns:
                    input_data = input_data.reset_index()
                input_data = input_data.rename(columns=input_columns)
                input_data = input_data[list(input_columns.values())]
            else:
                input_data = input_data[list(input_columns)]

        # Drop rows with invalid values.
        shape_before = input_data.shape
        input_data = input_data.dropna(how='any')
        if input_data.shape[0] != shape_before[0]:
            logger.warning('Dropped [%d] rows from the input due to NaNs', shape_before[0] - input_data.shape[0])
        return input_data

    def load_writer(self, properties: classes.AbstractKernelProperties, parent_dir: Path = None):
        if parent_dir is None:
            parent_dir = self._parent_dir
        if not isinstance(properties, classes.AbstractKernelProperties):
            raise TypeError(properties)

        prop_cls = type(properties)
        writer_cls = ephemeris.PROPERTIES_TO_WRITER.get(prop_cls, attitude.PROPERTIES_TO_WRITER.get(prop_cls, None))

        if writer_cls is None:
            for cls, wrtr in chain(ephemeris.PROPERTIES_TO_WRITER.values(), attitude.PROPERTIES_TO_WRITER.values()):
                if issubclass(writer_cls, cls):
                    writer_cls = wrtr
                    break

            if writer_cls is None:
                raise TypeError(f'Unable to find writer class that supports properties class [{prop_cls}]')

        writer_inst = writer_cls(properties, bin_dir=self.find_bin_dir(), parent_dir=parent_dir)
        logger.info('Constructed kernel writer [%s] for properties [%s]', writer_inst, properties)
        return writer_inst

    def write(self, properties: classes.AbstractKernelProperties, output_kernel: Path, input_data, input_columns=None):
        if not isinstance(properties, classes.AbstractKernelProperties):
            raise TypeError(properties)

        input_data = self.load_input_data(input_data, input_columns=input_columns)
        writer = self.load_writer(properties)
        writer(input_data, output_kernel, overwrite=self._overwrite, append=self._append)
        logger.info('Completed kernel [%s] for file: %s', 'APPEND' if self._append else 'WRITE', output_kernel)
        return output_kernel

    def write_from_json(self, properties_file: Path, output_kernel: Path = None, input_data=None, input_columns=None,
                        overrides=None):
        if isinstance(properties_file, str):
            properties_file = Path(properties_file)

        kernel_config = json.loads(properties_file.read_text())
        parent_dir = properties_file.parent if self._parent_dir is None else self._parent_dir

        # Define kernel properties based on a class path and dict.
        properties_def = kernel_config['kernel']
        properties_cls = self.find_class(properties_def['class_path'])
        if overrides:
            properties_def['properties'].update(overrides)
        properties = properties_cls(**properties_def['properties'], relative_dir=parent_dir)

        # Load the writer class.
        writer = self.load_writer(properties, parent_dir=parent_dir)

        # Optionally define the kernel data.
        if input_data is None:
            input_data = kernel_config.get('input_data', None)
            if not input_data:
                raise ValueError('Must define the `input_data` in the JSON or pass as a keyword argument!')

        if input_columns is None:
            input_columns = kernel_config.get('input_columns', None)

        input_data = self.load_input_data(input_data, parent_dir=parent_dir, input_columns=input_columns)

        # Optionally define the output file based on the config file path.
        if output_kernel is None:
            output_kernel = properties_file.with_suffix(writer.FILE_EXT)
            logger.debug('No output kernel file path specified, defaulting to: %s', output_kernel)
        if isinstance(output_kernel, str):
            output_kernel = Path(output_kernel)
        if output_kernel.is_dir():
            output_kernel = (output_kernel / properties_file.name).with_suffix(writer.FILE_EXT)
            logger.debug('No output kernel file name specified, defaulting to: %s', output_kernel)

        # Write to the kernel file!
        writer(input_data, output_kernel, overwrite=self._overwrite, append=self._append)
        logger.info('Completed kernel [%s] for file: %s from %s',
                    'APPEND' if self._append else 'WRITE', output_kernel, properties_file)
        return output_kernel


def batch_kernels(kernel_configs, output_kernels=None, time_range=None, time_format='utc', buffer_hours=0,
                  lag_days=None, overwrite=False, append=False):
    """Create SPICE kernels.

    The mission, data source (accessor) and kernel properties are defined in a
    JSON file.

    Parameters
    ----------
    kernel_configs : list[str or Path]
        One ore more kernel creation properties (JSON file).
    output_kernels : list[str or Path], optional
        File to save the kernel as. Default is to use the `kernel_config`, but
        with the appropriate file extension. Must be None if more than one
        kernel config is given.
    time_range : iter of (str or int), optional
        Time range of data to write to the kernel (exclusive end). Default is
        to write all available data.
    time_format : str, optional
        Time format code (ttype) of `time_range`. Default="utc".
    buffer_hours : float or list[float], optional
        Option to buffer `time_range`, in hours. Default=0.
    lag_days : int, optional
        Define the `time_range` to be a 24-hour window starting `lag_days` days
        prior to the start of the current day (UTC).
    overwrite : bool, optional
        Overwrite an existing file.
    append : bool, optional
        Append to an existing file.

    Returns
    -------
    list[str or Path]
        The output kernel files, same as `output_kernels` if it was given.

    """
    # TODO: Time args are no longer necessary?
    # Define the time range.
    if lag_days is not None and time_range is not None:
        raise ValueError('can not specify both `time_range` and `lag_days`')

    if lag_days is not None:
        dt = datetime.datetime.utcnow().date()
        dt0 = dt - datetime.timedelta(days=lag_days)
        dt1 = dt0 + datetime.timedelta(days=1)
        time_range = [dt0.isoformat(), dt1.isoformat()]
        time_format = 'utc'

    if time_range is not None:
        time_range = spicetime.adapt(time_range, from_=time_format, to='ugps')
        if len(time_range) != 2 or time_range[0] > time_range[1]:
            raise ValueError(f'If specified, `time_range` must contain two times in increasing order: {time_range}')

        if buffer_hours:
            if not isinstance(buffer_hours, (list, tuple)):
                buffer_hours = [buffer_hours, buffer_hours]
            time_range = [time_range[0] - int(buffer_hours[0] * 3.6e9),
                          time_range[1] + int(buffer_hours[1] * 3.6e9)]

    # Iterate through each config to create/append the kernels!
    n_config = len(kernel_configs)
    if output_kernels is not None:
        # TODO: Support output directory!
        if not isinstance(output_kernels, (list, tuple)):
            raise TypeError(f'`output_kernels` must be a list/tuple, not: {type(output_kernels)}')
        if not n_config == len(output_kernels):
            raise ValueError(f'`output_kernels` must be the same length as `kernel_configs` or None')
    else:
        output_kernels = [None for _ in range(n_config)]

    logger.info('Processing [%s] kernel configs...', n_config)
    for i, config in enumerate(kernel_configs):
        logger.info('Processing kernel config [%d/%d], time_range=[%s]: %s', i + 1, n_config, time_range, config)
        # file = create_kernel(config, output_kernel=output_kernels[i], time_range=time_range, overwrite=overwrite,
        #                      append=append)
        # if output_kernels[i] is None:
        #     output_kernels.pop(i)
        #     output_kernels.insert(i, file)
        raise NotImplementedError

    logger.info('Finished processing [%s] kernel configs!', n_config)
    return output_kernels

"""attitude

@author: Brandon Stone
"""
import logging
import os
import typing
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .classes import AbstractKernelWriter, AbstractKernelProperties, TypedDataDescriptor
from .. import spicetime, spicierpy


logger = logging.getLogger(__name__)


class AttitudeTypes(Enum):
    """Attitude kernel types.
    Note: `msopck` only supports types 1, 2 & 3.
    """
    DISCRETE_QUAT_AND_RATE = 1  # TODO: Remove QUAT from names? Not specific?
    CONTINUOUS_QUAT_AND_FIXED_RATE = 2
    LINEAR_QUAT = 3
    POLYNOMIAL_QUAT_AND_OPTIONAL_RATE = 4
    # Type-5 has four subtypes (0-3) for diff interp.
    # Type-6 is an optimized version of type-5s.


class AttitudeInputDataTypes(Enum):
    """Attitude input data types.
    """
    SPICE_QUAT = 'SPICE QUATERNIONS'  # (c, x, y, z)
    MSOP_QUAT = 'MSOP QUATERNIONS'  # (-x, -y, -z, c)
    FLIPSIGN_QUAT = 'FLIP SPICE QUATERNIONS'  # (c, -x, -y, -z) Invokes internal func to flip signs.
    EULER_ANGLES = 'EULER ANGLES'
    # MATRICES = 'MATRICES'


class AttitudeWriterTimeTypes(Enum):
    """Attitude writer time types.
    """
    SCLK = 'SCLK'
    UTC = 'UTC'
    TICKS = 'TICKS'
    DSCLK = 'DSCLK'
    ET = 'ET'


class AttitudeAngularRateTypes(Enum):
    """Attitude angular rate types.
    """
    PRESENT = 'YES'
    NOT_PRESENT = 'NO'
    MAKE_UP_AVG = 'MAKE UP'
    MAKE_UP_NO_AVG = 'MAKE UP/NO AVERAGING'


@dataclass
class AbstractAttitudeProperties(AbstractKernelProperties):
    """Attitude kernel common properties.
    """
    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = None

    # Input properties.
    # input_body: TypedDataDescriptor = TypedDataDescriptor(default=None, dtype=spicierpy.obj.AnyBodyOrFrame)
    # input_frame: TypedDataDescriptor = TypedDataDescriptor(default=None, dtype=spicierpy.obj.Frame)
    input_body: spicierpy.obj.AnyBodyOrFrame = None
    input_frame: spicierpy.obj.Frame = None
    input_time_type: str = 'ugps'
    input_data_type: TypedDataDescriptor = TypedDataDescriptor(
        default=AttitudeInputDataTypes.SPICE_QUAT, dtype=AttitudeInputDataTypes)
    input_angular_rate: TypedDataDescriptor = TypedDataDescriptor(
        default=AttitudeAngularRateTypes.PRESENT, dtype=AttitudeAngularRateTypes)
    input_time_columns: typing.List[str] = field(default_factory=lambda: ['ugps'])
    input_data_columns: typing.List[str] = None  # field(default_factory=lambda: [])
    input_rate_columns: typing.List[str] = field(default_factory=lambda: ['rate_x', 'rate_y', 'rate_z'])

    # Output properties.
    ck_type: TypedDataDescriptor = TypedDataDescriptor(default=AttitudeTypes.LINEAR_QUAT, dtype=AttitudeTypes)

    # Existing kernels that are required to build this one.
    leapsecond_kernel: str = None
    frame_kernel: str = None
    clock_kernel: str = None
    create_clock: bool = False  # If true, writes to `clock_kernel`.

    # Writer properties.
    writer_time_type: TypedDataDescriptor = TypedDataDescriptor(
        default=AttitudeWriterTimeTypes.UTC, dtype=AttitudeWriterTimeTypes)

    def __post_init__(self):
        super().__post_init__()
        self._update_paths(['frame_kernel', 'clock_kernel'])

        if not (isinstance(self.input_body, spicierpy.obj.AbstractObj) and
                isinstance(self.input_frame, spicierpy.obj.Frame)):
            loaded_krns = None
            if os.path.isfile(self.frame_kernel):
                loaded_krns = spicierpy.ext.load_kernel(self.frame_kernel)

            if not isinstance(self.input_body, spicierpy.obj.AbstractObj):
                self.input_body = spicierpy.obj.AnyBodyOrFrame(self.input_body)
            if not isinstance(self.input_frame, spicierpy.obj.Frame):
                self.input_frame = spicierpy.obj.Frame(self.input_frame)

            if loaded_krns is not None:
                loaded_krns.unload(clear=True)

    def to_dict(self):
        """Convert the properties class to a dict for creating kernels.
        """
        config = {}
        if any(val is None for val in (self.input_body, self.input_frame, self.input_data_columns, self.clock_kernel)):
            raise ValueError('Required fields may not be None!')

        # NOTE: The ID here can not be the ID for the spacecraft, because the
        # msopck program will //1000 to infer the clock ID but that's the same
        # as the S/C ID, so it errors out saying it's missing the sclk kernel.
        # The solution is to set this to a body if it's not the spacecraft
        # (e.g. instruments) or the spacecraft's frame. Here we try to infer.
        if self.input_body.is_frame:
            # All frame IDs are ok because `frame_id = sc_id * 1000 + ...`.
            pass
        elif self.input_body.id // 1000 < -100000:
            # The truncated ID was still in the standard range of S/C or body
            # IDs, so assume this isn't an S/C ID.
            pass
        else:
            # Likely an S/C ID, so look up the frame ID and use it instead.
            logger.info('Changing `input_body` (INSTRUMENT_ID) from [%s] to [%s], as the former is believed to be a'
                        ' spacecraft ID. The msopck program assumes `clock_id = input_id // 1000` and'
                        ' `clock_id == spacecraft_id`', self.input_body.id, self.input_body.frame.id)
            self.input_body = spicierpy.obj.AnyBodyOrFrame(self.input_body.frame, body=self.input_body)
        config['INSTRUMENT_ID'] = self.input_body.id

        config['ANGULAR_RATE_PRESENT'] = self.input_angular_rate.value
        config['REFERENCE_FRAME_NAME'] = self.input_frame.name
        config['INPUT_DATA_TYPE'] = self.input_data_type.value
        config['CK_TYPE'] = self.ck_type.value

        # Time format that spice expects. Internally, user data is converted
        #   from the format at "self.input_time_type" to this.
        config['INPUT_TIME_TYPE'] = self.writer_time_type.value

        # Required kernels.
        config['LSK_FILE_NAME'] = self.leapsecond_kernel
        config['MAKE_FAKE_SCLK' if self.create_clock else 'SCLK_FILE_NAME'] = self.clock_kernel
        if self.frame_kernel:
            config['FRAMES_FILE_NAME'] = self.frame_kernel
        elif spicierpy.frmnam(self.input_frame.id) == '':
            # TODO: Loop-hole if frame kernel is loaded. Spice error is obvious so...
            raise ValueError('The input Frame {} is not a built-in, so a "frame_kernel" must be supplied.'
                             ''.format(self.input_frame))

        config['VERSION'] = self.version
        config['PRODUCER_ID'] = self.author
        return config


@dataclass
class AttitudeQuaternionProperties(AbstractAttitudeProperties):
    """Attitude quaternion kernel properties.
    """
    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = (
        AttitudeInputDataTypes.SPICE_QUAT, AttitudeInputDataTypes.MSOP_QUAT, AttitudeInputDataTypes.FLIPSIGN_QUAT)

    # Input properties.
    input_data_columns: typing.List[str] = field(
        default_factory=lambda: ['quaternion_c', 'quaternion_x', 'quaternion_y', 'quaternion_z'])
    # TODO: `QUATERNION_NORM_ERROR`?


@dataclass
class AttitudeEulerProperties(AbstractAttitudeProperties):
    """Attitude euler angle kernel properties.
    """
    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = (AttitudeInputDataTypes.EULER_ANGLES,)

    # Input properties.
    input_data_type: TypedDataDescriptor = TypedDataDescriptor(
        default=AttitudeInputDataTypes.EULER_ANGLES, dtype=AttitudeInputDataTypes)
    input_rotations_order: typing.List[str] = field(default_factory=lambda: ['X', 'Y', 'Z'])
    input_rotations_type: str = 'SPACE'  # or 'BODY' (SPACE= ax * ay * az)
    input_angle_units: str = 'DEGREES'  # or 'RADIANS'
    input_data_columns: typing.List[str] = field(default_factory=lambda: ['angle_x', 'angle_y', 'angle_z'])

    def to_dict(self):
        config = super().to_dict()
        config['EULER_ROTATIONS_ORDER'] = self.input_rotations_order
        config['EULER_ROTATIONS_TYPE'] = self.input_rotations_type
        config['EULER_ANGLE_UNITS'] = self.input_angle_units

        # # TODO: Should these be available?
        # config['OFFSET_ROTATION_ANGLES'] = [0, 0, 0]
        # config['OFFSET_ROTATION_AXES'] = ['X', 'Y', 'Z']
        # config['OFFSET_ROTATION_UNITS'] = 'DEGREES'

        return config


class AttitudeWriter(AbstractKernelWriter):
    """Create or append to a SPICE attitude (CK) kernel.

    Notes
    -----
    Kernel configuration arguments set by this class:
        - TODO

    """
    KTYPE = 'ck'
    FILE_EXT = '.bc'

    def __init__(self, properties: AbstractAttitudeProperties, **kwargs):
        super().__init__(properties, **kwargs)
        if not isinstance(properties, AbstractAttitudeProperties):
            raise TypeError('Attitude writer only supports attitude properties')
        self.properties = properties

    def prepare_input_data(self, input_data: pd.DataFrame) -> typing.List[pd.DataFrame]:
        """Prepare an accessor that will provide input data for the kernel.
        """
        # Get pointing data (quaternions, etc.).
        columns = self.properties.input_time_columns + self.properties.input_data_columns
        if self.properties.input_angular_rate == AttitudeAngularRateTypes.PRESENT:
            columns += self.properties.input_rate_columns
        table = input_data[[col for col in columns if col in input_data.columns]]
        if table.index.name != self.properties.input_time_columns[0]:
            table = table.set_index(self.properties.input_time_columns)  # self.properties.input_time_type)
        if len(table.columns) == 0:
            raise ValueError(f'Input data must have at least one column! Original columns dropped:'
                             f' [{input_data.columns}]')

        # Can't continue if there was no input data.
        if table.size == 0:
            logger.warning('Query returned an empty dataset: %s', table.columns.values)
            return []

        # Pad columns if necessary.
        if len(table.columns) < len(columns) - 1:
            for col in columns:
                if col not in table.columns and col != table.index.name:
                    logger.info('Setting table column [%s] to 0.0', col)
                    table[col] = 0.0

        # Ensure the correct column order (ignore time index).
        if not (columns[1:] == table.columns).all():
            table = table.reindex(columns=columns[1:], copy=False)

        # Adjust to match SPICE's quat format.
        if self.properties.input_data_type == AttitudeInputDataTypes.FLIPSIGN_QUAT:
            table.loc[:, self.properties.input_data_columns[1:4]] *= -1
            self.properties.input_data_type = AttitudeInputDataTypes.SPICE_QUAT

        # Convert input type to the stored type (i.e., UTC).
        #   NOTE: Can't use "ET" with negative times ("-" breaks MSOPCK).
        table.index = pd.Index(
            spicetime.adapt(
                table.index,
                from_=self.properties.input_time_type,
                to=self.properties.writer_time_type.name,  # Name is supported by `spicetime`.
                date_format='%Y-%m-%dT%H:%M:%S.%f'
            ),
            name=self.properties.writer_time_type.value
        )

        # Remove any invalid quaternions (e.g. CTIM right after reset).
        if 'QUAT' in self.properties.input_data_type.name:
            is_invalid = table[self.properties.input_data_columns].sum(axis=1) == 0
            n_invalid = is_invalid.sum()
            if n_invalid > 0:
                dt_min_bad = table[is_invalid].index.min()
                dt_max_bad = table[is_invalid].index.max()
                logger.warning('Removing [%d/%d] quaternions that are invalid (zeros) with min/max range=[%s, %s]',
                               n_invalid, is_invalid.size, dt_min_bad, dt_max_bad)
                table = table[~is_invalid]

                # Was everything invalid?
                if table.size == 0:
                    logger.warning('Table empty after removing [%d] invalid values', n_invalid)
                    return []

        # Create chunks if a gap threshold was configured and gaps are present.
        chunks = self._chunk_table_by_gaps(table, index_dt_type=self.properties.writer_time_type.name)
        return chunks if chunks else [table]

    def write_input_data(self, fobj_or_str, input_data: pd.DataFrame):
        input_data.to_csv(fobj_or_str, sep=' ', header=False)  # , float_format='{:f}'.format)

    def _write_kernel(self, setup_file, input_file, kernel_file, append=False):
        """Setup the command to create a "CK" kernel using the external tool
        "msopck".

        Parameters
        ----------
        setup_file : str
            Setup file containing kernel creation properties (see "msopck").
        input_file : str
            Input data file to create kernel from.
        kernel_file : str
            Kernel filename to create.
        append : bool, optional
            Not supported by msopck (auto-appends). Public wrappers methods add
            support for not appending.

        Returns
        -------
        list of str
            List of subprocess commands and arguments.

        """
        # -----------------------------------------------------------------
        # Developer note: `msopck` is not nearly as flexible as `mkspk` is.
        # -----------------------------------------------------------------
        if os.path.isfile(kernel_file) and not append:
            raise ValueError('"msopck" auto-appends, meaning the kernel file can not exist if append is False.')

        # Full file paths are required by msopck. Use symlinks if the full real
        #   path is too long. Note that msopck always appends, so no keyword is
        #   passed.
        return [self._get_bin('msopck'),
                os.path.abspath(setup_file),
                os.path.abspath(input_file),
                os.path.abspath(kernel_file)]


PROPERTIES_TO_WRITER = {
    AttitudeQuaternionProperties: AttitudeWriter,
    AttitudeEulerProperties: AttitudeWriter,
}

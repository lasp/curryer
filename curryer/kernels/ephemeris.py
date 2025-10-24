"""ephemeris

@author: Brandon Stone
"""

import logging
import os
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from .. import spicetime, spicierpy
from .classes import AbstractKernelProperties, AbstractKernelWriter

logger = logging.getLogger(__name__)


class EphemerisTypes(Enum):
    """Ephemeris kernel types."""

    INTERP_LAGRANGE_EVEN = 8
    INTERP_LAGRANGE_UNEVEN = 9
    INTERP_HERMITE_EVEN = 12
    INTERP_HERMITE_UNEVEN = 13
    TLE = 10

    # MKSPK supports these too...
    # DISCRETE_TWO_BODY = 5
    # PRECESSING_CONIC = 15
    # EQUINOCTIAL_ELEMENTS = 17


class EphemerisInputDataTypes(Enum):
    """Ephemeris input data types."""

    # STATES type must contain records consisting of EPOCH and a full set of
    #   state vector parameters [X, Y, Z, VX, VY, Vz].
    #   If type=17, also all of [DPER/DT, DMPN/DT, DNOD/DT].
    STATES = "STATES"

    # TODO: The following are commented out as they are not fully implemented (properties differ).
    #   Probably should sub class the properties class. Parent with common prop, etc.
    #
    # # ELEMENTS type must contain records consisting of [EPOCH, E, INC, PER, NOD]
    # #   and one of [A, RP, T, P],
    # #   as well as one of [MEAN, EXAN, TRAN, EPOCHP, TAU].
    # #   If type=17, also all of [DPER/DT, DMPN/DT, DNOD/DT].
    # CONIC_ELEMENTS = 'ELEMENTS'
    #
    # # EQ_ELEMENTS type must contain EPOCH and nine equinoctial elements
    # #   [EQ_A, EQ_H, EQ_K, EQ_ML, EQ_P, EQ_Q, DPER/DT, DMPN/DT, DNOD/DT].
    # EQUINOCTIAL_ELEMENTS = 'EQ_ELEMENTS'

    # TL_ELEMENTS type has a pre-defined format and is processed by the program
    #   in a special way. Thus, it ignores many writer properties.
    #   For this, "spice_body" should be the TLE ID, not SPICE ID,
    #   (assumes NAIF_ID = -100000 - TLE_ID),
    #   OR use "TLE_INPUT_OBJ_ID" & "TLE_SPK_OBJ_ID" to define both
    #   (can't use OBJECT_NAME).
    TLE_ELEMENTS = "TL_ELEMENTS"


class EphemerisWriterTimeTypes(Enum):
    """Ephemeris writer time types."""

    UTC = "UTC"
    ET = "ETSECONDS"


@dataclass
class AbstractEphemerisProperties(AbstractKernelProperties):
    """Ephemeris kernel common properties."""

    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = None

    # Input properties.
    input_data_type: str = None
    input_center: str = None
    input_frame: str = None

    # Output properties.
    spk_type: str = None

    # Existing kernels that are required to build this one.
    leapsecond_kernel: str = None
    planet_kernels: list[str] = None  # Optional

    def __post_init__(self):
        super().__post_init__()
        self._update_paths(["planet_kernels"])

    def to_dict(self):
        """Convert the properties class to a dict for creating kernels."""
        config = {}
        if any(val is None for val in (self.input_data_type, self.input_center, self.input_frame, self.spk_type)):
            raise ValueError("Required fields may not be None!")

        # Input properties.
        config["INPUT_DATA_TYPE"] = EphemerisInputDataTypes[self.input_data_type].value
        config["CENTER_ID"] = spicierpy.obj.Body(self.input_center).id
        config["REF_FRAME_NAME"] = spicierpy.obj.Frame(self.input_frame).name

        # Output properties.
        config["OUTPUT_SPK_TYPE"] = EphemerisTypes[self.spk_type].value

        # Required kernels.
        config["LEAPSECONDS_FILE"] = self.leapsecond_kernel
        if self.planet_kernels:
            config["PCK_FILE"] = self.planet_kernels

        config["VERSION"] = self.version
        config["PRODUCER_ID"] = self.author

        # Optional setup arguments.  # TODO: Implement these?
        #   - SEGMENT_ID (str 40 char; default is input filename)
        #   - APPEND_TO_OUTPUT ("NO" or "YES")
        return config


@dataclass
class EphemerisStateProperties(AbstractEphemerisProperties):
    """Ephemeris state kernel properties."""

    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = (EphemerisInputDataTypes.STATES.name,)

    # Input properties.
    input_body: str = None
    input_data_type: str = EphemerisInputDataTypes.STATES.name
    input_data_units: dict[str, str] = field(default_factory=lambda: {"angles": "RADIANS", "distances": "METERS"})
    input_time_type: str = "ugps"
    input_time_columns: list[str] = field(default_factory=lambda: ["ugps"])
    input_data_columns: list[str] = field(
        default_factory=lambda: ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]
    )

    # Output properties.
    spk_type: str = EphemerisTypes.INTERP_HERMITE_UNEVEN.name
    polynom_degree: int = 3

    # Existing kernels that might be required to build this one.
    frame_kernel: str = None

    # Writer properties (change with caution!).
    writer_ignore_lines: int = 1
    writer_lines_per_record: int = 1
    writer_data_delimiter: str = ","
    writer_data_order: str = "epoch x y z vx vy vz"
    writer_time_type: str = EphemerisWriterTimeTypes.UTC.name

    def __post_init__(self):
        super().__post_init__()
        self._update_paths(["frame_kernel"])

    def to_dict(self):
        """Convert the properties class to a dict for creating kernels."""
        config = super().to_dict()
        if any(val is None for val in (self.input_body, self.input_data_columns, self.writer_data_order)):
            raise ValueError("Required fields may not be None!")

        # Input properties.
        config["OBJECT_ID"] = spicierpy.obj.Body(self.input_body).id
        config["INPUT_DATA_UNITS"] = [
            "ANGLES={}".format(self.input_data_units["angles"].upper()),  # TODO: Switch to enum?
            "DISTANCES={}".format(self.input_data_units["distances"].upper()),
        ]

        # Output properties.
        config["POLYNOM_DEGREE"] = self.polynom_degree

        # Required kernels.
        if self.frame_kernel:
            config["FRAME_DEF_FILE"] = self.frame_kernel
        elif spicierpy.frmnam(spicierpy.obj.Frame(self.input_frame).id) == "":
            raise ValueError(
                f'The input Frame {self.input_frame} is not a built-in, so a "frame_kernel" must be supplied.'
            )

        # Time format that spice expects. Internally, user data is converted
        #   from the format at "self.input_data_type" to this.
        config["TIME_WRAPPER"] = f"# {EphemerisWriterTimeTypes(self.writer_time_type).value}"
        config["IGNORE_FIRST_LINE"] = self.writer_ignore_lines
        config["LINES_PER_RECORD"] = self.writer_lines_per_record
        config["DATA_DELIMITER"] = self.writer_data_delimiter
        config["DATA_ORDER"] = self.writer_data_order

        return config


@dataclass
class EphemerisTLEProperties(AbstractEphemerisProperties):
    """Ephemeris state kernel properties."""

    SUPPORTED_INPUT_DATA_TYPES: typing.ClassVar = (EphemerisInputDataTypes.TLE_ELEMENTS.name,)

    # Input properties.
    input_tle: int = None  # TLE S/C ID.
    input_body: str = None  # NOTE: This maps to a diff name on write!
    input_data_type: str = EphemerisInputDataTypes.TLE_ELEMENTS.name

    # Output properties.
    spk_type: str = EphemerisTypes.TLE.name

    # TLE properties. Note: Padding (can't use with START_TIME or STOP_TIME).
    tle_start_pad: str = "12 hours"  # Default if omitted and no START_TIME.
    tle_stop_pad: str = "12 hours"  # Default if omitted and no STOP_TIME.

    # # Writer properties (change with caution!).
    # # TODO: Default generic tmp name?
    # #   Or not required if specified on the command-line?
    # writer_tle_file = TypedDataDescriptor(str, default=True)

    def to_dict(self):
        """Convert the properties class to a dict for creating kernels."""
        config = super().to_dict()
        if any(val is None for val in (self.input_tle, self.input_body)):
            raise ValueError("Required fields may not be None!")

        # Input properties.
        config["TLE_INPUT_OBJ_ID"] = self.input_tle
        config["TLE_SPK_OBJ_ID"] = spicierpy.obj.Body(self.input_body).id  # Note special name (not OBJECT_ID)!

        # TLE properties.
        config["TLE_START_PAD"] = self.tle_start_pad
        config["TLE_STOP_PAD"] = self.tle_stop_pad

        return config


class AbstractEphemerisWriter(AbstractKernelWriter):
    """Create or append to a SPICE ephemeris (SPK) kernel.

    Notes
    -----
    Kernel configuration arguments set by this class:
        - LEAPSECONDS_FILE : __init__, `accessor.properties.spice.kernels`
        - PCK_FILE : __init__, `accessor.properties.spice.kernels`

    """

    KTYPE = "spk"
    FILE_EXT = ".bsp"

    @abstractmethod
    def prepare_input_data(self, input_data: pd.DataFrame) -> list[pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def write_input_data(self, fobj_or_str, accessor):
        raise NotImplementedError

    def _write_kernel(self, setup_file, input_file, kernel_file, append=False):
        """Setup the command to create an "SPK" kernel using the external tool
        "mkspk".

        Parameters
        ----------
        setup_file : str
            Setup file containing kernel creation properties (see "mkspk").
        input_file : str
            Input data file to create kernel from.
        kernel_file : str
            Kernel filename to create.
        append : bool, optional
            Append data to an existing kernel (if exists). Default=False

        Returns
        -------
        list of str
            List of subprocess commands and arguments.

        """
        # Full file paths are required by mkspk. Use symlinks if the full real
        #   path is too long.
        cmd = [
            self._get_bin("mkspk"),
            "-setup",
            os.path.abspath(setup_file),
            "-input",
            os.path.abspath(input_file),
            "-output",
            os.path.abspath(kernel_file),
        ]
        if append:
            cmd.append("-append")
        return cmd


class EphemerisStateWriter(AbstractEphemerisWriter):
    """Create or append to a SPICE ephemeris (SPK) kernel using table-like input
    data sets (e.g. database or CSV file).

    Notes
    -----
    Kernel configuration arguments set by this class:
        - OBJECT_NAME : __init__, `accessor.properties.spice.name`  # TODO: Update
        - LEAPSECONDS_FILE : __init__, `accessor.properties.spice.kernels`
        - PCK_FILE : __init__, `accessor.properties.spice.kernels`
        - IGNORE_FIRST_LINE : __init__
        - LINES_PER_RECORD : __init__
        - DATA_DELIMITER : __init__
        - DATA_ORDER : prepare_input, time position velocity
        - TIME_WRAPPER : prepare_input, UTC string

    """

    def __init__(self, properties: EphemerisStateProperties, **kwargs):
        super().__init__(properties, **kwargs)
        if not isinstance(properties, EphemerisStateProperties):
            raise TypeError("State writer only supports State properties")
        self.properties = properties

    def prepare_input_data(self, input_data: pd.DataFrame) -> list[pd.DataFrame]:
        """Prepare an accessor that will provide input data for the kernel."""
        # Get position x/y/z and velocity x/y/z.
        columns = self.properties.input_time_columns + self.properties.input_data_columns
        table = input_data[[col for col in columns if col in input_data.columns]]
        if table.index.name != self.properties.input_time_columns[0]:
            table = table.set_index(self.properties.input_time_columns)  # self.properties.input_time_type)
        if len(table.columns) == 0:
            raise ValueError(
                f"Input data must have at least one column! Original columns dropped: [{input_data.columns}]"
            )

        # Can't continue if there was no input data.
        if table.size == 0:
            logger.warning("Query returned an empty dataset: %s", table.columns.values)
            return []

        # Pad columns if necessary.
        if len(table.columns) < len(columns) - 1:
            for col in columns:
                if col not in table.columns and col != table.index.name:
                    logger.info("Setting table column [%s] to 0.0", col)
                    table[col] = 0.0

            # Ensure the correct column order (ignore time index).
            if not (columns[1:] == table.columns).all():
                table = table.reindex(columns=columns[1:], copy=False)

        # Convert input type to the stored type (i.e., UTC).
        #   NOTE: Can't use "ET" with negative times ("-" breaks MSOPCK).
        time_type = EphemerisWriterTimeTypes[self.properties.writer_time_type]
        table.index = pd.Index(
            spicetime.adapt(
                table.index,
                from_=self.properties.input_time_type,
                to=time_type.name,  # Name is supported by `spicetime`.
                date_format="%Y-%m-%d %H:%M:%S.%f",
            ),
            name=time_type.value,
        )

        # Create chunks if a gap threshold was configured and gaps are present.
        chunks = self._chunk_table_by_gaps(table, index_dt_type=time_type.name)
        return chunks if chunks else [table]

    def write_input_data(self, fobj_or_str, input_data: pd.DataFrame):
        # Make a basic check that the configuration matches how we write the
        #   data. NOTE: Don't alter the configuration because it might be too
        #   late (i.e., `prepare_setup` was already called).
        if self.properties.writer_ignore_lines != 1:
            raise ValueError('Configuration "IGNORE_FIRST_LINE" should be 1.')
        if self.properties.writer_lines_per_record != 1:
            raise ValueError('Configuration "LINES_PER_RECORD" should be 1.')
        if self.properties.writer_data_delimiter != ",":
            raise ValueError('Configuration "DATA_DELIMITER" should be ",".')

        # Write to file.
        input_data.to_csv(fobj_or_str)


class EphemerisTLEWriter(AbstractEphemerisWriter):
    """Create or append to a SPICE ephemeris (SPK) kernel using TLE.

    Notes
    -----
    Kernel configuration arguments set by this class:
        - LEAPSECONDS_FILE : __init__, `accessor.properties.spice.kernels`
        - PCK_FILE : __init__, `accessor.properties.spice.kernels`

    """

    def __init__(self, properties: EphemerisTLEProperties, **kwargs):
        super().__init__(properties, **kwargs)
        if not isinstance(properties, EphemerisTLEProperties):
            raise TypeError("TLE writer only supports TLE properties")
        self.properties = properties

    def prepare_input_data(self, input_data: pd.DataFrame) -> list[pd.DataFrame]:
        if input_data.size == 0:
            logger.warning("Query returned an empty dataset: %s", input_data.columns.values)
            return []

        # Don't need to worry about chunks since data gaps aren't an issue. The
        # built-in TLE start/stop padding handles that for us.
        return [input_data]

    def write_input_data(self, fobj_or_str, input_data: pd.DataFrame):
        # TODO: Use `tle.TLERemoteAccessor.write`?
        if "tle_line1" not in input_data.columns or "tle_line2" not in input_data.columns:
            raise ValueError('Invalid existing query! Must include "TLE_LINE1" and "TLE_LINE2" columns!')

        if input_data.size == 0:
            raise ValueError("No TLE items to write!")

        # tle_txt = self.properties['mission']['name'].upper() + '\n'
        tle_txt = spicierpy.obj.Body(self.properties.input_body).name.upper() + "\n"
        tle_txt += "\n".join(row["tle_line1"] + "\n" + row["tle_line2"] for _, row in input_data.iterrows())
        tle_txt += "\n"  # SPICE will error out without this.

        if isinstance(fobj_or_str, (str, Path)):
            Path(fobj_or_str).write_text(tle_txt)
        else:
            fobj_or_str.write(tle_txt)


PROPERTIES_TO_WRITER = {
    EphemerisStateProperties: EphemerisStateWriter,
    EphemerisTLEProperties: EphemerisTLEWriter,
}

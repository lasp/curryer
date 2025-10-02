import logging
import typing
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import pandas as pd
import xarray as xr

from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import spatial
from curryer.kernels import create


logger = logging.getLogger(__name__)


class ParameterType(Enum):
    CONSTANT_KERNEL = auto()  # Set a specific value.
    OFFSET_KERNEL = auto()  # Modify input kernel data by an offset.
    OFFSET_TIME = auto()  # Modify input timetags by an offset


@dataclass
class ParameterConfig:
    ptype: ParameterType
    config_file: typing.Optional[Path]
    data: typing.Any


@dataclass
class GeolocationConfig:
    meta_kernel_file: Path
    generic_kernel_dir: Path
    dynamic_kernels: [Path]  # Kernels that are dynamic but *NOT* altered by param!
    instrument_name: str
    time_field: str


@dataclass
class MonteCarloConfig:
    seed: typing.Optional[int]  # Used to make param results reproducible.
    n_iterations: int
    parameters: typing.List[ParameterConfig]
    geo: GeolocationConfig
    # match: ImageMatchConfig
    # stats: ErrorStatsConfig


def load_param_sets(config: MonteCarloConfig) -> [ParameterConfig, typing.Any]:
    assert config.seed
    raise NotImplementedError


def load_telemetry(tlm_key: str, config: MonteCarloConfig) -> pd.DataFrame:
    raise NotImplementedError


def load_science(sci_key: str, config: MonteCarloConfig) -> xr.Dataset:
    raise NotImplementedError


def load_gcp(gcp_key: str, config: MonteCarloConfig) -> xr.Dataset:
    raise NotImplementedError


def apply_offset(config: ParameterConfig, param_data, input_data):
    raise NotImplementedError


def loop(config: MonteCarloConfig, work_dir: Path, tlm_sci_gcp_sets: [(str, str, str)]):
    # Initialize the entire set of parameters.
    params_set = load_param_sets(config)

    # Initialize return data structure...
    results = []  # TODO: Oversimplified...

    # Prepare meta kernel details and kernel writer.
    mkrn = meta.MetaKernel.from_json(
        config.geo.meta_kernel_file, relative=True, sds_dir=config.geo.generic_kernel_dir,
    )
    creator = create.KernelCreator(overwrite=True, append=False)

    # Process each pairing of image data to a GCP.
    for tlm_key, sci_key, gcp_key in tlm_sci_gcp_sets:

        # Load telemetry (L1) telemetry...
        tlm_dataset = load_telemetry(tlm_key, config)

        # Load science (L1A) dataset...
        sci_dataset = load_science(sci_key, config)
        ugps_times = sci_dataset[config.geo.time_field]  # Can be altered by later steps.

        # Load GCP data...
        gcp_dataset = load_gcp(gcp_key, config)

        # Create dynamic unmodified SPICE kernels...
        #   Aka: SC-SPK, SC-CK
        dynamic_kernels = []
        for kernel_config in config.geo.dynamic_kernels:
            dynamic_kernels.append(creator.write_from_json(
                kernel_config, output_kernel=work_dir, input_data=tlm_dataset,
            ))

        # Loop for each parameter set.
        for params in params_set:
            param_kernels = []

            # Apply each individual parameter change.
            for a_param, p_data in params:  # [ParameterConfig, typing.Any]

                # Create static changing SPICE kernels.
                if a_param.ptype == ParameterType.CONSTANT_KERNEL:
                    # Aka: BASE-CK, YOKE-CK, HYSICS-CK
                    param_kernels.append(creator.write_from_json(
                        a_param.config_file, output_kernel=work_dir, input_data=p_data,
                    ))

                # Create dynamic changing SPICE kernels.
                elif a_param.ptype == ParameterType.OFFSET_KERNEL:
                    # Aka: AZ-CK, EL-CK
                    tlm_dataset_alt = apply_offset(a_param, p_data, tlm_dataset)
                    param_kernels.append(creator.write_from_json(
                        a_param.config_file, output_kernel=work_dir, input_data=tlm_dataset_alt,
                    ))

                # Alter non-kernel data.
                elif a_param.ptype == ParameterType.OFFSET_TIME:
                    # Aka: Frame-times...
                    sci_dataset_alt = apply_offset(a_param, p_data, sci_dataset)
                    ugps_times = sci_dataset_alt[config.geo.time_field].values

                else:
                    raise NotImplementedError(a_param.ptype)

            # Geolocate.
            with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, dynamic_kernels, param_kernels]):
                geoloc_inst = spatial.Geolocate(config.geo.instrument_name)
                geo_dataset = geoloc_inst(ugps_times)

                # Image matching.
                #   sci_dataset...
                #   geo_dataset...
                #   gcp_dataset...
                #   ...match_dataset
                pass

                # Error stats.
                #   match_dataset...
                #   ...stats_dataset
                pass

                # Store results based on param_id & pair_id.
                #   params...
                #   sci_dataset...
                #   stats_dataset...
                #   ...result_dataset
                results.append((params, geo_dataset))  # TODO: Too simple, just for demo...

    return results

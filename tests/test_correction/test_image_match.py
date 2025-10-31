"""
Tests for image_match.py module

This module tests the integrated image matching functionality, including:
- Image grid loading from MATLAB files
- Geolocation error application for testing
- Image matching algorithm validation
- Cross-correlation and grid search

Running Tests:
-------------
# Via pytest (recommended)
pytest tests/test_correction/test_image_match.py -v

# Run specific test
pytest tests/test_correction/test_image_match.py::TestImageMatch::test_integrated_match_case_1 -v

# Standalone execution
python tests/test_correction/test_image_match.py

Requirements:
-----------------
These tests validate image matching algorithms against known test cases,
demonstrating that the Python implementation correctly identifies geolocation
errors through image cross-correlation.
"""

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import xarray as xr
from scipy.io import loadmat

from curryer import utils
from curryer.compute import constants
from curryer.correction.data_structures import (
    GeolocationConfig,
    ImageGrid,
    OpticalPSFEntry,
    SearchConfig,
)
from curryer.correction.image_match import integrated_image_match

logger = logging.getLogger(__name__)
utils.enable_logging(log_level=logging.INFO, extra_loggers=[__name__])

xr.set_options(display_width=120, display_max_rows=30)
np.set_printoptions(linewidth=120)


def image_grid_from_struct(mat_struct):
    return ImageGrid(
        data=np.asarray(mat_struct.data),
        lat=np.asarray(mat_struct.lat),
        lon=np.asarray(mat_struct.lon),
        h=np.asarray(mat_struct.h) if hasattr(mat_struct, "h") else None,
    )


def great_circle_displacement_deg(lat_km: float, lon_km: float, reference_lat_deg: float) -> tuple[float, float]:
    """Convert kilometer offsets to degree offsets using local radius of curvature."""
    earth_radius_km = constants.WGS84_SEMI_MAJOR_AXIS_KM
    lat_offset_deg = lat_km / earth_radius_km * (180.0 / np.pi)
    lon_radius_km = earth_radius_km * np.cos(np.deg2rad(reference_lat_deg))
    lon_offset_deg = lon_km / lon_radius_km * (180.0 / np.pi)
    return lat_offset_deg, lon_offset_deg


def apply_geolocation_error(subimage: ImageGrid, gcp: ImageGrid, lat_error_km: float, lon_error_km: float):
    """Return a copy of the subimage with imposed geolocation error."""
    mid_lat = float(gcp.lat[gcp.lat.shape[0] // 2, gcp.lat.shape[1] // 2])
    lat_offset_deg, lon_offset_deg = great_circle_displacement_deg(
        lat_error_km, lon_error_km, mid_lat
    )
    return ImageGrid(
        data=subimage.data.copy(),
        lat=subimage.lat + lat_offset_deg,
        lon=subimage.lon + lon_offset_deg,
        h=subimage.h.copy() if subimage.h is not None else None,
    )


class ImageMatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parent.parent.parent
        self.test_dir = root_dir / 'tests' / 'data' / 'clarreo' / 'image_match'
        self.assertTrue(self.test_dir.is_dir(), self.test_dir)

        self.__tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

    @staticmethod
    def run_image_match(subimg_file, gcp_file, ancil_file, psf_file, pix_vec_file, lat_lon_err):
        subimage_struct = loadmat(subimg_file, squeeze_me=True, struct_as_record=False)["subimage"]
        subimage = image_grid_from_struct(subimage_struct)

        gcp_struct = loadmat(gcp_file, squeeze_me=True, struct_as_record=False)["GCP"]
        gcp = image_grid_from_struct(gcp_struct)

        los_vectors = loadmat(pix_vec_file, squeeze_me=True)["b_HS"]
        r_iss = loadmat(ancil_file, squeeze_me=True)["R_ISS_midframe"].ravel()
        psf_struct = loadmat(psf_file, squeeze_me=True, struct_as_record=False)["PSF_struct_675nm"]

        psf_entries = [
            OpticalPSFEntry(
                data=np.asarray(entry.data),
                x=np.asarray(entry.x).ravel(),
                field_angle=np.asarray(entry.FA).ravel(),
            )
            for entry in np.atleast_1d(psf_struct)
        ]

        subimage_with_error = apply_geolocation_error(subimage, gcp, lat_lon_err[0], lat_lon_err[1])

        result = integrated_image_match(
            subimage=subimage_with_error,
            gcp=gcp,
            r_iss_midframe_m=r_iss,
            los_vectors_hs=los_vectors,
            optical_psfs=psf_entries,
            # lat_error_km=lat_lon_err[0],
            # lon_error_km=lat_lon_err[1],
            geolocation_config=GeolocationConfig(),
            search_config=SearchConfig(),
        )

        logger.info(f'Image file: {subimg_file.name}')
        logger.info(f'GCP file: {gcp_file.name}')
        logger.info(f'Lat Error (km): {result.lat_error_km:+6.3f} (exp={lat_lon_err[0]:+6.3f})')
        logger.info(f'Lon Error (km): {result.lon_error_km:+6.3f} (exp={lat_lon_err[1]:+6.3f})')
        logger.info(f'   CCV (final): {result.ccv_final}')
        logger.info(f'   Pixel (row): {result.final_index_row}')
        logger.info(f'   Pixel (col): {result.final_index_col}')
        return result

    def test_case_1a_unbinned(self):
        lat_lon_err = (3.0, -3.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "1" / "TestCase1a_subimage.mat",
            gcp_file=self.test_dir / "1" / "GCP12055Dili_resampled.mat",
            ancil_file=self.test_dir / "1" / "R_ISS_midframe_TestCase1.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 22)

    def test_case_1b_unbinned(self):
        lat_lon_err = (3.0, -3.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "1" / "TestCase1b_subimage.mat",
            gcp_file=self.test_dir / "1" / "GCP12055Dili_resampled.mat",
            ancil_file=self.test_dir / "1" / "R_ISS_midframe_TestCase1.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 21)
        np.testing.assert_allclose(result.final_index_col, 22)

    def test_case_1c_binned(self):
        lat_lon_err = (3.0, -3.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "1" / "TestCase1c_subimage_binned.mat",
            gcp_file=self.test_dir / "1" / "GCP12055Dili_resampled.mat",
            ancil_file=self.test_dir / "1" / "R_ISS_midframe_TestCase1.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 21)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_1d_binned(self):
        lat_lon_err = (3.0, -3.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "1" / "TestCase1d_subimage_binned.mat",
            gcp_file=self.test_dir / "1" / "GCP12055Dili_resampled.mat",
            ancil_file=self.test_dir / "1" / "R_ISS_midframe_TestCase1.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_2a_unbinned(self):
        lat_lon_err = (-3.0, 2.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "2" / "TestCase2a_subimage.mat",
            gcp_file=self.test_dir / "2" / "GCP10121Maracaibo_resampled.mat",
            ancil_file=self.test_dir / "2" / "R_ISS_midframe_TestCase2.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_2b_unbinned(self):
        lat_lon_err = (-3.0, 2.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "2" / "TestCase2b_subimage.mat",
            gcp_file=self.test_dir / "2" / "GCP10121Maracaibo_resampled.mat",
            ancil_file=self.test_dir / "2" / "R_ISS_midframe_TestCase2.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_2c_binned(self):
        lat_lon_err = (-3.0, 2.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "2" / "TestCase2c_subimage_binned.mat",
            gcp_file=self.test_dir / "2" / "GCP10121Maracaibo_resampled.mat",
            ancil_file=self.test_dir / "2" / "R_ISS_midframe_TestCase2.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_2d_binned(self):
        lat_lon_err = (-3.0, 2.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "2" / "TestCase2d_subimage_binned.mat",
            gcp_file=self.test_dir / "2" / "GCP10121Maracaibo_resampled.mat",
            ancil_file=self.test_dir / "2" / "R_ISS_midframe_TestCase2.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_3a_unbinned(self):
        lat_lon_err = (1.0, 1.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "3" / "TestCase3a_subimage.mat",
            gcp_file=self.test_dir / "3" / "GCP10665SantaRosa_resampled.mat",
            ancil_file=self.test_dir / "3" / "R_ISS_midframe_TestCase3.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_3b_unbinned(self):
        lat_lon_err = (1.0, 1.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "3" / "TestCase3b_subimage.mat",
            gcp_file=self.test_dir / "3" / "GCP10665SantaRosa_resampled.mat",
            ancil_file=self.test_dir / "3" / "R_ISS_midframe_TestCase3.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_3c_binned(self):
        lat_lon_err = (1.0, 1.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "3" / "TestCase3c_subimage_binned.mat",
            gcp_file=self.test_dir / "3" / "GCP10665SantaRosa_resampled.mat",
            ancil_file=self.test_dir / "3" / "R_ISS_midframe_TestCase3.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.07)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.07)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_3d_binned(self):
        lat_lon_err = (1.0, 1.0)
        result = self.run_image_match(
            subimg_file=self.test_dir / "3" / "TestCase3d_subimage_binned.mat",
            gcp_file=self.test_dir / "3" / "GCP10665SantaRosa_resampled.mat",
            ancil_file=self.test_dir / "3" / "R_ISS_midframe_TestCase3.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.07)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.07)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 20)

    def test_case_4a_unbinned(self):
        lat_lon_err = (-1.0, -2.5)
        result = self.run_image_match(
            subimg_file=self.test_dir / "4" / "TestCase4a_subimage.mat",
            gcp_file=self.test_dir / "4" / "GCP20484Morocco_resampled.mat",
            ancil_file=self.test_dir / "4" / "R_ISS_midframe_TestCase4.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 22)

    def test_case_4b_unbinned(self):
        lat_lon_err = (-1.0, -2.5)
        result = self.run_image_match(
            subimg_file=self.test_dir / "4" / "TestCase4b_subimage.mat",
            gcp_file=self.test_dir / "4" / "GCP20484Morocco_resampled.mat",
            ancil_file=self.test_dir / "4" / "R_ISS_midframe_TestCase4.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_4c_binned(self):
        lat_lon_err = (-1.0, -2.5)
        result = self.run_image_match(
            subimg_file=self.test_dir / "4" / "TestCase4c_subimage_binned.mat",
            gcp_file=self.test_dir / "4" / "GCP20484Morocco_resampled.mat",
            ancil_file=self.test_dir / "4" / "R_ISS_midframe_TestCase4.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 21)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_4d_binned(self):
        lat_lon_err = (-1.0, -2.5)
        result = self.run_image_match(
            subimg_file=self.test_dir / "4" / "TestCase4d_subimage_binned.mat",
            gcp_file=self.test_dir / "4" / "GCP20484Morocco_resampled.mat",
            ancil_file=self.test_dir / "4" / "R_ISS_midframe_TestCase4.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 23)

    def test_case_5a_unbinned(self):
        lat_lon_err = (2.5, 0.1)
        result = self.run_image_match(
            subimg_file=self.test_dir / "5" / "TestCase5a_subimage.mat",
            gcp_file=self.test_dir / "5" / "GCP10087Titicaca_resampled.mat",
            ancil_file=self.test_dir / "5" / "R_ISS_midframe_TestCase5.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 23)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_5b_unbinned(self):
        lat_lon_err = (2.5, 0.1)
        result = self.run_image_match(
            subimg_file=self.test_dir / "5" / "TestCase5b_subimage.mat",
            gcp_file=self.test_dir / "5" / "GCP10087Titicaca_resampled.mat",
            ancil_file=self.test_dir / "5" / "R_ISS_midframe_TestCase5.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 21)

    def test_case_5c_binned(self):
        lat_lon_err = (2.5, 0.1)
        result = self.run_image_match(
            subimg_file=self.test_dir / "5" / "TestCase5c_subimage_binned.mat",
            gcp_file=self.test_dir / "5" / "GCP10087Titicaca_resampled.mat",
            ancil_file=self.test_dir / "5" / "R_ISS_midframe_TestCase5.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 22)

    def test_case_5d_binned(self):
        lat_lon_err = (2.5, 0.1)
        result = self.run_image_match(
            subimg_file=self.test_dir / "5" / "TestCase5d_subimage_binned.mat",
            gcp_file=self.test_dir / "5" / "GCP10087Titicaca_resampled.mat",
            ancil_file=self.test_dir / "5" / "R_ISS_midframe_TestCase5.mat",
            psf_file=self.test_dir / "optical_PSF_675nm_3_pix_binned_upsampled.mat",
            pix_vec_file=self.test_dir / "b_HS.mat",
            lat_lon_err=lat_lon_err,
        )
        np.testing.assert_allclose(result.lat_error_km, lat_lon_err[0], atol=0.05)
        np.testing.assert_allclose(result.lon_error_km, lat_lon_err[1], atol=0.05)
        np.testing.assert_allclose(result.ccv_final, 1.0, atol=0.01)
        np.testing.assert_allclose(result.final_index_row, 22)
        np.testing.assert_allclose(result.final_index_col, 22)


if __name__ == '__main__':
    unittest.main()

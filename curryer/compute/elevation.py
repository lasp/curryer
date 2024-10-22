"""Elevation computations and DEM management.

References
----------
GMTED2010
    https://pubs.usgs.gov/of/2011/1073/pdf/of2011-1073.pdf
UNAVCO Geoid Height Calculator
    https://www.unavco.org/software/geodetic-utilities/geoid-height-calculator/geoid-height-calculator.html

@author: Brandon Stone
"""
import logging
import os
import re
from pathlib import Path
from typing import List, Union, Optional, Tuple, Set

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from pyproj import Transformer
from pyproj.transformer import TransformerGroup

from . import constants
from ..utils import track_performance


logger = logging.getLogger(__name__)

# Unique vertical datums in GMTED2010 (by area):
VDATUMS = {'EGM96', 'MSL', 'CVGD28', 'NGVD29', 'WGS 84', 'NAVD88', 'AHD'}
VDATUM_TO_EPSG = {
    'EGM96': 5773,  # Vert.
    'MSL': 5714,  # Vert.
    'CVGD28': 5713,  # Vert. (Source typo? EPSG says "CGVD28").
    'NGVD29': 7968,  # Vert.
    'WGS 84': 4979,  # 3D.
    'NAVD88': 5703,  # Vert.
    'AHD': 5711,  # Vert.
}


class Elevation:
    """High-level class for accessing elevation data from individual files.
    """

    LON_STEP = 30
    LAT_STEP = 20
    LON_MID = -180 % LON_STEP
    LAT_MID = -90 % LAT_STEP

    def __init__(self, data_dir: Union[str, Path] = None, meters=False, degrees=True):
        """Initialize elevation data handling.

        Parameters
        ----------
        data_dir: Path or str, optional
            Directory to scan and load DEM files from. Defaults to the
            environment variable "CURRYER_DATA_DIR" plus "/gmted", if set,
            otherwise the directory relative to the libraries root of
            "data/gmted".
        meters : bool, optional
            Assume in/out x/y/z/alt values are in meters. Default is False.
        degrees : bool, optional
            Assume in/out lon/lat values are in degrees. Default is True

        """
        if data_dir is None:
            data_dir = Path(os.getenv('CURRYER_DATA_DIR', Path(__file__).parents[2] / 'data')) / 'gmted'
            logger.info('Default elevation data directory: [%s]', data_dir)
        else:
            data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(f'Missing data directory [{data_dir}]! Use env var "CURRYER_DATA_DIR".')

        self.data_dir = data_dir
        self.meters = meters
        self.degrees = degrees
        self._files = None
        self._metadata = None
        self._cache = {}

        self.check_for_geoid_data()
        self.egm96_to_wgs84 = Transformer.from_crs(
            'EPSG:4326+5773', 'EPSG:4979', always_xy=True, only_best=True, allow_ballpark=False
        ).transform

    @staticmethod
    def check_for_geoid_data():
        """Ensure that third-party geoid data is available.
        """
        # EGM96.
        tg = TransformerGroup('EPSG:4326+5773', 'EPSG:4979')
        if len(tg.unavailable_operations):
            # Download extra data layers if missing.
            logger.info('Downloading EGM96 geoid data for pyproj transformer.')
            tg.download_grids(verbose=True)

    @track_performance
    def locate_files(self, pattern: str = '*_gmted_*.tif') -> List[Path]:
        """Locate DEM files on the filesystem (cached).

        Parameters
        ----------
        pattern : str, optional
            File name pattern to filter DEM files with.

        Returns
        -------
        list[Path]
            List of found DEM files. Cached after first execution.

        """
        if self._files is None:
            self._files = sorted(self.data_dir.glob(pattern))
        return self._files

    @track_performance
    def describe_files(self) -> pd.DataFrame:
        """Compute the metadata for the available DEM files (cached).

        Returns
        -------
        pd.DataFrame
            Metadata for the individual DEM files. Cached after first execution.

        """
        if self._metadata is not None:
            return self._metadata

        regex = re.compile(r'(?P<lat0>[0-9]{2})(?P<lat1>[NS])(?P<lon0>[0-9]{3})(?P<lon1>[EW])'
                           r'_20101117_gmted_(?P<stat>[a-z]+)(?P<res>[0-9]{3}).tif', re.I)
        metadata = []
        for filepath in self.locate_files():
            match = regex.fullmatch(filepath.name)
            if match is None:
                raise ValueError(f'Invalid file name pattern: {filepath}')

            lat = float(match.group('lat0'))
            if match.group('lat1').upper() == 'S':
                lat *= -1

            lon = float(match.group('lon0'))
            if match.group('lon1').upper() == 'W':
                lon *= -1

            metadata.append(dict(
                ll_lon=lon,
                ll_lat=lat,
                ur_lon=lon + self.LON_STEP,
                ur_lat=lat + self.LAT_STEP,
                arcsec=float(match.group('res')) / 10,
                stat=match.group('stat'),
                name=filepath,
            ))

        self._metadata = pd.DataFrame(metadata)

        # Set a multi-dim index for faster lookups (~8x faster).
        self._metadata = self._metadata.set_index(['ll_lon', 'll_lat', 'stat', 'arcsec']).sort_index()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Generated DEM metadata:\n%s', self._metadata.to_string())
        return self._metadata

    @track_performance
    def lookup_file(self, lon: float, lat: float, arcsec: float = None, stat: str = 'mea', wrap_lon=False,
                    degrees: Union[bool, None] = None) -> Optional[pd.Series]:
        """Find the DEM file for a given lon/lat point.

        Parameters
        ----------
        lon : float
            Longitude in degrees or radians, see `degrees`.
        lat : float
            Latitude in degrees or radians, see `degrees`.
        arcsec : float, optional
            Select the DEM resolution. Default is the highest available.
        stat : str, optional
            Select the type of DEM file. Default is "mea" (mean).
        wrap_lon : bool, optional
            Allow longitude values less than -180 and greater than 180,
            wrapping them around the globe (e.g., 190 = -170). Default is False.
        degrees : None or bool, optional
            Assume in/out lon/lat are in degrees. If not specified or None
            (default), uses the instance value of `degrees` (default is True).

        Returns
        -------
        pd.Series or None
            Matching DEM metadata entry or None if no match was found.

        """
        if self._metadata is None:
            self.describe_files()

        if degrees is None:
            degrees = self.degrees
        if not degrees:
            lon = np.rad2deg(lon)
            lat = np.rad2deg(lat)

        # Option to wrap longitudes beyond +/- 180 deg. Don't default to this
        # because the user needs to be aware of the file bounds not numerically
        # overlapping the input lon.
        if wrap_lon:
            if lon < -180:
                lon %= 360
            elif lon >= 180:
                lon %= -360

        # Look up files with a matching lower-left corner.
        try:
            match = self._metadata.loc[(
                lon - (lon - self.LON_MID) % self.LON_STEP,
                lat - (lat - self.LAT_MID) % self.LAT_STEP,
                stat
            )]
        except KeyError:
            return

        if match.size == 0:
            return

        # Default to the highest resolution if multiple are available. Assumes
        # the separate file description step sorts by ascending arcsec.
        if arcsec is not None:
            if match.shape[0] == 1:
                return match.iloc[0] if arcsec in match.index else None

            return match.loc[arcsec] if arcsec in match.index else None

        if match.ndim == 1:
            return match
        return match.iloc[0]

    @track_performance
    def load_file(self, filepath: Path) -> xr.DataArray:
        """Load data from a DEM file (cached).

        Parameters
        ----------
        filepath : Path
            Path to the DEM file to load.

        Returns
        -------
        xr.DataArray
            Loaded elevation data. Cached for each unique file path.

        """
        name = filepath.name
        if name not in self._cache:
            logger.info('Reading DEM file: %s', filepath)
            self._cache[name] = rioxarray.open_rasterio(filepath).sel(band=1)
        return self._cache[name]

    @track_performance
    def get_geoid_height(self, lon: Union[np.ndarray, float], lat: Union[np.ndarray, float],
                         degrees: Union[bool, None] = None) -> Union[np.ndarray, float]:
        """Query the geoid height for a given lon/lat(s).

        Parameters
        ----------
        lon : np.ndarray or float
            Longitude in degrees or radians, see `degrees`.
        lat : np.ndarray or float
            Latitude in degrees or radians, see `degrees`.
        degrees : None or bool, optional
            Assume in/out lon/lat are in degrees. If not specified or None
            (default), uses the instance value of `degrees` (default is True).

        Returns
        -------
        np.ndarray or float
            Geoid height above the ellipsoid. Default is kilometers unless
            init variable `meters` was True.

        """
        # TODO: Future improvements:
        #   1) Determine if a lesser used geoid needs to be used.
        if degrees is None:
            degrees = self.degrees
        if not degrees:
            lon = np.rad2deg(lon)
            lat = np.rad2deg(lat)
        if np.isscalar(lon):
            _, _, hts = self.egm96_to_wgs84(lon, lat, 0.0)
        else:
            _, _, hts = self.egm96_to_wgs84(lon, lat, np.zeros(len(lon)))
        return hts if self.meters else hts / 1e3

    @track_performance
    def get_dem_height(self, lon: float, lat: float, filepath: Path, degrees: Union[bool, None] = None) -> float:
        """Query the DEM height above the geoid for a given lon/lat.

        Parameters
        ----------
        lon : float
            Longitude in degrees or radians, see `degrees`.
        lat : float
            Latitude in degrees or radians, see `degrees`.
        filepath : Path
            DEM file to query data from.
        degrees : None or bool, optional
            Assume in/out lon/lat are in degrees. If not specified or None
            (default), uses the instance value of `degrees` (default is True).

        Returns
        -------
        float
            DEM height above the geoid. Default is kilometers unless
            init variable `meters` was True.

        """
        # TODO: Future improvements:
        #   1) Support arrays of lon/lat (sel returns every combo, not pairs?).
        #   2) Handle fill values now instead of later (`rds._FillValue`)?
        if degrees is None:
            degrees = self.degrees
        if not degrees:
            lon = np.rad2deg(lon)
            lat = np.rad2deg(lat)
        rds = self.load_file(filepath)
        hts = rds.sel(x=lon, y=lat, method='nearest')
        hts = hts.item()
        return hts if self.meters else hts / 1e3

    @track_performance
    def query(self, lon, lat, orthometric=False):
        """Query the surface (DEM + geoid) height above the ellipsoid a given
        lon/lat.

        Parameters
        ----------
        lon : float
            Longitude in degrees or radians, see init `degrees`.
        lat : float
            Latitude in degrees or radians, see init `degrees`.
        orthometric : bool, optional
            Exclude the geoid height as part of the surface height. Default is
            False.

        Returns
        -------
        float
            Surface (DEM and geoid (unless orthometric=True) height above the
            ellipsoid. Default is kilometers unless init variable `meters` was
            True.

        """
        match = self.lookup_file(lon, lat, stat='mea')
        if match is None:
            raise ValueError(f'Failed to find DEM file for lon=[{lon}], lat=[{lat}], stat=[mea]')

        dem_ht = self.get_dem_height(lon, lat, match['name'])
        if orthometric:
            return dem_ht

        geo_ht = self.get_geoid_height(lon, lat)
        return geo_ht + dem_ht

    def _pad_lonlat(self, lon: float, lat: float, pad: float) -> Tuple[float, float, float, float]:
        """Intelligently pad lon/lat values with a meter/kilometer distance.
        """
        if self.meters:
            pad /= 1e3
        km_per_deg = 2 * np.pi * constants.WGS84_SEMI_MAJOR_AXIS_KM / 360  # At equator.
        pad /= km_per_deg
        if not self.degrees:
            pad = np.deg2rad(pad)

        min_lon = lon - pad
        max_lon = lon + pad
        min_lat = lat - pad
        max_lat = lat + pad
        return min_lon, max_lon, min_lat, max_lat

    def _regional_files(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float, allow_empty=False
                        ) -> Set[Path]:
        """Find all DEM files within a min/max lon/lat region.
        """
        if not self.degrees:
            min_lon, max_lon, min_lat, max_lat = np.rad2deg((min_lon, max_lon, min_lat, max_lat))

        start_lon = min_lon - (min_lon - self.LON_MID) % self.LON_STEP
        start_lat = min_lat - (min_lat - self.LAT_MID) % self.LAT_STEP

        # Special case of wrapping the dateline.
        if min_lon > max_lon:
            stop_lon = 360 + max_lon
        else:
            stop_lon = max_lon

        files = set()
        for alon in np.arange(start_lon, stop_lon, self.LON_STEP):
            for alat in np.arange(start_lat, max_lat, self.LAT_STEP):
                # TODO: Request lowest available res? Defaults highest.
                match = self.lookup_file(alon, alat, stat='mea', wrap_lon=True, degrees=True)
                if match is not None:
                    files.add(match['name'])

        if not allow_empty and not files:
            raise ValueError(f'Invalid lon/lat [{min_lon}:{max_lon}, {min_lat}:{max_lat}]. No DEMs intersect region!')

        return files

    def _slice_file(self, rds: xr.Dataset, min_lon: float, max_lon: float, min_lat: float, max_lat: float,
                    orthometric=False, fill_val: float = None) -> xr.DataArray:
        """Subset a file by a min/max lon/lat region.
        """
        # Edge case that longitude wrapped the international dateline.
        if rds['x'].min().item() >= max_lon:
            dem_hts = rds.rio.slice_xy(minx=min_lon % 360, maxx=180, miny=min_lat, maxy=max_lat)

        elif rds['x'].max().item() < min_lon:
            dem_hts = rds.rio.slice_xy(minx=-180, maxx=max_lon % -360, miny=min_lat, maxy=max_lat)

        else:
            dem_hts = rds.rio.slice_xy(minx=min_lon, maxx=max_lon, miny=min_lat, maxy=max_lat)

        # Handle fill-values.
        if fill_val is not None and fill_val != rds.attrs['_FillValue']:
            raise ValueError(f'DEM files have different fill values! [{fill_val}] != [{rds.attrs["_FillValue"]}]')

        dem_hts = dem_hts.where(dem_hts != rds.attrs['_FillValue'], other=0.0)

        if not self.meters:
            dem_hts = dem_hts / 1e3

        if orthometric:
            return dem_hts

        # Compute the geoid heights.
        x = dem_hts['x'].values
        y = dem_hts['y'].values

        xx = np.tile(x, (y.size, 1))
        yy = np.tile(y, (x.size, 1)).T

        geoid_hts = self.get_geoid_height(xx.ravel(), yy.ravel(), degrees=True)
        geoid_hts = geoid_hts.reshape(xx.shape)

        ellps_hts = dem_hts + geoid_hts
        return ellps_hts

    @track_performance
    def local_minmax(self, lon, lat, pad, orthometric=False) -> Tuple[float, float]:
        """Compute the elevation min/max around a buffered point.

        Parameters
        ----------
        lon : float
            Longitude in degrees or radians, see init `degrees`.
        lat : float
            Latitude in degrees or radians, see init `degrees`.
        pad : float
            Padding to add to the point, in meters or kilometers, see init
            `meters`.
        orthometric : bool, optional
            Exclude the geoid heights from the result.

        Returns
        -------
        (float, float)
            Min and max elevation around the specified padding point.

        """
        min_lon, max_lon, min_lat, max_lat = self._pad_lonlat(lon, lat, pad)
        files = self._regional_files(min_lon, max_lon, min_lat, max_lat, allow_empty=False)

        ellps_minmax = None
        for filepath in files:
            rds = self.load_file(filepath)
            ellps_hts = self._slice_file(rds, min_lon, max_lon, min_lat, max_lat, orthometric=orthometric)

            local_minmax = (ellps_hts.min(skipna=True).item(),
                            ellps_hts.max(skipna=True).item())

            if ellps_minmax is None:
                ellps_minmax = local_minmax
            else:
                ellps_minmax = (min(ellps_minmax[0], local_minmax[0]),
                                max(ellps_minmax[1], local_minmax[1]))

        return ellps_minmax

    @track_performance
    def local_region(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float, orthometric=False):
        """Create a regional subset, supporting SIGNIFICANTLY faster queries.

        Parameters
        ----------
        min_lon : float
            Minimum longitude extent of the region, see init `degrees`.
        max_lon : float
            Maximum longitude extent of the region, see init `degrees`.
        min_lat : float
            Minimum latitude extent of the region, see init `degrees`.
        max_lat : float
            Maximum latitude extent of the region, see init `degrees`.
        orthometric : bool, optional
            Exclude the geoid heights from the result.

        Returns
        -------
        ElevationRegion
            Elevation subset that only supports queries within its region, but
            SIGNIFICANTLY faster within the region!

        """
        files = self._regional_files(min_lon, max_lon, min_lat, max_lat, allow_empty=False)
        files = sorted(files)

        if not self.degrees:
            min_lon, max_lon, min_lat, max_lat = np.rad2deg((min_lon, max_lon, min_lat, max_lat))

        # Load the height arrays for each file.
        fill_val = None
        tiles = []
        resolutions = []
        for filepath in files:
            rds = self.load_file(filepath)
            tile_hts = self._slice_file(
                rds, min_lon, max_lon, min_lat, max_lat, orthometric=orthometric, fill_val=fill_val,
            )
            resolutions.append(rds.rio.resolution())
            fill_val = rds.attrs['_FillValue']

            # Special case of wrapping the dateline.
            if min_lon > max_lon and tile_hts['x'].max().item() < min_lon:
                tile_hts['x'] = tile_hts['x'] + 360

            tiles.append(tile_hts)

        # Handle rare case of mismatched resolutions (near poles).
        if not all(resolutions[0][0] == x and resolutions[0][1] == y for x, y in resolutions):

            xmin_res = min(x for x, y in resolutions)  # Positive.
            ymin_res = max(y for x, y in resolutions)  # Negative.

            bounds = []
            xmin, xmax = None, None
            ymin, ymax = None, None
            for ith, (xres, yres) in enumerate(resolutions):
                tile = tiles[ith]
                bounds.append((
                    tile['x'].min().item() - xres / 2,
                    tile['x'].max().item() + xres / 2,
                    tile['y'].min().item() + yres / 2,
                    tile['y'].max().item() - yres / 2,
                ))
                if ith == 0:
                    xmin, xmax, ymin, ymax = bounds[-1]
                else:
                    xmin = min(xmin, bounds[-1][0])
                    xmax = max(xmax, bounds[-1][1])
                    ymin = min(ymin, bounds[-1][2])
                    ymax = max(ymax, bounds[-1][3])

            xcoord = np.arange(xmin + xmin_res / 2, xmax, xmin_res)  # Increasing.
            ycoord = np.arange(ymax + ymin_res / 2, ymin, ymin_res)  # Decreasing.

            new_tiles = []
            for ith, ((xmin, xmax, ymin, ymax), tile) in enumerate(zip(bounds, tiles)):
                ix = (xcoord > xmin) & (xcoord < xmax)
                iy = (ycoord > ymin) & (ycoord < ymax)
                new_tiles.append(tile.interp(
                    dict(x=xcoord[ix], y=ycoord[iy]), method='nearest', kwargs=dict(fill_value='extrapolate')
                ))
            tiles = new_tiles

        hts = xr.combine_by_coords(tiles, combine_attrs='drop_conflicts')

        if not self.degrees:
            hts['x'] = np.deg2rad(hts['x'])
            hts['y'] = np.deg2rad(hts['y'])

        return ElevationRegion.from_xarray(
            hts, orthometric=orthometric, degrees=self.degrees, meters=self.meters,
        )


class ElevationRegion:
    """High-level class for accessing elevation data from an optimized
    pre-cached dataset.
    """

    def __init__(self, data: np.ndarray, ulx: float, uly: float, dx: float, dy: float, orthometric=None,
                 meters: Union[bool, None] = None, degrees: Union[bool, None] = None):
        """Initialize the region's elevation data.

        Parameters
        ----------
        data : np.ndarray
        ulx : float
            Upper left longitude point.
        uly : float
            Upper left latitude point.
        dx : float
            Longitude pixel width.
        dy : float
            Latitude pixel width.
        orthometric : bool, optional
            Exclude the geoid heights from the result.
        meters : bool, optional
            Height in kilometers or meters.
        degrees : bool, optional
            Expect lon/lat queries in degrees or radians.

        """
        self.data = data
        self.ulx = ulx
        self.uly = uly
        self.dx = dx
        self.dy = dy

        self.orthometric = orthometric
        self.meters = meters
        self.degrees = degrees

        self.lrx = ulx + dx * data.shape[1]
        self.lry = uly + dy * data.shape[0]
        self._wrap = 360 if degrees else np.deg2rad(360)

    def __repr__(self):
        return (f'{self.__class__.__name__}(ul=[{self.ulx:.6f}, {self.uly:.6f}], lr=[{self.lrx:.6f}, {self.lry:.6f}]'
                f', res=[{self.dx:.6f}, {self.dy:.6f}], shape=[{self.data.shape}])')

    @classmethod
    def from_xarray(cls, hts: xr.DataArray, **kwargs):
        """Create a region from an xarray data array.

        Parameters
        ----------
        hts : xr.DataArray
            Elevation data with lon/lat coordinates (x/y).
        kwargs : dict
            See init arguments.

        Returns
        -------
        ElevationRegion
            Elevation region from an xarray data array.

        """
        if 'x' not in hts.dims or 'y' not in hts.dims:
            raise ValueError(f'Unable to create a region without "x" and "y" dimensions: {hts.dims}')
        if hts.sizes['x'] == 0 or hts.sizes['y'] == 0:
            raise ValueError(f'Unable to create a region with zero-sized dimension: {hts.sizes}')

        dx = hts['x'].values[1] - hts['x'].values[0]
        dy = hts['y'].values[1] - hts['y'].values[0]

        ulx = hts['x'].values[0]
        uly = hts['y'].values[0]

        data = hts.values

        return cls(data, ulx, uly, dx, dy, **kwargs)

    @track_performance
    def query(self, lon: Union[np.ndarray, float], lat: Union[np.ndarray, float], orthometric=None
              ) -> Union[np.ndarray, float]:
        """Query the surface (DEM + geoid) height above the ellipsoid a given
        lon/lat.

        Parameters
        ----------
        lon : np.ndarray or float
            Longitude in degrees or radians, see init `degrees`.
        lat : np.ndarray or float
            Latitude in degrees or radians, see init `degrees`.
        orthometric : bool, optional
            Included for backwards compatibility with the slower Elevation
            implementation. Ignored if not specified, otherwise it must match
            the value used to create this region.

        Returns
        -------
        np.ndarray or float
            Surface (DEM and geoid (unless orthometric=True)) height above the
            ellipsoid. Default is kilometers unless init variable `meters` was
            True.

        """
        if orthometric is not None and self.orthometric is not None and orthometric != self.orthometric:
            raise ValueError(f'Region was created with orthometric={self.orthometric}, unable to change!')

        in_scalar = np.isscalar(lon)
        if in_scalar != np.isscalar(lat) or (not in_scalar and (len(lon) != len(lat) or len(lon) == 0)):
            raise ValueError('Inputs must be of same length and not empty!')

        # Edge case of crossing the dateline.
        if in_scalar and lon < self.ulx:
            lon = lon + self._wrap
        elif not in_scalar:
            idx_wrap = lon < self.ulx
            if idx_wrap.any():
                lon = lon.copy()
                lon[idx_wrap] += self._wrap

        ix = np.int64((lon - self.ulx + self.dx / 2) / self.dx)
        iy = np.int64((lat - self.uly + self.dy / 2) / self.dy)

        if in_scalar:
            assert (self.ulx <= lon < self.lrx), (self.ulx, lon, self.lrx)
            assert (self.uly >= lat > self.lry), (self.uly, lat, self.lry)
            assert (0 <= ix < self.data.shape[1]), (ix, self.data.shape[1])
            assert (0 <= iy < self.data.shape[0]), (iy, self.data.shape[0])
        else:
            assert all((self.ulx <= lon) & (lon < self.lrx)), (self.ulx, lon, self.lrx)
            assert all((self.uly >= lat) & (lat > self.lry)), (self.uly, lat, self.lry)
            assert all((0 <= ix) & (ix < self.data.shape[1])), (ix, self.data.shape[1])
            assert all((0 <= iy) & (iy < self.data.shape[0])), (iy, self.data.shape[0])

        val = self.data[iy, ix]
        return val

    @track_performance
    def local_minmax(self, lon=None, lat=None, pad=None) -> Tuple[float, float]:
        """Compute local min/max elevation. Simply assumes the region is the
        same as the locality, returning its min/max.

        Parameters
        ----------
        lon, lat, pad
            Ignored.

        Returns
        -------
        float, float
            Min and max elevation for the region.

        """
        # TODO: Support sub-setting the region? Can't expand, so why bother?!
        # TODO: Throws `RuntimeWarning` if all NaNs, check in init?
        vmin = np.nanmin(self.data)
        vmax = np.nanmax(self.data)
        return vmin, vmax

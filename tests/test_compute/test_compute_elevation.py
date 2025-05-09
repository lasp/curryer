import logging
import re
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr
from pyhdf.SD import SD, SDC
from pyproj import Transformer

from curryer import utils
from curryer.compute import elevation, constants


logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])

xr.set_options(display_width=120)
np.set_printoptions(linewidth=120)


class ElevationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        self.generic_dir = root_dir / 'data' / 'generic'
        self.data_dir = root_dir / 'data'
        self.test_dir = root_dir / 'tests' / 'data'
        self.assertTrue(self.generic_dir.is_dir())
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(self.test_dir.is_dir())

    def test_elevation_locate_files(self):
        elev = elevation.Elevation()
        self.assertTrue(elev.data_dir.is_dir())
        self.assertIsNone(elev._files)

        files = elev.locate_files()
        self.assertIsInstance(files, list)
        self.assertEqual(108, len(files))
        for filepath in files:
            self.assertIsInstance(filepath, Path)
        self.assertEqual(files, elev._files)

        elev._files = None
        files = elev.locate_files(pattern='*N*W*_gmted_*.tif')
        self.assertEqual(24, len(files))
        self.assertEqual(files, elev._files)

    def test_elevation_describe_files(self):
        elev = elevation.Elevation()
        self.assertIsNone(elev._metadata)

        metadata = elev.describe_files()
        self.assertIsInstance(metadata, pd.DataFrame)
        self.assertIsInstance(elev._metadata, pd.DataFrame)
        self.assertListEqual(metadata.columns.to_list(), ['ur_lon', 'ur_lat', 'name'])
        self.assertTupleEqual(metadata.shape, (108, 3))

        metadata = metadata.reset_index()
        self.assertListEqual(metadata.columns.to_list(),
                             ['ll_lon', 'll_lat', 'stat', 'arcsec', 'ur_lon', 'ur_lat', 'name'])
        self.assertTupleEqual(metadata.shape, (108, 7))
        self.assertTrue((metadata['stat'] == 'mea').all())
        self.assertTrue(((metadata['ur_lon'] - metadata['ll_lon']) == 30).all())
        self.assertTrue(((metadata['ur_lat'] - metadata['ll_lat']) == 20).all())
        self.assertEqual((metadata['arcsec'] == 15).sum(), 84)
        self.assertEqual((metadata['arcsec'] == 30).sum(), 24)

    def test_elevation_lookup_files(self):
        elev = elevation.Elevation()
        self.assertIsNone(elev._metadata)

        _ = elev.describe_files()
        self.assertIsInstance(elev._metadata, pd.DataFrame)

        match = elev.lookup_file(2, 3)
        self.assertIsNotNone(match)
        self.assertIsInstance(match, pd.Series)
        self.assertEqual(match['ur_lon'], 30)
        self.assertEqual(match['ur_lat'], 10)
        self.assertEqual(match['name'].name, '10S000E_20101117_gmted_mea150.tif')

        for lon in range(-180, 180, 15):
            for lat in range(-90, 90, 15):
                match = elev.lookup_file(lon, lat)
                self.assertIsNotNone(match, msg=f'lon={lon}, lat={lat}')
                self.assertListEqual(list(match.index), ['ur_lon', 'ur_lat', 'name'])
                self.assertTupleEqual(match.shape, (3,))

        self.assertIsNone(elev.lookup_file(180, 0))
        self.assertIsNone(elev.lookup_file(-180 - 1e-12, 0))
        self.assertIsNone(elev.lookup_file(0, 90))
        self.assertIsNone(elev.lookup_file(0, -90 - 1e-12))

        match = elev.lookup_file(180, 0, wrap_lon=True)
        self.assertIsNotNone(match)
        self.assertEqual(match['name'].name, '10S180W_20101117_gmted_mea150.tif')

        match = elev.lookup_file(-180 - 1e-12, 0, wrap_lon=True)
        self.assertIsNotNone(match)
        self.assertEqual(match['name'].name, '10S150E_20101117_gmted_mea150.tif')

        # Modify the metadata to catch more edge cases.
        match1 = elev.lookup_file(0, 10)
        self.assertEqual(match1['name'].name, '10N000E_20101117_gmted_mea150.tif')

        match2 = match1.copy()
        match2['name'] = match2['name'].parent / re.sub(r'150', '075', match2['name'].name)
        elev._metadata.loc[(0, 10, 'mea', 7.5)] = match2

        match3 = match1.copy()
        match3['name'] = match3['name'].parent / re.sub(r'mea', 'std', match3['name'].name)
        elev._metadata.loc[(0, 10, 'std', 15)] = match3

        elev._metadata.sort_index(inplace=True)

        # Defaults to mean and highest resolution.
        match = elev.lookup_file(0, 10)
        self.assertEqual(match['name'].name, match2['name'].name)

        # Force lower resolution.
        match = elev.lookup_file(0, 10, arcsec=15)
        self.assertEqual(match['name'].name, match1['name'].name)

        # Force stat other than mean.
        match = elev.lookup_file(0, 10, stat='std')
        self.assertEqual(match['name'].name, match3['name'].name)

    def test_elevation_query_height_simple(self):
        elev = elevation.Elevation()

        ellps_ht = elev.query(-105.25, 40)  # aka Boulder, CO.
        npt.assert_allclose(ellps_ht, 1.607, rtol=1e-3)

        ortho_ht = elev.query(-105.25, 40, orthometric=True)  # aka Boulder, CO.
        self.assertEqual(ortho_ht, 1.622)

        geoid_ht = ellps_ht - ortho_ht
        npt.assert_allclose(geoid_ht, -0.015438, rtol=1e-4)

    def test_elevation_query_height_independent(self):
        elev = elevation.Elevation(meters=True)

        # Values taken from an Eng implementation.
        data = pd.DataFrame(np.asarray([
            [43.1148139831317, -84.6976955990196, 168.224880218506, 201.75, -33.5251197814941],
            [30.7142335714838, -118.408357681129, -41.5095481872559, -1, -40.5095481872559],
            [46.9825861173755, -106.153850751956, 787.063784599304, 802.75, -15.6862154006958],
            [48.679864955151, -117.691430468442, 1285.94964790344, 1302.25, -16.3003520965576],
            [43.5747030971555, -115.143410938208, 1953.82563877106, 1967.5, -13.6743612289429],
            [45.1548026115667, -78.8271085836354, 298.847255706787, 334.5, -35.6527442932129],
            [44.8626493624983, -85.2585688512091, 198.102993011475, 232.75, -34.6470069885254],
            [37.8445403906834, -104.145025996957, 1461.82157897949, 1482.75, -20.9284210205078],
            [43.1095578035511, -72.4888975580822, 288.799005508423, 316.75, -27.9509944915771],
            [33.4237337562312, -118.277695974855, -38.2005424499512, -1, -37.2005424499512],
        ]), columns=['lat', 'lon', 'elev', 'ortho', 'geoid'])

        out = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.float64)
        self.assertEqual(out.shape[0], data.shape[0])

        for ith, pnt in data.iterrows():
            elev_ht = elev.query(pnt['lon'], pnt['lat'])
            ortho_ht = elev.query(pnt['lon'], pnt['lat'], orthometric=True)
            out.loc[ith] = [pnt['lat'], pnt['lon'], elev_ht, ortho_ht, elev_ht - ortho_ht]

        npt.assert_allclose(data, out, rtol=1e-1, atol=100)

    def test_elevation_local_minmax(self):
        elev = elevation.Elevation()
        pad = 2 * np.pi * constants.WGS84_SEMI_MAJOR_AXIS_KM / 360  # KM per degree.

        out = elev.local_minmax(-105.25, 40, pad=pad)  # aka Boulder, CO.
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        npt.assert_allclose(out[0], 1.341, rtol=1e-3)
        npt.assert_allclose(out[1], 4.279, rtol=1e-3)

        # Point with region overlapping four DEM files.
        out = elev.local_minmax(0.05, 10.05, pad=pad)
        npt.assert_allclose(out[0], 0.1107, rtol=1e-3)
        npt.assert_allclose(out[1], 0.7901, rtol=1e-3)

        # Point with region overlapping international dateline near New Zealand.
        out = elev.local_minmax(-179.75, -40.0, pad=300)
        npt.assert_allclose(out[0], 0.01169, rtol=1e-3)
        npt.assert_allclose(out[1], 1.672, rtol=1e-3)

        # Point with region overlapping international dateline and north pole.
        # NOTE: DEM is all NANs near the poles!
        out = elev.local_minmax(-179.75, 89.75, pad=pad)
        npt.assert_allclose(out[0], 0.0126, rtol=1e-3)
        npt.assert_allclose(out[1], 0.0136, rtol=1e-3)

    def test_elevation_local_region_four(self):
        elev = elevation.Elevation()

        elev_region = elev.local_region(-110, -85, 25, 45)
        self.assertIsInstance(elev_region, elevation.ElevationRegion)
        # hts.sel(x=-105, y=40, method='nearest')

        ellps_ht = elev.query(-105.25, 40)  # aka Boulder, CO.
        npt.assert_allclose(ellps_ht, 1.607, rtol=1e-3)

        ellps_ht2 = elev_region.query(-105.25, 40)
        npt.assert_allclose(ellps_ht, ellps_ht2, rtol=1e-3)

    def test_elevation_local_region_mismatched_res_easy(self):
        # Tiles near the south pole are lower resolution (<-50 deg lat).
        elev = elevation.Elevation()

        # Boundary that perfectly aligns with both resolutions.
        elev_region = elev.local_region(-90, -55, -60, -40)
        self.assertIsInstance(elev_region, elevation.ElevationRegion)

        # https://en.wikipedia.org/wiki/Southern_Patagonian_Ice_Field
        lon, lat = -73.5, -50.0
        as30 = 1 / 60 / 2  # 30 arc seconds.

        ellps_ht1 = elev.query(lon, lat + as30)  # Higher res source.
        ellps_ht2 = elev.query(lon, lat - as30)  # Lower res source.
        npt.assert_allclose(ellps_ht1, 2.06581854, rtol=1e-8)
        npt.assert_allclose(ellps_ht2, 2.04077713, rtol=1e-8)

        ellps_ht11 = elev_region.query(lon, lat + as30)
        ellps_ht22 = elev_region.query(lon, lat - as30)
        npt.assert_allclose(ellps_ht11, ellps_ht1, rtol=1e-6)  # Geoid diffs.
        npt.assert_allclose(ellps_ht22, ellps_ht2, rtol=1e-6)

    def test_elevation_local_region_mismatched_res_hard(self):
        # Tiles near the south pole are lower resolution (<-50 deg lat).
        elev = elevation.Elevation()

        # Boundary that causes the lower res tile to partially overlap.
        elev_region = elev.local_region(-90 + (1 / 60 / 60 * 8), -55, -60, -40)
        self.assertIsInstance(elev_region, elevation.ElevationRegion)

        # https://en.wikipedia.org/wiki/Southern_Patagonian_Ice_Field
        lon, lat = -73.5, -50.0
        as30 = 1 / 60 / 2  # 30 arc seconds.

        ellps_ht1 = elev.query(lon, lat + as30)  # Higher res source.
        ellps_ht2 = elev.query(lon, lat - as30)  # Lower res source.
        npt.assert_allclose(ellps_ht1, 2.06581854, rtol=1e-8)
        npt.assert_allclose(ellps_ht2, 2.04077713, rtol=1e-8)

        ellps_ht11 = elev_region.query(lon, lat + as30)
        ellps_ht22 = elev_region.query(lon, lat - as30)
        npt.assert_allclose(ellps_ht11, ellps_ht1, rtol=1e-6)  # Geoid diffs.
        npt.assert_allclose(ellps_ht22, ellps_ht2, rtol=1e-6)

    def test_elevation_local_region_dateline(self):
        # Region overlapping the international dateline (-180/180 lon).
        elev = elevation.Elevation()

        elev_region = elev.local_region(170, -170, -5, 5)
        self.assertIsInstance(elev_region, elevation.ElevationRegion)

        ht_expt1 = elev.query(179, 0)
        npt.assert_allclose(ht_expt1, 0.0219113, rtol=1e-6)

        ht_calc1 = elev_region.query(179, 0)
        npt.assert_allclose(ht_expt1, ht_calc1, rtol=1e-3)

        ht_expt2 = elev.query(-179, 0)
        npt.assert_allclose(ht_expt2, 0.02005082, rtol=1e-6)

        ht_calc2a = elev_region.query(-179, 0)
        npt.assert_allclose(ht_expt2, ht_calc2a, rtol=1e-3)

        ht_calc2b = elev_region.query(181, 0)
        npt.assert_allclose(ht_expt2, ht_calc2b, rtol=1e-3)
        npt.assert_allclose(ht_calc2a, ht_calc2b, rtol=1e-13)

        lons = np.array([179, -179, 181])
        ht_calc12 = elev_region.query(lons, np.zeros(3))
        npt.assert_allclose(ht_calc1, ht_calc12[0], rtol=1e-13)
        npt.assert_allclose(ht_calc2a, ht_calc12[1], rtol=1e-13)
        npt.assert_allclose(ht_calc2b, ht_calc12[2], rtol=1e-13)

    # TODO: Delete or move else where. Used to investigate DEM data sources.
    @unittest.skip
    def test_dem_source_assumptions(self):
        import rioxarray
        import shapefile
        from pyproj import CRS, Geod

        pix_sz = 1 / 60 / 4  # 15-arc sec to degrees
        arc15min = 4 * 60

        # Unmodified GMTED2010 DEM from USGS at 15-arcmin resolution.
        usgs_fn = '30N120W_20101117_gmted_mea150.tif'
        usgs_rds = rioxarray.open_rasterio(self.test_dir / 'tmp' / 'usgs' / usgs_fn)
        usgs_hts_full = usgs_rds.values[0, :, :]

        # NASA SDP ToolKit preprocessed DEMs.
        nasa_fn = 'dem15ARC_W120N45.hdf'
        nasa_dobj = SD(str(self.test_dir / 'tmp' / '15ARC' / 'GMTED2010' / nasa_fn), SDC.READ)
        nasa_hts_full = nasa_dobj.select('Elevation')[:]  # 15-arcmin resolution.
        nasa_geoid_full = nasa_dobj.select('Geoid')[:]  # Global map at 1-deg resolution.

        # Overlapping area (lon=-120:-90, lat=45:30).
        usgs_hts = usgs_hts_full[int(5 / pix_sz):, :]
        nasa_hts = nasa_hts_full[:-int(30 / pix_sz), :int(30 / pix_sz)]

        nasa_geoid_1deg = nasa_geoid_full[90 - 45:90 - 30, 180 - 120:180 - 90]
        nasa_geoid = np.repeat(np.repeat(nasa_geoid_1deg, 4 * 60, axis=0), 4 * 60, axis=1)

        self.assertTupleEqual((3600, 7200), usgs_hts.shape)
        self.assertTupleEqual(nasa_hts.shape, usgs_hts.shape)
        self.assertTupleEqual(nasa_geoid.shape, usgs_hts.shape)

        # Height values are exactly the same.
        self.assertEqual(np.abs(nasa_hts - usgs_hts).max(), 0)

        # NASA pre-computed geoid heights are undocumented.
        self.assertGreaterEqual(nasa_geoid.min(), -39)
        self.assertLessEqual(nasa_geoid.max(), -16)

        # Compute Geoid heights assuming all are EGM96.
        transformer = Transformer.from_crs(
            'EPSG:4326+5773', 'EPSG:4979', always_xy=True, only_best=True, allow_ballpark=False
        )
        idx = usgs_hts != usgs_rds._FillValue

        # Pixel centers with a 1-arcsec N/W offset from -180/90.
        x = usgs_rds['x'].values
        y = usgs_rds['y'].values[int(5 / pix_sz):]

        xx = np.tile(x, (y.size, 1))
        yy = np.tile(y, (x.size, 1)).T

        _, _, wgs84_z = transformer.transform(xx[idx], yy[idx], usgs_hts[idx])
        wgs84_hts = np.full(usgs_hts.shape, np.NaN, dtype=np.float64)  # Source is int16!!!
        wgs84_hts[idx] = wgs84_z

        _, _, geoid_z = transformer.transform(xx[idx], yy[idx], np.zeros(xx[idx].size))
        geoid_hts = np.full(usgs_hts.shape, np.NaN, dtype=np.float64)  # Source is int16!!!
        geoid_hts[idx] = geoid_z

        # Geoid has some small differences.
        diffs = nasa_geoid - geoid_hts
        self.assertLessEqual(np.abs(diffs).mean(), 2.4)  # Meters.
        self.assertLessEqual(np.abs(diffs).max(), 9.6)  # Meters.

        # Note the variance within the NASA 1-deg pixel.
        imax = np.where(np.abs(diffs) == np.abs(diffs).max())
        ilat, ilon = imax[0][0] // arc15min, imax[1][0] // arc15min
        self.assertEqual(nasa_geoid_1deg[ilat, ilon], nasa_geoid[imax])  # Sanity check.

        calc_hts_1deg = geoid_hts[ilat * arc15min: (ilat + 1) * arc15min, ilon * arc15min: (ilon + 1) * arc15min]
        self.assertTupleEqual((arc15min, arc15min), calc_hts_1deg.shape)

        diffs_1deg_pix = calc_hts_1deg - nasa_geoid_1deg[ilat, ilon]
        self.assertGreaterEqual(diffs_1deg_pix.min(), 7.7)
        self.assertLessEqual(diffs_1deg_pix.max(), 9.6)
        self.assertLessEqual(diffs_1deg_pix.mean(), 8.7)

        # Much less than the actual DEM variance.
        nasa_hts_1deg_pix = nasa_hts[ilat * arc15min: (ilat + 1) * arc15min, ilon * arc15min: (ilon + 1) * arc15min]
        self.assertGreaterEqual(nasa_hts_1deg_pix.min(), 1711)
        self.assertLessEqual(nasa_hts_1deg_pix.max(), 3331)

        # Inspect the original metadata.
        usgs_shp_name = 'GMTED2010_Spatial_Metadata.zip'
        sf = shapefile.Reader(self.test_dir / 'tmp' / 'usgs' / usgs_shp_name)

        verts = set()
        for sr in sf.iterShapeRecords(fields=['VERT_DATUM'], bbox=[-120, 30, -90, 45]):
            verts.add(sr.record['VERT_DATUM'])

        print(verts)

        # Compute area of various vertical datums.
        # WARNING: Takes ~2-min to compute.
        geod = Geod(ellps='WGS84')

        area = {}
        count = {}
        invalid = []
        for sr in sf.iterShapeRecords(fields=['VERT_DATUM']):
            vd = sr.record['VERT_DATUM']
            if vd not in area:
                area[vd] = 0.0
                count[vd] = 0

            a_km2 = geod.polygon_area_perimeter(*zip(*sr.shape.points))[0] / (1e3 ** 2) * -1
            if not np.isfinite(a_km2):
                # One is invalid because it has 44k points? South Pole.
                invalid.append(sr)
            else:
                area[vd] += abs(a_km2)
            count[vd] += 1

        total_km2 = sum(val for val in area.values())
        parea = {k: v / total_km2 * 100 for k, v in area.items()}
        for key, val in sorted(parea.items(), key=lambda a: a[1] * -1):
            print(f'{key}[{count[key]}]: {val:.2f}%')
        return


if __name__ == '__main__':
    unittest.main()

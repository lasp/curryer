"""kernels.coverage - Unit test

@author: Matthew Maclay
"""

import logging
import tempfile
import unittest
import warnings
from pathlib import Path

from curryer import spicetime, spicierpy, utils
from curryer.kernels import coverage
from curryer.spicierpy import ext, obj

logger = logging.getLogger(__name__)
utils.enable_logging(extra_loggers=[__name__])

BODY_A = -999001
BODY_B = -999002
BODY_UNRELATED = -999009
PCK_CLASS = 3000


class CoverageTestCase(unittest.TestCase):
    def setUp(self):
        self.__tmp_dir = tempfile.TemporaryDirectory(prefix="/tmp/")
        self.addCleanup(self.__tmp_dir.cleanup)
        self.tmp_dir = Path(self.__tmp_dir.name)

        # Synthetic kernels with controlled windows (ET seconds).
        self.spk_a1 = self._write_spk("a1.bsp", BODY_A, (0.0, 2000.0))
        self.spk_a2 = self._write_spk("a2.bsp", BODY_A, (3000.0, 4000.0))
        self.spk_b = self._write_spk("b.bsp", BODY_B, (1000.0, 3500.0))
        self.spk_c = self._write_spk("c.bsp", BODY_A, (1500.0, 2500.0))
        self.spk_unrelated = self._write_spk("u.bsp", BODY_UNRELATED, (10.0, 20.0))
        self.pck_p = self._write_pck("p.bpc", PCK_CLASS, 500.0, 3600.0)

    def _write_spk(self, basename, body, *segments):
        """Write a minimal type-8 SPK with one segment for `body` per ET (first, last) pair."""
        filename = self.tmp_dir / basename
        handle = spicierpy.spkopn(str(filename), "test-spk", 0)
        for first, last in segments:
            n, degree = 2, 1
            states = [[0.0] * 6 for _ in range(n)]
            spicierpy.spkw08(handle, body, 399, "J2000", first, last, "testseg", degree, n, states, first, last - first)
        spicierpy.spkcls(handle)
        return filename

    def _write_pck(self, basename, classid, first, last):
        """Write a minimal type-2 binary PCK with one segment for `classid` over ET [first, last]."""
        filename = self.tmp_dir / basename
        handle = spicierpy.pckopn(str(filename), "test-bpc", 0)
        polydg = 1
        cdata = [0.0] * (3 * (polydg + 1))
        spicierpy.pckw02(handle, classid, "J2000", first, last, "testseg", last - first, 1, polydg, cdata, first)
        spicierpy.pckcls(handle)
        return filename

    @staticmethod
    def _ugps(et_value):
        return int(spicetime.adapt(et_value, "et", "ugps"))

    def _ugps_windows(self, *et_windows):
        return tuple((self._ugps(start), self._ugps(stop)) for start, stop in et_windows)

    def test_object_coverage_union(self):
        result = coverage.object_coverage(BODY_A, kernels=[self.spk_a1, self.spk_a2])
        self.assertEqual(self._ugps_windows((0.0, 2000.0), (3000.0, 4000.0)), result.windows)
        self.assertEqual((str(self.spk_a1), str(self.spk_a2)), result.kernels)

    def test_valid_window_intersection(self):
        windows = coverage.valid_window([BODY_A, BODY_B], kernels=[self.spk_a1, self.spk_a2, self.spk_b])
        self.assertEqual(self._ugps_windows((1000.0, 2000.0), (3000.0, 3500.0)), windows)

    def test_valid_window_includes_pck(self):
        windows = coverage.valid_window(
            [BODY_A, BODY_B, PCK_CLASS],
            kernels=[self.spk_a1, self.spk_a2, self.spk_b, self.pck_p],
        )
        self.assertEqual(self._ugps_windows((1000.0, 2000.0), (3000.0, 3500.0)), windows)

    def test_object_coverage_pck_frame_object(self):
        # Frame targets map to the frame *class* ID (ITRF93: frame ID 13000,
        # class ID 3000) and must match the coverage of the class-ID query.
        result = coverage.object_coverage(obj.Frame("ITRF93"), kernels=[self.pck_p])
        self.assertEqual(self._ugps_windows((500.0, 3600.0)), result.windows)
        self.assertEqual((str(self.pck_p),), result.kernels)

        # Frame names resolve the same way.
        named = coverage.object_coverage("ITRF93", kernels=[self.pck_p])
        self.assertEqual(result.windows, named.windows)

    def test_object_coverage_ck(self):
        # CK coverage requires the spacecraft clock kernel (and LSK) loaded.
        root_dir = Path(__file__).parents[2]
        tsis_dir = root_dir / "tests" / "data" / "tsis1"
        ck_file = tsis_dir / "iss_sc_v01.attitude.ck.20210610.bc"
        support = [root_dir / "data" / "generic" / "naif0012.tls", tsis_dir / "iss_v01.fakeclock.sclk.tsc"]
        with ext.load_kernel(support):
            result = coverage.object_coverage(-125544000, kernels=[ck_file])
            self.assertEqual((str(ck_file),), result.kernels)
            self.assertTrue(result.windows)
            direct = ext.kernel_coverage(ck_file, -125544000, to_fmt="ugps")
            self.assertEqual(int(direct[0]), result.windows[0][0])
            self.assertEqual(int(direct[-1]), result.windows[-1][1])

    def test_object_coverage_multi_segment_kernel(self):
        # Disjoint segments within one kernel stay disjoint windows.
        multi = self._write_spk("multi.bsp", BODY_A, (0.0, 1000.0), (2000.0, 3000.0))
        result = coverage.object_coverage(BODY_A, kernels=[multi])
        self.assertEqual(self._ugps_windows((0.0, 1000.0), (2000.0, 3000.0)), result.windows)
        self.assertEqual((str(multi),), result.kernels)

    def test_valid_window_no_common_coverage(self):
        # A target absent from every kernel yields no valid window — and warns,
        # since it is otherwise indistinguishable from a genuine coverage hole.
        with self.assertWarns(UserWarning) as caught:
            windows = coverage.valid_window([BODY_A, BODY_B], kernels=[self.spk_a1])
        self.assertEqual((), windows)
        self.assertIn("no kernel contains it", str(caught.warning))
        self.assertIn(str(BODY_B), str(caught.warning))

        # Disjoint coverage windows yield no valid window (both targets are
        # covered somewhere, so no warning).
        spk_b_early = self._write_spk("b_early.bsp", BODY_B, (0.0, 500.0))
        with warnings.catch_warnings(record=True) as silent:
            warnings.simplefilter("always")
            windows = coverage.valid_window([BODY_A, BODY_B], kernels=[self.spk_a2, spk_b_early])
        self.assertEqual((), windows)
        self.assertEqual([], [warning for warning in silent if issubclass(warning.category, UserWarning)])

    def test_unresolvable_target_warns_and_raises(self):
        # An undefined name resolves in no ID space: warned by default,
        # raised with error=True so a typo cannot pass as missing data.
        with self.assertWarns(UserWarning) as caught:
            windows = coverage.valid_window(["NOT_A_REAL_TARGET"], kernels=[self.spk_a1])
        self.assertEqual((), windows)
        self.assertIn("could not be resolved", str(caught.warning))

        with self.assertRaises(ValueError) as raised:
            coverage.coverage_gaps(
                ["NOT_A_REAL_TARGET"], self._ugps(0.0), self._ugps(100.0), kernels=[self.spk_a1], error=True
            )
        self.assertIn("NOT_A_REAL_TARGET", str(raised.exception))

    def test_explicit_non_coverage_kernel_warns(self):
        # Text kernels carry no time coverage: silently scoped out of the
        # pool default, but warned about when passed explicitly.
        text_kernel = self.tmp_dir / "vals.tpc"
        text_kernel.write_text("KPL/PCK\n\n")
        with self.assertWarns(UserWarning) as caught:
            result = coverage.object_coverage(BODY_A, kernels=[self.spk_a1, text_kernel])
        self.assertIn("vals.tpc", str(caught.warning))
        self.assertEqual((str(self.spk_a1),), result.kernels)

    def test_scoping_unrelated_kernel_ignored(self):
        scoped = coverage.valid_window([BODY_A, BODY_B], kernels=[self.spk_a1, self.spk_a2, self.spk_b])
        with_unrelated = coverage.valid_window(
            [BODY_A, BODY_B],
            kernels=[self.spk_a1, self.spk_a2, self.spk_b, self.spk_unrelated],
        )
        self.assertEqual(scoped, with_unrelated)

        result = coverage.object_coverage(BODY_A, kernels=[self.spk_a1, self.spk_unrelated])
        self.assertNotIn(str(self.spk_unrelated), result.kernels)

    def test_coverage_gaps_warns_by_default(self):
        kernels = [self.spk_a1, self.spk_a2, self.spk_b]
        with self.assertWarns(UserWarning) as caught:
            gaps = coverage.coverage_gaps([BODY_A, BODY_B], self._ugps(0.0), self._ugps(4000.0), kernels=kernels)
        self.assertEqual(self._ugps_windows((0.0, 1000.0), (2000.0, 3000.0), (3500.0, 4000.0)), gaps)
        self.assertIn("not fully covered", str(caught.warning))

        with self.assertRaises(ValueError) as raised:
            coverage.coverage_gaps([BODY_A, BODY_B], self._ugps(0.0), self._ugps(4000.0), kernels=kernels, error=True)
        self.assertIn("3 uncovered sub-range(s)", str(raised.exception))

    def test_coverage_gaps_inverted_window_raises(self):
        with self.assertRaises(ValueError) as raised:
            coverage.coverage_gaps([BODY_A], self._ugps(100.0), self._ugps(0.0), kernels=[self.spk_a1])
        self.assertIn("inverted", str(raised.exception))

    def test_coverage_gaps_fully_covered_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gaps = coverage.coverage_gaps(
                [BODY_A, BODY_B],
                self._ugps(1200.0),
                self._ugps(1800.0),
                kernels=[self.spk_a1, self.spk_b],
            )
        self.assertEqual((), gaps)
        self.assertEqual([], [warning for warning in caught if issubclass(warning.category, UserWarning)])

    def test_coverage_rollup(self):
        records = coverage.coverage_rollup(kernels=[self.spk_a1, self.spk_b, self.pck_p])
        self.assertEqual(3, len(records))
        by_id = {record.object_id: record for record in records}
        self.assertEqual(self._ugps_windows((0.0, 2000.0))[0], by_id[BODY_A].window)
        self.assertEqual("SPK", by_id[BODY_A].ktype)
        self.assertEqual(self._ugps_windows((1000.0, 3500.0))[0], by_id[BODY_B].window)
        self.assertEqual(self._ugps_windows((500.0, 3600.0))[0], by_id[PCK_CLASS].window)
        self.assertEqual("PCK", by_id[PCK_CLASS].ktype)

    def test_pairwise_overlap(self):
        overlap = coverage.pairwise_overlap(self.spk_a1, self.spk_c, BODY_A)
        self.assertEqual(self._ugps_windows((1500.0, 2000.0)), overlap)

        disjoint = coverage.pairwise_overlap(self.spk_a1, self.spk_a2, BODY_A)
        self.assertEqual((), disjoint)

    def test_pool_default_matches_explicit(self):
        explicit = coverage.valid_window([BODY_A, BODY_B], kernels=[self.spk_a1, self.spk_a2, self.spk_b])
        with ext.load_kernel([self.spk_a1, self.spk_a2, self.spk_b, self.spk_unrelated]):
            from_pool = coverage.valid_window([BODY_A, BODY_B])
        self.assertEqual(explicit, from_pool)


if __name__ == "__main__":
    unittest.main()

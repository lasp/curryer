"""Test short temp directory functionality and path shortening.

@author: Assistant (2024-12-19)
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path

from curryer.kernels.path_utils import get_short_temp_dir
from curryer.kernels.writer import update_invalid_paths

logger = logging.getLogger(__name__)


class TestShortTempDir(unittest.TestCase):
    """Test get_short_temp_dir() helper function."""

    def setUp(self):
        """Clean up any existing CURRYER_TEMP_DIR env var."""
        self.orig_env = os.environ.get("CURRYER_TEMP_DIR")
        if "CURRYER_TEMP_DIR" in os.environ:
            del os.environ["CURRYER_TEMP_DIR"]

    def tearDown(self):
        """Restore original env var."""
        if self.orig_env:
            os.environ["CURRYER_TEMP_DIR"] = self.orig_env
        elif "CURRYER_TEMP_DIR" in os.environ:
            del os.environ["CURRYER_TEMP_DIR"]

    def test_default_temp_dir(self):
        """Test that default temp directory is short."""
        temp_dir = get_short_temp_dir()

        # Should return /tmp on Unix-like systems
        if os.name != "nt":
            self.assertEqual(str(temp_dir), "/tmp")

        # Path should be short (< 20 chars)
        self.assertLess(len(str(temp_dir)), 20, f"Default temp dir should be short: {temp_dir}")

        # Directory should exist
        self.assertTrue(temp_dir.exists())

    def test_custom_temp_dir(self):
        """Test CURRYER_TEMP_DIR environment variable."""
        custom_path = "/tmp/curryer_test_custom"
        os.environ["CURRYER_TEMP_DIR"] = custom_path

        try:
            temp_dir = get_short_temp_dir()

            # Should use custom path
            self.assertEqual(str(temp_dir), custom_path)

            # Directory should be created
            self.assertTrue(temp_dir.exists())

        finally:
            # Clean up
            if Path(custom_path).exists():
                Path(custom_path).rmdir()

    def test_multiple_calls_same_result(self):
        """Test that multiple calls return the same directory."""
        temp_dir1 = get_short_temp_dir()
        temp_dir2 = get_short_temp_dir()

        self.assertEqual(temp_dir1, temp_dir2)


class TestUpdateInvalidPaths(unittest.TestCase):
    """Test update_invalid_paths() with try_copy strategy."""

    def test_short_path_unchanged(self):
        """Test that short paths are not modified."""
        config = {"properties": {"clock_kernel": "/tmp/short.tsc"}}

        result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

        # Short path should remain unchanged
        self.assertEqual(result["properties"]["clock_kernel"], "/tmp/short.tsc")

        # No temp files should be created for short paths
        self.assertEqual(len(temp_files), 0)

    def test_long_path_shortened(self):
        """Test that long paths are copied to short temp location."""
        # Create a test file with long path
        with tempfile.TemporaryDirectory() as tmpdir:
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel.tsc"
            test_file.write_text("DUMMY KERNEL CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Original path should be long
            self.assertGreater(len(str(test_file)), 80)

            # Fix with update_invalid_paths
            result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

            fixed_path = result["properties"]["clock_kernel"]

            # Fixed path should be shorter
            self.assertLess(len(fixed_path), 80, f"Fixed path should be < 80 chars: {fixed_path}")

            # Fixed path should exist
            self.assertTrue(Path(fixed_path).exists())

            # Should have tracked the temp file
            self.assertEqual(len(temp_files), 1, "Should track one temp file")
            self.assertIn(fixed_path, temp_files, "Temp file list should contain the shortened path")

            # Fixed path should start with /tmp (on Unix)
            if os.name != "nt":
                self.assertTrue(fixed_path.startswith("/tmp/"), f"Fixed path should be in /tmp: {fixed_path}")

    def test_multiple_kernel_properties(self):
        """Test that multiple kernel properties are all shortened."""
        with tempfile.TemporaryDirectory() as tmpdir:
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            clock_file = long_path_dir / "clock_kernel.tsc"
            frame_file = long_path_dir / "frame_kernel.tf"
            leap_file = long_path_dir / "leapsecond.tls"

            clock_file.write_text("CLOCK")
            frame_file.write_text("FRAME")
            leap_file.write_text("LEAP")

            config = {
                "properties": {
                    "clock_kernel": str(clock_file),
                    "frame_kernel": str(frame_file),
                    "leapsecond_kernel": str(leap_file),
                }
            }

            # Fix all paths
            result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

            # All should be shortened
            for key in ["clock_kernel", "frame_kernel", "leapsecond_kernel"]:
                fixed_path = result["properties"][key]
                self.assertLess(len(fixed_path), 80, f"{key} should be < 80 chars: {fixed_path}")
                self.assertTrue(Path(fixed_path).exists())

            # Should have tracked 3 temp files
            self.assertEqual(len(temp_files), 3, "Should track three temp files")

    def test_try_copy_false_uses_wrap(self):
        """Test that try_copy=False falls back to wrapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very long directory names to ensure path > 80 chars
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel_with_long_filename.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Verify path is actually long enough
            actual_len = len(str(test_file))
            self.assertGreater(actual_len, 80, f"Test path not long enough: {actual_len} chars - {test_file}")

            # With try_copy=False, should wrap instead
            result, temp_files = update_invalid_paths(config, max_len=80, try_copy=False, try_wrap=True)

            fixed_value = result["properties"]["clock_kernel"]

            # No temp files should be created when wrapping
            self.assertEqual(len(temp_files), 0, "Wrapping should not create temp files")

            # Should be wrapped (list of strings with +)
            self.assertIsInstance(fixed_value, list, "Should return wrapped path as list")

    def test_temp_file_cleanup_tracking(self):
        """Test that temp files are properly tracked for cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very long directory names to ensure path > 80 chars
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel_with_very_long_filename.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Verify path is actually long enough
            actual_len = len(str(test_file))
            self.assertGreater(actual_len, 80, f"Test path not long enough: {actual_len} chars - {test_file}")

            # Generate shortened path
            result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

            # Verify temp file was tracked
            self.assertEqual(len(temp_files), 1)
            temp_file_path = temp_files[0]

            # Verify temp file exists
            self.assertTrue(Path(temp_file_path).exists(), "Temp file should exist after creation")

            # Simulate cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Verify temp file was deleted
            self.assertFalse(Path(temp_file_path).exists(), "Temp file should be deleted after cleanup")


if __name__ == "__main__":
    unittest.main()

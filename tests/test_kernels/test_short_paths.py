"""Test short temp directory functionality and path shortening."""

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


class TestCleanupTempKernelFiles(unittest.TestCase):
    """Test _cleanup_temp_kernel_files() exception handling."""

    def test_cleanup_with_missing_file(self):
        """Test that cleanup handles missing files gracefully."""
        from unittest.mock import MagicMock

        from curryer.kernels.classes import AbstractKernelWriter

        # Create a mock kernel instance with temp files
        kernel = MagicMock(spec=AbstractKernelWriter)
        kernel._temp_kernel_files = ["/tmp/nonexistent_file_12345.tsc"]

        # Manually call the real cleanup method
        AbstractKernelWriter._cleanup_temp_kernel_files(kernel)

        # Should have cleared the list even though file didn't exist
        self.assertEqual(len(kernel._temp_kernel_files), 0, "Temp files list should be empty after cleanup")

    def test_cleanup_with_permission_error(self):
        """Test that cleanup handles permission errors gracefully."""
        from unittest.mock import MagicMock, patch

        from curryer.kernels.classes import AbstractKernelWriter

        # Create a temporary file to test cleanup
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsc") as tmp:
            temp_file_path = tmp.name
            tmp.write("TEST CONTENT")

        try:
            # Create a mock kernel instance
            kernel = MagicMock(spec=AbstractKernelWriter)
            kernel._temp_kernel_files = [temp_file_path]

            # Mock os.remove to raise OSError (simulating permission error)
            with patch("os.remove", side_effect=OSError("Permission denied")):
                with patch("os.path.exists", return_value=True):
                    # Should not raise, just log warning
                    with self.assertLogs("curryer.kernels.classes", level="WARNING") as log_context:
                        AbstractKernelWriter._cleanup_temp_kernel_files(kernel)

                    # Verify warning was logged
                    self.assertTrue(
                        any("Failed to clean up" in message for message in log_context.output),
                        "Should log warning about cleanup failure",
                    )

            # Verify list was cleared despite error
            self.assertEqual(len(kernel._temp_kernel_files), 0, "Temp files list should be empty after cleanup")

        finally:
            # Clean up the test file if it still exists
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_cleanup_successful_removal(self):
        """Test successful cleanup of temp files."""
        from unittest.mock import MagicMock

        from curryer.kernels.classes import AbstractKernelWriter

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsc") as tmp:
            temp_file_path = tmp.name
            tmp.write("TEST CONTENT")

        try:
            # Verify file exists before cleanup
            self.assertTrue(os.path.exists(temp_file_path))

            # Create a mock kernel instance
            kernel = MagicMock(spec=AbstractKernelWriter)
            kernel._temp_kernel_files = [temp_file_path]

            # Perform cleanup - should succeed and log debug message
            with self.assertLogs("curryer.kernels.classes", level="DEBUG") as log_context:
                AbstractKernelWriter._cleanup_temp_kernel_files(kernel)

            # Verify debug message was logged
            self.assertTrue(
                any("Cleaned up temp kernel file" in message for message in log_context.output),
                "Should log debug message about successful cleanup",
            )

            # Verify file was actually removed
            self.assertFalse(os.path.exists(temp_file_path), "Temp file should be removed after cleanup")

            # Verify list was cleared
            self.assertEqual(len(kernel._temp_kernel_files), 0, "Temp files list should be empty after cleanup")

        finally:
            # Clean up if somehow it still exists
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_cleanup_multiple_files_with_mixed_results(self):
        """Test cleanup with multiple files where some fail."""
        from unittest.mock import MagicMock

        from curryer.kernels.classes import AbstractKernelWriter

        # Create two temporary files
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsc") as tmp1:
            temp_file1 = tmp1.name
            tmp1.write("FILE1")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsc") as tmp2:
            temp_file2 = tmp2.name
            tmp2.write("FILE2")

        nonexistent_file = "/tmp/nonexistent_file_99999.tsc"

        try:
            # Create mock kernel with 3 files: 2 exist, 1 doesn't
            kernel = MagicMock(spec=AbstractKernelWriter)
            kernel._temp_kernel_files = [temp_file1, nonexistent_file, temp_file2]

            # Perform cleanup
            with self.assertLogs("curryer.kernels.classes", level="DEBUG") as log_context:
                AbstractKernelWriter._cleanup_temp_kernel_files(kernel)

            # Verify both existing files were removed
            self.assertFalse(os.path.exists(temp_file1), "First temp file should be removed")
            self.assertFalse(os.path.exists(temp_file2), "Second temp file should be removed")

            # Verify debug messages for successful cleanups (2 files)
            cleanup_messages = [msg for msg in log_context.output if "Cleaned up temp kernel file" in msg]
            self.assertEqual(len(cleanup_messages), 2, "Should log 2 successful cleanups")

            # Verify list was cleared
            self.assertEqual(len(kernel._temp_kernel_files), 0, "Temp files list should be empty after cleanup")

        finally:
            # Clean up any remaining files
            for path in [temp_file1, temp_file2]:
                if os.path.exists(path):
                    os.remove(path)

    def test_cleanup_with_read_only_file(self):
        """Test cleanup handles read-only file errors."""
        from unittest.mock import MagicMock, patch

        from curryer.kernels.classes import AbstractKernelWriter

        # Create a temporary file and make it read-only
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsc") as tmp:
            temp_file_path = tmp.name
            tmp.write("TEST CONTENT")

        try:
            # Make file read-only to simulate permission issue
            os.chmod(temp_file_path, 0o444)

            # Create mock kernel
            kernel = MagicMock(spec=AbstractKernelWriter)
            kernel._temp_kernel_files = [temp_file_path]

            # On some systems, removing read-only files might fail
            # Use mock to guarantee we test the exception path
            with patch("os.remove", side_effect=OSError("Read-only file")):
                with patch("os.path.exists", return_value=True):
                    with self.assertLogs("curryer.kernels.classes", level="WARNING") as log_context:
                        # Should not raise exception
                        AbstractKernelWriter._cleanup_temp_kernel_files(kernel)

                    # Verify warning was logged
                    warning_messages = [msg for msg in log_context.output if "Failed to clean up" in msg]
                    self.assertEqual(len(warning_messages), 1, "Should log one warning")
                    self.assertIn("Read-only file", warning_messages[0])

            # Verify list was still cleared
            self.assertEqual(len(kernel._temp_kernel_files), 0, "Temp files list should be empty after cleanup")

        finally:
            # Restore permissions and clean up
            try:
                os.chmod(temp_file_path, 0o644)
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except (OSError, PermissionError):
                pass  # Best effort cleanup


if __name__ == "__main__":
    unittest.main()

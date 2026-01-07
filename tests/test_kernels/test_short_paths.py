"""Test short temp directory functionality and path shortening."""

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from curryer.kernels.classes import AbstractKernelWriter
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
                shutil.rmtree(custom_path, ignore_errors=True)

    def test_multiple_calls_same_result(self):
        """Test that multiple calls return the same directory."""
        temp_dir1 = get_short_temp_dir()
        temp_dir2 = get_short_temp_dir()

        self.assertEqual(temp_dir1, temp_dir2)

    def test_custom_temp_dir_too_long(self):
        """Test that CURRYER_TEMP_DIR raises ValueError if path is too long."""
        # Create a path longer than 50 characters
        long_path = "/tmp/" + "a" * 60
        os.environ["CURRYER_TEMP_DIR"] = long_path

        try:
            with self.assertRaises(ValueError) as context:
                get_short_temp_dir()

            self.assertIn("too long", str(context.exception))
            self.assertIn("Must be ≤50 characters", str(context.exception))
        finally:
            if "CURRYER_TEMP_DIR" in os.environ:
                del os.environ["CURRYER_TEMP_DIR"]

    def test_custom_temp_dir_not_writable(self):
        """Test that CURRYER_TEMP_DIR raises ValueError if path is not writable."""
        # Use a short path that we can mock to simulate write failure
        if os.name != "nt":
            test_path = "/tmp/test_rw"
            os.environ["CURRYER_TEMP_DIR"] = test_path

            try:
                # Mock the touch() operation to raise PermissionError
                with patch("pathlib.Path.touch", side_effect=PermissionError("Permission denied")):
                    with self.assertRaises(ValueError) as context:
                        get_short_temp_dir()

                    self.assertIn("not writable", str(context.exception))
            finally:
                if "CURRYER_TEMP_DIR" in os.environ:
                    del os.environ["CURRYER_TEMP_DIR"]
        else:
            # On Windows, skip this test as it's hard to find a reliably unwritable path
            self.skipTest("Not applicable on Windows")

    def test_custom_temp_dir_sensitive_directory(self):
        """Test that CURRYER_TEMP_DIR raises ValueError if pointing to sensitive directory."""
        if os.name != "nt":
            # Try to use /etc which is a sensitive system directory
            sensitive_path = "/etc/curryer_test"
            os.environ["CURRYER_TEMP_DIR"] = sensitive_path

            try:
                with self.assertRaises(ValueError) as context:
                    get_short_temp_dir()

                self.assertIn("sensitive system directory", str(context.exception))
            finally:
                if "CURRYER_TEMP_DIR" in os.environ:
                    del os.environ["CURRYER_TEMP_DIR"]
        else:
            # On Windows, test C:\Windows
            sensitive_path = "C:\\Windows\\curryer_test"
            os.environ["CURRYER_TEMP_DIR"] = sensitive_path

            try:
                with self.assertRaises(ValueError) as context:
                    get_short_temp_dir()

                self.assertIn("sensitive system directory", str(context.exception))
            finally:
                if "CURRYER_TEMP_DIR" in os.environ:
                    del os.environ["CURRYER_TEMP_DIR"]


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

            # With try_copy=False and try_symlink=False, should wrap instead
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=False, try_copy=False, try_wrap=True
            )

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
        # Create a mock kernel instance with temp files
        kernel = MagicMock(spec=AbstractKernelWriter)
        kernel._temp_kernel_files = ["/tmp/nonexistent_file_12345.tsc"]

        # Manually call the real cleanup method
        AbstractKernelWriter._cleanup_temp_kernel_files(kernel)

        # Should have cleared the list even though file didn't exist
        self.assertEqual(len(kernel._temp_kernel_files), 0, "Temp files list should be empty after cleanup")

    def test_cleanup_with_permission_error(self):
        """Test that cleanup handles permission errors gracefully."""

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


class TestSymlinkStrategy(unittest.TestCase):
    """Test symlink-based path shortening."""

    def test_symlink_creation_success(self):
        """Test successful symlink creation on Unix/macOS."""
        # Skip on Windows if not supported
        if os.name == "nt":
            self.skipTest("Symlinks may require admin on Windows")

        # Create test file with long path
        with tempfile.TemporaryDirectory() as tmpdir:
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel.tls"
            test_file.write_text("DUMMY KERNEL CONTENT FOR TESTING")

            config = {"properties": {"leapsecond_kernel": str(test_file)}}

            # Original path should be long
            self.assertGreater(len(str(test_file)), 80)

            # Call update_invalid_paths with try_symlink=True
            result, temp_files = update_invalid_paths(config, max_len=80, try_symlink=True)

            fixed_path = result["properties"]["leapsecond_kernel"]

            # Verify symlink was created and path shortened
            self.assertLess(len(fixed_path), 80, f"Symlink path should be < 80 chars: {fixed_path}")

            # Verify symlink exists
            self.assertTrue(Path(fixed_path).exists(), "Symlink should exist")

            # Verify symlink points to original file
            self.assertTrue(Path(fixed_path).is_symlink(), "Should be a symlink")

            # Verify original file content accessible through symlink
            with open(fixed_path) as f:
                content = f.read()
            self.assertEqual(content, "DUMMY KERNEL CONTENT FOR TESTING")

            # Verify symlink tracked for cleanup
            self.assertEqual(len(temp_files), 1)
            self.assertEqual(temp_files[0], fixed_path)

            # Clean up symlink
            if os.path.exists(fixed_path):
                os.remove(fixed_path)

    def test_symlink_fallback_to_copy(self):
        """Test fallback to copy when symlink fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel.tls"
            test_file.write_text("CONTENT")

            config = {"properties": {"leapsecond_kernel": str(test_file)}}

            # Mock os.symlink to raise OSError (simulate symlink failure)
            with patch("os.symlink", side_effect=OSError("Operation not permitted")):
                # Should fall back to copy strategy
                result, temp_files = update_invalid_paths(
                    config, max_len=80, try_symlink=True, try_copy=True, try_wrap=False
                )

                fixed_path = result["properties"]["leapsecond_kernel"]

                # Verify file copy strategy was used (not a symlink)
                self.assertFalse(Path(fixed_path).is_symlink(), "Should not be a symlink")
                self.assertTrue(Path(fixed_path).exists(), "Copied file should exist")

                # Verify path is shortened
                self.assertLess(len(fixed_path), 80)

                # Clean up
                if os.path.exists(fixed_path):
                    os.remove(fixed_path)

    def test_symlink_cleanup(self):
        """Test symlinks are removed after kernel generation."""
        # Skip on Windows if not supported
        if os.name == "nt":
            self.skipTest("Symlinks may require admin on Windows")

        with tempfile.TemporaryDirectory() as tmpdir:
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel.tls"
            test_file.write_text("CONTENT")

            config = {"properties": {"leapsecond_kernel": str(test_file)}}

            # Create symlink
            result, temp_files = update_invalid_paths(config, max_len=80, try_symlink=True)

            symlink_path = result["properties"]["leapsecond_kernel"]

            # Verify symlink exists during generation
            self.assertTrue(Path(symlink_path).exists())
            self.assertTrue(Path(symlink_path).is_symlink())

            # Simulate cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Verify symlink removed after cleanup
            self.assertFalse(Path(symlink_path).exists())

    def test_symlink_disabled_via_env_var(self):
        """Test CURRYER_DISABLE_SYMLINKS environment variable."""
        # Skip on Windows if not supported
        if os.name == "nt":
            self.skipTest("Symlinks may require admin on Windows")

        os.environ["CURRYER_DISABLE_SYMLINKS"] = "true"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                long_path_dir = (
                    Path(tmpdir)
                    / "very_long_directory_name_for_testing"
                    / "another_long_directory_name"
                    / "yet_another_long_directory_name"
                    / "and_one_more_long_directory"
                )
                long_path_dir.mkdir(parents=True, exist_ok=True)

                test_file = long_path_dir / "test_kernel.tls"
                test_file.write_text("CONTENT")

                config = {"properties": {"leapsecond_kernel": str(test_file)}}

                # Symlink strategy should be skipped
                result, temp_files = update_invalid_paths(
                    config, max_len=80, try_symlink=True, try_copy=True, try_wrap=False
                )

                fixed_path = result["properties"]["leapsecond_kernel"]

                # Verify symlink strategy was skipped (should use copy instead)
                self.assertFalse(Path(fixed_path).is_symlink(), "Should not be a symlink")
                self.assertTrue(Path(fixed_path).exists(), "Should use copy strategy")

                # Clean up
                if os.path.exists(fixed_path):
                    os.remove(fixed_path)
        finally:
            del os.environ["CURRYER_DISABLE_SYMLINKS"]


class TestContinuationCharacterStrategy(unittest.TestCase):
    """Test continuation character (+) path wrapping."""

    def test_wrap_success_multiple_short_segments(self):
        """Test wrapping succeeds when all segments are short."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with long total but short components - ensure >80 chars
            long_path_dir = (
                Path(tmpdir)
                / "directory1"
                / "directory2"
                / "directory3"
                / "directory4"
                / "directory5"
                / "directory6"
                / "directory7"
                / "directory8"
                / "directory9"
                / "directory10"
                / "directory11"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel_file.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Verify path is long enough
            self.assertGreater(len(str(test_file)), 80, f"Path should be >80 chars: {len(str(test_file))}")

            # Try wrapping (disable symlink and copy to test wrap)
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=False, try_wrap=True, try_copy=False
            )

            fixed_value = result["properties"]["clock_kernel"]

            # Verify wrapping applied
            self.assertIsInstance(fixed_value, list, "Should return wrapped path as list")

            # Verify each line <= 80 chars
            for line in fixed_value:
                self.assertLessEqual(len(line), 80, f"Each wrapped line should be <= 80 chars: {line}")

    def test_wrap_failure_single_long_segment(self):
        """Test wrapping fails when a single directory exceeds limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with one directory name >80 chars
            long_dir_name = "a" * 85  # Directory name longer than 80 chars
            long_path_dir = Path(tmpdir) / long_dir_name
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "kernel.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Try wrapping - should fail and fall back to copy
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=False, try_wrap=True, try_copy=True
            )

            fixed_path = result["properties"]["clock_kernel"]

            # Verify wrapping was skipped (should use copy instead)
            self.assertIsInstance(fixed_path, str, "Should return string (not wrapped list)")
            self.assertLess(len(fixed_path), 80, "Should use copy strategy")

            # Clean up
            if os.path.exists(fixed_path):
                os.remove(fixed_path)

    def test_wrap_preferred_over_copy(self):
        """Test wrapping is tried before copying."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with short segments (wrappable) - ensure >80 chars
            long_path_dir = (
                Path(tmpdir)
                / "directory1"
                / "directory2"
                / "directory3"
                / "directory4"
                / "directory5"
                / "directory6"
                / "directory7"
                / "directory8"
                / "directory9"
                / "directory10"
                / "directory11"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel_file.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Verify path is long enough
            self.assertGreater(len(str(test_file)), 80, f"Path should be >80 chars: {len(str(test_file))}")

            # Enable both try_wrap and try_copy (disable symlink)
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=False, try_wrap=True, try_copy=True
            )

            fixed_value = result["properties"]["clock_kernel"]

            # Verify wrap was used (returns list), not copy (returns string)
            self.assertIsInstance(fixed_value, list, "Should use wrap strategy, not copy")

            # No temp files should be created
            self.assertEqual(len(temp_files), 0, "Wrap strategy should not create temp files")


class TestRelativePathStrategy(unittest.TestCase):
    """Test relative path optimization."""

    def test_relative_path_shortening(self):
        """Test relative path used when shorter than absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a deeply nested temp directory to make absolute path long
            # But use short relative path from a subdirectory
            deep_tmpdir = Path(tmpdir) / "very_long_directory_name_for_testing" / "another_long_directory_name"
            deep_tmpdir.mkdir(parents=True, exist_ok=True)

            # Create file in a subdirectory with short name
            test_dir = deep_tmpdir / "data"
            test_dir.mkdir(parents=True, exist_ok=True)

            test_file = test_dir / "k.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Verify absolute path is long enough
            self.assertGreater(len(str(test_file)), 80, f"Absolute path should be >80: {len(str(test_file))}")

            # Try relative path from deep_tmpdir (making relative path short)
            result, temp_files = update_invalid_paths(
                config,
                max_len=80,
                try_symlink=False,
                try_wrap=False,
                try_relative=True,
                try_copy=False,
                relative_dir=deep_tmpdir,
            )

            fixed_path = result["properties"]["clock_kernel"]

            # Verify relative path is used
            self.assertNotEqual(fixed_path, str(test_file), "Should use relative path, not absolute")

            # Verify it's the relative path
            expected_rel = "data/k.tsc"
            self.assertEqual(fixed_path, expected_rel, "Should be the short relative path")

            # Verify length is under limit
            self.assertLessEqual(len(fixed_path), 80, "Relative path should be under limit")

    def test_relative_path_still_too_long(self):
        """Test fallback when relative path also exceeds limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create path where both absolute and relative are >80 chars
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "kernel.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Both absolute and relative paths >80 chars from root
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=False, try_wrap=False, try_relative=True, try_copy=True, relative_dir="/"
            )

            fixed_path = result["properties"]["clock_kernel"]

            # Verify relative path was attempted but copy was used
            self.assertLess(len(fixed_path), 80, "Should fall back to copy")

            # Clean up
            if os.path.exists(fixed_path):
                os.remove(fixed_path)


class TestStrategyPriorityOrder(unittest.TestCase):
    """Test strategies execute in correct priority order."""

    def test_strategy_order_symlink_wrap_relative_copy(self):
        """Test default priority: symlink → wrap → relative → copy."""
        # Skip on Windows if not supported
        if os.name == "nt":
            self.skipTest("Symlinks may require admin on Windows")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create path with short segments (wrappable) - ensure >80 chars
            long_path_dir = (
                Path(tmpdir)
                / "directory1"
                / "directory2"
                / "directory3"
                / "directory4"
                / "directory5"
                / "directory6"
                / "directory7"
                / "directory8"
                / "directory9"
                / "directory10"
                / "directory11"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "test_kernel_file.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Verify path is long enough
            self.assertGreater(len(str(test_file)), 80, f"Path should be >80 chars: {len(str(test_file))}")

            # Enable all strategies
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=True, try_wrap=True, try_relative=True, try_copy=True
            )

            fixed_path = result["properties"]["clock_kernel"]

            # First successful strategy (symlink) should be used
            self.assertTrue(Path(fixed_path).is_symlink(), "Should use symlink strategy (priority 1)")

            # Clean up
            if os.path.exists(fixed_path):
                os.remove(fixed_path)

    def test_first_successful_strategy_stops_chain(self):
        """Test that first successful strategy prevents later attempts."""
        # Skip on Windows if not supported
        if os.name == "nt":
            self.skipTest("Symlinks may require admin on Windows")

        with tempfile.TemporaryDirectory() as tmpdir:
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
                / "and_one_more_long_directory"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "kernel.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            # Enable all strategies - symlink should succeed first
            result, temp_files = update_invalid_paths(
                config, max_len=80, try_symlink=True, try_wrap=True, try_relative=True, try_copy=True
            )

            fixed_path = result["properties"]["clock_kernel"]

            # Verify symlink was created (first strategy)
            self.assertTrue(Path(fixed_path).is_symlink(), "Should use symlink")

            # Verify no file copy occurred (only one temp file - the symlink)
            self.assertEqual(len(temp_files), 1, "Only symlink should be tracked")

            # Clean up
            if os.path.exists(fixed_path):
                os.remove(fixed_path)


class TestEnvironmentVariableConfig(unittest.TestCase):
    """Test environment variable configuration."""

    def test_curryer_disable_symlinks(self):
        """Test CURRYER_DISABLE_SYMLINKS=true."""
        os.environ["CURRYER_DISABLE_SYMLINKS"] = "true"

        try:
            from curryer.kernels.path_utils import get_path_strategy_config

            config = get_path_strategy_config()

            self.assertTrue(config["disable_symlinks"], "Should disable symlinks")

        finally:
            del os.environ["CURRYER_DISABLE_SYMLINKS"]

    def test_curryer_path_strategy(self):
        """Test custom strategy priority via CURRYER_PATH_STRATEGY."""
        os.environ["CURRYER_PATH_STRATEGY"] = "copy,symlink,wrap,relative"

        try:
            from curryer.kernels.path_utils import get_path_strategy_config

            config = get_path_strategy_config()

            self.assertEqual(config["strategy_order"], ["copy", "symlink", "wrap", "relative"])

        finally:
            del os.environ["CURRYER_PATH_STRATEGY"]

    def test_curryer_warn_on_copy_default(self):
        """Test CURRYER_WARN_ON_COPY default value."""
        from curryer.kernels.path_utils import get_path_strategy_config

        config = get_path_strategy_config()

        self.assertTrue(config["warn_on_copy"], "Should warn on copy by default")

    def test_curryer_warn_copy_threshold(self):
        """Test CURRYER_WARN_COPY_THRESHOLD environment variable."""
        os.environ["CURRYER_WARN_COPY_THRESHOLD"] = "50"

        try:
            from curryer.kernels.path_utils import get_path_strategy_config

            config = get_path_strategy_config()

            self.assertEqual(config["warn_copy_threshold_mb"], 50, "Should use custom threshold")

        finally:
            del os.environ["CURRYER_WARN_COPY_THRESHOLD"]

    def test_warn_on_large_file_copy(self):
        """Test that warnings are logged when copying large files."""
        os.environ["CURRYER_WARN_ON_COPY"] = "true"
        os.environ["CURRYER_WARN_COPY_THRESHOLD"] = "1"  # 1 MB threshold
        os.environ["CURRYER_PATH_STRATEGY"] = "copy"  # Force copy strategy

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a large file (2 MB)
                long_path_dir = (
                    Path(tmpdir)
                    / "very_long_directory_name_for_testing"
                    / "another_long_directory_name"
                    / "yet_another_long_directory_name"
                    / "and_one_more_long_directory"
                )
                long_path_dir.mkdir(parents=True, exist_ok=True)

                test_file = long_path_dir / "large_test.tls"
                # Write 2 MB of data
                with open(test_file, "wb") as f:
                    f.write(b"X" * (2 * 1024 * 1024))

                config = {"properties": {"leapsecond_kernel": str(test_file)}}

                # Capture log output
                with self.assertLogs("curryer.kernels.writer", level="WARNING") as log_context:
                    result, temp_files = update_invalid_paths(config, max_len=80)

                # Verify warning was logged
                self.assertTrue(
                    any("Copying large file" in message for message in log_context.output),
                    "Should log warning about large file copy",
                )

                # Verify file was copied
                fixed_path = result["properties"]["leapsecond_kernel"]
                self.assertLess(len(fixed_path), 80, "Path should be shortened")

                # Clean up
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)

        finally:
            del os.environ["CURRYER_WARN_ON_COPY"]
            del os.environ["CURRYER_WARN_COPY_THRESHOLD"]
            del os.environ["CURRYER_PATH_STRATEGY"]


if __name__ == "__main__":
    unittest.main()

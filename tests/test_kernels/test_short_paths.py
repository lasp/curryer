"""Test short temp directory functionality and path shortening.

This module comprehensively tests the SPICE path shortening functionality
that automatically handles paths exceeding the 80-character limit through
two strategies: symlink → copy.
"""

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from curryer.kernels.classes import AbstractKernelWriter
from curryer.kernels.path_utils import (
    _convert_paths_to_strings,
    _is_file_property,
    copy_to_short_path,
    get_path_strategy_config,
    get_short_temp_dir,
    update_invalid_paths,
)

logger = logging.getLogger(__name__)


class TestShortTempDir(unittest.TestCase):
    """Test get_short_temp_dir() helper function.

    Tests the temporary directory helper that provides short base paths
    for temporary files to avoid SPICE's 80-character limit.
    Covers platform-specific defaults, custom paths, and path length validation.
    """

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
        """Test that default temp directory uses shortest available path."""
        temp_dir = get_short_temp_dir()

        # Should prioritize short paths: /tmp on Unix, C:\Temp on Windows
        if os.name != "nt":
            # On Unix/macOS, should use /tmp (4 chars) for maximum filename space
            self.assertEqual(str(temp_dir), "/tmp", "Should use /tmp on Unix for shortest path")
        else:
            # On Windows, should try C:\Temp first, or fall back to system temp
            self.assertTrue(
                str(temp_dir) in ["C:\\Temp", tempfile.gettempdir()],
                f"Should use C:\\Temp or system temp on Windows: {temp_dir}",
            )

        # Directory should exist
        self.assertTrue(temp_dir.exists())

        # Directory should be writable
        self.assertTrue(os.access(str(temp_dir), os.W_OK))

        # On Unix, verify we have maximum space for filenames
        if os.name != "nt" and str(temp_dir) == "/tmp":
            # /tmp is 4 chars, leaving 75 chars for filenames (plenty!)
            self.assertEqual(len(str(temp_dir)), 4, "Unix /tmp should be exactly 4 chars")

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


class TestUpdateInvalidPaths(unittest.TestCase):
    """Test update_invalid_paths() with path shortening strategies.

    Tests the core path shortening function that detects paths exceeding
    the maximum length and applies appropriate strategies. Covers symlink
    creation, file copying, and tracking of temporary files.
    """

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
    """Test _cleanup_temp_kernel_files() exception handling.

    Tests the cleanup functionality that removes temporary files created
    during path shortening. Covers successful cleanup, missing files,
    permission errors, and partial cleanup scenarios.
    """

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
    """Test symlink-based path shortening (Strategy #1 - preferred).

    Tests the symlink strategy which creates symbolic links in a short
    temp directory. This is the preferred strategy as it has zero storage
    overhead and zero I/O cost. Covers creation, fallback to copy, and cleanup.
    """

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

            # Call update_invalid_paths - symlink is always tried first
            result, temp_files = update_invalid_paths(config, max_len=80)

            fixed_path = result["properties"]["leapsecond_kernel"]

            # Verify symlink was created and path shortened
            self.assertLess(len(fixed_path), 80, f"Symlink path should be < 80 chars: {fixed_path}")

            # Verify symlink exists
            self.assertTrue(Path(fixed_path).exists(), "Symlink should exist")

            # Verify symlink points to original file
            self.assertTrue(Path(fixed_path).is_symlink(), "Should be a symlink")

            # Note: We don't check logs here since assertLogs would interfere with update_invalid_paths call

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

            # Mock os.symlink in curryer.kernels.path_utils to raise OSError (simulate symlink failure)
            with patch("curryer.kernels.path_utils.os.symlink", side_effect=OSError("Operation not permitted")):
                # Should fall back to copy strategy
                result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

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
            result, temp_files = update_invalid_paths(config, max_len=80)

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


class TestStrategyPriorityOrder(unittest.TestCase):
    """Test strategies execute in correct priority order.

    Tests the fixed strategy priority (symlink → copy) and ensures that
    strategies are tried in order, with the first successful strategy
    preventing later attempts. Also verifies the complete fallback chain
    when earlier strategies fail.

    Strategy Priority (fixed order):
    1. Symlink - Zero overhead, preferred when available
    2. Copy - Bulletproof fallback, always works but creates temp files
    """

    def test_strategy_order_symlink_copy(self):
        """Test default priority: symlink → copy."""
        # Skip on Windows if not supported
        if os.name == "nt":
            self.skipTest("Symlinks may require admin on Windows")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create long path - ensure >80 chars
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

            # Enable copy strategy (symlink always tried first by default)
            result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

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

            # Enable copy strategy (symlink always tried first)
            result, temp_files = update_invalid_paths(config, max_len=80, try_copy=True)

            fixed_path = result["properties"]["clock_kernel"]

            # Verify symlink was created (first strategy)
            self.assertTrue(Path(fixed_path).is_symlink(), "Should use symlink")

            # Verify no file copy occurred (only one temp file - the symlink)
            self.assertEqual(len(temp_files), 1, "Only symlink should be tracked")

            # Clean up
            if os.path.exists(fixed_path):
                os.remove(fixed_path)

    def test_full_strategy_chain_fallback(self):
        """Test complete strategy cascade: symlink → copy.

        This test explicitly verifies that when symlink fails,
        the copy strategy is tried as the bulletproof fallback.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a long path
            long_path_dir = (
                Path(tmpdir)
                / "very_long_directory_name_for_testing"
                / "another_long_directory_name"
                / "yet_another_long_directory_name"
            )
            long_path_dir.mkdir(parents=True, exist_ok=True)

            test_file = long_path_dir / "kernel.tsc"
            test_file.write_text("CONTENT")

            config = {"properties": {"clock_kernel": str(test_file)}}

            self.assertGreater(len(str(test_file)), 80, "Path should be >80 chars")

            # Mock symlink to fail (simulating Windows/restricted environment)
            with patch("curryer.kernels.path_utils.os.symlink", side_effect=OSError("Operation not permitted")):
                # Capture log output to verify strategy attempts
                with self.assertLogs("curryer.kernels.path_utils", level="INFO") as log_context:
                    result, temp_files = update_invalid_paths(
                        config,
                        max_len=80,
                        try_copy=True,  # Will succeed
                    )

                fixed_path = result["properties"]["clock_kernel"]

                # Verify strategies were attempted
                log_output = "\n".join(log_context.output)

                # Strategy 2: Copy succeeded (should see "Using copy:" message)
                self.assertIn("Using copy:", log_output, "Should log 'Using copy:' when copy succeeds")

                # Verify the file was actually copied (not symlinked)
                self.assertFalse(Path(fixed_path).is_symlink(), "Should not be a symlink")
                self.assertTrue(Path(fixed_path).exists(), "Copied file should exist")
                self.assertLess(len(fixed_path), 80, "Final path should be <80 chars")

                # Verify copy strategy was used (temp file tracked)
                self.assertEqual(len(temp_files), 1, "Should have one temp file from copy")

                # Clean up
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)


class TestEnvironmentVariableConfig(unittest.TestCase):
    """Test environment variable configuration.

    Tests configuration via environment variables:
    - CURRYER_DISABLE_COPY: Disable file copy strategy (for AWS/cloud)
    - CURRYER_TEMP_DIR: Custom temp directory (tested in TestShortTempDir)
    """

    def test_curryer_disable_copy(self):
        """Test CURRYER_DISABLE_COPY=true prevents file copying."""
        os.environ["CURRYER_DISABLE_COPY"] = "true"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create long path that would need copy strategy
                long_path_dir = (
                    Path(tmpdir)
                    / "very_long_directory_name_for_testing"
                    / "another_long_directory_name"
                    / "yet_another_long_directory_name"
                    / "and_one_more_long_directory"
                )
                long_path_dir.mkdir(parents=True, exist_ok=True)

                test_file = long_path_dir / "test.txt"
                test_file.write_text("test")

                config = {"properties": {"test_file": str(test_file)}}

                # Mock symlink to fail (force fallback to copy)
                with patch("curryer.kernels.path_utils.os.symlink", side_effect=OSError("Not supported")):
                    result, temp_files = update_invalid_paths(config, max_len=50)

                    # Path should remain unchanged (no copy fallback)
                    # Since symlink failed and copy is disabled, path stays long
                    self.assertEqual(result["properties"]["test_file"], str(test_file))
                    self.assertEqual(len(temp_files), 0, "No temp files should be created when copy disabled")

        finally:
            if "CURRYER_DISABLE_COPY" in os.environ:
                del os.environ["CURRYER_DISABLE_COPY"]

    def test_curryer_disable_copy_false_by_default(self):
        """Test that copy strategy is enabled by default."""

        # Ensure env var is not set
        if "CURRYER_DISABLE_COPY" in os.environ:
            del os.environ["CURRYER_DISABLE_COPY"]

        config = get_path_strategy_config()

        self.assertFalse(config["disable_copy"], "Copy should be enabled by default")
        self.assertTrue(config["try_copy"], "try_copy should be True by default")


class TestAdditionalCoverage(unittest.TestCase):
    """Additional tests to achieve comprehensive coverage of path_utils.py."""

    def setUp(self):
        """Clean up environment variables."""
        self.orig_tmpdir = os.environ.get("TMPDIR")
        self.orig_curryer_temp = os.environ.get("CURRYER_TEMP_DIR")

    def tearDown(self):
        """Restore environment variables."""
        if self.orig_tmpdir:
            os.environ["TMPDIR"] = self.orig_tmpdir
        elif "TMPDIR" in os.environ:
            del os.environ["TMPDIR"]

        if self.orig_curryer_temp:
            os.environ["CURRYER_TEMP_DIR"] = self.orig_curryer_temp
        elif "CURRYER_TEMP_DIR" in os.environ:
            del os.environ["CURRYER_TEMP_DIR"]

    def test_temp_directory_prioritizes_short_paths(self):
        """Test that get_short_temp_dir() prioritizes /tmp on Unix systems."""
        if os.name != "nt":
            # On Unix/macOS without CURRYER_TEMP_DIR set, should use /tmp
            if "CURRYER_TEMP_DIR" in os.environ:
                del os.environ["CURRYER_TEMP_DIR"]

            temp_dir = get_short_temp_dir()

            # Should use /tmp which is the shortest option (4 chars)
            self.assertEqual(str(temp_dir), "/tmp")
            self.assertEqual(len(str(temp_dir)), 4)

            # This leaves maximum space for filenames (75 chars with 80 char limit)
            filename_space = 80 - len(str(temp_dir)) - 1  # -1 for the path separator
            self.assertEqual(filename_space, 75)

    @unittest.skipIf(os.name != "nt", "Windows-specific test")
    def test_windows_c_temp_creation(self):
        """Test Windows C:\\Temp path creation."""
        # This test only runs on Windows

        if "CURRYER_TEMP_DIR" in os.environ:
            del os.environ["CURRYER_TEMP_DIR"]

        temp_dir = get_short_temp_dir()

        # On Windows, should prefer C:\Temp if possible
        self.assertTrue(temp_dir.exists())
        self.assertTrue(os.access(str(temp_dir), os.W_OK))

    def test_copy_to_short_path_with_long_temp_dir(self):
        """Test copy_to_short_path when resulting path is still too long."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            # Create a very long temp directory path
            long_temp = Path(tmpdir) / ("x" * 50)
            long_temp.mkdir(parents=True)

            # Try to copy with a very short max_len to trigger the warning
            with self.assertLogs("curryer.kernels.path_utils", level="WARNING") as log_context:
                result = copy_to_short_path(test_file, long_temp, max_len=20)

                # Should return None and log warning
                self.assertIsNone(result, "Should return None when temp path too long")
                self.assertTrue(
                    any("Temp directory base path too long" in msg for msg in log_context.output),
                    "Should warn about temp directory being too long",
                )

    def test_copy_to_short_path_oserror(self):
        """Test copy_to_short_path handles OSError gracefully."""

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()

            # Mock mkstemp to raise OSError
            with patch("tempfile.mkstemp", side_effect=OSError("Mock error")):
                with self.assertLogs("curryer.kernels.path_utils", level="DEBUG") as log_context:
                    result = copy_to_short_path(test_file, temp_dir, max_len=80)

                    # Should return None and log error
                    self.assertIsNone(result)
                    self.assertTrue(any("Copy failed" in msg for msg in log_context.output))

    def test_update_invalid_paths_with_non_file_properties(self):
        """Test update_invalid_paths skips non-file properties."""

        config = {
            "properties": {
                "some_number": 42,
                "some_bool": True,
                "some_dict": {"nested": "value"},
                "kernel_path": "/tmp/test.txt",  # This is a file property
            }
        }

        result, temp_files = update_invalid_paths(config, max_len=80)

        # Non-file properties should be unchanged
        self.assertEqual(result["properties"]["some_number"], 42)
        self.assertEqual(result["properties"]["some_bool"], True)
        self.assertEqual(result["properties"]["some_dict"], {"nested": "value"})

    def test_update_invalid_paths_with_path_objects(self):
        """Test update_invalid_paths converts Path objects to strings."""

        config = {"properties": {"short_path": Path("/tmp/short.txt"), "another_path": Path("/tmp/another.txt")}}

        result, temp_files = update_invalid_paths(config, max_len=80)

        # Path objects should be converted to strings
        self.assertIsInstance(result["properties"]["short_path"], str)
        self.assertIsInstance(result["properties"]["another_path"], str)
        self.assertEqual(result["properties"]["short_path"], "/tmp/short.txt")

    def test_update_invalid_paths_with_list_of_paths(self):
        """Test update_invalid_paths handles lists of paths."""

        config = {
            "properties": {"planet_kernels": [Path("/tmp/kernel1.bsp"), "/tmp/kernel2.bsp", Path("/tmp/kernel3.bsp")]}
        }

        result, temp_files = update_invalid_paths(config, max_len=80)

        # All paths in list should be strings
        kernels = result["properties"]["planet_kernels"]
        self.assertEqual(len(kernels), 3)
        for kernel in kernels:
            self.assertIsInstance(kernel, str)

    def test_update_invalid_paths_without_properties_key(self):
        """Test update_invalid_paths works on top-level config without 'properties' key."""

        config = {"kernel_path": "/tmp/test.txt", "another_key": "value"}

        result, temp_files = update_invalid_paths(config, max_len=80)

        # Should work on top-level dict
        self.assertEqual(result["kernel_path"], "/tmp/test.txt")
        self.assertEqual(result["another_key"], "value")

    def test_update_invalid_paths_with_relative_paths(self):
        """Test update_invalid_paths resolves relative paths with parent_dir."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test_kernel.txt"
            test_file.write_text("test")

            config = {
                "properties": {
                    "kernel_path": "test_kernel.txt"  # Relative path
                }
            }

            result, temp_files = update_invalid_paths(config, max_len=80, parent_dir=tmpdir)

            # Should resolve relative path
            result_path = result["properties"]["kernel_path"]
            self.assertTrue(os.path.isabs(result_path) or result_path == "test_kernel.txt")

    def test_update_invalid_paths_parent_dir_is_file(self):
        """Test update_invalid_paths handles parent_dir that is a file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_file = Path(tmpdir) / "parent.txt"
            parent_file.write_text("parent")

            config = {"properties": {"kernel_path": "/tmp/test.txt"}}

            # Should use parent directory of the file
            result, temp_files = update_invalid_paths(
                config,
                max_len=80,
                parent_dir=parent_file,  # Pass a file, not a directory
            )

            # Should not crash
            self.assertIsNotNone(result)

    def test_update_invalid_paths_all_strategies_fail(self):
        """Test update_invalid_paths when both symlink and copy fail."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a long path that exceeds max_len
            long_dir = Path(tmpdir) / ("x" * 30)
            long_dir.mkdir(parents=True)
            test_file = long_dir / "test_kernel.txt"
            test_file.write_text("test")

            # Verify the file exists and path is long
            self.assertTrue(test_file.exists())
            self.assertGreater(len(str(test_file)), 50)

            config = {"properties": {"leapsecond_kernel": str(test_file)}}

            # Mock both strategies to fail
            with patch("curryer.kernels.path_utils.create_short_symlink", return_value=None):
                with patch("curryer.kernels.path_utils.copy_to_short_path", return_value=None):
                    with self.assertLogs("curryer.kernels.path_utils", level="INFO") as log_context:
                        result, temp_files = update_invalid_paths(config, max_len=50, try_copy=True)

                        # Should log warning about failure
                        log_output = "\n".join(log_context.output)
                        self.assertTrue(
                            "Failed to shorten path" in log_output,
                            f"Expected warning about failed shortening. Got logs:\n{log_output}",
                        )

                        # Path should remain unchanged
                        self.assertEqual(result["properties"]["leapsecond_kernel"], str(test_file))

    def test_convert_paths_to_strings(self):
        """Test _convert_paths_to_strings helper function."""

        # Test with nested structure
        data = {
            "path": Path("/tmp/test.txt"),
            "nested": {"another_path": Path("/tmp/nested.txt"), "number": 42},
            "list": [Path("/tmp/list1.txt"), Path("/tmp/list2.txt"), "string"],
            "plain_string": "unchanged",
        }

        result = _convert_paths_to_strings(data)

        # All Path objects should be converted to strings
        self.assertIsInstance(result["path"], str)
        self.assertIsInstance(result["nested"]["another_path"], str)
        self.assertIsInstance(result["list"][0], str)
        self.assertIsInstance(result["list"][1], str)
        self.assertEqual(result["nested"]["number"], 42)
        self.assertEqual(result["plain_string"], "unchanged")

    def test_is_file_property(self):
        """Test _is_file_property helper function."""

        # Should match _FILE pattern
        self.assertTrue(_is_file_property("LEAPSECONDS_FILE"))
        self.assertTrue(_is_file_property("INPUT_DATA_FILE_NAME"))
        self.assertTrue(_is_file_property("PCK_FILE"))

        # Should match explicit list
        self.assertTrue(_is_file_property("clock_kernel"))
        self.assertTrue(_is_file_property("frame_kernel"))
        self.assertTrue(_is_file_property("leapsecond_kernel"))
        self.assertTrue(_is_file_property("meta_kernel"))
        self.assertTrue(_is_file_property("planet_kernels"))

        # Should not match non-file properties
        self.assertFalse(_is_file_property("some_number"))
        self.assertFalse(_is_file_property("input_data"))
        self.assertFalse(_is_file_property("version"))


if __name__ == "__main__":
    unittest.main()

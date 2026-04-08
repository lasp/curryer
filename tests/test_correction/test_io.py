"""Tests for curryer.correction.io — unified path resolution."""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from curryer.correction.io import _download_from_s3, _temp_files, resolve_path


class TestResolvePathLocal:
    """Tests for local file resolution."""

    def test_existing_file_returns_path(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        result = resolve_path(f)
        assert result == f

    def test_existing_file_from_string(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        result = resolve_path(str(f))
        assert result == f

    def test_returns_path_type(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x")
        result = resolve_path(f)
        assert isinstance(result, Path)

    def test_nonexistent_local_file_raises_filenotfounderror(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            resolve_path("/nonexistent/path/to/file.csv")


class TestResolvePathS3:
    """Tests for S3 URI resolution with injected client."""

    def _make_client(self, content: str = "downloaded content") -> MagicMock:
        """Return a mock S3 client that writes *content* to the target path."""
        mock_client = MagicMock()

        def fake_download(bucket, key, local_path):
            Path(local_path).write_text(content)

        mock_client.download_file.side_effect = fake_download
        return mock_client

    def test_s3_uri_downloads_to_temp_file(self):
        mock_client = self._make_client("downloaded content")

        result = resolve_path("s3://my-bucket/path/to/file.mat", s3_client=mock_client)

        try:
            assert result.exists()
            assert result.read_text() == "downloaded content"
            assert result.suffix == ".mat"
            mock_client.download_file.assert_called_once_with("my-bucket", "path/to/file.mat", str(result))
        finally:
            result.unlink(missing_ok=True)

    def test_s3_uri_preserves_file_extension(self):
        mock_client = self._make_client("")

        result = resolve_path("s3://bucket/data/telemetry.nc", s3_client=mock_client)
        try:
            assert result.suffix == ".nc"
        finally:
            result.unlink(missing_ok=True)

    def test_s3_uri_no_key_raises_valueerror(self):
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="must include an object key"):
            resolve_path("s3://bucket-only", s3_client=mock_client)

    def test_s3_uri_empty_key_raises_valueerror(self):
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="must include an object key"):
            resolve_path("s3://bucket/", s3_client=mock_client)

    def test_s3_uri_without_boto3_raises_importerror(self, monkeypatch):
        """When no s3_client is injected AND boto3 is missing, ImportError is raised."""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pip install boto3"):
            resolve_path("s3://bucket/key/file.csv")

    def test_s3_download_failure_cleans_up_partial_temp_file(self):
        """On download failure the temp file must not be left on disk."""
        mock_client = MagicMock()
        mock_client.download_file.side_effect = RuntimeError("S3 failure")

        # Capture any temp path that would have been created
        created_paths: list[Path] = []
        import tempfile as _tempfile

        real_ntf = _tempfile.NamedTemporaryFile

        def capturing_ntf(*args, **kwargs):
            f = real_ntf(*args, **kwargs)
            created_paths.append(Path(f.name))
            return f

        import curryer.correction.io as _io

        original = _io.tempfile.NamedTemporaryFile
        _io.tempfile.NamedTemporaryFile = capturing_ntf
        try:
            with pytest.raises(RuntimeError, match="S3 failure"):
                _download_from_s3("s3://bucket/key/file.csv", s3_client=mock_client)
        finally:
            _io.tempfile.NamedTemporaryFile = original

        # Temp file should have been deleted by the except block
        for path in created_paths:
            assert not path.exists(), f"Temp file was not cleaned up: {path}"

    def test_temp_files_registered_for_atexit_cleanup(self):
        """Successful downloads register the temp path in _temp_files."""
        mock_client = self._make_client("")

        initial_count = len(_temp_files)
        result = resolve_path("s3://bucket/key/file.csv", s3_client=mock_client)
        try:
            assert len(_temp_files) == initial_count + 1
            assert _temp_files[-1] == result
        finally:
            result.unlink(missing_ok=True)
            if result in _temp_files:
                _temp_files.remove(result)

"""
Model Downloader

Downloads TTS model files with resume capability, multi-mirror fallback,
checksum verification, and progress tracking.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger("ModelDownloader")


@dataclass
class DownloadProgress:
    """Progress information for a download"""
    filename: str
    total_bytes: int
    downloaded_bytes: int
    speed_bps: float  # Bytes per second
    eta_seconds: float
    current_mirror: str
    status: str  # "downloading", "verifying", "completed", "failed"

    @property
    def percent(self) -> float:
        """Download percentage (0-100)"""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100

    @property
    def speed_mbps(self) -> float:
        """Download speed in MB/s"""
        return self.speed_bps / (1024 * 1024)


class ModelDownloader:
    """
    Model file downloader with resume and multi-mirror fallback

    Features:
    - Resume interrupted downloads
    - Multi-mirror fallback on failure
    - SHA256 checksum verification
    - Progress tracking with callbacks
    - Timeout and retry logic
    """

    def __init__(
        self,
        manifest_path: Path,
        download_dir: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        chunk_size: int = 8192,
        timeout: int = 300,  # Increased to 5 minutes for large files
        max_retries: int = 3
    ):
        """
        Initialize model downloader

        Args:
            manifest_path: Path to model_manifest.json
            download_dir: Directory to download models to
            progress_callback: Optional callback for progress updates
            chunk_size: Download chunk size in bytes
            timeout: Request timeout in seconds (applies to each chunk read)
            max_retries: Maximum retry attempts per mirror
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available - required for downloads")

        self.manifest_path = manifest_path
        self.download_dir = download_dir
        self.progress_callback = progress_callback
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.max_retries = max_retries

        # Load manifest
        self.manifest = self._load_manifest()

        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict:
        """Load model manifest from JSON file"""
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            logger.info(f"Loaded manifest: {len(manifest.get('models', {}))} models")
            return manifest
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            raise

    def _report_progress(self, progress: DownloadProgress):
        """Report progress via callback"""
        if self.progress_callback:
            try:
                self.progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify file SHA256 checksum

        Args:
            file_path: Path to file
            expected_sha256: Expected SHA256 hex digest

        Returns:
            True if checksum matches, False otherwise
        """
        logger.info(f"Verifying checksum for {file_path.name}...")

        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    sha256.update(chunk)

            actual = sha256.hexdigest()
            matches = actual == expected_sha256

            if matches:
                logger.info(f"✓ Checksum verified: {file_path.name}")
            else:
                logger.error(f"✗ Checksum mismatch for {file_path.name}")
                logger.error(f"  Expected: {expected_sha256}")
                logger.error(f"  Actual:   {actual}")

            return matches

        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

    def _download_with_resume(
        self,
        url: str,
        file_path: Path,
        expected_size: int,
        model_name: str
    ) -> bool:
        """
        Download file with resume capability

        Args:
            url: Download URL
            file_path: Destination file path
            expected_size: Expected file size in bytes
            model_name: Model name for progress reporting

        Returns:
            True if download successful, False otherwise
        """
        # Check if partial download exists
        start_byte = 0
        if file_path.exists():
            start_byte = file_path.stat().st_size
            if start_byte >= expected_size:
                logger.info(f"File already downloaded: {file_path.name}")
                return True
            logger.info(f"Resuming download from byte {start_byte}")

        # Prepare headers for resume
        headers = {}
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'

        try:
            # Start download
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Check if server supports resume
            if start_byte > 0 and response.status_code != 206:
                logger.warning("Server does not support resume, starting from beginning")
                start_byte = 0
                file_path.unlink(missing_ok=True)
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()

            # Open file in append or write mode
            mode = 'ab' if start_byte > 0 else 'wb'

            # Download with progress tracking
            downloaded = start_byte
            start_time = time.time()
            last_update = start_time

            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress every 0.5 seconds
                        now = time.time()
                        if now - last_update >= 0.5:
                            elapsed = now - start_time
                            speed = (downloaded - start_byte) / elapsed if elapsed > 0 else 0
                            remaining = expected_size - downloaded
                            eta = remaining / speed if speed > 0 else 0

                            progress = DownloadProgress(
                                filename=model_name,
                                total_bytes=expected_size,
                                downloaded_bytes=downloaded,
                                speed_bps=speed,
                                eta_seconds=eta,
                                current_mirror=url,
                                status="downloading"
                            )
                            self._report_progress(progress)
                            last_update = now

            # Final progress update
            elapsed = time.time() - start_time
            speed = (downloaded - start_byte) / elapsed if elapsed > 0 else 0
            progress = DownloadProgress(
                filename=model_name,
                total_bytes=expected_size,
                downloaded_bytes=downloaded,
                speed_bps=speed,
                eta_seconds=0,
                current_mirror=url,
                status="completed"
            )
            self._report_progress(progress)

            logger.info(f"Download completed: {file_path.name} ({downloaded} bytes in {elapsed:.1f}s)")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed from {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False

    def download_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Download a single model with multi-mirror fallback

        Args:
            model_name: Model name from manifest (e.g., "kokoro-v1.0.onnx")

        Returns:
            (success, message)
        """
        if model_name not in self.manifest['models']:
            return False, f"Model not found in manifest: {model_name}"

        model_info = self.manifest['models'][model_name]
        file_path = self.download_dir / model_info['filename']
        expected_size = model_info['size_bytes']
        expected_sha256 = model_info['sha256']
        mirrors = model_info['mirrors']

        logger.info(f"Downloading {model_name} ({expected_size / 1e6:.0f} MB)...")

        # Check if file already exists and is valid
        if file_path.exists():
            if file_path.stat().st_size == expected_size:
                if self._verify_checksum(file_path, expected_sha256):
                    logger.info(f"Model already downloaded and verified: {model_name}")
                    return True, "Already downloaded"
                else:
                    logger.warning(f"Existing file has invalid checksum, re-downloading")
                    file_path.unlink()

        # Try each mirror with retries
        for mirror_idx, mirror_url in enumerate(mirrors):
            logger.info(f"Trying mirror {mirror_idx + 1}/{len(mirrors)}: {mirror_url}")

            for attempt in range(self.max_retries):
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")

                # Attempt download
                success = self._download_with_resume(
                    mirror_url,
                    file_path,
                    expected_size,
                    model_name
                )

                if success:
                    # Verify checksum
                    progress = DownloadProgress(
                        filename=model_name,
                        total_bytes=expected_size,
                        downloaded_bytes=expected_size,
                        speed_bps=0,
                        eta_seconds=0,
                        current_mirror=mirror_url,
                        status="verifying"
                    )
                    self._report_progress(progress)

                    if self._verify_checksum(file_path, expected_sha256):
                        progress.status = "completed"
                        self._report_progress(progress)
                        return True, f"Downloaded from mirror {mirror_idx + 1}"
                    else:
                        logger.error(f"Checksum verification failed, trying next mirror")
                        file_path.unlink(missing_ok=True)
                        break  # Try next mirror

                # Download failed, retry or next mirror
                time.sleep(2 ** attempt)  # Exponential backoff

        # All mirrors failed
        progress = DownloadProgress(
            filename=model_name,
            total_bytes=expected_size,
            downloaded_bytes=file_path.stat().st_size if file_path.exists() else 0,
            speed_bps=0,
            eta_seconds=0,
            current_mirror="",
            status="failed"
        )
        self._report_progress(progress)

        return False, f"All mirrors failed after {self.max_retries} retries each"

    def download_all_models(self) -> Tuple[bool, List[str], List[str]]:
        """
        Download all models in manifest

        Returns:
            (all_success, successful_models, failed_models)
        """
        successful = []
        failed = []

        for model_name in self.manifest['models'].keys():
            success, message = self.download_model(model_name)

            if success:
                successful.append(model_name)
                logger.info(f"✓ {model_name}: {message}")
            else:
                failed.append(model_name)
                logger.error(f"✗ {model_name}: {message}")

        all_success = len(failed) == 0
        return all_success, successful, failed

    def get_total_download_size(self) -> int:
        """
        Get total download size for all models

        Returns:
            Total size in bytes
        """
        return self.manifest.get('total_download_size_bytes', 0)

    def get_models_for_feature(self, feature: str) -> List[str]:
        """
        Get list of models required for a specific feature

        Args:
            feature: Feature name (e.g., "tts")

        Returns:
            List of model names
        """
        features = self.manifest.get('features_enabled', {})
        return features.get(feature, [])

    def is_model_downloaded(self, model_name: str) -> bool:
        """
        Check if a model is already downloaded and verified

        Args:
            model_name: Model name from manifest

        Returns:
            True if downloaded and verified
        """
        if model_name not in self.manifest['models']:
            return False

        model_info = self.manifest['models'][model_name]
        file_path = self.download_dir / model_info['filename']

        if not file_path.exists():
            return False

        # Quick size check
        if file_path.stat().st_size != model_info['size_bytes']:
            return False

        # Full checksum verification (expensive)
        return self._verify_checksum(file_path, model_info['sha256'])

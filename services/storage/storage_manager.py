"""Storage abstraction layer for local and future cloud persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Protocol
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class StoredObject:
    """Represents a stored binary asset."""

    object_id: str
    file_name: str
    location: str
    metadata: dict[str, str]


class StorageError(Exception):
    """Raised when storage operations fail."""


class StorageBackend(Protocol):
    """Protocol for storage backends."""

    def save_bytes(self, file_name: str, content: bytes) -> StoredObject:
        """Persist bytes and return storage metadata."""

    def read_bytes(self, location: str) -> bytes:
        """Read raw bytes from persisted storage."""


class LocalStorageManager:
    """Persist uploaded documents on local disk in a cloud-portable manner."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, file_name: str, content: bytes) -> StoredObject:
        """Save uploaded bytes to disk with a collision-resistant name."""

        if not file_name:
            raise StorageError("file_name is required for storage.")

        object_id = uuid4().hex
        target_path = self.base_dir / f"{object_id}_{Path(file_name).name}"

        try:
            target_path.write_bytes(content)
        except OSError as exc:
            raise StorageError(f"Failed to save object locally: {exc}") from exc

        return StoredObject(
            object_id=object_id,
            file_name=Path(file_name).name,
            location=str(target_path),
            metadata={"backend": "local"},
        )

    def read_bytes(self, location: str) -> bytes:
        """Read bytes back from local storage."""

        try:
            return Path(location).expanduser().resolve().read_bytes()
        except OSError as exc:
            raise StorageError(f"Failed to read local object: {exc}") from exc

    def delete(self, location: str) -> None:
        """Delete a locally stored object if it exists."""

        path = Path(location).expanduser().resolve()
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise StorageError(f"Failed to delete local object: {exc}") from exc

    def copy_from_path(self, source_path: str | Path) -> StoredObject:
        """Copy an existing file into managed local storage."""

        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise StorageError("Source file does not exist.")

        object_id = uuid4().hex
        target_path = self.base_dir / f"{object_id}_{source.name}"
        try:
            shutil.copy2(source, target_path)
        except OSError as exc:
            raise StorageError(f"Failed to copy file into storage: {exc}") from exc

        return StoredObject(
            object_id=object_id,
            file_name=source.name,
            location=str(target_path),
            metadata={"backend": "local"},
        )


class S3StorageManager:
    """S3-backed storage implementation for future cloud deployment."""

    def __init__(self, bucket_name: str, region_name: str | None = None) -> None:
        self.bucket_name = bucket_name
        self.region_name = region_name

        try:
            import boto3
        except ImportError as exc:
            raise StorageError(
                "boto3 is required to use the S3 storage backend."
            ) from exc

        self._client = boto3.client("s3", region_name=region_name)

    def save_bytes(self, file_name: str, content: bytes) -> StoredObject:
        """Save bytes to S3 and return the resulting object metadata."""

        object_id = uuid4().hex
        key = f"uploads/{object_id}_{Path(file_name).name}"

        try:
            self._client.put_object(Bucket=self.bucket_name, Key=key, Body=content)
        except Exception as exc:  # noqa: BLE001 - external SDK exception surface
            raise StorageError(f"Failed to save object to S3: {exc}") from exc

        return StoredObject(
            object_id=object_id,
            file_name=Path(file_name).name,
            location=key,
            metadata={"backend": "s3", "bucket": self.bucket_name},
        )

    def read_bytes(self, location: str) -> bytes:
        """Read bytes from S3."""

        try:
            response = self._client.get_object(Bucket=self.bucket_name, Key=location)
            return response["Body"].read()
        except Exception as exc:  # noqa: BLE001 - external SDK exception surface
            raise StorageError(f"Failed to read object from S3: {exc}") from exc


def build_storage_manager(
    backend: str,
    local_base_dir: str | Path,
    bucket_name: str | None = None,
    region_name: str | None = None,
) -> StorageBackend:
    """Construct the configured storage backend."""

    if backend == "local":
        return LocalStorageManager(local_base_dir)

    if backend == "s3":
        if not bucket_name:
            raise StorageError("aws_bucket_name is required for the S3 backend.")
        return S3StorageManager(bucket_name=bucket_name, region_name=region_name)

    raise StorageError(f"Unsupported storage backend: {backend}")

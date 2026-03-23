"""Unit tests for local storage behavior."""

from pathlib import Path

from services.storage.storage_manager import LocalStorageManager


def test_local_storage_round_trip(tmp_path: Path) -> None:
    storage = LocalStorageManager(tmp_path)
    stored = storage.save_bytes("sample.txt", b"hello world")

    assert stored.file_name == "sample.txt"
    assert Path(stored.location).exists()
    assert storage.read_bytes(stored.location) == b"hello world"

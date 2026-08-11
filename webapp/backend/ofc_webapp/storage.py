"""Optional durable SQLite backup in a dedicated Google Cloud Storage object."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile


class GCSDatabaseBackup:
    """Restore/upload a consistent SQLite snapshot.

    The adapter is inert when ``OFC_GCS_BUCKET`` is not configured, which
    keeps local development self-contained.  Cloud Run config supplies a
    separate recording bucket and object name.
    """

    def __init__(
        self,
        bucket: str | None = None,
        object_name: str | None = None,
    ) -> None:
        self.bucket_name = (
            bucket if bucket is not None else os.environ.get("OFC_GCS_BUCKET", "")
        ).strip()
        self.object_name = (
            object_name
            if object_name is not None
            else os.environ.get("OFC_GCS_OBJECT", "state/ofc-webapp.sqlite3")
        ).strip()

    @property
    def enabled(self) -> bool:
        return bool(self.bucket_name)

    def restore(self, destination: Path) -> bool:
        if not self.enabled:
            return False
        from google.api_core.exceptions import NotFound
        from google.cloud import storage

        destination.parent.mkdir(parents=True, exist_ok=True)
        client = storage.Client()
        blob = client.bucket(self.bucket_name).blob(self.object_name)
        try:
            blob.download_to_filename(str(destination))
        except NotFound:
            return False
        return True

    def backup(self, source: Path) -> None:
        if not self.enabled:
            return
        from google.cloud import storage

        if not source.is_file():
            raise FileNotFoundError(f"SQLite database is missing: {source}")
        with tempfile.TemporaryDirectory(prefix="ofc-gcs-backup-") as directory:
            snapshot = Path(directory) / "snapshot.sqlite3"
            with sqlite3.connect(source) as origin, sqlite3.connect(snapshot) as copy:
                origin.backup(copy)
            client = storage.Client()
            blob = client.bucket(self.bucket_name).blob(self.object_name)
            blob.upload_from_filename(
                str(snapshot),
                content_type="application/vnd.sqlite3",
                timeout=120,
            )


__all__ = ["GCSDatabaseBackup"]

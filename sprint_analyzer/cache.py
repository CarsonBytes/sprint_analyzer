"""
Disk-based narrative cache.

Keyed by the SHA-256 of the CSV bytes that were uploaded/loaded. Each sprint
keeps its own cached narrative — switching between samples or uploading the
same file again will retrieve the previously-generated narrative.

The cache is invalidated implicitly whenever the underlying CSV changes
(different bytes → different hash → different cache file).

Stored as plain JSON under `.cache/narratives/<sha>.json` at the project root.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


_CACHE_BASE = Path(__file__).resolve().parent.parent / ".cache"
CACHE_ROOT = _CACHE_BASE / "narratives"
UPLOAD_ROOT = _CACHE_BASE / "uploads"


@dataclass
class CachedNarrative:
    narrative: str
    sprint_name: Optional[str]
    provider: str
    model: str
    extra_context: Optional[str]
    generated_at: str        # ISO-8601 UTC
    csv_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_cache_key(csv_bytes: bytes) -> str:
    """SHA-256 hex digest of the raw CSV bytes."""
    return hashlib.sha256(csv_bytes).hexdigest()


def _path_for(key: str) -> Path:
    return CACHE_ROOT / f"{key}.json"


def load(key: str) -> Optional[CachedNarrative]:
    """Return the cached narrative for `key`, or None if missing/corrupt."""
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CachedNarrative(**data)
    except Exception:
        return None


def save(
    key: str,
    *,
    narrative: str,
    sprint_name: Optional[str],
    provider: str,
    model: str,
    extra_context: Optional[str],
) -> CachedNarrative:
    """Persist a narrative to disk. Returns the stored record."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    record = CachedNarrative(
        narrative=narrative,
        sprint_name=sprint_name,
        provider=provider,
        model=model,
        extra_context=extra_context,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        csv_sha256=key,
    )
    _path_for(key).write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def delete(key: str) -> bool:
    """Remove a cached narrative. Returns True if a file was deleted."""
    path = _path_for(key)
    if path.exists():
        path.unlink()
        return True
    return False


def list_cached() -> list[CachedNarrative]:
    """Enumerate all cached narratives (useful for debugging / management UI)."""
    if not CACHE_ROOT.exists():
        return []
    out: list[CachedNarrative] = []
    for p in CACHE_ROOT.glob("*.json"):
        try:
            out.append(CachedNarrative(**json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


# ---------- Uploaded-file persistence ----------

@dataclass
class UploadRecord:
    sha256: str
    filename: str
    uploaded_at: str           # ISO-8601 UTC
    size_bytes: int


def _upload_data_path(key: str) -> Path:
    return UPLOAD_ROOT / f"{key}.bin"


def _upload_meta_path(key: str) -> Path:
    return UPLOAD_ROOT / f"{key}.json"


def save_upload(filename: str, csv_bytes: bytes) -> UploadRecord:
    """Persist an uploaded file to disk under its SHA-256. Idempotent."""
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    sha = compute_cache_key(csv_bytes)
    _upload_data_path(sha).write_bytes(csv_bytes)
    record = UploadRecord(
        sha256=sha,
        filename=filename,
        uploaded_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        size_bytes=len(csv_bytes),
    )
    _upload_meta_path(sha).write_text(
        json.dumps(record.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def load_upload(sha256: str) -> Optional[tuple[UploadRecord, bytes]]:
    """Return (metadata, raw bytes) for a previously-saved upload, or None."""
    meta_path = _upload_meta_path(sha256)
    data_path = _upload_data_path(sha256)
    if not meta_path.exists() or not data_path.exists():
        return None
    try:
        meta = UploadRecord(**json.loads(meta_path.read_text(encoding="utf-8")))
    except Exception:
        return None
    return meta, data_path.read_bytes()


def list_uploads() -> list[UploadRecord]:
    """Return all stored uploads, newest first."""
    if not UPLOAD_ROOT.exists():
        return []
    out: list[UploadRecord] = []
    for p in UPLOAD_ROOT.glob("*.json"):
        try:
            out.append(UploadRecord(**json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return sorted(out, key=lambda r: r.uploaded_at, reverse=True)


def delete_upload(sha256: str) -> bool:
    """Remove the upload's data and metadata files. Returns True if anything was deleted."""
    meta_path = _upload_meta_path(sha256)
    data_path = _upload_data_path(sha256)
    removed = False
    for p in (meta_path, data_path):
        if p.exists():
            p.unlink()
            removed = True
    return removed

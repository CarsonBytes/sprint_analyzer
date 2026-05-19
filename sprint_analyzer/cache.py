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


CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "narratives"


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

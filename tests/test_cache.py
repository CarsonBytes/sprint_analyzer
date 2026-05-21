"""
Tests for the narrative cache module.
Uses monkeypatch to redirect CACHE_ROOT into a tmp_path so tests don't
pollute the real .cache directory.
"""
import json
from pathlib import Path

import pytest

from sprint_analyzer import cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect every test in this module to use a tmp cache root."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "narratives")
    monkeypatch.setattr(cache, "UPLOAD_ROOT", tmp_path / "uploads")
    yield


class TestComputeCacheKey:
    def test_same_bytes_same_key(self):
        b = b"id,title\nT-1,Hello\n"
        assert cache.compute_cache_key(b) == cache.compute_cache_key(b)

    def test_different_bytes_different_key(self):
        a = b"id,title\nT-1,Hello\n"
        b = b"id,title\nT-1,World\n"
        assert cache.compute_cache_key(a) != cache.compute_cache_key(b)

    def test_key_is_hex_sha256(self):
        key = cache.compute_cache_key(b"x")
        assert len(key) == 64
        int(key, 16)  # should parse as hex


class TestLoadSavRoundTrip:
    def test_load_missing_returns_none(self):
        assert cache.load("deadbeef") is None

    def test_save_then_load_roundtrip(self):
        key = cache.compute_cache_key(b"some-csv")
        record = cache.save(
            key,
            narrative="## Hello",
            sprint_name="Sprint X",
            provider="anthropic",
            model="claude-sonnet-4-5",
            extra_context="prev sprint = 32 pts",
        )
        loaded = cache.load(key)
        assert loaded is not None
        assert loaded.narrative == "## Hello"
        assert loaded.sprint_name == "Sprint X"
        assert loaded.model == "claude-sonnet-4-5"
        assert loaded.extra_context == "prev sprint = 32 pts"
        assert loaded.csv_sha256 == key
        # generated_at should be ISO-8601-ish
        assert "T" in loaded.generated_at

    def test_save_creates_directory(self):
        # Cache root doesn't exist yet
        assert not cache.CACHE_ROOT.exists()
        cache.save(
            "abc",
            narrative="x",
            sprint_name=None,
            provider="anthropic",
            model="m",
            extra_context=None,
        )
        assert cache.CACHE_ROOT.exists()


class TestDelete:
    def test_delete_missing_returns_false(self):
        assert cache.delete("nope") is False

    def test_delete_existing_returns_true(self):
        key = "key1"
        cache.save(key, narrative="x", sprint_name=None,
                   provider="p", model="m", extra_context=None)
        assert cache.delete(key) is True
        assert cache.load(key) is None


class TestListCached:
    def test_empty(self):
        assert cache.list_cached() == []

    def test_returns_all_saved(self):
        for i in range(3):
            cache.save(f"k{i}", narrative=f"n{i}", sprint_name=None,
                       provider="p", model="m", extra_context=None)
        items = cache.list_cached()
        assert len(items) == 3
        assert {item.narrative for item in items} == {"n0", "n1", "n2"}

    def test_skips_corrupt_files(self):
        cache.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        (cache.CACHE_ROOT / "good.json").write_text(
            json.dumps({
                "narrative": "ok", "sprint_name": None, "provider": "p",
                "model": "m", "extra_context": None,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "csv_sha256": "abc",
            })
        )
        (cache.CACHE_ROOT / "corrupt.json").write_text("not json {{{")
        items = cache.list_cached()
        assert len(items) == 1
        assert items[0].narrative == "ok"


# ---------- Upload-cache tests ----------

class TestSaveUpload:
    def test_save_then_load_roundtrip(self):
        data = b"id,title,status\nT-1,Hello,Done\n"
        record = cache.save_upload("voyager.csv", data)
        assert record.filename == "voyager.csv"
        assert record.size_bytes == len(data)
        loaded = cache.load_upload(record.sha256)
        assert loaded is not None
        meta, raw = loaded
        assert meta.filename == "voyager.csv"
        assert raw == data

    def test_idempotent_save(self):
        data = b"id,title\nT-1,Hello\n"
        r1 = cache.save_upload("a.csv", data)
        r2 = cache.save_upload("a.csv", data)
        assert r1.sha256 == r2.sha256
        # Only one stored entry per hash
        assert len(cache.list_uploads()) == 1

    def test_save_creates_directory(self):
        assert not cache.UPLOAD_ROOT.exists()
        cache.save_upload("x.csv", b"data")
        assert cache.UPLOAD_ROOT.exists()


class TestListUploads:
    def test_empty(self):
        assert cache.list_uploads() == []

    def test_orders_newest_first(self):
        import time
        cache.save_upload("first.csv", b"1")
        time.sleep(1.05)  # uploaded_at is second-precision; force a gap
        cache.save_upload("second.csv", b"2")
        items = cache.list_uploads()
        assert len(items) == 2
        assert items[0].filename == "second.csv"
        assert items[1].filename == "first.csv"

    def test_skips_corrupt_meta(self):
        cache.UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        (cache.UPLOAD_ROOT / "bad.json").write_text("not json {{{")
        cache.save_upload("good.csv", b"good")
        items = cache.list_uploads()
        assert len(items) == 1
        assert items[0].filename == "good.csv"


class TestDeleteUpload:
    def test_delete_missing_returns_false(self):
        assert cache.delete_upload("nope") is False

    def test_delete_existing_removes_both_files(self):
        record = cache.save_upload("a.csv", b"data")
        assert cache.delete_upload(record.sha256) is True
        assert cache.load_upload(record.sha256) is None
        assert not cache._upload_data_path(record.sha256).exists()
        assert not cache._upload_meta_path(record.sha256).exists()


class TestLoadUpload:
    def test_missing_returns_none(self):
        assert cache.load_upload("deadbeef") is None

    def test_load_with_only_meta_missing_returns_none(self):
        # Simulate a half-corrupt state: data exists but meta is gone
        record = cache.save_upload("a.csv", b"data")
        cache._upload_meta_path(record.sha256).unlink()
        assert cache.load_upload(record.sha256) is None

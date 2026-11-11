"""Security and correctness tests for the on-disk LRS analysis cache."""

import pickle
import time

import pytest

from lrs.enterprise.performance_optimization import LRSCache


def test_cache_round_trip_and_index_reload(tmp_path):
    cache_dir = tmp_path / "nested" / "cache"
    cache = LRSCache(cache_dir=str(cache_dir))
    params = {"path": "example"}
    value = {"files": ["a.py", "b.py"], "success": True, "score": 0.75}

    cache.put("analyze", params, value)
    assert cache.get("analyze", params) == value

    reloaded = LRSCache(cache_dir=str(cache_dir))
    assert reloaded.get("analyze", params) == value


def test_cache_rejects_pickle_globals_in_index(tmp_path):
    (tmp_path / "cache_index.pkl").write_bytes(pickle.dumps(eval))

    cache = LRSCache(cache_dir=str(tmp_path))

    assert cache._cache_index == {}
    assert cache.get("missing", {}) is None


def test_cache_rejects_malformed_index_data(tmp_path):
    index = {"invalid": {"operation": "analyze"}}
    (tmp_path / "cache_index.pkl").write_bytes(pickle.dumps(index))

    cache = LRSCache(cache_dir=str(tmp_path))

    assert cache._cache_index == {}


def test_cache_rejects_pickle_globals_in_entry(tmp_path):
    cache = LRSCache(cache_dir=str(tmp_path))
    params = {"path": "example"}
    cache.put("analyze", params, {"safe": True})
    key = cache._get_cache_key("analyze", params)
    (tmp_path / f"{key}.pkl").write_bytes(pickle.dumps(eval))

    assert cache.get("analyze", params) is None


def test_cache_rejects_oversized_files_without_reading(tmp_path, monkeypatch):
    from lrs.enterprise import performance_optimization

    cache = LRSCache(cache_dir=str(tmp_path))
    cache.put("analyze", {"name": "x"}, {"value": "large"})
    key = cache._get_cache_key("analyze", {"name": "x"})
    monkeypatch.setattr(performance_optimization, "_MAX_CACHE_BYTES", 5)

    assert cache.get("analyze", {"name": "x"}) is None


def test_cache_eviction_works_when_capacity_is_one(tmp_path):
    cache = LRSCache(cache_dir=str(tmp_path), max_size=1)
    cache.put("analyze", {"id": 1}, {"value": 1})
    cache.put("analyze", {"id": 2}, {"value": 2})

    assert cache.get("analyze", {"id": 1}) is None
    assert cache.get("analyze", {"id": 2}) == {"value": 2}
    assert len(cache._cache_index) == 1


def test_cache_rejects_zero_capacity(tmp_path):
    with pytest.raises(ValueError, match="max_size"):
        LRSCache(cache_dir=str(tmp_path), max_size=0)


def test_cache_expiration_and_clear(tmp_path):
    cache = LRSCache(cache_dir=str(tmp_path))
    cache.put("analyze", {"id": 1}, {"value": 1})
    key = cache._get_cache_key("analyze", {"id": 1})
    cache._cache_index[key]["timestamp"] = time.time() - 86401

    assert cache.get("analyze", {"id": 1}) is None
    cache.clear()
    assert cache._cache_index == {}
    assert list(tmp_path.glob("*.pkl")) == [tmp_path / "cache_index.pkl"]

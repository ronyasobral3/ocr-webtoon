"""Tests for screen_capture.cache.TranslationCache."""
from __future__ import annotations

import json
import threading

import pytest

import screen_capture.cache as cache_module
from screen_capture.cache import TranslationCache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_FILE", tmp_path / "cache.json")
    return TranslationCache()


# ── chave de normalização ────────────────────────────────────────────────────

def test_key_is_case_insensitive(cache):
    cache.set("Hello", "Olá")
    assert cache.get("HELLO") == "Olá"
    assert cache.get("hello") == "Olá"


def test_key_strips_whitespace(cache):
    cache.set("  hello  ", "Olá")
    assert cache.get("hello") == "Olá"
    assert cache.get("  hello  ") == "Olá"


def test_key_case_and_strip_combined(cache):
    cache.set("  HELLO  ", "Olá")
    assert cache.get("hello") == "Olá"


# ── get / set / len ──────────────────────────────────────────────────────────

def test_get_missing_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_set_and_get_round_trip(cache):
    cache.set("I'm sorry", "Sinto muito")
    assert cache.get("I'm sorry") == "Sinto muito"


def test_len_empty(cache):
    assert len(cache) == 0


def test_len_after_set(cache):
    cache.set("a", "1")
    cache.set("b", "2")
    assert len(cache) == 2


def test_overwrite_same_key(cache):
    cache.set("hi", "oi")
    cache.set("hi", "olá")
    assert cache.get("hi") == "olá"
    assert len(cache) == 1


# ── persistência em disco ────────────────────────────────────────────────────

def test_flush_persists_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_FILE", tmp_path / "cache.json")
    c = TranslationCache()
    c.set("hello", "olá")
    c.flush()

    data = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert "olá" in data.values()


def test_set_alone_does_not_write_immediately(tmp_path, monkeypatch):
    # A escrita é adiada (debounce) — set() sozinho não deve tocar o disco.
    monkeypatch.setattr(cache_module, "_CACHE_FILE", tmp_path / "cache.json")
    c = TranslationCache()
    c.set("hello", "olá")
    assert not (tmp_path / "cache.json").exists()


def test_flush_without_changes_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_FILE", tmp_path / "cache.json")
    c = TranslationCache()
    c.flush()  # nada sujo — não deve criar arquivo nem lançar
    assert not (tmp_path / "cache.json").exists()


def test_load_reads_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    import hashlib
    key = hashlib.md5("hello".encode()).hexdigest()
    path.write_text(json.dumps({key: "olá"}), encoding="utf-8")

    monkeypatch.setattr(cache_module, "_CACHE_FILE", path)
    c = TranslationCache()
    assert c.get("hello") == "olá"


def test_load_ignores_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    path.write_text("not valid json", encoding="utf-8")

    monkeypatch.setattr(cache_module, "_CACHE_FILE", path)
    c = TranslationCache()
    assert len(c) == 0


# ── clear ────────────────────────────────────────────────────────────────────

def test_clear_empties_store(cache):
    cache.set("a", "1")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_clear_removes_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    monkeypatch.setattr(cache_module, "_CACHE_FILE", path)
    c = TranslationCache()
    c.set("x", "y")
    c.flush()
    assert path.exists()
    c.clear()
    assert not path.exists()


def test_clear_idempotent(cache):
    cache.clear()
    cache.clear()  # deve funcionar sem erro


# ── thread-safety ────────────────────────────────────────────────────────────

def test_concurrent_writes_do_not_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_FILE", tmp_path / "cache.json")
    c = TranslationCache()

    errors = []

    def worker(i):
        try:
            c.set(f"key{i}", f"val{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(c) == 20

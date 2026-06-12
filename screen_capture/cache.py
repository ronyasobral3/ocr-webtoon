from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

_CACHE_FILE = Path(__file__).parent.parent / ".translation_cache.json"


class TranslationCache:
    """Cache de tradução persistido em disco entre sessões."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    def _key(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _load(self) -> None:
        if _CACHE_FILE.exists():
            try:
                self._store = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
                logging.info("Cache de tradução carregado: %d entradas", len(self._store))
            except Exception as exc:
                logging.warning("Falha ao carregar cache: %s", exc)

    def _save(self) -> None:
        try:
            _CACHE_FILE.write_text(
                json.dumps(self._store, ensure_ascii=False, indent=None),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Falha ao salvar cache: %s", exc)

    def get(self, text: str) -> str | None:
        return self._store.get(self._key(text))

    def set(self, text: str, translation: str) -> None:
        with self._lock:
            self._store[self._key(text)] = translation
            self._save()

    def clear(self) -> None:
        self._store.clear()
        _CACHE_FILE.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._store)

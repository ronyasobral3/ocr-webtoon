from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
from pathlib import Path

_CACHE_FILE = Path(__file__).parent.parent / ".translation_cache.json"
# Janela de debounce da escrita em disco: agrupa os sets de um pipeline
# (vários balões) em uma única gravação, em vez de reescrever o JSON inteiro
# a cada tradução (O(n) por set, no meio do pipeline).
_FLUSH_DELAY_S = 2.0


class TranslationCache:
    """Cache de tradução persistido em disco entre sessões.

    `set()` só marca o cache como sujo; a gravação acontece num timer com
    debounce (e no exit do processo via atexit), então múltiplas traduções
    seguidas custam uma única escrita.
    """

    def __init__(self):
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._flush_timer: threading.Timer | None = None
        self._load()
        atexit.register(self.flush)

    def _key(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _load(self) -> None:
        if _CACHE_FILE.exists():
            try:
                self._store = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
                logging.info("Cache de tradução carregado: %d entradas", len(self._store))
            except Exception as exc:
                logging.warning("Falha ao carregar cache: %s", exc)

    def _schedule_flush_locked(self) -> None:
        """Agenda um flush futuro se ainda não houver um pendente. Chamar com o lock."""
        if self._flush_timer is None:
            timer = threading.Timer(_FLUSH_DELAY_S, self.flush)
            timer.daemon = True
            self._flush_timer = timer
            timer.start()

    def flush(self) -> None:
        """Grava o cache em disco se houver mudanças pendentes."""
        with self._lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
            if not self._dirty:
                return
            data = json.dumps(self._store, ensure_ascii=False, indent=None)
            self._dirty = False
        try:
            _CACHE_FILE.write_text(data, encoding="utf-8")
        except Exception as exc:
            logging.warning("Falha ao salvar cache: %s", exc)

    def get(self, text: str) -> str | None:
        return self._store.get(self._key(text))

    def set(self, text: str, translation: str) -> None:
        with self._lock:
            self._store[self._key(text)] = translation
            self._dirty = True
            self._schedule_flush_locked()

    def clear(self) -> None:
        with self._lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
            self._store.clear()
            self._dirty = False
        _CACHE_FILE.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._store)

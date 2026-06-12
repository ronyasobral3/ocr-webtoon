from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from deep_translator import GoogleTranslator

from .cache import TranslationCache


class Translator:
    def __init__(self, source: str = "en", target: str = "pt"):
        self._source = source
        self._target = target
        self._cache = TranslationCache()

    def translate(self, text: str) -> str:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        try:
            # Nova instância por chamada: sem estado compartilhado entre threads
            result = GoogleTranslator(source=self._source, target=self._target).translate(text)
        except Exception as exc:
            logging.warning("Tradução falhou: %s", exc)
            result = text

        logging.debug("  Tradução '%s': %.3fs", text[:40], time.perf_counter() - t0)
        self._cache.set(text, result)
        return result

    def translate_many(self, texts: list[str]) -> list[str]:
        """Traduz múltiplos textos em paralelo, um GoogleTranslator por thread."""
        if len(texts) <= 1:
            return [self.translate(t) for t in texts]
        with ThreadPoolExecutor(max_workers=len(texts)) as pool:
            return list(pool.map(self.translate, texts))

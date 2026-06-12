from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from deep_translator import GoogleTranslator

from .cache import TranslationCache


class _GoogleBackend:
    def __init__(self, source: str, target: str) -> None:
        self._source = source
        self._target = target

    def translate_batch(self, texts: list[str], _context: list[tuple[str, str]]) -> list[str]:
        def _one(text: str) -> str:
            try:
                return GoogleTranslator(source=self._source, target=self._target).translate(text)
            except Exception as exc:
                logging.warning("GoogleTranslator falhou: %s", exc)
                return text

        if len(texts) == 1:
            return [_one(texts[0])]
        with ThreadPoolExecutor(max_workers=len(texts)) as pool:
            return list(pool.map(_one, texts))


class _OllamaBackend:
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    def _build_prompt(self, texts: list[str], context: list[tuple[str, str]]) -> str:
        ctx_lines = ""
        if context:
            ctx_block = json.dumps(
                [{"en": e, "pt": p} for e, p in context[-10:]],
                ensure_ascii=False,
            )
            ctx_lines = f"\nRecent dialogue context (use for consistency and tone):\n{ctx_block}\n"

        texts_json = json.dumps(texts, ensure_ascii=False)
        n = len(texts)
        return (
            "You are an expert translator for Korean fantasy/action manhwa and manga.\n"
            f"Translate {n} English speech bubble(s) to natural, fluent Brazilian Portuguese (PT-BR).\n"
            "\n"
            "Guidelines:\n"
            "- Prioritize natural, idiomatic Portuguese over literal translation.\n"
            "  BAD: 'IS MINE TO LEAD' → 'É MEU PARA LIDERAR'  (literal)\n"
            "  GOOD: 'IS MINE TO LEAD' → 'É MEU PARA GOVERNAR' or 'CABE A MIM GOVERNAR'\n"
            "- The input may have OCR errors (garbled letters from stylized fonts). "
            "Infer the correct word from context and translate the intended meaning.\n"
            "- Preserve character names unchanged. Match the emotional intensity of the original.\n"
            "- '...' ellipsis at the start means the sentence continues from a previous bubble — keep it.\n"
            f"{ctx_lines}\n"
            f"Return ONLY a JSON array with exactly {n} string(s). No explanation, no markdown.\n"
            "Example: [\"Texto traduzido aqui\"]\n"
            "\n"
            f"Input: {texts_json}"
        )

    def translate_batch(self, texts: list[str], context: list[tuple[str, str]]) -> list[str]:
        try:
            import ollama  # optional dependency
        except ImportError:
            logging.error("Pacote 'ollama' não instalado. Execute: pip install ollama")
            return texts

        prompt = self._build_prompt(texts, context)
        try:
            t0 = time.perf_counter()
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 512},
            )
            raw = resp["message"]["content"].strip()
            logging.debug("Ollama %.3fs — resposta: %s", time.perf_counter() - t0, raw[:120])
            result = self._parse_json_array(raw, len(texts))
            if result is None:
                logging.warning("Ollama: parse falhou, usando Google Translate como fallback")
                return _GoogleBackend("en", "pt").translate_batch(texts, context)
            return self._fix_untranslated(texts, result, context)
        except Exception as exc:
            logging.warning("Ollama falhou (%s), usando Google Translate como fallback", exc)
            return _GoogleBackend("en", "pt").translate_batch(texts, context)

    def _is_untranslated(self, original: str, result: str) -> bool:
        """Retorna True se o resultado parece inglês (Ollama esqueceu de traduzir)."""
        if original.strip().lower() == result.strip().lower():
            return True
        orig_words = re.findall(r'[a-zA-Z]{4,}', original.lower())
        if not orig_words:
            return False
        res_words = set(re.findall(r'[a-zA-Z]{4,}', result.lower()))
        overlap = sum(1 for w in orig_words if w in res_words)
        return overlap / len(orig_words) >= 0.6

    def _fix_untranslated(
        self,
        originals: list[str],
        results: list[str],
        context: list[tuple[str, str]],
    ) -> list[str]:
        """Para cada item que parece não traduzido, substitui com Google Translate."""
        fallback_idx = [i for i, (o, r) in enumerate(zip(originals, results)) if self._is_untranslated(o, r)]
        if not fallback_idx:
            return results

        fallback_texts = [originals[i] for i in fallback_idx]
        logging.warning(
            "Ollama manteve inglês em %d item(ns), usando Google como fallback: %s",
            len(fallback_idx), fallback_texts,
        )
        fallback_results = _GoogleBackend("en", "pt").translate_batch(fallback_texts, context)

        final = list(results)
        for i, trans in zip(fallback_idx, fallback_results):
            final[i] = trans
        return final

    def _repair_json_array(self, raw: str) -> str:
        """Tenta fechar um array JSON truncado pelo modelo."""
        s = raw.strip()
        if not s.startswith("["):
            return s
        # conta strings abertas e fecha se necessário
        in_str = False
        escaped = False
        for ch in s:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_str = not in_str
        if in_str:
            s += '"'   # fecha string aberta
        if not s.endswith("]"):
            s += "]"   # fecha array
        return s

    def _parse_json_array(self, raw: str, expected: int) -> list[str] | None:
        # tentativa 1: JSON direto
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == expected:
                return [str(s) for s in parsed]
        except json.JSONDecodeError:
            pass

        # tentativa 2: extrair o primeiro [...] da resposta (modelo adicionou texto extra)
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and len(parsed) == expected:
                    return [str(s) for s in parsed]
            except json.JSONDecodeError:
                pass

        # tentativa 3: reparar JSON truncado (falta " ou ] no final)
        repaired = self._repair_json_array(raw)
        if repaired != raw:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, list) and len(parsed) == expected:
                    logging.debug("Ollama: JSON reparado com sucesso")
                    return [str(s) for s in parsed]
            except json.JSONDecodeError:
                pass

        # tentativa 4: resposta é uma string simples (sem array) — válido para expected==1
        if expected == 1:
            cleaned = raw.strip().strip('"').strip("'")
            if cleaned and not cleaned.startswith("["):
                logging.debug("Ollama: resposta simples usada diretamente")
                return [cleaned]

        return None  # sinaliza fallback

    def test_connection(self) -> tuple[bool, str]:
        try:
            import ollama
        except ImportError:
            return False, "Pacote 'ollama' não instalado. Execute: pip install ollama"
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": 'Translate "Hello" to PT-BR. Reply with one word only.'}],
                options={"temperature": 0},
            )
            word = resp["message"]["content"].strip()
            return True, f"OK — {self.model} respondeu: {word}"
        except Exception as exc:
            return False, str(exc)


class Translator:
    def __init__(self, source: str = "en", target: str = "pt") -> None:
        self._source = source
        self._target = target
        self._cache = TranslationCache()
        self._google = _GoogleBackend(source, target)
        self._ollama = _OllamaBackend()
        self._active: _GoogleBackend | _OllamaBackend = self._google
        self._context: deque[tuple[str, str]] = deque(maxlen=15)

    # ── configuração ──────────────────────────────────────────────────────────

    def set_backend(self, backend: str) -> None:
        if backend == "ollama":
            self._active = self._ollama
            logging.info("Tradutor: Ollama (%s @ %s)", self._ollama.model, self._ollama.host)
        else:
            self._active = self._google
            logging.info("Tradutor: Google Translate")

    def set_ollama_model(self, model: str) -> None:
        self._ollama.model = model
        logging.info("Ollama: modelo alterado para '%s'", model)

    def test_ollama(self) -> tuple[bool, str]:
        return self._ollama.test_connection()

    def clear_context(self) -> None:
        self._context.clear()
        logging.info("Janela de contexto do Ollama limpa.")

    @property
    def backend_name(self) -> str:
        return "ollama" if self._active is self._ollama else "google"

    # ── tradução ──────────────────────────────────────────────────────────────

    def translate_many(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        results: list[str | None] = [None] * len(texts)
        uncached_idx: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            t0 = time.perf_counter()
            translated = self._active.translate_batch(uncached_texts, list(self._context))
            logging.debug("Tradução (%s): %.3fs — %d texto(s)", self.backend_name, time.perf_counter() - t0, len(uncached_texts))
            for idx, orig, trans in zip(uncached_idx, uncached_texts, translated):
                results[idx] = trans
                self._cache.set(orig, trans)
                if self._is_clean_context_entry(orig, trans):
                    self._context.append((orig, trans))
                else:
                    logging.debug("Contexto ignorado (entrada suja): '%s' → '%s'", orig[:30], trans[:30])

        return [r if r is not None else t for r, t in zip(results, texts)]

    def _is_clean_context_entry(self, original: str, translated: str) -> bool:
        """Retorna False se a tradução parecer corrompida e não deve entrar no histórico."""
        # artefatos JSON ou lixo de parse
        if any(c in translated for c in ('[', ']', '{', '}')):
            return False
        # tradução vazia ou mínima
        if len(translated.strip()) < 2:
            return False
        # muito do texto original inglês foi preservado sem traduzir
        # (palavras >3 chars do original presentes na tradução = provável não-tradução)
        orig_words = {w.lower() for w in re.findall(r'[a-zA-Z]{4,}', original)}
        trans_words = {w.lower() for w in re.findall(r'[a-zA-Z]{4,}', translated)}
        if orig_words and len(orig_words & trans_words) / len(orig_words) > 0.5:
            return False
        return True

    # compatibilidade com código antigo que chamava translate() diretamente
    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

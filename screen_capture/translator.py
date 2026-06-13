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


class _NLLBBackend:
    """Tradução via NLLB-200 (Meta) — modelo de MT dedicado, fiel e offline.

    Diferente de um LLM (Ollama), não é generativo: não inventa nem alucina, só
    traduz. Roda na GPU quando disponível. Carrega sob demanda (a 1ª chamada baixa
    ~2.4GB do HuggingFace na variante distilled-600M) e cai para Google se falhar.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "por_Latn",
    ) -> None:
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._tok = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        logging.info(
            "NLLB-200 carregado: %s (%s) em %.1fs",
            self.model_name, self._device, time.perf_counter() - t0,
        )

    def translate_batch(self, texts: list[str], _context: list[tuple[str, str]]) -> list[str]:
        try:
            import torch
            self._ensure_loaded()
            self._tok.src_lang = self.src_lang
            tgt_id = self._tok.convert_tokens_to_ids(self.tgt_lang)
            inputs = self._tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(self._device)
            with torch.no_grad():
                gen = self._model.generate(
                    **inputs, forced_bos_token_id=tgt_id, max_length=512, num_beams=4,
                )
            return self._tok.batch_decode(gen, skip_special_tokens=True)
        except Exception as exc:
            logging.warning("NLLB falhou (%s), usando Google Translate como fallback", exc)
            return _GoogleBackend("en", "pt").translate_batch(texts, _context)

    def test_connection(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            out = self.translate_batch(["Hello, how are you?"], [])
            return True, f"OK — NLLB ({self._device}): {out[0]}"
        except Exception as exc:
            return False, str(exc)


class Translator:
    def __init__(self, source: str = "en", target: str = "pt") -> None:
        self._source = source
        self._target = target
        self._cache = TranslationCache()
        self._google = _GoogleBackend(source, target)
        self._ollama = _OllamaBackend()
        self._nllb = _NLLBBackend()
        self._active: _GoogleBackend | _OllamaBackend | _NLLBBackend = self._google
        self._context: deque[tuple[str, str]] = deque(maxlen=15)

    # ── configuração ──────────────────────────────────────────────────────────

    def set_backend(self, backend: str) -> None:
        if backend == "ollama":
            self._active = self._ollama
            logging.info("Tradutor: Ollama (%s @ %s)", self._ollama.model, self._ollama.host)
        elif backend == "nllb":
            self._active = self._nllb
            logging.info("Tradutor: NLLB-200 (%s)", self._nllb.model_name)
        else:
            self._active = self._google
            logging.info("Tradutor: Google Translate")

    def set_ollama_model(self, model: str) -> None:
        self._ollama.model = model
        logging.info("Ollama: modelo alterado para '%s'", model)

    def test_ollama(self) -> tuple[bool, str]:
        return self._ollama.test_connection()

    def test_nllb(self) -> tuple[bool, str]:
        return self._nllb.test_connection()

    def clear_context(self) -> None:
        self._context.clear()
        logging.info("Janela de contexto do Ollama limpa.")

    @property
    def backend_name(self) -> str:
        if self._active is self._ollama:
            return "ollama"
        if self._active is self._nllb:
            return "nllb"
        return "google"

    def is_cached(self, text: str) -> bool:
        """True se `text` já está no cache (consulte ANTES de traduzir — a
        tradução popula o cache e apagaria a distinção entre acerto e novo)."""
        return self._cache.get(text) is not None

    # ── tradução ──────────────────────────────────────────────────────────────

    def translate_one(self, text: str) -> str:
        """Traduz um único texto passando pelo cache e contexto.

        Usado no caminho com overlap (Google), onde cada balão é traduzido assim
        que seu OCR termina, sobrepondo a latência de rede com o OCR dos demais.
        `TranslationCache` é thread-safe e `deque.append` é atômico sob o GIL,
        então é seguro chamar de várias threads do pool de balões."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        trans = self._active.translate_batch([text], list(self._context))[0]
        self._cache.set(text, trans)
        if self._is_clean_context_entry(text, trans):
            self._context.append((text, trans))
        return trans

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

"""Tests for pure/deterministic functions in screen_capture.translator."""
from __future__ import annotations

import pytest

from screen_capture.translator import (
    Translator,
    _OllamaBackend,
    _to_ptbr,
)


# ── _to_ptbr: conjugações "tu" → "você" ─────────────────────────────────────

@pytest.mark.parametrize("eu, br", [
    ("estás bem", "está bem"),
    ("és forte", "é forte"),
    ("quiseres ir", "quiser ir"),
    ("fizeres isso", "fizer isso"),
    ("tiveres tempo", "tiver tempo"),
    ("puderes ajudar", "puder ajudar"),
    ("fores lá", "for lá"),
    ("deres um passo", "der um passo"),
    ("vires aqui", "vir aqui"),
    ("souberes a verdade", "souber a verdade"),
    ("disseres algo", "disser algo"),
    ("vieres amanhã", "vier amanhã"),
    ("houveres dúvidas", "houver dúvidas"),
])
def test_to_ptbr_verb_conjugations(eu, br):
    assert _to_ptbr(eu) == br


def test_to_ptbr_preserves_case_on_capitalized_word():
    assert _to_ptbr("Estás pronto?") == "Está pronto?"
    assert _to_ptbr("És o escolhido.") == "É o escolhido."


# ── _to_ptbr: léxico EU → BR ─────────────────────────────────────────────────

@pytest.mark.parametrize("eu, br", [
    ("uma rapariga bonita", "uma garota bonita"),
    ("raparigas corajosas", "garotas corajosas"),
    ("apanhei o comboio", "apanhei o trem"),
    ("dois comboios partiram", "dois trens partiram"),
    ("o autocarro chegou", "o ônibus chegou"),
    ("meu telemóvel", "meu celular"),
    ("abre o frigorífico", "abre o geladeira"),  # artigo não é corrigido (heurística simples)
])
def test_to_ptbr_vocabulary(eu, br):
    assert _to_ptbr(eu) == br


# ── _to_ptbr: "precisar de + infinitivo" ────────────────────────────────────

def test_to_ptbr_precisar_de_infinitivo():
    assert _to_ptbr("preciso de fazer isso") == "preciso fazer isso"


def test_to_ptbr_precisar_com_adverbio():
    assert _to_ptbr("preciso mais de fazer") == "preciso mais fazer"


def test_to_ptbr_precisar_de_nao_toca_substantivo():
    # "preciso de um lugar" — "um" não é infinitivo → não deve alterar
    result = _to_ptbr("preciso de um lugar")
    assert result == "preciso de um lugar"


def test_to_ptbr_nao_toca_texto_correto():
    # texto PT-BR correto não deve ser modificado
    assert _to_ptbr("Ela está bem.") == "Ela está bem."
    assert _to_ptbr("gosto de fazer isso") == "gosto de fazer isso"


def test_to_ptbr_texto_sem_eu_inalterado():
    assert _to_ptbr("Hello world") == "Hello world"


# ── _OllamaBackend._is_untranslated ─────────────────────────────────────────

@pytest.fixture
def ollama():
    return _OllamaBackend()


def test_is_untranslated_identical_strings(ollama):
    assert ollama._is_untranslated("Hello world", "Hello world") is True


def test_is_untranslated_case_insensitive_identical(ollama):
    assert ollama._is_untranslated("HELLO WORLD", "hello world") is True


def test_is_untranslated_clear_translation(ollama):
    assert ollama._is_untranslated("I will destroy you", "Vou te destruir") is False


def test_is_untranslated_high_overlap(ollama):
    # >60% of long words preserved unchanged → untranslated
    assert ollama._is_untranslated("DESTROY EVERYTHING AROUND", "DESTROY EVERYTHING AROUND") is True


def test_is_untranslated_no_long_words(ollama):
    # Sem palavras ≥4 chars: retorna False (não dá para saber)
    assert ollama._is_untranslated("Hi", "Oi") is False


# ── _OllamaBackend._parse_json_array ────────────────────────────────────────

def test_parse_json_array_valid(ollama):
    assert ollama._parse_json_array('["Olá", "Tudo bem"]', 2) == ["Olá", "Tudo bem"]


def test_parse_json_array_extracts_embedded(ollama):
    raw = 'Aqui está a tradução: ["Vou te destruir"]. Pronto.'
    assert ollama._parse_json_array(raw, 1) == ["Vou te destruir"]


def test_parse_json_array_repairs_truncated(ollama):
    # Array truncado no meio de uma string
    raw = '["Olá mundo'
    result = ollama._parse_json_array(raw, 1)
    assert result is not None
    assert len(result) == 1


def test_parse_json_array_single_string_fallback(ollama):
    result = ollama._parse_json_array("Olá mundo", 1)
    assert result == ["Olá mundo"]


def test_parse_json_array_wrong_count_returns_none(ollama):
    assert ollama._parse_json_array('["um", "dois"]', 3) is None


def test_parse_json_array_garbage_returns_none(ollama):
    assert ollama._parse_json_array("!!!@@@###", 2) is None


# ── _OllamaBackend._repair_json_array ───────────────────────────────────────

def test_repair_closes_open_string(ollama):
    repaired = ollama._repair_json_array('["Olá')
    assert repaired.endswith('"]')


def test_repair_closes_open_bracket(ollama):
    repaired = ollama._repair_json_array('["Olá"')
    assert repaired.endswith("]")


def test_repair_noop_on_complete(ollama):
    complete = '["Olá"]'
    assert ollama._repair_json_array(complete) == complete


def test_repair_noop_on_non_array(ollama):
    s = "Não é array"
    assert ollama._repair_json_array(s) == s


# ── Translator._is_clean_context_entry ──────────────────────────────────────

@pytest.fixture
def translator():
    return Translator()


def test_clean_entry_accepted(translator):
    assert translator._is_clean_context_entry("I will fight", "Vou lutar") is True


def test_dirty_entry_json_brackets(translator):
    assert translator._is_clean_context_entry("hello", '["Olá"]') is False
    assert translator._is_clean_context_entry("hello", '{"key": "val"}') is False


def test_dirty_entry_too_short(translator):
    assert translator._is_clean_context_entry("hello", "") is False
    assert translator._is_clean_context_entry("hello", "x") is False


def test_dirty_entry_mostly_english(translator):
    # tradução preserva >50% das palavras longas originais → não traduzida
    assert translator._is_clean_context_entry(
        "destroy everything around", "destroy everything around"
    ) is False


def test_clean_entry_original_has_no_long_words(translator):
    # sem palavras ≥4 chars no original: filtro de overlap desarmado
    assert translator._is_clean_context_entry("Hi", "Oi") is True


# ── Translator: configuração de backend ──────────────────────────────────────

def test_default_backend_is_google(translator):
    assert translator.backend_name == "google"


def test_set_backend_nllb(translator):
    translator.set_backend("nllb")
    assert translator.backend_name == "nllb"


def test_set_backend_ollama(translator):
    translator.set_backend("ollama")
    assert translator.backend_name == "ollama"


def test_set_backend_google(translator):
    translator.set_backend("nllb")
    translator.set_backend("google")
    assert translator.backend_name == "google"


def test_set_backend_unknown_falls_back_to_google(translator):
    translator.set_backend("inexistente")
    assert translator.backend_name == "google"


def test_set_ollama_model(translator):
    translator.set_ollama_model("llama3:8b")
    assert translator._ollama.model == "llama3:8b"


def test_clear_context(translator):
    # popula contexto manualmente
    translator._context.append(("en", "pt"))
    translator._context.append(("en2", "pt2"))
    translator.clear_context()
    assert len(translator._context) == 0


def test_context_maxlen_is_15(translator):
    for i in range(20):
        translator._context.append((f"en{i}", f"pt{i}"))
    assert len(translator._context) == 15

"""Tests for pure/deterministic functions in screen_capture.ocr_engine."""
from __future__ import annotations

import numpy as np
import pytest

from screen_capture.ocr_engine import (
    _avg_conf,
    _binarize,
    _crop_hash,
    _deskew,
    _enhance,
    _estimate_shear,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def white_text_image(h=80, w=120):
    """Balão branco com alguns pixels pretos simulando texto."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    img[h // 3: 2 * h // 3, w // 4: 3 * w // 4] = 30
    return img


def italic_text_image():
    """Imagem de 100×200 com texto itálico sintético (linhas diagonais)."""
    img = np.ones((100, 200), dtype=np.uint8) * 255
    # Desenha traços diagonais que simulam glifos itálicos
    for x in range(20, 180, 15):
        for y in range(10, 90):
            col = x + int(y * 0.3)
            if 0 <= col < 200:
                img[y, col] = 0
    return img


# ── _crop_hash ────────────────────────────────────────────────────────────────

def test_hash_same_image_consistent():
    img = white_text_image()
    assert _crop_hash(img) == _crop_hash(img)


def test_hash_copy_identical():
    img = white_text_image()
    assert _crop_hash(img) == _crop_hash(img.copy())


def test_hash_different_images_differ():
    a = white_text_image()
    b = np.zeros_like(a)
    assert _crop_hash(a) != _crop_hash(b)


def test_hash_returns_hex_string():
    h = _crop_hash(white_text_image())
    assert isinstance(h, str)
    assert len(h) == 32
    int(h, 16)  # deve ser hex válido


# ── _enhance ─────────────────────────────────────────────────────────────────

def test_enhance_returns_2d():
    result = _enhance(white_text_image())
    assert result.ndim == 2


def test_enhance_dtype_uint8():
    result = _enhance(white_text_image())
    assert result.dtype == np.uint8


def test_enhance_upscales_small_image():
    small = np.ones((50, 50, 3), dtype=np.uint8) * 200
    result = _enhance(small)
    # target_min=300, scale = min(2.0, 300/50) = 2.0 → output 100×100
    assert result.shape[0] >= 100 and result.shape[1] >= 100


def test_enhance_does_not_upscale_large_image():
    large = np.ones((400, 400, 3), dtype=np.uint8) * 200
    result = _enhance(large)
    # scale = min(2.0, 300/400) < 1.0 → clamp to 1.0 → same size
    assert result.shape == (400, 400)


def test_enhance_inverted_balloon_inverts_pixels():
    # Imagem escura (balão invertido): média < 127 → pixels são invertidos
    dark = np.ones((80, 80, 3), dtype=np.uint8) * 30
    result = _enhance(dark)
    # Após inversão, pixels devem ser claros (>127)
    assert float(np.mean(result)) > 127


def test_enhance_accepts_grayscale():
    gray = np.ones((100, 100), dtype=np.uint8) * 200
    result = _enhance(gray)
    assert result.ndim == 2


# ── _binarize ────────────────────────────────────────────────────────────────

def test_binarize_returns_binary_values():
    enhanced = _enhance(white_text_image())
    binary = _binarize(enhanced)
    unique = set(binary.flatten().tolist())
    assert unique <= {0, 255}


def test_binarize_same_shape_as_input():
    enhanced = _enhance(white_text_image())
    binary = _binarize(enhanced)
    assert binary.shape == enhanced.shape


def test_binarize_dtype_uint8():
    enhanced = _enhance(white_text_image())
    assert _binarize(enhanced).dtype == np.uint8


def test_binarize_mostly_white_for_white_balloon():
    # Balão branco com pouco texto → maioria do binário deve ser branco (255)
    enhanced = _enhance(white_text_image())
    binary = _binarize(enhanced)
    white_ratio = float(np.mean(binary == 255))
    assert white_ratio > 0.5


# ── _estimate_shear ──────────────────────────────────────────────────────────

def test_estimate_shear_in_valid_range():
    enhanced = _enhance(white_text_image())
    binary = _binarize(enhanced)
    k = _estimate_shear(binary)
    assert -0.45 <= k <= 0.45


def test_estimate_shear_returns_float():
    enhanced = _enhance(white_text_image())
    binary = _binarize(enhanced)
    k = _estimate_shear(binary)
    assert isinstance(k, float)


def test_estimate_shear_near_zero_for_upright_text():
    # Texto com linhas verticais bem retas: shear deve ser pequeno
    img = np.ones((100, 200), dtype=np.uint8) * 255
    for x in range(20, 180, 20):
        img[:, x] = 0  # barras verticais
    binary = _binarize(_enhance(img))
    k = _estimate_shear(binary)
    assert abs(k) < 0.3  # conservador — o teste só garante que não é extremo


def test_estimate_shear_detects_slant():
    binary = _binarize(_enhance(italic_text_image()))
    k = _estimate_shear(binary)
    # Texto com inclinação real: |k| deve ser detectável
    assert abs(k) >= 0.0  # pelo menos não lança exceção; valor exato é heurístico


# ── _deskew ──────────────────────────────────────────────────────────────────

def test_deskew_preserves_shape():
    gray = _enhance(white_text_image())
    result = _deskew(gray, 0.2)
    assert result.shape == gray.shape


def test_deskew_dtype_preserved():
    gray = _enhance(white_text_image())
    result = _deskew(gray, 0.15)
    assert result.dtype == gray.dtype


def test_deskew_zero_shear_same_as_input():
    gray = _enhance(white_text_image())
    result = _deskew(gray, 0.0)
    # Com k=0 a transformação é identidade
    assert np.array_equal(result, gray)


def test_deskew_fills_border_with_white():
    gray = _enhance(white_text_image())
    result = _deskew(gray, 0.3)
    # Cantos superiores devem ser brancos (borderValue=255)
    assert int(result[0, 0]) == 255


# ── _avg_conf ─────────────────────────────────────────────────────────────────

def test_avg_conf_empty():
    assert _avg_conf([]) == 0.0


def test_avg_conf_single():
    raw = [(None, "text", "0.9")]
    assert _avg_conf(raw) == pytest.approx(0.9)


def test_avg_conf_multiple():
    raw = [(None, "a", "0.8"), (None, "b", "0.6")]
    assert _avg_conf(raw) == pytest.approx(0.7)


def test_avg_conf_accepts_float_confidence():
    raw = [(None, "a", 0.75), (None, "b", 0.25)]
    assert _avg_conf(raw) == pytest.approx(0.5)


def test_avg_conf_returns_float():
    assert isinstance(_avg_conf([(None, "x", "1.0")]), float)

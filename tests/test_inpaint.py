"""Tests for screen_capture.inpaint.inpaint_text."""
from __future__ import annotations

import numpy as np
import pytest

from screen_capture.inpaint import inpaint_text


# ── rejeições antecipadas ────────────────────────────────────────────────────

def test_returns_none_for_none():
    assert inpaint_text(None) is None


def test_returns_none_for_empty_array():
    assert inpaint_text(np.array([])) is None


def test_returns_none_for_zero_size():
    assert inpaint_text(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_returns_none_for_too_small_height():
    img = np.ones((4, 100, 3), dtype=np.uint8) * 255
    assert inpaint_text(img) is None


def test_returns_none_for_too_small_width():
    img = np.ones((100, 4, 3), dtype=np.uint8) * 255
    assert inpaint_text(img) is None


def test_returns_none_for_5x5_exactly():
    img = np.ones((5, 5, 3), dtype=np.uint8) * 255
    assert inpaint_text(img) is None


def test_accepts_6x6():
    img = np.ones((6, 6, 3), dtype=np.uint8) * 255
    # Pode retornar None (máscara trivial) ou tuple — não deve lançar exceção
    result = inpaint_text(img)
    assert result is None or isinstance(result, tuple)


# ── rejeição por máscara excessiva (>55%) ───────────────────────────────────

def test_returns_none_for_dense_checkerboard():
    # Checkerboard: após Otsu + dilation, máscara cobre ~80% → aborta
    img = np.zeros((100, 100), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    assert inpaint_text(img) is None


# ── caminho normal: balão branco com texto preto ─────────────────────────────

@pytest.fixture
def white_balloon_with_text():
    """Balão branco 100×100 com bloco de texto preto 20×20 no centro."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img[40:60, 40:60] = 0
    return img


def test_normal_balloon_returns_tuple(white_balloon_with_text):
    result = inpaint_text(white_balloon_with_text)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_normal_balloon_clean_is_ndarray(white_balloon_with_text):
    clean, _, _ = inpaint_text(white_balloon_with_text)
    assert isinstance(clean, np.ndarray)
    assert clean.shape == white_balloon_with_text.shape


def test_normal_balloon_center_in_range(white_balloon_with_text):
    h, w = white_balloon_with_text.shape[:2]
    _, center, _ = inpaint_text(white_balloon_with_text)
    assert center is not None
    cx, cy = center
    assert 0 <= cx <= w
    assert 0 <= cy <= h


def test_normal_balloon_center_near_text_block(white_balloon_with_text):
    # O centroide deve estar próximo do bloco 40:60,40:60 → centro ~(50, 50)
    _, center, _ = inpaint_text(white_balloon_with_text)
    cx, cy = center
    assert 35 <= cx <= 65
    assert 35 <= cy <= 65


def test_normal_balloon_interior_is_rgb_tuple(white_balloon_with_text):
    _, _, interior = inpaint_text(white_balloon_with_text)
    assert isinstance(interior, tuple)
    assert len(interior) == 3
    assert all(isinstance(v, int) for v in interior)
    assert all(0 <= v <= 255 for v in interior)


def test_normal_balloon_interior_is_light(white_balloon_with_text):
    # Interior do balão é branco → cor amostrada deve ser clara
    _, _, (r, g, b) = inpaint_text(white_balloon_with_text)
    assert r > 150 and g > 150 and b > 150


# ── imagem grayscale (ndim == 2) ─────────────────────────────────────────────

def test_accepts_grayscale_input():
    img = np.ones((80, 80), dtype=np.uint8) * 255
    img[30:50, 30:50] = 0
    result = inpaint_text(img)
    assert result is None or isinstance(result, tuple)

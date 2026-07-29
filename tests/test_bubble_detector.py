"""Tests for pure/deterministic functions in screen_capture.bubble_detector."""
from __future__ import annotations

import numpy as np
import pytest

from screen_capture.bubble_detector import (
    _iou,
    _remove_overlapping,
    _sample_bg_color,
    _isolate_bubbles,
)


# ── _iou ────────────────────────────────────────────────────────────────────

def test_iou_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert _iou(a, b) == 0.0


def test_iou_identical_boxes():
    a = (0, 0, 10, 10)
    assert _iou(a, a) == pytest.approx(1.0)


def test_iou_half_overlap_horizontal():
    a = (0, 0, 10, 10)   # 10×10 = 100
    b = (5, 0, 15, 10)   # 10×10 = 100, intersect = 5×10 = 50
    # union = 100 + 100 - 50 = 150 → iou = 50/150 ≈ 0.333
    assert _iou(a, b) == pytest.approx(50 / 150)


def test_iou_fully_contained():
    outer = (0, 0, 20, 20)   # 400
    inner = (5, 5, 15, 15)   # 100
    # intersect = 100, union = 400 + 100 - 100 = 400 → iou = 0.25
    assert _iou(outer, inner) == pytest.approx(100 / 400)


def test_iou_touching_edge_no_overlap():
    a = (0, 0, 10, 10)
    b = (10, 0, 20, 10)
    assert _iou(a, b) == 0.0


def test_iou_symmetric(
):
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    assert _iou(a, b) == pytest.approx(_iou(b, a))


# ── _remove_overlapping ──────────────────────────────────────────────────────

def test_remove_overlapping_empty():
    assert _remove_overlapping([]) == []


def test_remove_overlapping_single():
    assert _remove_overlapping([(0, 0, 10, 10)]) == [(0, 0, 10, 10)]


def test_remove_overlapping_keeps_largest():
    big = (0, 0, 20, 20)    # 400
    small = (1, 1, 15, 15)  # 196 — sobreposto com big
    result = _remove_overlapping([big, small])
    assert big in result
    assert small not in result


def test_remove_overlapping_keeps_non_overlapping():
    a = (0, 0, 10, 10)
    b = (50, 50, 60, 60)
    result = _remove_overlapping([a, b])
    assert len(result) == 2
    assert a in result
    assert b in result


def test_remove_overlapping_threshold():
    # IoU exatamente na fronteira: dois boxes com IoU = 0.5 (>0.3 → remove)
    a = (0, 0, 10, 10)  # 100
    b = (5, 0, 15, 10)  # 100, intersect=50, union=150, iou≈0.33 → remove
    result = _remove_overlapping([a, b])
    assert len(result) == 1


# ── _sample_bg_color ─────────────────────────────────────────────────────────

def test_sample_bg_color_uniform_white():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    r, g, b = _sample_bg_color(img)
    assert r == 255 and g == 255 and b == 255


def test_sample_bg_color_uniform_black():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    r, g, b = _sample_bg_color(img)
    assert r == 0 and g == 0 and b == 0


def test_sample_bg_color_bgr_to_rgb_conversion():
    # Cria imagem com B=10, G=20, R=30 (todos pixels) — esperado: (R=30, G=20, B=10)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 10  # B
    img[:, :, 1] = 20  # G
    img[:, :, 2] = 30  # R
    r, g, b = _sample_bg_color(img)
    assert r == 30 and g == 20 and b == 10


def test_sample_bg_color_uses_corners():
    # Centro diferente dos cantos — deve retornar cor dos cantos
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # cantos cinza
    img[20:80, 20:80] = 50  # centro escuro
    r, g, b = _sample_bg_color(img)
    assert r > 100  # cor dos cantos (200), não do centro (50)


def test_sample_bg_color_returns_tuple_of_three_ints():
    img = np.ones((60, 60, 3), dtype=np.uint8) * 128
    result = _sample_bg_color(img)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(v, int) for v in result)


# ── _isolate_bubbles ─────────────────────────────────────────────────────────

def test_isolate_bubbles_all_white_does_not_crash():
    # Imagem toda branca sem anéis de balão: função não deve lançar exceção.
    # A dilatação da borda preta do padding cria um anel zero ao redor,
    # mas a região interior permanece branca (bg_labels vazio → sem remoção).
    binary = np.ones((100, 100), dtype=np.uint8) * 255
    result = _isolate_bubbles(binary)
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8


def test_isolate_bubbles_preserves_isolated_bubble():
    # Círculo branco no centro com borda preta ao redor → balão isolado
    binary = np.zeros((100, 100), dtype=np.uint8)
    import cv2
    cv2.circle(binary, (50, 50), 20, 255, -1)
    result = _isolate_bubbles(binary)
    # Centro deve permanecer branco
    assert result[50, 50] == 255


def test_isolate_bubbles_output_same_size():
    binary = np.zeros((80, 120), dtype=np.uint8)
    result = _isolate_bubbles(binary)
    assert result.shape == binary.shape

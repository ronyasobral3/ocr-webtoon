from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

_MIN_CONFIDENCE = 0.4
_YOLO_IMGSZ = 640  # 1024 é o tamanho de treino mas 640 é ~2.5× mais rápido na CPU sem perda relevante para balões grandes

_HF_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
_HF_FILE = "comic-speech-bubble-detector.pt"


def _sample_bg_color(crop: np.ndarray) -> tuple[int, int, int]:
    """Amostra cor de fundo do balão pelos quatro cantos do crop (BGR→RGB)."""
    h, w = crop.shape[:2]
    s = max(1, min(h, w) // 6)
    corners = np.concatenate([
        crop[:s, :s].reshape(-1, 3),
        crop[:s, w - s:].reshape(-1, 3),
        crop[h - s:, :s].reshape(-1, 3),
        crop[h - s:, w - s:].reshape(-1, 3),
    ])
    bgr = np.median(corners, axis=0).astype(int)
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _download_bubble_model() -> Path | None:
    """Baixa o modelo do HuggingFace Hub na primeira execução (cache local)."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        return Path(path)
    except Exception as exc:
        logging.warning("Não foi possível baixar o modelo YOLOv8 (%s) — usando detector OpenCV.", exc)
        return None


def _isolate_bubbles(binary: np.ndarray) -> np.ndarray:
    """Isola interiores de balões removendo o fundo da página.

    Adiciona padding preto para selar balões cortados pelas bordas da imagem
    (caso contrário, o interior branco do balão se conecta ao fundo branco da
    página pela borda da frame). Identifica o fundo da página como a(s)
    componente(s) branca(s) que tocam os quatro lados da imagem original —
    o fundo envolve tudo, enquanto balões (mesmo unidos em figura-8 ou
    cortados pela borda) tocam no máximo 1-2 lados."""
    pad = 2
    padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    dark = 255 - padded
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.dilate(dark, k, iterations=2)
    sealed = 255 - dark

    _, labels = cv2.connectedComponents(sealed, connectivity=4)

    # Bordas da imagem ORIGINAL em coordenadas pós-padding
    h, w = sealed.shape
    inner_top, inner_bottom = pad, h - pad - 1
    inner_left, inner_right = pad, w - pad - 1

    top = set(labels[inner_top, inner_left:inner_right + 1].tolist())
    bottom = set(labels[inner_bottom, inner_left:inner_right + 1].tolist())
    left = set(labels[inner_top:inner_bottom + 1, inner_left].tolist())
    right = set(labels[inner_top:inner_bottom + 1, inner_right].tolist())

    # Fundo da página = componentes brancas que tocam os 4 lados (label 0 = preto)
    bg_labels = (top & bottom & left & right) - {0}

    if bg_labels:
        mask = np.isin(labels, list(bg_labels))
        sealed[mask] = 0

    return sealed[pad:-pad, pad:-pad]


def _binary_from_gray(detection_gray: np.ndarray) -> np.ndarray:
    """Threshold adaptativo + supressão de preto puro → binary pronto para _isolate_bubbles."""
    binary = cv2.adaptiveThreshold(
        detection_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    # Em regiões uniformes escuras, threshold ≈ pixel-10 → todo pixel passa → fundo
    # preto vira branco. Percentil-3 estima o assoalho escuro; suprime até p3+15.
    p3 = int(np.percentile(detection_gray, 3))
    if p3 < 30:
        binary[detection_gray <= p3 + 15] = 0
    return _isolate_bubbles(binary)


def _binary_dark_from_gray(gray: np.ndarray) -> np.ndarray:
    """Threshold global para isolar interiores muito escuros (balões invertidos).

    Usa threshold absoluto (< 60) em vez de adaptativo: o interior escuro
    de um balão invertido é genuinamente preto (0-30), bem distinto de qualquer
    fundo colorido (fogo, batalha etc., que fica em 80+). O threshold adaptativo
    falha aqui porque inverte o fundo colorido e produz binário ruidoso."""
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    return _isolate_bubbles(mask)


def _analyze_contours(
    binary: np.ndarray,
    brightness_gray: np.ndarray,
    frame_hw: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Filtra contornos de um binary por forma, brilho e presença de texto.

    brightness_gray — gray original da imagem, usado para medir brilho/texto
    real do balão (independente de como o binary foi gerado).
    """
    h, w = frame_hw
    frame_area = w * h
    min_area = frame_area * 0.005
    max_area = frame_area * 0.70
    border = 1

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)

        if area < min_area or area > max_area:
            continue
        if x <= border and y <= border and (x + bw) >= (w - border) and (y + bh) >= (h - border):
            continue
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.2 or aspect > 4:
            continue
        if bh > h * 0.85 or bw > w * 0.85:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        hull_circularity = (4 * np.pi * hull_area / (hull_perimeter ** 2)) if hull_perimeter > 0 else 0
        if hull_circularity < 0.35:
            continue

        # Mede brightness/text nos pixels DENTRO do contorno sobre o gray ORIGINAL —
        # evita que cantos pretos do bounding rect derrubem a média em fundos escuros.
        roi = np.zeros(brightness_gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(roi, [cnt], -1, 255, cv2.FILLED)
        interior = brightness_gray[roi > 0]
        if interior.size == 0:
            continue
        brightness = float(np.mean(interior))

        is_normal   = brightness >= 190
        is_inverted = brightness <= 80
        if not is_normal and not is_inverted:
            continue

        text_ratio = np.mean(interior > 180) if is_inverted else np.mean(interior < 80)
        if text_ratio < 0.02:
            continue

        logging.debug("  ACEITO box=(%d,%d,%d,%d) hull_circ=%.2f bright=%.1f text=%.3f inv=%s",
                      x, y, x+bw, y+bh, hull_circularity, brightness, text_ratio, is_inverted)
        candidates.append((x, y, x + bw, y + bh))

    return candidates


def _detect_from_gray(
    detection_gray: np.ndarray,
    brightness_gray: np.ndarray,
    frame_hw: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    return _analyze_contours(_binary_from_gray(detection_gray), brightness_gray, frame_hw)


def _opencv_detect(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detecta balões de fala em dois passes.

    Passe normal (adaptativo): interiores CLAROS isolados por borda escura —
    balões brancos clássicos e balões em fundo preto sólido.

    Passe escuro (threshold global): interiores MUITO ESCUROS (< 60) isolados
    das bordas da imagem — balões pretos em fundos coloridos (cenas de ação,
    fogo). Threshold absoluto evita o ruído que "255-gray" gera ao inverter
    fundos coloridos saturados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hw = image.shape[:2]

    normal = _detect_from_gray(gray, gray, hw)
    dark   = _analyze_contours(_binary_dark_from_gray(gray), gray, hw)

    logging.debug("  [detector] frame=%dx%d normal=%d dark=%d",
                  hw[1], hw[0], len(normal), len(dark))

    return _remove_overlapping(normal + dark)


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union entre dois boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _remove_overlapping(
    boxes: list[tuple], iou_threshold: float = 0.3
) -> list[tuple]:
    """Remove boxes sobrepostos, mantendo o maior."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for box in boxes:
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


class BubbleDetector:
    """
    Detecta balões de fala.
    Por padrão faz download automático do modelo YOLOv8 treinado em speech
    bubbles (ogkalu/comic-speech-bubble-detector-yolov8m, ~52 MB, cached).
    Cai para o detector OpenCV se o download ou carga falhar.
    Passe `model_path` explicitamente para usar um modelo local.
    """

    def __init__(self, model_path: str | Path | None = None):
        self._model = None

        resolved = Path(model_path) if model_path is not None else _download_bubble_model()

        if resolved is not None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(resolved))
                logging.info("Modelo YOLOv8 carregado: %s", resolved.name)
            except Exception as exc:
                logging.warning("Falha ao carregar modelo YOLOv8 (%s) — usando detector OpenCV.", exc)

    @property
    def using_yolo(self) -> bool:
        return self._model is not None

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._model is not None:
            results = self._model(image, verbose=False, conf=_MIN_CONFIDENCE, imgsz=_YOLO_IMGSZ)
            boxes = [tuple(map(int, b)) for b in results[0].boxes.xyxy.cpu().numpy()]
            if boxes:
                return boxes
            logging.debug("YOLOv8 retornou 0 detecções — tentando detector OpenCV.")
        return _opencv_detect(image)

    def crop_bubbles(
        self, image: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray, tuple[int, int, int]]]:
        boxes = self.detect(image)
        result = []
        for box in boxes:
            crop = image[box[1]: box[3], box[0]: box[2]]
            bg_color = _sample_bg_color(crop) if crop.size > 0 else (255, 255, 255)
            result.append((box, crop, bg_color))
        return result

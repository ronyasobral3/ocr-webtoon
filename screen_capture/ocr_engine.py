from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

_MIN_CONFIDENCE = 0.5
_MIN_ALNUM_RATIO = 0.4
_VOWELS = frozenset("aeiouAEIOU")


def _crop_hash(img: np.ndarray) -> str:
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _enhance(image: np.ndarray) -> np.ndarray:
    """Upscale + MIN(R,G,B) + CLAHE + unsharp → gray normalizado.

    Retorna grayscale (texto escuro sobre fundo claro) sem converter para BGR,
    permitindo reusar o resultado tanto no caminho padrão quanto no binarizado.

    - Lanczos4: preserva bordas de glifos melhor que CUBIC em fontes decorativas.
    - MIN(R,G,B): mantém texto colorido (laranja, vermelho) como pixel escuro.
    - CLAHE: normaliza contraste em fundos com gradiente ou textura.
    - Unsharp: realça bordas de glifos itálicos/negrito para segmentação."""
    h, w = image.shape[:2]

    target_min = 300
    scale = min(2.0, max(1.0, target_min / min(h, w)))
    if scale > 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    gray = np.min(image, axis=2).astype(np.uint8) if image.ndim == 3 else image.copy()

    if np.mean(gray) < 127:
        # Balão invertido: apara 8% de cada borda antes de inverter para remover
        # o glow/borda branca que, após inversão, vira cinza escuro e o OCR lê
        # como texto fantasma (ex: "SIHL" de artefatos da borda oval).
        mh, mw = max(1, h // 12), max(1, w // 12)
        gray = gray[mh:h - mh, mw:w - mw]
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = 255 - gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


def _binarize(enhanced: np.ndarray) -> np.ndarray:
    """Binarização Otsu sobre o gray já melhorado.

    Fontes com sombra criam pixels cinza ao redor dos glifos. Otsu mapeia
    esse halo cinza para branco (fundo), deixando apenas o núcleo escuro de
    cada caractere. Close 2×2 fecha brechas dentro dos traços causadas por
    sombras ou outlines grossos."""
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)


def _avg_conf(raw) -> float:
    return sum(float(r[2]) for r in raw) / len(raw) if raw else 0.0


class OCREngine:
    def __init__(self):
        self._engine = RapidOCR()
        self._cache: dict[str, list[dict]] = {}

    def extract(self, image: np.ndarray) -> list[dict]:
        key = _crop_hash(image)
        if key in self._cache:
            return self._cache[key]
        result = self._extract_uncached(image)
        self._cache[key] = result
        return result

    def _ocr_pass(self, gray: np.ndarray):
        # use_cls=False: texto de webtoon é sempre na horizontal; o passo de
        # classificação de ângulo do RapidOCR é uma inferência extra inútil aqui.
        raw, _ = self._engine(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), use_cls=False)
        return raw

    def _extract_uncached(self, image: np.ndarray) -> list[dict]:
        enhanced = _enhance(image)
        binary = _binarize(enhanced)

        # Dois passes concorrentes — padrão (enhanced) e binarizado (Otsu) — e
        # ficamos com o de maior confiança média. O binarizado remove o halo
        # cinza de fontes com sombra/outline grosso, recuperando glifos que o
        # passe padrão erra. Rodar os dois em paralelo corta a latência de pior
        # caso quando há cores livres (poucos balões); em páginas densas o ganho
        # diminui porque o onnxruntime já satura os cores por inferência.
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_bin = pool.submit(self._ocr_pass, binary)
            raw = self._ocr_pass(enhanced)
            raw_bin = f_bin.result()
        if _avg_conf(raw_bin) > _avg_conf(raw):
            raw = raw_bin

        if not raw:
            return []

        detections = []
        for box, text, confidence in raw:
            if float(confidence) < _MIN_CONFIDENCE:
                continue
            text = text.strip()
            if len(text) < 2:
                continue
            alnum_ratio = sum(c.isalnum() for c in text) / len(text)
            if alnum_ratio < _MIN_ALNUM_RATIO:
                continue
            # Rejeita tokens sem nenhuma vogal — lixo de OCR (ex: "Lsnr", "w,i")
            words = [w for w in text.split() if len(w) > 1]
            if words and not any(_VOWELS & set(w) for w in words):
                continue
            detections.append({
                "text": text,
                "box": [list(map(int, pt)) for pt in box],
                "confidence": float(confidence),
            })

        return detections

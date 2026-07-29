from __future__ import annotations

import hashlib
import logging
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


def _binarize_adaptive(enhanced: np.ndarray) -> np.ndarray:
    """Limiarização adaptativa (janela 51px) para fontes decorativas/manuscritas.

    Melhor que o Otsu global quando o fundo tem gradiente, textura ou quando a
    fonte tem outline grosso — o threshold é calculado localmente por vizinhança,
    então diferenças globais de iluminação não interferem."""
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 8
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)


def _binarize_blackhat(enhanced: np.ndarray) -> np.ndarray:
    """Black top-hat isola strokes escuros em fundo texturizado ou com gradiente lento.

    `close(img) - img` extrai feições escuras menores que o kernel — remove o fundo
    gradiente antes do threshold, deixando só o núcleo dos glifos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh)


def _avg_conf(raw) -> float:
    return sum(float(r[2]) for r in raw) / len(raw) if raw else 0.0


def _estimate_shear(binary: np.ndarray) -> float:
    """Estima o cisalhamento do texto (itálico) maximizando a variância da
    projeção vertical: glifos itálicos endireitados alinham a tinta de cada
    coluna → picos mais nítidos. Texto já reto tem ótimo em k≈0, então a
    estimativa é autolimitada e não distorce o que já está bom.

    `binary` = Otsu (texto=0, fundo=255). Retorna k em [-0.45, 0.45]."""
    ink = cv2.bitwise_not(binary)  # texto = 255
    h, w = ink.shape
    best_k, best_score = 0.0, -1.0
    for k in np.arange(-0.45, 0.46, 0.075):
        # Largura fixa (desloca -k*h/2 para centrar) → variâncias comparáveis.
        M = np.float32([[1, k, -k * h / 2], [0, 1, 0]])
        sheared = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST)
        score = float(sheared.sum(axis=0, dtype=np.float64).var())
        if score > best_score:
            best_score, best_k = score, float(k)
    return best_k


def _deskew(gray: np.ndarray, k: float) -> np.ndarray:
    """Endireita texto itálico aplicando o cisalhamento estimado (fundo claro)."""
    h, w = gray.shape
    M = np.float32([[1, k, -k * h / 2], [0, 1, 0]])
    return cv2.warpAffine(gray, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


class OCREngine:
    def __init__(self, force_extra_passes: bool = False):
        self._engine = RapidOCR()
        self._cache: dict[str, list[dict]] = {}
        # Manga EN: sempre ativa os passes adaptativos sem esperar por baixa confiança.
        self._force_extra = force_extra_passes

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

        # Passes padrão: enhanced (CLAHE+unsharp) e binarizado Otsu.
        # Texto itálico/inclinado: adiciona passe endireitado se |shear| >= 0.12.
        candidates = [enhanced, binary]
        shear = _estimate_shear(binary)
        if abs(shear) >= 0.12:
            candidates.append(_deskew(enhanced, shear))

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            raws = list(pool.map(self._ocr_pass, candidates))
        best_raw = max(raws, key=_avg_conf)

        # Passes extras para fontes decorativas/manuscritas. Ativados sempre em
        # modo Manga EN (force_extra=True) ou quando a confiança padrão é baixa
        # (< 0.65) — webtoon com fonte limpa (0.8+) não paga custo adicional.
        if self._force_extra or _avg_conf(best_raw) < 0.65:
            extra = [_binarize_adaptive(enhanced), _binarize_blackhat(enhanced)]
            with ThreadPoolExecutor(max_workers=len(extra)) as pool:
                extra_raws = list(pool.map(self._ocr_pass, extra))
            raw = max(raws + extra_raws, key=_avg_conf)
        else:
            raw = best_raw

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


class MangaOCREngine:
    """OCR especializado para mangá japonês via manga-ocr (transformers).

    Lida nativamente com texto vertical e horizontal em japonês. Retorna o
    mesmo formato de lista de dicts que OCREngine para ser intercambiável no
    pipeline. O modelo (~400 MB) é baixado automaticamente no primeiro uso.
    """

    def __init__(self) -> None:
        self._mocr = None
        self._cache: dict[str, list[dict]] = {}

    def _ensure_loaded(self) -> None:
        if self._mocr is not None:
            return
        from manga_ocr import MangaOcr  # pip install manga-ocr
        logging.info("Carregando MangaOCR (primeira execução pode demorar)...")
        self._mocr = MangaOcr()
        logging.info("MangaOCR carregado.")

    def extract(self, image: np.ndarray) -> list[dict]:
        key = _crop_hash(image)
        if key in self._cache:
            return self._cache[key]
        result = self._extract_uncached(image)
        self._cache[key] = result
        return result

    def _extract_uncached(self, image: np.ndarray) -> list[dict]:
        from PIL import Image
        try:
            self._ensure_loaded()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            text = self._mocr(pil).strip()
            if not text:
                return []
            h, w = image.shape[:2]
            box = [[0, 0], [w, 0], [w, h], [0, h]]
            return [{"text": text, "box": box, "confidence": 1.0}]
        except Exception as exc:
            logging.warning("MangaOCR falhou: %s", exc)
            return []

from __future__ import annotations

import cv2
import numpy as np


def inpaint_text(
    crop: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float] | None, tuple[int, int, int]] | None:
    """Remove o texto de um balão por inpainting.

    Devolve `(fundo_limpo, centro, cor_interior)`:

    - `centro` = (cx, cy) em coordenadas do crop, o centroide da tinta do texto
      original. O overlay usa isso para ancorar a tradução onde estava o lettering
      — relevante em balões "boneco de neve" (dois ovais empilhados), onde o centro
      geométrico cai no pescoço vazio entre os ovais, mas o centroide da tinta puxa
      para o oval com mais texto.
    - `cor_interior` = (R,G,B) amostrada do fundo já limpo ao redor do centroide.
      É a cor REAL do interior do balão, usada para decidir a cor do texto por
      contraste. Os cantos do crop (que `_sample_bg_color` usa) podem cair fora do
      oval, sobre fundo escuro, e inverter a escolha — deixando texto claro sobre
      interior branco (ilegível).

    Diferente de cobrir o balão com um retângulo opaco, isto reconstrói o
    *próprio* fundo do balão (cor, gradiente, textura) e preserva o contorno
    oval — então o overlay redesenha apenas o texto traduzido por cima, e o
    resultado acompanha a forma real do balão capturado.

    A máscara de texto vem do contraste com o fundo (Otsu): texto escuro sobre
    fundo claro em balões normais, texto claro sobre fundo escuro nos invertidos.
    Não depende das caixas do OCR (que vivem no espaço já reescalado do
    `ocr_engine`), o que mantém o alinhamento pixel-perfeito com o crop original.

    Retorna `None` quando não há o que limpar com segurança (crop minúsculo,
    máscara cobrindo quase tudo → provável arte, não texto) para o overlay cair
    no preenchimento sólido tradicional.
    """
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 6 or w < 6:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    inverted = float(np.mean(gray)) < 127

    # Otsu separa texto e fundo. Em balão normal o texto fica escuro (0); no
    # invertido fica claro (255). Normalizamos para máscara onde texto = 255.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = otsu if inverted else cv2.bitwise_not(otsu)

    # Dilata para cobrir o anti-aliasing/halo em volta dos glifos.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_mask = cv2.dilate(text_mask, k, iterations=2)

    # Zera um anel nas bordas para não apagar o contorno do balão (mantém o oval
    # visível) nem inpaint contra vizinhos de fora do balão.
    b = max(2, min(h, w) // 18)
    text_mask[:b, :] = 0
    text_mask[-b:, :] = 0
    text_mask[:, :b] = 0
    text_mask[:, -b:] = 0

    # Se a máscara cobre quase tudo, o crop provavelmente contém arte (cabelo,
    # cenário) e não um balão limpo — abortar é mais seguro que borrar.
    if float(np.mean(text_mask > 0)) > 0.55:
        return None

    ys, xs = np.nonzero(text_mask)
    center = (float(xs.mean()), float(ys.mean())) if xs.size else None

    bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    clean = cv2.inpaint(np.ascontiguousarray(bgr), text_mask, 4, cv2.INPAINT_TELEA)

    # Cor do interior do balão: mediana de um patch no fundo limpo ao redor do
    # centroide (garante estar DENTRO do balão, não nos cantos externos).
    cx, cy = (int(center[0]), int(center[1])) if center else (w // 2, h // 2)
    rad = max(3, min(h, w) // 10)
    patch = clean[max(0, cy - rad):cy + rad, max(0, cx - rad):cx + rad].reshape(-1, 3)
    med = np.median(patch, axis=0).astype(int) if patch.size else np.array([255, 255, 255])
    interior = (int(med[2]), int(med[1]), int(med[0]))  # BGR → RGB

    return clean, center, interior

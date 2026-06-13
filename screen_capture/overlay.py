from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QRect, QRectF
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QWidget


class OverlayLabel:
    """Representa uma caixa de tradução posicionada sobre um balão.

    `bg_image` (quando presente) é o balão original com o texto removido por
    inpainting — o overlay o pinta no lugar do balão e desenha o texto traduzido
    por cima, preservando forma/cor/textura. Sem ele, cai no retângulo opaco.
    """

    def __init__(
        self,
        rect: QRect,
        text: str,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        bg_image: QImage | None = None,
        text_center: tuple[float, float] | None = None,
    ):
        self.rect = rect
        self.text = text
        self.bg_color = bg_color
        self.bg_image = bg_image
        # (cx, cy) em coords do crop: centroide da tinta original; ancora o texto
        # traduzido no oval certo de balões "boneco de neve".
        self.text_center = text_center


def _np_to_qimage(bgr: np.ndarray) -> QImage | None:
    """Converte um crop BGR (OpenCV) em QImage RGB independente do buffer numpy."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return None
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # BGR → RGB
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return qimg.copy()  # destaca do buffer numpy, que será coletado


def _fit_font_size(text: str, avail: QRect, family: str = "Arial Black") -> int:
    """Maior tamanho de fonte (pt) em que o texto, com quebra de linha, cabe na
    área útil `avail` (já com margem). Teto baixo (16pt) para o lettering não
    encostar nas bordas do balão e parecer natural — sem o teto, o tamanho fica
    limitado só pela largura e o texto enche o balão (parece esticado)."""
    if avail.width() < 4 or avail.height() < 4 or not text:
        return 8

    flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
    lo, hi, best = 7, 16, 7
    while lo <= hi:
        mid = (lo + hi) // 2
        font = QFont(family, mid)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.5)
        br = QFontMetrics(font).boundingRect(avail, flags, text)
        if br.width() <= avail.width() and br.height() <= avail.height():
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class OverlayWindow(QWidget):
    """
    Janela transparente fullscreen que renderiza as traduções sobre os balões.
    - Sem bordas e sempre no topo.
    - Click-through: eventos de mouse passam para as janelas abaixo.
    """

    def __init__(self):
        super().__init__()
        self._labels: list[OverlayLabel] = []
        self._offset_x = 0
        self._offset_y = 0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")

        geo = QApplication.primaryScreen().geometry()
        self.setGeometry(geo)
        self._offset_x = -geo.x()
        self._offset_y = -geo.y()
        self.show()

    def reposition(self, screen) -> None:
        """Move o overlay para cobrir o monitor indicado, limpando labels antigas."""
        self._labels = []
        geo = screen.geometry()
        self.setGeometry(geo)
        # Labels usam coordenadas absolutas de tela; offset converte para coords locais
        self._offset_x = -geo.x()
        self._offset_y = -geo.y()

    def update_labels(self, labels: list[OverlayLabel]) -> None:
        self._labels = labels
        self.update()

    def clear(self) -> None:
        self._labels = []
        self.update()

    def paintEvent(self, _event) -> None:
        if not self._labels:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        for label in self._labels:
            rect = label.rect.translated(self._offset_x, self._offset_y)
            r, g, b = label.bg_color

            if label.bg_image is not None and not label.bg_image.isNull():
                # Balão real com texto removido — pinta no lugar do balão original.
                painter.drawImage(rect, label.bg_image)
            else:
                # Fallback: retângulo arredondado opaco com a cor amostrada.
                radius = min(rect.width(), rect.height()) * 0.22
                path = QPainterPath()
                path.addRoundedRect(QRectF(rect), radius, radius)
                painter.fillPath(path, QColor(r, g, b))
                border = QColor(160, 160, 160) if (r + g + b) < 382 else QColor(90, 90, 90)
                painter.setPen(QPen(border, 1.5))
                painter.drawPath(path)

            # Texto traduzido: caixa-alta estilo lettering, cor por luminância do fundo.
            is_dark = (r + g + b) < 382
            text_color = QColor(245, 245, 245) if is_dark else QColor(15, 15, 15)
            display = label.text.upper()

            # Margem proporcional: o balão é oval, então o texto precisa de folga
            # nas bordas (~15% horizontal, ~14% vertical) para não estourar a curva.
            pad_x = max(8, int(rect.width() * 0.15))
            pad_y = max(8, int(rect.height() * 0.14))
            avail = rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)

            font_size = _fit_font_size(display, avail)
            font = QFont("Arial Black", font_size)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.5)
            painter.setFont(font)
            painter.setPen(text_color)

            # Ancora o texto verticalmente no centroide da tinta original (quando
            # disponível): centraliza nesse y mantendo a simetria dentro de `avail`.
            draw_rect = avail
            if label.text_center is not None:
                cy = rect.top() + int(label.text_center[1])
                cy = max(avail.top(), min(avail.bottom(), cy))
                half = min(cy - avail.top(), avail.bottom() - cy)
                if half > 6:
                    draw_rect = QRect(avail.left(), cy - half, avail.width(), 2 * half)

            painter.drawText(
                draw_rect,
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                display,
            )


def build_labels(
    detections: list[dict],
    region_origin: tuple[int, int],
) -> list[OverlayLabel]:
    """
    Converte detecções {box, translated_text, bg_color, clean_image} em
    OverlayLabel com posição absoluta na tela.
    """
    ox, oy = region_origin
    labels = []
    for det in detections:
        box = det["box"]  # [[x,y], [x,y], [x,y], [x,y]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        rect = QRect(ox + x1, oy + y1, x2 - x1, y2 - y1)
        bg_color = det.get("bg_color", (255, 255, 255))
        clean = det.get("clean_image")
        bg_image = _np_to_qimage(clean) if clean is not None else None
        text_center = det.get("text_center")
        labels.append(OverlayLabel(rect, det["translated_text"], bg_color, bg_image, text_center))
    return labels

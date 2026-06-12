from __future__ import annotations

from PyQt6.QtCore import Qt, QRect, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QWidget


class OverlayLabel:
    """Representa uma caixa de tradução posicionada sobre um balão."""

    def __init__(
        self,
        rect: QRect,
        text: str,
        bg_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.rect = rect
        self.text = text
        self.bg_color = bg_color


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

        for label in self._labels:
            rect = label.rect.translated(self._offset_x, self._offset_y)
            r, g, b = label.bg_color

            # Texto claro em fundo escuro, escuro em fundo claro
            is_dark = (r + g + b) < 382
            text_color  = QColor(240, 240, 240) if is_dark else QColor(15, 15, 15)
            border_color = QColor(160, 160, 160) if is_dark else QColor(90, 90, 90)

            # Rounded rect com raio proporcional à menor dimensão
            radius = min(rect.width(), rect.height()) * 0.22
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), radius, radius)

            # Fundo opaco — cobre o texto original completamente
            painter.fillPath(path, QColor(r, g, b))

            # Borda sutil
            painter.setPen(QPen(border_color, 1.5))
            painter.drawPath(path)

            # Texto traduzido, fonte condensada estilo scanlation
            font_size = max(8, min(18, rect.height() // 6))
            font = QFont("Arial Black", font_size)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.5)
            painter.setFont(font)
            painter.setPen(text_color)
            painter.drawText(
                rect.adjusted(6, 6, -6, -6),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                label.text,
            )


def build_labels(
    detections: list[dict],
    region_origin: tuple[int, int],
) -> list[OverlayLabel]:
    """
    Converte detecções {box, translated_text, bg_color} em OverlayLabel com
    posição absoluta na tela.
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
        labels.append(OverlayLabel(rect, det["translated_text"], bg_color))
    return labels

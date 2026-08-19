from __future__ import annotations

import mss
import numpy as np

from PyQt6.QtCore import Qt, QRect, QPoint, QEventLoop, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QScreen
from PyQt6.QtWidgets import QApplication, QWidget


class _RegionSelector(QWidget):
    """Janela fullscreen em um monitor específico para seleção de região."""

    closed = pyqtSignal()

    def __init__(self, screen: QScreen):
        super().__init__()
        self.selected: QRect | None = None
        self._start = QPoint()
        self._end = QPoint()
        self._dragging = False

        self._bg = screen.grabWindow(0)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(screen.geometry())
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._bg)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self._dragging:
            rect = QRect(self._start, self._end).normalized()
            painter.drawPixmap(rect, self._bg, rect)
            painter.setPen(QPen(QColor(30, 144, 255), 2))
            painter.drawRect(rect)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.pos()
            self._end = event.pos()
            self._dragging = True

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            rect = QRect(self._start, event.pos()).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.selected = rect
            self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()


def select_region_on_screen(screen: QScreen) -> dict | None:
    """
    Abre seletor fullscreen no monitor especificado.
    Retorna coordenadas absolutas compatíveis com mss, ou None se cancelado.
    """
    loop = QEventLoop()
    selector = _RegionSelector(screen)
    selector.closed.connect(loop.quit)  # closeEvent → sinal → loop sai
    loop.exec()

    if selector.selected is None:
        return None

    r = selector.selected
    geo = screen.geometry()
    # Converte coordenadas locais do widget para coordenadas absolutas da tela
    return {
        "left": geo.x() + r.x(),
        "top": geo.y() + r.y(),
        "width": r.width(),
        "height": r.height(),
    }


def select_region() -> dict:
    """Seleciona região no monitor principal. Lança RuntimeError se cancelado."""
    screen = QApplication.primaryScreen()
    region = select_region_on_screen(screen)
    if region is None:
        raise RuntimeError("Nenhuma regiao selecionada.")
    return region


class ScreenCapture:
    def __init__(self, region: dict):
        self._region = region
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """Retorna o frame atual da região como array BGRA (formato nativo do mss).

        Não converte para BGR aqui: o loop de captura roda a ~20 fps só para
        detectar movimento, e a conversão de frame inteiro a cada tick seria
        custo puro. Quem precisa de BGR (o pipeline) converte no disparo."""
        raw = self._sct.grab(self._region)
        return np.array(raw)

    def close(self):
        self._sct.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

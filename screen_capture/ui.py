from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QFile, QObject, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineScript
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from .capture import select_region_on_screen

_HTML = Path(__file__).parent.parent / "ui" / "dashboard.html"


def _inject_qwebchannel(view: QWebEngineView) -> None:
    """Injeta qwebchannel.js (recurso interno do Qt) na página antes do carregamento."""
    f = QFile(":/qtwebchannel/qwebchannel.js")
    if not f.open(QFile.OpenModeFlag.ReadOnly):
        return
    src = bytes(f.readAll()).decode()
    f.close()

    s = QWebEngineScript()
    s.setName("qwebchannel")
    s.setSourceCode(src)
    s.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    s.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    view.page().scripts().insert(s)


class BackendBridge(QObject):
    """
    Exposto ao JavaScript via QWebChannel como `window.backend`.

    Sinais Python→JS: statusChanged, translationAdded, processingStarted,
                      pipelineDone, regionChanged, ollamaTestResult.
    Slots  JS→Python: start(), stop(), selectRegion(), getMonitors(), setMonitor(i),
                      setTranslationBackend(str), setOllamaModel(str), testOllama().
    """

    # Python → JS
    statusChanged     = pyqtSignal(str)
    translationAdded  = pyqtSignal(str)   # JSON {en, pt, cached}
    processingStarted = pyqtSignal()
    pipelineDone      = pyqtSignal()
    regionChanged     = pyqtSignal(int, int, int, int)  # left, top, w, h
    ollamaTestResult  = pyqtSignal(bool, str)           # ok, message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._start_cb             = None
        self._stop_cb              = None
        self._select_cb            = None
        self._set_monitor_cb       = None
        self._set_backend_cb       = None
        self._set_ollama_model_cb  = None
        self._test_ollama_cb       = None
        self._clear_context_cb     = None

    # JS → Python
    @pyqtSlot()
    def start(self) -> None:
        if self._start_cb:
            self._start_cb()

    @pyqtSlot()
    def stop(self) -> None:
        if self._stop_cb:
            self._stop_cb()

    @pyqtSlot()
    def selectRegion(self) -> None:
        if self._select_cb:
            self._select_cb()

    @pyqtSlot(result=str)
    def getMonitors(self) -> str:
        screens = QApplication.screens()
        primary = QApplication.primaryScreen()
        monitors = []
        for i, screen in enumerate(screens):
            g = screen.geometry()
            monitors.append({
                "index":   i,
                "label":   f"Monitor {i + 1}  —  {g.width()}×{g.height()}",
                "primary": screen == primary,
            })
        return json.dumps(monitors)

    @pyqtSlot(int)
    def setMonitor(self, index: int) -> None:
        if self._set_monitor_cb:
            self._set_monitor_cb(index)

    @pyqtSlot(str)
    def setTranslationBackend(self, backend: str) -> None:
        if self._set_backend_cb:
            self._set_backend_cb(backend)

    @pyqtSlot(str)
    def setOllamaModel(self, model: str) -> None:
        if self._set_ollama_model_cb:
            self._set_ollama_model_cb(model)

    @pyqtSlot()
    def testOllama(self) -> None:
        if self._test_ollama_cb:
            self._test_ollama_cb()

    @pyqtSlot()
    def clearOllamaContext(self) -> None:
        if self._clear_context_cb:
            self._clear_context_cb()


class ControlPanel(QWidget):
    """
    Painel de controle baseado em QWebEngineView.
    Interface externa idêntica à versão anterior (start_requested, stop_requested,
    set_status, get_region), mais o atributo `bridge` para conexões adicionais
    do main.py.
    """

    start_requested = pyqtSignal()
    stop_requested  = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._region: dict | None = None
        self._monitor_index: int = 0
        self.bridge = BackendBridge(self)
        self._build_ui()

        self.bridge._start_cb        = self.start_requested.emit
        self.bridge._stop_cb         = self.stop_requested.emit
        self.bridge._select_cb       = self._do_select
        self.bridge._set_monitor_cb  = self._set_monitor

    def _build_ui(self) -> None:
        self.setWindowTitle("Webtoon OCR")
        self.resize(1280, 800)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )

        view = QWebEngineView(self)
        channel = QWebChannel(view.page())
        channel.registerObject("backend", self.bridge)
        view.page().setWebChannel(channel)
        _inject_qwebchannel(view)
        view.load(QUrl.fromLocalFile(str(_HTML.resolve())))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(view)

    # ── Interface pública (compatível com main.py) ────────────────────────────

    def set_status(self, text: str) -> None:
        self.bridge.statusChanged.emit(text)

    def get_region(self) -> dict | None:
        return self._region

    def notify_detections(self, detections: list[dict]) -> None:
        """Envia cada detecção EN→PT para o log de traduções no dashboard."""
        for det in detections:
            self.bridge.translationAdded.emit(json.dumps({
                "en":     det.get("text", ""),
                "pt":     det.get("translated_text", ""),
                "cached": det.get("cached", False),
                "engine": det.get("engine", "google"),
            }))
        self.bridge.pipelineDone.emit()

    # ── Interno ───────────────────────────────────────────────────────────────

    def _set_monitor(self, index: int) -> None:
        screens = QApplication.screens()
        if 0 <= index < len(screens):
            self._monitor_index = index

    def _do_select(self) -> None:
        screens = QApplication.screens()
        screen = (
            screens[self._monitor_index]
            if self._monitor_index < len(screens)
            else QApplication.primaryScreen()
        )
        region = select_region_on_screen(screen)
        if region:
            self._region = region
            self.bridge.regionChanged.emit(
                region["left"], region["top"],
                region["width"], region["height"],
            )

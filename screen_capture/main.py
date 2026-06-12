from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

from PyQt6.QtCore import QPoint, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from .bubble_detector import BubbleDetector
from .capture import ScreenCapture
from .motion_detector import MotionDetector
from .ocr_engine import OCREngine
from .overlay import OverlayWindow, build_labels
from .translator import Translator
from .ui import ControlPanel

_CAPTURE_INTERVAL = 0.05  # ~20 fps


class ProcessingThread(QThread):
    labels_ready      = pyqtSignal(list)
    status_update     = pyqtSignal(str)
    detections_ready  = pyqtSignal(list)   # [{text, translated_text, ...}]
    processing_started = pyqtSignal()

    def __init__(self, region: dict, translator: Translator):
        super().__init__()
        self._region = region
        self._translator = translator
        self._running = True

    def run(self) -> None:
        self.status_update.emit("Carregando modelo de detecção...")
        capture = ScreenCapture(self._region)
        motion = MotionDetector()
        detector = BubbleDetector()
        ocr = OCREngine()
        mode = "YOLOv8" if detector.using_yolo else "OpenCV"
        self.status_update.emit(f"OCR em execução ({mode})...")
        translator = self._translator

        origin = (self._region["left"], self._region["top"])

        already_processed = False
        motion_frames = 0  # frames consecutivos de movimento detectado
        # A overlay aparecendo gera 1-2 frames de movimento; rolagem real gera muitos.
        # Só resetamos após _RESET_FRAMES frames consecutivos para evitar loop de feedback.
        _RESET_FRAMES = 6

        with capture:
            while self._running:
                frame = capture.grab()
                is_stable = motion.update(frame)

                if not is_stable:
                    motion_frames += 1
                    if already_processed and motion_frames >= _RESET_FRAMES:
                        self.labels_ready.emit([])
                        already_processed = False
                        motion_frames = 0
                    time.sleep(_CAPTURE_INTERVAL)
                    continue

                motion_frames = 0

                if already_processed:
                    # Tela continua parada — OCR já foi feito, não repete
                    time.sleep(_CAPTURE_INTERVAL)
                    continue

                # Transição movimento → estável: executa OCR uma única vez
                already_processed = True
                self.processing_started.emit()
                self.status_update.emit("Processando...")
                t0 = time.perf_counter()

                bubbles = detector.crop_bubbles(frame)
                t_detect = time.perf_counter()
                logging.debug("Detecção de balões: %.3fs — %d balão(ões)", t_detect - t0, len(bubbles))

                def _run_ocr(item: tuple) -> tuple | None:
                    box, crop, bg_color = item
                    try:
                        t_ocr = time.perf_counter()
                        lines = ocr.extract(crop)
                        logging.debug("  OCR (balão %s): %.3fs — %d linha(s)", box, time.perf_counter() - t_ocr, len(lines))
                        if not lines:
                            return None
                        # Ordena top→bottom pelo topo do bounding box de cada linha.
                        # RapidOCR não garante ordem de leitura; sem sort o texto
                        # fica embaralhado antes de chegar ao tradutor.
                        lines_sorted = sorted(lines, key=lambda d: min(pt[1] for pt in d["box"]))
                        logging.debug("  OCR linhas: %s", [d['text'] for d in lines_sorted])
                        full_text = " ".join(d["text"] for d in lines_sorted)
                        return (box, full_text, bg_color)
                    except Exception as exc:
                        logging.warning("  OCR falhou para balão %s: %s", box, exc)
                        return None

                # Etapa 1: OCR em paralelo (gargalo principal)
                # pytesseract spawna subprocessos reais — mais workers = mais paralelismo
                _workers = max(4, os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=_workers) as pool:
                    ocr_results = [r for r in pool.map(_run_ocr, bubbles) if r is not None]

                logging.debug("OCR paralelo: %d/%d balão(ões) com texto", len(ocr_results), len(bubbles))

                # Etapa 2: tradução em paralelo (instância independente por thread)
                t_tr = time.perf_counter()
                texts = [full_text for _, full_text, _ in ocr_results]
                translated_texts = translator.translate_many(texts)
                logging.debug("Tradução paralela: %.3fs — %d texto(s)", time.perf_counter() - t_tr, len(texts))

                engine = translator.backend_name
                detections = []
                for (box, full_text, bg_color), translated in zip(ocr_results, translated_texts):
                    x1, y1, x2, y2 = box
                    detections.append({
                        "text": full_text,
                        "translated_text": translated,
                        "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        "bg_color": bg_color,
                        "engine": engine,
                    })

                t_total = time.perf_counter()
                logging.debug("Total do pipeline: %.3fs — %d texto(s)", t_total - t0, len(detections))

                self.detections_ready.emit(detections)
                self.labels_ready.emit(build_labels(detections, origin))
                status = f"OCR em execução — {len(detections)} texto(s) detectado(s)."
                self.status_update.emit(status)
                time.sleep(_CAPTURE_INTERVAL)

    def stop(self) -> None:
        self._running = False
        self.wait()


def main() -> None:
    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    panel = ControlPanel()

    translator = Translator()

    worker: ProcessingThread | None = None

    def on_start() -> None:
        nonlocal worker
        region = panel.get_region()
        if region is None:
            return
        # Reposiciona o overlay no monitor onde está a região capturada
        target_screen = (
            app.screenAt(QPoint(region["left"], region["top"]))
            or app.primaryScreen()
        )
        overlay.reposition(target_screen)

        worker = ProcessingThread(region, translator)
        worker.labels_ready.connect(overlay.update_labels)
        worker.status_update.connect(panel.set_status)
        worker.processing_started.connect(panel.bridge.processingStarted)
        worker.detections_ready.connect(panel.notify_detections)
        worker.start()

    def on_stop() -> None:
        nonlocal worker
        if worker:
            worker.stop()
            worker = None
        overlay.clear()
        panel.set_status("OCR pausado.")

    def on_screen_removed(_removed) -> None:
        """Para o OCR se o monitor da região capturada for desconectado."""
        if worker is None:
            return
        region = panel.get_region()
        if region and app.screenAt(QPoint(region["left"], region["top"])) is None:
            on_stop()
            panel.set_status("Monitor desconectado. Selecione uma nova área.")

    app.screenRemoved.connect(on_screen_removed)

    def on_set_backend(backend: str) -> None:
        translator.set_backend(backend)

    def on_set_ollama_model(model: str) -> None:
        translator.set_ollama_model(model)

    def on_test_ollama() -> None:
        ok, msg = translator.test_ollama()
        panel.bridge.ollamaTestResult.emit(ok, msg)

    panel.bridge._set_backend_cb      = on_set_backend
    panel.bridge._set_ollama_model_cb = on_set_ollama_model
    panel.bridge._test_ollama_cb      = on_test_ollama
    panel.bridge._clear_context_cb    = translator.clear_context

    panel.start_requested.connect(on_start)
    panel.stop_requested.connect(on_stop)
    panel.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

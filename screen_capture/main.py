from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

# Silencia o DEBUG de bibliotecas de terceiros (httpcore/httpx ao baixar o modelo,
# matplotlib/PIL, huggingface_hub) que inundam o log e escondem o pipeline.
for _noisy in ("httpcore", "httpx", "urllib3", "matplotlib", "PIL",
               "huggingface_hub", "filelock", "fontTools"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import cv2
from PyQt6.QtCore import QPoint, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from .bubble_detector import BubbleDetector
from .capture import ScreenCapture
from .inpaint import inpaint_text
from .motion_detector import MotionDetector
from .ocr_engine import MangaOCREngine, OCREngine
from .overlay import OverlayWindow, build_labels
from .settings import Settings
from .translator import Translator
from .ui import ControlPanel

_CAPTURE_INTERVAL = 0.05  # ~20 fps


class ProcessingThread(QThread):
    labels_ready       = pyqtSignal(list)
    status_update      = pyqtSignal(str)
    detections_ready   = pyqtSignal(list)   # [{text, translated_text, ...}]
    processing_started = pyqtSignal()
    pipeline_cancelled = pyqtSignal()        # scroll detectado durante OCR

    # Modelos são caros de carregar (YOLO ~2s do disco, sessões ONNX) e são
    # thread-safe para inferência — compartilhados entre instâncias para que
    # Stop/Start (trocar região, pausar) não pague o load de novo.
    _models_lock = threading.Lock()
    _shared_detector: BubbleDetector | None = None
    _shared_ocr: dict[str, object] = {}

    def __init__(self, region: dict, translator: Translator, debounce_ms: int = 300,
                 ocr_mode: str = "en"):
        super().__init__()
        self._region = region
        self._translator = translator
        self._debounce_s = max(0.05, debounce_ms / 1000.0)
        self._ocr_mode = ocr_mode
        self._running = True
        # Evento do pipeline MAIS RECENTE. Cada pipeline recebe seu próprio
        # Event no dispatch — um evento compartilhado permitiria que o clear()
        # de um pipeline novo "descancelasse" um antigo ainda em execução.
        self._cancel = threading.Event()

    @classmethod
    def _get_detector(cls) -> BubbleDetector:
        with cls._models_lock:
            if cls._shared_detector is None:
                cls._shared_detector = BubbleDetector()
            return cls._shared_detector

    @classmethod
    def _get_ocr(cls, mode: str):
        with cls._models_lock:
            engine = cls._shared_ocr.get(mode)
            if engine is None:
                if mode == "ja":
                    engine = MangaOCREngine()
                elif mode == "manga_en":
                    engine = OCREngine(force_extra_passes=True)
                else:
                    engine = OCREngine()
                cls._shared_ocr[mode] = engine
            return engine

    def run(self) -> None:
        self.status_update.emit("Carregando modelo de detecção...")
        capture = ScreenCapture(self._region)
        motion = MotionDetector(debounce=self._debounce_s)
        detector = self._get_detector()
        ocr = self._get_ocr(self._ocr_mode)
        mode = "MangaOCR" if self._ocr_mode == "ja" else ("YOLOv8" if detector.using_yolo else "OpenCV")
        self.status_update.emit(f"OCR em execução ({mode})...")
        translator = self._translator
        origin = (self._region["left"], self._region["top"])

        already_processed = False
        motion_frames = 0
        motion_start_t = 0.0
        # Repaints de UI (overlay, dashboard) causam ~100-200ms de movimento detectado.
        # Scroll real do usuário dura 400ms+. Usamos duração mínima, não contagem de
        # frames, para distinguir os dois casos e evitar o loop de feedback.
        _SCROLL_RESET_S = 0.40

        with capture:
            while self._running:
                frame = capture.grab()  # BGRA cru — sem conversão no tick ocioso
                is_stable = motion.update(frame)

                if not is_stable:
                    if motion_frames == 0:
                        motion_start_t = time.monotonic()
                    motion_frames += 1
                    if already_processed and (time.monotonic() - motion_start_t) >= _SCROLL_RESET_S:
                        # Movimento sustentado ≥ 400ms → scroll real detectado.
                        # Cancela pipeline em andamento, limpa overlay e reseta estado.
                        self._cancel.set()
                        self.labels_ready.emit([])
                        already_processed = False
                        motion_frames = 0
                        motion_start_t = 0.0
                    time.sleep(_CAPTURE_INTERVAL)
                    continue

                motion_frames = 0
                motion_start_t = 0.0

                if already_processed:
                    # Tela continua parada — OCR já foi feito, não repete
                    time.sleep(_CAPTURE_INTERVAL)
                    continue

                # Transição movimento → estável: dispara pipeline uma única vez.
                # Roda em thread daemon para que o loop principal continue capturando
                # frames e detectando movimento mesmo durante OCR/tradução.
                already_processed = True
                cancel = threading.Event()
                self._cancel = cancel  # scroll/stop cancelam o pipeline mais recente
                self.processing_started.emit()
                self.status_update.emit("Processando...")

                # Conversão BGRA→BGR só aqui (o pipeline precisa de BGR); a
                # cópia resultante também serve de snapshot antes de ceder o loop.
                frame_snap = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                def _worker(f=frame_snap, det=detector, eng=ocr, tr=translator,
                            orig=origin, cn=cancel):
                    self._do_pipeline(f, det, eng, tr, orig, cn)

                threading.Thread(target=_worker, daemon=True).start()

                time.sleep(_CAPTURE_INTERVAL)

    def _do_pipeline(
        self,
        frame,
        detector: BubbleDetector,
        ocr,
        translator: Translator,
        origin: tuple,
        cancel: threading.Event,
    ) -> None:
        """Executa detecção → OCR → tradução em thread daemon.

        Verifica `cancel` (evento próprio deste pipeline) em vários pontos; se
        setado (scroll detectado), aborta sem emitir resultados e dispara
        `pipeline_cancelled`.
        """
        t0 = time.perf_counter()

        if cancel.is_set():
            self.pipeline_cancelled.emit()
            return

        bubbles = detector.crop_bubbles(frame)
        logging.debug("Detecção de balões: %.3fs — %d balão(ões)", time.perf_counter() - t0, len(bubbles))

        if cancel.is_set():
            self.pipeline_cancelled.emit()
            return

        overlap = translator.backend_name == "google"

        def _run_ocr(item: tuple) -> tuple | None:
            if cancel.is_set():
                return None
            box, crop, bg_color = item
            try:
                t_ocr = time.perf_counter()
                lines = ocr.extract(crop)
                logging.debug("  OCR (balão %s): %.3fs — %d linha(s)", box, time.perf_counter() - t_ocr, len(lines))
                if not lines or cancel.is_set():
                    return None
                lines_sorted = sorted(lines, key=lambda d: min(pt[1] for pt in d["box"]))
                logging.debug("  OCR linhas: %s", [d['text'] for d in lines_sorted])
                full_text = " ".join(d["text"] for d in lines_sorted)
                if cancel.is_set():
                    return None
                pack = inpaint_text(crop)
                return (box, full_text, bg_color, pack)
            except Exception as exc:
                logging.warning("  OCR falhou para balão %s: %s", box, exc)
                return None

        def _run_pipeline_item(item: tuple) -> tuple | None:
            res = _run_ocr(item)
            if res is None or not overlap or cancel.is_set():
                return res
            box, full_text, bg_color, pack = res
            cached = translator.is_cached(full_text)
            return (box, full_text, bg_color, pack, translator.translate_one(full_text), cached)

        _workers = max(4, os.cpu_count() or 4)
        t_tr = time.perf_counter()
        with ThreadPoolExecutor(max_workers=_workers) as pool:
            results = [r for r in pool.map(_run_pipeline_item, bubbles) if r is not None]

        if cancel.is_set():
            self.pipeline_cancelled.emit()
            return

        if overlap:
            ocr_results     = [(b, t, bg, pk) for (b, t, bg, pk, _tr, _c) in results]
            translated_texts = [tr for (_b, _t, _bg, _pk, tr, _c) in results]
            cached_flags    = [c  for (_b, _t, _bg, _pk, _tr, c) in results]
            logging.debug("OCR+tradução (overlap): %.3fs — %d texto(s)", time.perf_counter() - t_tr, len(results))
        else:
            ocr_results = results
            texts = [ft for _, ft, _, _ in ocr_results]
            cached_flags = [translator.is_cached(t) for t in texts]
            translated_texts = translator.translate_many(texts)
            logging.debug("Tradução batch: %.3fs — %d texto(s)", time.perf_counter() - t_tr, len(texts))

        if cancel.is_set():
            self.pipeline_cancelled.emit()
            return

        logging.debug("OCR: %d/%d balão(ões) com texto", len(ocr_results), len(bubbles))

        engine = translator.backend_name
        detections = []
        for (box, full_text, bg_color, pack), translated, cached in zip(
            ocr_results, translated_texts, cached_flags
        ):
            x1, y1, x2, y2 = box
            clean_img   = pack[0] if pack else None
            text_center = pack[1] if pack else None
            if pack:
                bg_color = pack[2]
            detections.append({
                "text":           full_text,
                "translated_text": translated,
                "box":            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "bg_color":       bg_color,
                "clean_image":    clean_img,
                "text_center":    text_center,
                "cached":         cached,
                "engine":         engine,
            })

        logging.debug("Total do pipeline: %.3fs — %d texto(s)", time.perf_counter() - t0, len(detections))
        self.detections_ready.emit(detections)
        self.labels_ready.emit(build_labels(detections, origin))
        self.status_update.emit(f"OCR em execução — {len(detections)} texto(s) detectado(s).")

    def stop(self) -> None:
        self._running = False
        self._cancel.set()   # encerra pipeline em andamento se houver
        self.wait()


def main() -> None:
    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    panel = ControlPanel()

    settings = Settings()
    translator = Translator()

    # Aplica as pré-definições salvas ao tradutor antes do primeiro OCR.
    saved = settings.get_all()
    if saved.get("ollamaModel"):
        translator.set_ollama_model(saved["ollamaModel"])
    if saved.get("engine"):
        translator.set_backend(saved["engine"])
    if saved.get("ocrMode"):
        translator.set_ocr_mode(saved["ocrMode"])
    # Carrega o NLLB (engine padrão) em background antes do primeiro OCR —
    # evita o stall de ~15s no primeiro balão da sessão.
    translator.preload()

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

        debounce_ms = int(settings.get_all().get("debounce", 300))
        ocr_mode = settings.get_all().get("ocrMode", "en")
        worker = ProcessingThread(region, translator, debounce_ms, ocr_mode)
        worker.labels_ready.connect(overlay.update_labels)
        worker.status_update.connect(panel.set_status)
        worker.processing_started.connect(panel.bridge.processingStarted)
        worker.pipeline_cancelled.connect(panel.bridge.pipelineCancelled)
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

    def on_set_ocr_mode(mode: str) -> None:
        translator.set_ocr_mode(mode)

    def on_set_backend(backend: str) -> None:
        translator.set_backend(backend)

    def on_set_ollama_model(model: str) -> None:
        translator.set_ollama_model(model)

    def on_test_ollama() -> None:
        ok, msg = translator.test_ollama()
        panel.bridge.ollamaTestResult.emit(ok, msg)

    def on_test_nllb() -> None:
        ok, msg = translator.test_nllb()
        panel.bridge.nllbTestResult.emit(ok, msg)

    panel.bridge._set_ocr_mode_cb     = on_set_ocr_mode
    panel.bridge._set_backend_cb      = on_set_backend
    panel.bridge._set_ollama_model_cb = on_set_ollama_model
    panel.bridge._test_ollama_cb      = on_test_ollama
    panel.bridge._test_nllb_cb        = on_test_nllb
    panel.bridge._clear_context_cb    = translator.clear_context
    panel.bridge._get_settings_cb     = settings.get_all
    panel.bridge._save_settings_cb    = settings.update

    panel.start_requested.connect(on_start)
    panel.stop_requested.connect(on_stop)
    panel.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

import time
import cv2
import numpy as np

# Percentual mínimo de pixels alterados para considerar movimento
_MOTION_THRESHOLD = 0.02
# Tempo de quietude (segundos) antes de liberar o processamento
_DEBOUNCE_SECONDS = 0.3


class MotionDetector:
    def __init__(
        self,
        threshold: float = _MOTION_THRESHOLD,
        debounce: float = _DEBOUNCE_SECONDS,
    ):
        self._threshold = threshold
        self._debounce = debounce
        self._prev_gray: np.ndarray | None = None
        self._last_motion_time: float = 0.0

    def update(self, frame: np.ndarray) -> bool:
        """
        Recebe o frame atual e retorna True se a tela está estável
        (sem movimento por pelo menos `debounce` segundos).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = cv2.absdiff(self._prev_gray, gray)
        changed_ratio = np.count_nonzero(diff > 25) / diff.size
        self._prev_gray = gray

        if changed_ratio > self._threshold:
            self._last_motion_time = time.monotonic()
            return False

        return (time.monotonic() - self._last_motion_time) >= self._debounce

    def reset(self):
        self._prev_gray = None
        self._last_motion_time = 0.0

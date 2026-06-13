from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

_SETTINGS_FILE = Path(__file__).parent.parent / ".ui_settings.json"


class Settings:
    """Pré-definições do painel persistidas em disco entre sessões.

    O `QWebEngineProfile` padrão do Qt é off-the-record (memória), então
    `localStorage` no dashboard se perderia ao reiniciar. Guardamos as escolhas
    do usuário (engine de tradução, modelo Ollama, monitor, debounce, toggles)
    aqui no Python — a fonte da verdade — e o dashboard lê/escreve via bridge.
    """

    def __init__(self) -> None:
        self._data: dict = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if _SETTINGS_FILE.exists():
            try:
                self._data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                logging.info("Pré-definições carregadas: %d chave(s)", len(self._data))
            except Exception as exc:
                logging.warning("Falha ao carregar pré-definições: %s", exc)

    def _save(self) -> None:
        try:
            _SETTINGS_FILE.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Falha ao salvar pré-definições: %s", exc)

    def get_all(self) -> dict:
        return dict(self._data)

    def update(self, data: dict) -> None:
        with self._lock:
            self._data.update(data)
            self._save()

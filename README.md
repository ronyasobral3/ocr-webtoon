# ocr-webtoon

Overlay de tradução em tempo real para leitura de manhwa, webtoon e mangá.

Captura uma região da tela, detecta balões de fala, extrai o texto em inglês via OCR e sobrepõe a tradução em português diretamente sobre os balões — sem modificar a imagem original. O OCR é pausado durante o scroll e disparado ~300 ms depois que o usuário para, minimizando uso de CPU/GPU e chamadas de API.

---

## Funcionalidades

- **Painel de controle** — interface gráfica com seletor de monitor, seletor de área e botão iniciar/parar, renderizada via `QWebEngineView` com dashboard HTML/CSS/JS
- **Seleção de região** — janela fullscreen de arrastar-e-soltar, com suporte a múltiplos monitores
- **Captura contínua** — frames via `mss` com coordenadas absolutas (~20 fps)
- **Detecção de movimento** — diff de frames; overlay é limpo ao detectar scroll
- **Debounce de 300 ms** — OCR dispara uma única vez na transição movimento→estável, evitando loop de feedback com o próprio overlay
- **Detecção de balões** — YOLOv8 (`ogkalu/comic-speech-bubble-detector-yolov8m`, ~52 MB, download automático via `huggingface_hub`); fallback para detector OpenCV com `adaptiveThreshold` + `connectedComponents`
- **OCR** — `rapidocr-onnxruntime` (~0,35 s/balão, sem dependência de Tesseract); paralelo por balão com `ThreadPoolExecutor`
- **Pré-processamento** — upscale, MIN(R,G,B), CLAHE, unsharp mask; inversão automática para texto branco em fundo escuro
- **Tradução EN → PT-BR** — Google Translate via `deep-translator`; cache persistido em disco (`.translation_cache.json`), chamada de API apenas na primeira ocorrência de cada texto
- **Overlay transparente** — janela PyQt6 click-through sempre no topo; fundo reescrito com cor amostrada dos cantos do balão (estilo scanlation); fonte proporcional à altura do balão
- **Suporte a GPU** — YOLOv8 usa CUDA automaticamente se disponível

---

## Estrutura

```
ocr-webtoon/
├── screen_capture/
│   ├── main.py            # ponto de entrada, event loop Qt, wiring de componentes
│   ├── ui.py              # ControlPanel (QWebEngineView) + BackendBridge (QWebChannel)
│   ├── capture.py         # captura de frames (mss) e seletor de região (PyQt6)
│   ├── motion_detector.py # diff de frames para detectar scroll
│   ├── bubble_detector.py # YOLOv8 + fallback OpenCV
│   ├── ocr_engine.py      # RapidOCR com pré-processamento
│   ├── translator.py      # Google Translate + cache em disco
│   ├── cache.py           # TranslationCache (MD5, persistência JSON, thread-safe)
│   └── overlay.py         # OverlayWindow transparente click-through
└── ui/
    └── dashboard.html     # painel de controle (HTML/CSS/JS, conectado via QWebChannel)
```

---

## Como rodar

```powershell
# 1. Instalar dependências
pip install setuptools
pip install -r requirements.txt

# 1b. GPU NVIDIA (opcional — ~10× mais rápido na detecção de balões)
pip install torch==2.12.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 2. Executar
python -m screen_capture.main
```

Na primeira execução, o modelo YOLOv8 (~52 MB) é baixado automaticamente do HuggingFace e armazenado em cache local.

---

## Stack

| Camada | Biblioteca |
|---|---|
| Captura de tela | `mss` |
| Processamento de imagem | `opencv-python`, `numpy` |
| Detecção de balões | `ultralytics` (YOLOv8), fallback OpenCV |
| OCR | `rapidocr-onnxruntime` |
| Tradução | `deep-translator` (Google Translate) |
| Cache | JSON em disco, hash MD5 |
| Overlay / GUI | `PyQt6` |
| Dashboard | `PyQt6-WebEngine`, `QWebChannel`, HTML/CSS/JS |
| Download de modelo | `huggingface_hub` |

---

## Melhorias futuras

- Tradução contextual via LLM (preservar gírias, nomes próprios, tom narrativo)
- Reescrita completa do balão: remover texto original, redesenhar fundo, inserir tradução estilizada (scanlation-style)
- Validação com fontes muito decorativas (manuscritas, com sombra)

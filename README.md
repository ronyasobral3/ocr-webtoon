# ocr-webtoon

Overlay de tradução em tempo real para leitura de manhwa, webtoon e mangá.

Captura uma região da tela, detecta balões de fala, extrai o texto via OCR (webtoon em inglês, mangá em inglês com fonte estilizada, ou mangá em japonês) e sobrepõe a tradução em português diretamente sobre os balões — sem modificar a imagem original. O balão é reescrito via inpainting (o texto original é removido, não só coberto). O OCR é pausado durante o scroll e disparado ~300 ms depois que o usuário para; rolar a tela durante o processamento cancela o balão em andamento em vez de esperar terminar.

---

## Funcionalidades

- **Painel de controle** — interface gráfica com seletor de monitor, seletor de área e botão iniciar/parar, renderizada via `QWebEngineView` com dashboard HTML/CSS/JS
- **Seleção de região** — janela fullscreen de arrastar-e-soltar, com suporte a múltiplos monitores
- **Captura contínua** — frames via `mss` com coordenadas absolutas (~20 fps)
- **Detecção de movimento** — diff de frames; overlay é limpo ao detectar scroll; um pipeline de OCR/tradução em andamento é cancelado se o scroll for sustentado, em vez de bloquear a captura até terminar
- **Debounce configurável (padrão 300 ms)** — OCR dispara uma única vez na transição movimento→estável
- **Detecção de balões** — YOLOv8 (`ogkalu/comic-speech-bubble-detector-yolov8m`, ~52 MB, download automático via `huggingface_hub`); fallback para detector OpenCV com `adaptiveThreshold` + `connectedComponents`
- **Três modos de OCR** (seletor SOURCE LANG no dashboard):
  - **WEBTOON (EN)** — `rapidocr-onnxruntime`; passes extras de binarização só entram se a confiança ficar baixa
  - **MANGA (EN)** — mesmo engine, mas sempre com os passes extras (fontes mais estilizadas que webtoon)
  - **MANGA (JP)** — `manga-ocr`, lida nativamente com texto vertical e horizontal em japonês
- **Pré-processamento** — upscale, MIN(R,G,B), CLAHE, unsharp mask, deskew automático para texto itálico, threshold adaptativo/black top-hat para fontes decorativas; inversão automática para texto branco em fundo escuro
- **Tradução** — três engines configuráveis pelo dashboard:
  - **Google Translate** (padrão) — via `deep-translator`, sem instalação extra
  - **NLLB-200** — modelo de tradução dedicado (`facebook/nllb-200-distilled-600M`, ~2,4 GB, baixa no 1º uso), local e offline, roda na GPU se disponível; heurística de pós-processamento aproxima a saída (europeia) do PT-BR
  - **Ollama** (opcional) — LLM local com janela de contexto de 15 falas; corrige erros de OCR por inferência; fallback automático para Google se o modelo mantiver o inglês
- **Cache de tradução** — persistido em disco (`.translation_cache.json`), hash MD5; chamada de engine apenas na primeira ocorrência de cada texto
- **Reescrita completa do balão** — `inpaint.py` remove o texto original via `cv2.inpaint` (preserva a arte/textura real do balão) e o overlay redesenha só a tradução por cima, com contorno de letreiramento para legibilidade sobre fundos texturizados; cai para um retângulo opaco quando o inpainting não é seguro
- **Pré-definições persistidas** — engine, modo de OCR, modelo Ollama, debounce, toggles e monitor salvos em `.ui_settings.json` e restaurados no próximo início
- **Suporte a GPU** — YOLOv8 e NLLB-200 usam CUDA automaticamente se disponível

---

## Estrutura

```
ocr-webtoon/
├── screen_capture/
│   ├── main.py            # ponto de entrada, event loop Qt, wiring de componentes
│   ├── ui.py               # ControlPanel (QWebEngineView) + BackendBridge (QWebChannel)
│   ├── capture.py          # captura de frames (mss) e seletor de região (PyQt6)
│   ├── motion_detector.py  # diff de frames para detectar scroll
│   ├── bubble_detector.py  # YOLOv8 + fallback OpenCV
│   ├── ocr_engine.py       # RapidOCR (+ MangaOCR) com pré-processamento
│   ├── inpaint.py          # remove o texto original do balão via cv2.inpaint
│   ├── translator.py       # Google Translate + NLLB-200 + Ollama LLM + heurística PT-BR
│   ├── cache.py            # TranslationCache (MD5, persistência JSON, thread-safe)
│   ├── settings.py         # pré-definições do usuário (JSON em disco)
│   └── overlay.py          # OverlayWindow transparente click-through
├── ui/
│   ├── dashboard.html      # painel de controle (HTML/CSS/JS, conectado via QWebChannel)
│   └── assets/logo/        # marca do projeto (logo.svg, app_icon.ico)
├── tests/                  # suíte de testes unitários (pytest, sem GPU/rede)
├── run_app.py              # ponto de entrada para empacotamento (PyInstaller)
└── WebtoonOCR.spec         # spec do PyInstaller para o executável standalone
```

---

## Como rodar

### A partir do código-fonte

```powershell
# 1. Instalar dependências
pip install setuptools
pip install -r requirements.txt

# 1b. GPU NVIDIA (opcional — ~10× mais rápido na detecção de balões)
# IMPORTANTE: instale o torchvision do MESMO índice CUDA, senão o pip traz um
# torchvision CPU-only e `torchvision::nms` não tem kernel CUDA → o YOLO
# quebra na GPU e cai pro detector OpenCV.
pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 2. Executar
python -m screen_capture.main
```

Na primeira execução, os modelos (YOLOv8 ~52 MB, e NLLB-200 ~2,4 GB ou manga-ocr ~400 MB conforme o engine escolhido) são baixados automaticamente e armazenados em cache local.

### Executável (Windows, sem instalar Python)

Baixe o `.zip` mais recente na [página de releases](https://github.com/ronyasobral3/ocr-webtoon/releases), extraia e rode `WebtoonOCR.exe` dentro da pasta extraída (os arquivos ao lado do `.exe` são necessários). É um build **CPU-only**; para aceleração via GPU, rode a partir do código-fonte com o passo 1b acima.

### Tradução com Ollama (opcional)

```powershell
# Instalar pacote Python
pip install ollama

# Baixar um modelo (recomendado: qwen2.5:7b para melhor qualidade)
ollama pull qwen2.5:3b   # ~2 GB, mais rápido
ollama pull qwen2.5:7b   # ~4.7 GB, melhor qualidade
```

No dashboard: **Settings → TRANS ENGINE → OLLAMA**, configure o nome do modelo e clique em **TEST** para verificar a conexão. O botão **⌫ CLEAR CONTEXT** limpa o histórico de diálogo — útil ao trocar de capítulo.

---

## Testes

```powershell
pytest
```

130 testes unitários isolados (sem GPU, sem rede, sem download de modelo) cobrindo cache, tradução, detecção de balões, inpainting e OCR — rodam em menos de 1 segundo.

---

## Stack

| Camada | Biblioteca |
|---|---|
| Captura de tela | `mss` |
| Processamento de imagem | `opencv-python`, `numpy` |
| Detecção de balões | `ultralytics` (YOLOv8), fallback OpenCV |
| OCR | `rapidocr-onnxruntime`, `manga-ocr` (mangá JP) |
| Tradução | `deep-translator` (Google), `transformers` (NLLB-200), `ollama` (opcional) |
| Cache / pré-definições | JSON em disco, hash MD5 |
| Overlay / GUI | `PyQt6` |
| Dashboard | `PyQt6-WebEngine`, `QWebChannel`, HTML/CSS/JS |
| Download de modelo | `huggingface_hub` |
| Testes | `pytest` |
| Empacotamento | `pyinstaller` (veja `WebtoonOCR.spec`) |

---

## Melhorias futuras

- NLLB-200 usa `por_Latn` (português genérico), que ainda puxa para construções europeias fora do que a heurística de pós-processamento cobre hoje.
- O executável distribuído é CPU-only — empacotar uma variante com CUDA aumentaria bastante o tamanho do download (torch+torchvision com CUDA somam alguns GB a mais) e só beneficiaria quem tem GPU NVIDIA compatível.

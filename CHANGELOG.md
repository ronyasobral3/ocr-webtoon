# CHANGELOG — ocr-webtoon

> **Status: ATIVO** — v2 em andamento desde 2026-06-13

---

## v2 — 2026-06-14

### Aba PIPELINE — visualização da arquitetura em tempo real

O dashboard ganhou um sistema de abas (`DASHBOARD | PIPELINE`).

A aba **PIPELINE** substitui o bloco mini-pipeline que ficava na sidebar esquerda por uma visualização expandida do fluxo completo:

- Cinco cards empilhados: **IMAGE CAPTURE → PRE-PROCESSING → YOLOV8 BUBBLE DETECTION → OCR TEXT EXTRACTION → TRANSLATE**
- Cada card tem três estados visuais: idle (cinza) → ativo (ciano pulsante + sweep animado) → concluído (verde)
- Métricas ao vivo por card: latência (ms), confiança (BUBBLE), score (OCR)
- Preview expansível: BUBBLE mostra thumbnails dos balões detectados; OCR e TRANSLATE mostram o texto extraído/traduzido
- Conectores entre cards acendem progressivamente com o avanço do pipeline
- Terminal na base com log timestamped de cada evento (`INFO`, `SUCCESS`, `WARN`)
- Badge `● LIVE STREAMING` na topbar do pipeline, sincronizado com o estado START/STOP

A animação é acionada pelo sinal `processingStarted` (modo app) ou por um loop de simulação com 20% de chance de cancelamento mid-pipeline (modo browser/demo).

### Cancelamento de pipeline por scroll mid-run

Antes desta mudança, o loop principal do `ProcessingThread` bloqueava durante a execução do OCR/tradução (~0.5-2s). Qualquer scroll durante esse período não era detectado até o pipeline terminar.

**Arquitetura nova (`main.py`):**
- O pipeline (`_do_pipeline`) agora roda numa `threading.Thread(daemon=True)`, liberando o loop principal para continuar capturando frames e detectando movimento em tempo real.
- `threading.Event self._cancel` é setado pelo loop principal ao detectar scroll sustentado enquanto `already_processed = True`.
- `_do_pipeline` verifica `cancel.is_set()` em 5 pontos: antes da detecção de balões, após detecção, dentro de cada OCR por balão (via `_run_ocr`), após o `pool.map`, e antes de emitir resultados — abortando sem mostrar resultados stale.
- `stop()` agora também seta o cancel para encerrar qualquer pipeline em andamento.

**Novo sinal `pipeline_cancelled` (`ui.py`, `main.py`):**
- `ProcessingThread.pipeline_cancelled = pyqtSignal()` emitido quando o daemon thread aborta por cancel.
- `BackendBridge.pipelineCancelled = pyqtSignal()` exposto ao JavaScript.
- Dashboard: `cancelPipeline()` marca os cards ativos com estado `cancelled` (flash vermelho × 3), loga no terminal e reseta após ~1.3s.

### Fix: loop infinito após pipeline completar (feedback de UI)

**Causa:** Com o pipeline em daemon thread, o loop principal continua rodando enquanto o OCR processa. Quando o pipeline termina, ele emite múltiplos sinais Qt (`translationAdded` × N balões + `pipelineDone` + `statusChanged` + repaint do overlay). Cada sinal causa um repaint do `QWebEngineView`/overlay que o `mss` captura como um frame de movimento. Com ≥ 3 balões → ≥ 6 sinais → `motion_frames` atingia o threshold 6 → `already_processed = False` → novo pipeline → loop infinito.

**Fix:** Substituição de contagem de frames (`_RESET_FRAMES = 6`) por duração mínima de movimento contínuo (`_SCROLL_RESET_S = 0.40s`):

- Repaints de UI causam movimento detectado por ~100-200ms → nunca atingem 400ms → ignorados
- Scroll real do usuário dura 400ms+ → detectado e cancela normalmente
- Implementado via `motion_start_t = time.monotonic()` no início de cada streak de movimento; o reset só ocorre quando `time.monotonic() - motion_start_t >= 0.40`

---

## v2 — 2026-06-13

### Fontes manuscritas/decorativas — novos passes de binarização

Adicionados dois passes extras ao voting de confiança do `OCREngine`:

- `_binarize_adaptive()` — `cv2.adaptiveThreshold` com janela 51 px: melhor que Otsu global para fundos com gradiente, textura ou outline grosso (threshold calculado localmente por vizinhança).
- `_binarize_blackhat()` — black top-hat + Otsu: isola strokes escuros em fundos texturizados/gradiente removendo o fundo morfologicamente antes do threshold.

Os passes extras só são ativados quando a confiança dos passes padrão é < 0.65 — webtoon com fonte limpa (confiança > 0.8) não paga custo adicional.

### PT-BR via NLLB — `_to_ptbr()` expandido

Três adições conservadoras ao pós-passe sobre a saída do NLLB:

1. **`estar/andar a + infinitivo` → gerúndio BR**: `"estou a fazer"` → `"estou fazendo"` (construção exclusivamente EU-PT, seguro converter sempre).
2. **`ter de + infinitivo` → `ter que + infinitivo`**: `"tenho de ir"` → `"tenho que ir"`.
3. **Vocabulário EU→BR expandido**: `chávena→xícara`, `ecrã→tela`, `miúdo→garoto`, `miúda→garota`, `fixe→legal`, `bué→muito` (total: 13 entradas, todas whole-word e sem ambiguidade).

### Manga EN — modo dedicado para mangá em inglês

Adicionada terceira opção ao seletor SOURCE LANG: `MANGA (EN)`.

Usa `OCREngine` (RapidOCR, inglês) como o modo Webtoon, mas força os passes extras de binarização (`_binarize_adaptive` + `_binarize_blackhat`) **sempre**, independente da confiança — mangá em inglês usa fontes mais estilizadas que webtoon e se beneficia do threshold adaptativo em qualquer caso. Implementado via `OCREngine(force_extra_passes=True)`.

| Opção | Engine | Passes adaptativos | Tradução |
|---|---|---|---|
| WEBTOON (EN) | RapidOCR | Só se conf < 0.65 | EN → PT |
| MANGA (EN) | RapidOCR | Sempre | EN → PT |
| MANGA (JP) | manga-ocr | N/A | JP → PT |

### Suporte a mangá japonês (texto vertical)

Pipeline alternativo ativado pelo seletor **SOURCE LANG** no dashboard:

- **`MangaOCREngine`** (`ocr_engine.py`) — wrapper em torno de `manga-ocr` (kha-white/manga-ocr, ~400 MB, baixa no 1º uso). Lida nativamente com texto vertical e horizontal japonês. Retorna o mesmo formato de lista de dicts que `OCREngine`, intercambiável no pipeline.
- **`Translator.set_ocr_mode(mode)`** (`translator.py`) — quando `mode = "ja"`, configura o NLLB para usar `jpn_Jpan` como língua de origem (JA→PT-BR direto, sem passo intermediário EN→PT).
- **Dashboard** (`dashboard.html`) — novo seletor `SOURCE LANG`: `WEBTOON (EN)` / `MANGA (JP)`. Atualiza a topbar (`EN → PT` ↔ `JP → PT`) e persiste em `settings.json` como `ocrMode`.
- **`ProcessingThread`** lê `ocrMode` das settings no `on_start()` e instancia o engine correto.

Nova dependência: `manga-ocr` (adicionada ao `requirements.txt`).

---

> **Status da v1: FECHADO** — 2026-06-13
> Todos os itens planejados foram concluídos. Este documento é o registro histórico do que foi construído na v1.

---

## Pós-v1 — 2026-06-13

### Testes automatizados

Adicionada suíte de testes unitários isolados (sem GPU, sem rede, sem download de modelos). **130 testes, 0 falhas, ~0.8s.**

| Arquivo | Testes | Cobre |
|---|---|---|
| `tests/test_cache.py` | 16 | Normalização de chave (MD5, case-insensitive, strip), get/set/clear, persistência em disco, thread-safety |
| `tests/test_translator_utils.py` | 48 | `_to_ptbr` (conjugações "tu", léxico EU→BR, "precisar de + infinitivo"), `_is_untranslated`, `_parse_json_array` (4 tentativas de parse), `_repair_json_array`, `_is_clean_context_entry`, configuração de backend |
| `tests/test_bubble_detector.py` | 19 | `_iou`, `_remove_overlapping` (NMS), `_sample_bg_color` (BGR→RGB), `_isolate_bubbles` |
| `tests/test_inpaint.py` | 14 | Rejeições antecipadas (None, vazio, < 6px, máscara > 55%), caminho normal (shape, centroide, cor interior) |
| `tests/test_ocr_engine_utils.py` | 33 | `_crop_hash`, `_enhance` (upscale, inversão de balão escuro, grayscale), `_binarize`, `_estimate_shear`, `_deskew`, `_avg_conf` |

Dependência `pytest` adicionada ao `requirements.txt`. Configuração em `pytest.ini`.

---

## Estado da v1 (entrega final)

App funcional end-to-end com painel de controle renderizado via `QWebEngineView` (dashboard HTML/CSS/JS conectado ao backend Python via `QWebChannel`). Usuário seleciona monitor e região, OCR roda na transição movimento→estável e exibe tradução em overlay transparente. Testado com webtoon real.

---

## O que foi construído

- [x] Estrutura de pastas e `requirements.txt`
- [x] Painel de controle PyQt6 com seletor de monitor, botão de área e toggle Iniciar/Parar
- [x] Seleção de região via janela fullscreen no monitor escolhido (clique e arraste)
- [x] Suporte a múltiplos monitores (lista todos via `QApplication.screens()`)
- [x] Captura contínua de frames via `mss` com coordenadas absolutas
- [x] Detecção de movimento com debounce de 300ms
- [x] OCR dispara **uma única vez** na transição movimento→estável (evita loop de feedback com overlay)
- [x] OCR com `rapidocr-onnxruntime` (~0.35s/balão, ONNX, sem Tesseract)
- [x] Cache de imagem por MD5 no `OCREngine` — balões já vistos retornam em <1ms
- [x] Detecção de balões via OpenCV: dilata borda escura → `connectedComponents` remove fundo → isola interior branco
- [x] Filtros: área, cirularidade, aspect ratio, brilho, presença de texto; aceita balões normais (fundo claro) e invertidos (fundo escuro)
- [x] Todo o texto de um balão agrupado em uma frase única antes de traduzir
- [x] OCR paralelizado por balão com `ThreadPoolExecutor(max_workers=os.cpu_count())`
- [x] Tradução separada do OCR (sequencial) — evita race condition no GoogleTranslator
- [x] Tradução EN→PT-BR via `deep-translator` (Google Translate)
- [x] Cache de tradução persistido em disco (`json`) — rede chamada apenas na 1ª ocorrência de cada texto
- [x] Overlay PyQt6 transparente e click-through com status em tempo real
- [x] Tamanho de fonte do overlay proporcional à altura do balão (mín 8pt, máx 18pt, `altura ÷ 6`)
- [x] Loop principal em `QThread` com sinais Qt
- [x] `Translator` inicializado no startup da UI (não bloqueia o primeiro OCR)
- [x] Painel de controle migrado de widgets Qt nativos para `QWebEngineView` com dashboard HTML/CSS/JS
- [x] `BackendBridge(QObject)` exposto via `QWebChannel` como `window.backend` no JS
- [x] Sinais `processing_started` e `detections_ready` no `ProcessingThread` alimentam o dashboard em tempo real (animação de pipeline, log de traduções, stats)
- [x] Dashboard mantém modo de simulação como fallback quando aberto no browser

### Qualidade da detecção de balões

- [x] Balões sobre fundo escuro / com bordas coloridas (rosa, azul) — resolvido com `adaptiveThreshold`
- [x] Balões com cauda pronunciada — resolvido com `hull_circularity` no convex hull
- [x] Balões cortados pelas bordas da imagem — resolvido com padding preto antes da morfologia
- [x] Balões unidos em "figura-8" — fundo identificado como componente que toca os 4 lados
- [x] Falso positivo: área branca da página detectada como balão quando captura inclui margens escuras do leitor — resolvido com check `bh > h*0.85 or bw > w*0.85`
- [x] Modelo YOLOv8 para speech bubbles — `ogkalu/comic-speech-bubble-detector-yolov8m` (~52 MB) baixado automaticamente via `huggingface_hub` na primeira execução; cai para OpenCV se falhar; `imgsz=1024` na inferência

### Qualidade do OCR

- [x] Pipeline de pré-processamento: upscale, MIN(R,G,B), CLAHE, unsharp mask
- [x] Texto branco sobre fundo escuro (balões invertidos) — detector aceita `brightness ≤ 80`; OCR inverte crop quando `mean(gray) < 127`
- [x] Tokens OCR sem vogal descartados ("Lsnr", "w,i") — filtro pós-OCR evita lixo na tradução
- [x] Fontes muito decorativas (manuscritas, com sombra) — multi-pass binarizado (Otsu), upscale Lanczos4 para 300px, corte de borda em balões invertidos para evitar OCR de artefatos do glow
- [x] Fontes itálicas/inclinadas (RapidOCR duplica/funde glifos: "IT'S NOT" → "IT'S S NOT") — passe de **deskew** adicionado à votação de passes (`_estimate_shear` maximiza a variância da projeção vertical; autolimitado, k≈0 em texto reto; só adiciona o passe quando |k|≥0.12). Sobe a confiança de 0.64 → 0.83 no caso de teste.

### Overlay

- [x] Alternar entre monitores durante execução — `overlay.reposition(screen)` em `on_start()`; `screenRemoved` para o OCR se o monitor da captura sumir; `_populate_monitors` atualiza lista ao conectar/desconectar monitores

### Performance

- [x] OCR substituído: Tesseract → `rapidocr-onnxruntime` (~0.35s/balão, ~5-9× mais rápido)
- [x] Cache de imagem MD5 + cache de tradução em disco
- [x] `ThreadPoolExecutor(os.cpu_count())` workers
- [x] `connectedComponents` em vez de flood-fill pixel-a-pixel
- [x] `logging.INFO` em produção
- [x] GPU via CUDA para YOLOv8 — `torch 2.12.0+cu126` instalado (RTX 3050, driver 591.74, CUDA 13.1); ultralytics detecta e usa GPU automaticamente; sem mudança de código necessária. **Importante:** instalar `torchvision==0.27.0+cu126` do mesmo índice CUDA — senão vem CPU-only e `torchvision::nms` quebra na GPU.

### Overlay estilo scanlation e melhorias

- [x] Reescrita do balão estilo scanlation (versão simples): fundo opaco com cor amostrada dos cantos do crop (`_sample_bg_color`), rounded rect proporcional, fonte "Arial Black" condensada, cor de texto auto (claro/escuro por luminância do fundo)
- [x] Reescrita completa do balão (versão avançada): `inpaint.py` **remove** o texto original do crop capturado via `cv2.inpaint` (máscara por Otsu, anel de borda preservado p/ manter o contorno do balão; aborta se a máscara cobre >55% → provável arte). O overlay pinta o crop limpo (`OverlayLabel.bg_image`, `QImage`) no lugar do balão e redesenha só a tradução por cima — acompanha forma/cor/textura reais, sem retângulo colado. Fallback para o rounded rect opaco quando o inpaint retorna `None`. Detalhes do texto:
  - CAIXA-ALTA, fonte auto-dimensionada (`_fit_font_size`, busca binária com **teto 16pt** + margem proporcional ~15% para não estourar o oval).
  - Ancorado no **centroide da tinta original** (devolvido pelo inpaint) — em balões "boneco de neve" (dois ovais), a tradução cai no oval com mais texto, não no pescoço vazio.
  - Cor do texto decidida pela **cor real do interior** do balão (amostrada do fundo limpo no centroide), não pelos cantos do crop — corrige texto claro-sobre-claro quando os cantos do bbox caem fora do oval, sobre fundo escuro.
- [x] Persistência de pré-definições: `settings.py` (`Settings`, JSON em `.ui_settings.json`) exposto via bridge (`getSettings`/`saveSettings`); o dashboard restaura engine, modelo Ollama, debounce, toggles e monitor no load e salva a cada mudança. Engine/modelo/debounce aplicados ao `Translator`/`MotionDetector` no startup. Necessário porque o `QWebEngineProfile` padrão é off-the-record (localStorage não persiste). Slider de debounce agora é **funcional** (passado a `MotionDetector(debounce=...)`).
- [x] Tradução local fiel via **NLLB-200** (Meta) — engine `nllb` no dashboard (`_NLLBBackend`). Modelo de MT dedicado (`facebook/nllb-200-distilled-600M`, ~2.4GB, baixa no 1º uso), roda na GPU via `transformers`. Fiel e offline — sem a alucinação do Ollama 3b nem a dependência de rede do Google. Carga sob demanda, fallback para Google se falhar, botão TEST/LOAD no painel. Vira o engine padrão. O `por_Latn` puxa para PT-europeu, então um pós-passe leve `_to_ptbr` (whole-word, conservador) aproxima do PT-BR.
- [x] Log de traduções marca **cache vs. novo**: o backend consulta o cache ANTES de traduzir (`Translator.is_cached`) e envia a flag `cached`; o dashboard mostra badge CACHED (ciano) e os contadores CACHED/API CALLS. Badge NLLB (ciano) e GOOGLE/OLLAMA também.
- [x] Tradução contextual via LLM — Ollama como engine opcional configurável no dashboard; janela de contexto de 15 falas (EN→PT); prompt com domínio manhwa/fantasy; correção de erros OCR por inferência do modelo; fallback por item para Google quando o modelo mantém inglês (≥60% palavras originais preservadas); filtro de entradas corrompidas no histórico; botão CLEAR CONTEXT no dashboard; badge OLLAMA/GOOGLE no log de traduções.

---

## Problemas resolvidos

| Problema | Causa | Solução |
|---|---|---|
| Overlay sempre no monitor principal, ignorando monitor da captura | `OverlayWindow` usava `primaryScreen()` hardcoded | `reposition(screen)` chamado em `on_start()` via `screenAt(region_origin)` |
| Labels desenhadas fora da tela em monitor não-principal | `_offset_x/y` fixos em 0; labels em coords absolutas desenhadas em coords locais | `reposition()` define `_offset_x = -geo.x()`, `_offset_y = -geo.y()` |
| Monitor da captura desconectado sem feedback | Nenhum handler para `screenRemoved` | `on_screen_removed` para OCR e notifica usuário se a região ficou sem tela |
| Lista de monitores desatualizada em runtime | `_populate_monitors` nunca era re-chamado | Conectado a `app.screenAdded` / `app.screenRemoved`; `clear()` antes de repopular |
| Balões invertidos (fundo escuro, texto branco) não detectados | Filtro `brightness < 190` rejeitava interiores escuros; `MIN(R,G,B)` deixava texto branco invisível | Detector aceita `brightness ≤ 80` + checa pixels claros como texto; `_preprocess` inverte crop quando `mean(gray) < 127` |
| Falso positivo: página branca do webtoon detectada como balão | Margens escuras do leitor impediam a página de tocar os 4 lados do frame → `_isolate_bubbles` não descartava como fundo | Check `bh > h*0.85 or bw > w*0.85` rejeita blobs que cobrem quase todo o frame |
| Tokens OCR sem sentido traduzidos ("Lsnr w,i") | Fontes estilizadas/baixo contraste geravam caracteres sem vogal | Filtro pós-OCR descarta linhas onde todos os tokens >1 char não têm vogal |
| `opencv-python-headless` sem GUI | `ultralytics` sobrescreve `opencv-python` | Seleção de região migrada para PyQt6 |
| `python-bidi` 0.6+ falha ao compilar no Python 3.13 | Sem wheels cp313-win_amd64, requer Rust+MSVC | Trocado EasyOCR por Tesseract, depois por RapidOCR |
| Seleção de área não registrava | `selector.destroyed` nunca disparava | Trocado para `closeEvent` + `pyqtSignal` próprio |
| OCR reiniciava com tela parada | Loop de feedback: overlay capturado pelo `mss` | OCR roda apenas na transição movimento→estável (`already_processed` flag) |
| Overlay piscando após detecção | Aparição da overlay gerava 1-2 frames de movimento | Exige 6 frames consecutivos de movimento antes de limpar overlay |
| Objetos do fundo sendo "traduzidos" | Fundo branco do painel detectado como balão | Filtros de cirularidade, brilho, texto; `connectedComponents` remove fundo |
| Balão não detectado (fundo branco da página) | Interior branco do balão conectado ao fundo da página | Dilata borda escura antes de isolar o interior |
| `yolov8n.pt` não detectava balões | Modelo genérico COCO, não treinado em speech bubbles | Removido como padrão; OpenCV virou detector principal |
| Tesseract lento (1-3s/balão) | Engine pesado, sem aceleração de hardware | Substituído por `rapidocr-onnxruntime` (~0.35s/balão) |
| Dois balões não traduzidos simultaneamente | `GoogleTranslator` compartilhado entre threads | OCR paralelo, tradução sequencial, `try/except` por balão |
| Tradução por linha isolada | Cada linha traduzida separadamente, sem contexto | Texto do balão agrupado numa frase antes de traduzir |
| Cache de tradução perdido entre sessões | Cache apenas em memória | Cache persistido em disco como JSON |
| Balões com cauda pronunciada rejeitados | Cauda longa aumenta perímetro → `circularity < 0.25` | Trocado para `hull_circularity` (convex hull fecha a cauda) |
| Balões cortados pelo topo zerados | Interior conectava ao fundo pela borda da frame | Padding preto de 2px sela balões cortados |
| Dois balões unidos em figura-8 perdidos | Heurística "maior componente = fundo" removia o par | Fundo identificado como componente que toca os 4 lados |
| Balões com borda colorida não detectados | `threshold(gray, 230)` assumia bordas pretas | Trocado por `adaptiveThreshold` (contraste local) |
| Balões em fundo preto puro não detectados | `adaptiveThreshold` em região uniforme escura → threshold ≈ -10 → fundo preto vira branco, conecta-se ao interior, `_isolate_bubbles` descarta tudo | `binary[gray <= p3+15] = 0` suprime pixels escuros antes do isolamento |
| Balões "boneco de neve" (dois ovais conectados) não detectados | YOLO retornava 0 sem acionar fallback OpenCV; brightness medido sobre bounding rect incluía cantos pretos | YOLO cai para OpenCV se retornar lista vazia; brightness medido só nos pixels dentro do contorno |
| Balão preto em fundo colorido (fogo, batalha) não detectado | `255-gray` inverte fundo colorido → binário ruidoso, `_isolate_bubbles` não isola interior | Passe `_binary_dark_from_gray`: threshold absoluto `< 60` isola só pixels genuinamente escuros |
| Palavras embaralhadas na tradução | `" ".join(lines)` sem ordenar; RapidOCR não garante ordem top→bottom | Sort por Y mínimo do bounding box antes de juntar as linhas |
| Texto fantasma na borda do balão invertido ("SIHL") | Borda branca/glow → após inversão em `_enhance`, vira cinza escuro → OCR lê artefatos como texto | Apara 8% de cada lado antes de inverter (só em balões escuros, `mean(gray) < 127`) |
| `_avg_conf` lançava TypeError | RapidOCR retorna confidence como `str`; `sum()` inicia com `int(0)` | Cast explícito `float(r[2])` em `_avg_conf` |
| App fechava (crash nativo, exit 9) ao clicar START | `torchvision` instalado CPU-only com `torch` cu126 → `torchvision::nms` sem kernel CUDA → exceção na 1ª inferência YOLO derrubava a thread | `detect()` envolve a inferência em try/except e degrada para OpenCV (desliga o YOLO); raiz: instalar `torchvision==0.27.0+cu126` do mesmo índice CUDA |
| Ollama "não funcionava" (caía sempre no Google) | Servidor Ollama instalado, mas o pacote Python `ollama` faltava no venv → `ImportError` silencioso | `pip install ollama` no venv |
| Cor do texto clara sobre balão claro (ilegível) | `_sample_bg_color` amostrava os cantos do crop, que caem fora do oval (fundo escuro) → escolhia texto claro | Cor decidida pelo interior real do balão (centroide do fundo limpo pelo inpaint) |
| Fonte do overlay enchendo o balão (parecia esticada) | `_fit_font_size` sem teto + padding mínimo → tamanho limitado só pela largura | Teto de 16pt + margem proporcional (~15%) |
| Log de traduções mostrava só o cabeçalho das entradas (sem EN/PT) | `.log-scroll` é flex column; com muitas entradas o flexbox encolhia cada `.t-entry` (`flex-shrink:1`) e o `overflow:hidden` cortava tudo abaixo da 1ª linha | `flex-shrink: 0` nas entradas → mantêm altura natural e o painel rola |

---

## Como rodar

```powershell
# 1. Instalar dependências Python
pip install setuptools
pip install -r requirements.txt

# 1b. GPU NVIDIA (opcional, ~10× mais rápido na detecção)
# Instale o torchvision do MESMO índice CUDA — senão o pip puxa torchvision CPU-only
# e `torchvision::nms` não tem kernel CUDA → YOLO quebra na GPU e cai pro OpenCV.
pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 2. Executar
python -m screen_capture.main
```

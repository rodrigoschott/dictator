# 🎙️ Dictator - Voice to Text Windows Service

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)
![Poetry](https://img.shields.io/badge/dependency-poetry-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Serviço Windows para transcrição de voz em texto usando **faster-whisper** (Whisper AI otimizado) com aceleração GPU local.

## ✨ Características

### 🎯 Core Features
- ✅ **Windows Service** - Inicia automaticamente com o Windows
- ✅ **Mouse/Keyboard Trigger** - Botão lateral do mouse (padrão) ou hotkey customizável
- ✅ **System Tray** - Controle completo via ícone na bandeja
- ✅ **Visual Overlay** - Indicador visual colorido durante gravação/processamento
- ✅ **GPU Accelerated** - faster-whisper com CTranslate2 para transcrição ultrarrápida
- ✅ **100% Local** - Sem APIs externas, privacidade total
- ✅ **Auto-paste** - Cola texto automaticamente no campo em foco

### 🎤 Advanced Voice Features
- ✅ **Push-to-Talk Mode** - Grava enquanto segura o botão
- ✅ **Toggle Mode** - Clique para iniciar/parar gravação
- ✅ **VAD (Voice Activity Detection)** - Para automaticamente após silêncio
- ✅ **TTS (Text-to-Speech)** - Kokoro-ONNX para síntese de voz local de alta qualidade
- ✅ **TTS Interrupt** - Para TTS instantaneamente ao pressionar hotkey (~170ms latência)
- ✅ **Event-Driven Architecture** - Zero polling, processamento eficiente via fila de eventos

### 🤖 LLM Integration (Voice Assistant Mode)
- ✅ **Ollama Integration** - Conecte com modelos locais (llama, qwen, deepseek, etc.)
- ✅ **Dynamic Model Discovery** - Modelos Ollama descobertos automaticamente no menu
- ✅ **Thinking Models Support** - Filtra tags `<think>` de modelos como Qwen3 e DeepSeek-R1
- ✅ **Context Preservation** - Mantém histórico de conversação
- ✅ **Auto-Restart** - Mudanças de modelo/provider reiniciam serviço automaticamente

### ⚙️ Configuration & Management
- ✅ **YAML Config** - Configuração completa e fácil personalização
- ✅ **Poetry** - Gerenciamento moderno de dependências
- ✅ **Multiple Triggers** - Mouse (Side1/Side2/Middle) ou Teclado (Ctrl+Alt+V)
- ✅ **Multi-language** - Suporta múltiplos idiomas (PT-BR padrão)

## 📋 Requisitos

- **Windows 10/11**
- **Python 3.10+** (< 3.14)
- **NVIDIA GPU** com CUDA (RTX series recomendado)
  - Funciona com RTX 5080 out-of-the-box
  - Mínimo 2GB VRAM (depende do modelo)
- **Poetry** (instalado automaticamente pelo installer)
- **Arquivos de modelo** (incluídos no repo via Git LFS):
  - `kokoro-v1.0.onnx` (310 MB) - Modelo TTS
  - `voices-v1.0.bin` (43 MB) - Vozes TTS

## 🚀 Instalação Rápida

### Passo 0: Instalar Dependências (PRIMEIRA VEZ)

**Execute APENAS UMA VEZ antes de usar:**
```batch
setup.bat
```

Este script irá:
- ✅ Verificar/instalar Poetry
- ✅ Instalar todas as dependências Python
- ✅ Instalar faster-whisper com suporte CUDA
- ✅ Instalar CTranslate2 (motor de inferência otimizado)
- ✅ Instalar kokoro-onnx com suporte GPU

**Nota:** Os modelos ONNX (`kokoro-v1.0.onnx` e `voices-v1.0.bin`) já estão incluídos no repositório via Git LFS.

### Passo 1: Testar Localmente (Recomendado)

Antes de instalar como serviço, teste se tudo funciona:
```batch
run_local.bat
```

**⚠️ Limitação:** Modo local não funciona em apps elevados (terminal admin) devido ao UIPI do Windows.

**Solução temporária:** Clique com botão direito em `run_local_admin.bat` → **"Executar como Administrador"**

Se funcionar corretamente, prossiga para instalar como serviço (funciona em todos os apps).

### Passo 2: Instalar como Serviço Windows

1. **Instalar como serviço Windows**:
   - Clique com botão direito em `install_service.bat`
   - Selecione **"Executar como Administrador"**
   - O instalador irá automaticamente:
     - ✅ Verificar Python
     - ✅ Instalar Poetry (se necessário)
     - ✅ Baixar/instalar NSSM (gerenciador de serviço)
     - ✅ Instalar todas as dependências via Poetry
     - ✅ Configurar o serviço Windows
     - ✅ Iniciar o serviço

2. **Procure o ícone de microfone na bandeja do sistema** 🎤

## 🎯 Como Usar

### Modo Padrão (Mouse Side Button - Toggle)

1. **Clique** no **botão lateral do mouse** (Back button)
2. **Fale** o que deseja transcrever
3. **Clique novamente** no botão lateral para parar
4. **Aguarde** a transcrição (1-3 segundos com GPU)
5. **Texto será colado** automaticamente no campo em foco! 🎉

### Modo Push-to-Talk (Opcional)

1. **Segure** o botão lateral do mouse
2. **Fale** enquanto segura
3. **Solte** o botão para processar
4. Texto colado automaticamente!

### Visual Feedback

Durante a operação, você verá um **indicador colorido** no canto da tela:
- 🔴 **Vermelho** - Gravando áudio
- 🟠 **Laranja** - Processando transcrição
- 🟢 **Verde** - TTS falando (se habilitado)

### Menu da Bandeja

Clique com **botão direito** no ícone do microfone 🎤:

**Informações:**
- Trigger atual (Mouse/Keyboard)
- Modelo Whisper em uso

**Change Trigger:**
- 🖱️ Mouse Side 1 (Back) - *padrão*
- 🖱️ Mouse Side 2 (Forward)
- 🖱️ Mouse Middle (Scroll click)
- ⌨️ Keyboard (Ctrl+Alt+V)

**Modos de Gravação:**
- ☑️ **Push-to-Talk Mode** - Grava enquanto segura
- ☑️ **Auto-Stop (VAD)** - Para após silêncio
- ☑️ **LLM Mode** - Habilita assistente de voz com LLM

**LLM Configuration (Voice Assistant):**
- 🦙 **Ollama Models** - Lista dinâmica de modelos instalados
- 🔄 **LLM Provider** - Escolha entre Ollama, Claude Direct, ou Claude CLI
- 🎙️ **VAD Toggle** - Liga/desliga detecção de voz

**Ações:**
- **Open Config** - Editar `config.yaml`
- **Open Logs** - Ver `logs/dictator.log`
- **Restart Service** - Reiniciar serviço automaticamente
- **Exit** - Sair do serviço

## ⚙️ Configuração Completa

Edite `config.yaml` para personalizar o comportamento:

### 🎤 Whisper (Transcrição)
```yaml
whisper:
  model: "large-v3"    # tiny, base, small, medium, large, large-v3
  language: "pt"       # pt, en, es, fr, de, etc.
  device: "cuda"       # cuda (GPU) ou cpu
```

**Modelos disponíveis:**
- `tiny` - Mais rápido, menor precisão (~1GB VRAM)
- `base` - Balanceado (~1GB VRAM)
- `small` - Bom custo-benefício (~2GB VRAM)
- `medium` - Recomendado (~5GB VRAM)
- `large` - Melhor precisão (~10GB VRAM)
- `large-v3` - **Melhor modelo atual** (~10GB VRAM)

### 🖱️ Hotkey/Trigger
```yaml
hotkey:
  type: "mouse"                # "mouse" ou "keyboard"
  mouse_button: "side1"        # side1, side2, middle
  keyboard_trigger: "ctrl+alt+v"  # Usado se type = "keyboard"
  mode: "toggle"               # "toggle" ou "push_to_talk"
  
  # Voice Activity Detection (Auto-stop)
  vad_enabled: false           # true = para após silêncio
  vad_threshold: 0.002         # Sensibilidade (0.001 - 0.01)
  auto_stop_silence: 2.0       # Segundos de silêncio para parar
  max_duration: 60             # Máximo de segundos de gravação
```

### 📋 Auto-paste
```yaml
paste:
  auto_paste: true     # false = apenas copia para clipboard
  delay: 0.5           # Segundos de delay antes de colar
```

### 🔊 TTS (Text-to-Speech)
```yaml
tts:
  enabled: true                      # Ativar TTS
  engine: "kokoro-onnx"              # Motor de TTS
  volume: 0.8                        # Volume (0.0 - 1.0)
  interrupt_on_speech: true          # Para TTS ao iniciar gravação
  
  kokoro:
    model_path: "kokoro-v1.0.onnx"   # Caminho do modelo
    voices_path: "voices-v1.0.bin"   # Caminho das vozes
    voice: "pf_dora"                 # Voz padrão (Portuguese Female)
    language: "pt-br"                # pt-br, en-us, en-gb, es, fr, it, ja, zh, hi
    speed: 1.25                      # Velocidade (0.5 - 2.0)
```

**56 vozes disponíveis em múltiplos idiomas:**

🇵🇹 **Português:** `pf_dora`, `pm_alex`, `pm_santa`  
🇺🇸 **American:** `af_alloy`, `af_bella`, `af_nova`, `am_adam`, `am_onyx`, etc.  
🇬🇧 **British:** `bf_alice`, `bf_emma`, `bm_daniel`, `bm_george`  
🇪🇸 **Spanish:** `ef_dora`, `em_alex`  
🇫🇷 **French:** `ff_siwis`  
🇮🇹 **Italian:** `if_sara`, `im_nicola`  
🇯🇵 **Japanese:** `jf_alpha`, `jf_gongitsune`, `jm_kumo`  
🇨🇳 **Chinese:** `zf_xiaobei`, `zm_yunxi`, etc.

**Teste vozes com:**
```batch
poetry run python test_portuguese_voices.py
```

### 🎨 Visual Overlay
```yaml
overlay:
  enabled: true        # Mostrar indicador visual
  size: 15             # Tamanho em pixels
  position: "top-right"  # top-right, top-left, bottom-right, bottom-left
  padding: 20          # Distância da borda em pixels
```

### 🔧 Service
```yaml
service:
  auto_start: true     # Iniciar com Windows
  notifications: true  # Notificações do sistema
  log_level: "INFO"    # DEBUG, INFO, WARNING, ERROR
  log_file: ""         # Vazio = logs/dictator.log
```

### 🎵 Audio
```yaml
audio:
  sample_rate: 16000   # Hz (16000 recomendado para Whisper)
  channels: 1          # 1 = mono, 2 = stereo
```

**Após editar, reinicie o serviço pelo menu da bandeja ou execute:**
```batch
restart_dictator.bat
```

## 🧪 Testar Sem Instalar

Para testar antes de instalar como serviço:

```batch
# Modo normal (não funciona em apps elevados)
run_local.bat

# Modo administrador (funciona em todos os apps)
run_local_admin.bat  # (Executar como Administrador)
```

**Nota:** O modo local é útil para debugging e testes rápidos.

## 🗑️ Desinstalar

1. Clique com botão direito em `uninstall_service.bat`
2. Selecione **"Executar como Administrador"**
3. Opcionalmente, remova logs e configurações

## 📖 Documentação Completa

Veja [SERVICE.md](SERVICE.md) para:
- Instalação manual
- Seleção de modelos
- Troubleshooting completo
- Configurações avançadas
- Comandos úteis

## 🛠️ Desenvolvimento

### Setup do Ambiente

```batch
# Clone o repositório
git clone https://github.com/rodrigoschott/dictator.git
cd dictator

# Instalar Poetry (se necessário)
python -m pip install poetry

# Instalar dependências
poetry install

# Executar localmente com system tray
poetry run python src/dictator/tray.py config.yaml

# Ou executar apenas o serviço (sem tray)
poetry run python src/dictator/service.py config.yaml
```

### Estrutura Técnica

**Stack Principal:**
- **faster-whisper** - Whisper otimizado com CTranslate2 (não usa PyTorch!)
- **CTranslate2** - Motor de inferência otimizado para CPU/GPU
- **kokoro-onnx** - TTS de alta qualidade com ONNX Runtime
- **pynput** - Captura global de hotkeys (mouse/keyboard)
- **pystray** - System tray integration
- **sounddevice** - Captura de áudio
- **tkinter** - Visual overlay

### Arquitetura

```
src/dictator/
├── main.py          # Script original standalone
├── service.py       # Core service (recording + transcription)
├── tray.py          # System tray GUI + service integration
├── overlay.py       # Visual feedback overlay
└── tts_engine.py    # Text-to-Speech engine (Kokoro)
```

**Fluxo de Execução:**
1. `tray.py` inicia `service.py` e `overlay.py`
2. `service.py` monitora hotkey/mouse button
3. Ao detectar trigger, grava áudio
4. `overlay.py` mostra status visual
5. Audio é transcrito com faster-whisper
6. Texto é colado automaticamente
7. (Opcional) TTS fala o texto transcrito

### Comandos Úteis

```batch
# Verificar dependências instaladas
poetry run python verify_deps.py

# Ver logs em tempo real
tail -f logs/dictator.log  # Linux/WSL
Get-Content logs/dictator.log -Wait  # PowerShell

# Limpar cache do Poetry
poetry cache clear . --all

# Atualizar dependências
poetry update
```

## 🔧 Comandos de Serviço

```batch
# Iniciar serviço
nssm start Dictator

# Parar serviço
nssm stop Dictator

# Reiniciar serviço
nssm restart Dictator

# Status do serviço
sc query Dictator
```

## 📁 Estrutura do Projeto

```
Dictator/
├── src/
│   └── dictator/
│       ├── __init__.py
│       ├── main.py                  # Script original standalone
│       ├── service.py               # Core service (gravação + transcrição)
│       ├── tray.py                  # System tray GUI + dynamic model menu
│       ├── overlay.py               # Visual feedback overlay
│       ├── tts_engine.py            # Text-to-Speech engine (Kokoro)
│       └── voice/
│           ├── __init__.py
│           ├── events.py            # Event-driven architecture
│           ├── llm_caller.py        # LLM integration + thinking tag filter
│           ├── session_manager.py   # Voice session event processor
│           ├── vad_processor.py     # Voice Activity Detection
│           └── sentence_chunker.py  # Real-time sentence chunking
├── logs/
│   └── dictator.log                 # Logs do serviço
├── config.yaml                      # Configuração principal
├── pyproject.toml                   # Poetry dependencies
├── poetry.lock                      # Lock file
├── .gitattributes                   # Git LFS config
├── .gitignore               # Git ignore rules
│
├── kokoro-v1.0.onnx         # Modelo TTS (310 MB - via Git LFS)
├── voices-v1.0.bin          # Vozes TTS (43 MB - via Git LFS)
│
├── setup.bat                        # Instalar dependências
├── install_service.bat              # Instalador Windows Service
├── uninstall_service.bat            # Desinstalador
├── run_local.bat                    # Teste local
├── run_local_admin.bat              # Teste local (admin)
├── restart_dictator.bat             # Reiniciar serviço
├── verify_deps.py                   # Verificar dependências
│
├── test_portuguese_voices.py        # Teste de vozes Kokoro
├── test_thinking_tags.py            # Teste filtro thinking models
├── test_vad_tts_interrupt.py        # Teste interrupção TTS
├── test_auto_restart.py             # Teste auto-restart
│
├── ANALYSIS_VAD_TTS_INTERRUPT.md    # Análise técnica interrupção
├── IMPLEMENTATION_VAD_TTS_FIX.md    # Documentação implementação
├── SERVICE.md                       # Documentação técnica completa
└── README.md                        # Este arquivo
```

## 🔐 Privacidade & Segurança

- ✅ **100% Local** - Todo processamento acontece na sua máquina
- ✅ **Sem Internet** - Não envia dados para nenhum servidor
- ✅ **Sem APIs** - Não usa serviços de terceiros
- ✅ **Sem Telemetria** - Zero tracking ou coleta de dados
- ✅ **Open Source** - Código auditável
- ✅ **Modelos Locais** - Whisper e Kokoro armazenados no seu PC

**Seus dados nunca saem do seu computador!**

## 💰 Custo

**TOTALMENTE GRATUITO!** 🎉

- ✅ Sem APIs pagas (OpenAI, Google, etc.)
- ✅ Sem limites de uso
- ✅ Sem assinaturas
- ✅ Apenas custo de hardware (GPU local)

**Economize milhares por ano** comparado a serviços pagos de transcrição!

## 🎮 Performance & Hardware

### GPU Recomendada
- **RTX 4060** ou superior - Excelente performance
- **RTX 3060** - Bom para modelos small/medium
- **RTX 5080** - Performance excepcional (testado)

### Tempo de Transcrição (RTX 5080)
- **Modelo tiny** - ~0.5s por minuto de áudio
- **Modelo small** - ~0.8s por minuto de áudio
- **Modelo medium** - ~1.5s por minuto de áudio
- **Modelo large-v3** - ~2.5s por minuto de áudio

### Uso de VRAM
- **tiny** - ~1GB
- **small** - ~2GB
- **medium** - ~5GB
- **large/large-v3** - ~10GB

## ❓ FAQ

### Por que mouse side button ao invés de teclado?
Mouse side button é mais ergonômico para uso contínuo e não interfere com atalhos de aplicativos. Mas você pode facilmente trocar para teclado pelo menu da bandeja!

### Os modelos ONNX são baixados automaticamente?
Não! Os arquivos `kokoro-v1.0.onnx` e `voices-v1.0.bin` já estão incluídos no repositório via Git LFS. Ao clonar o repo, eles são baixados automaticamente.

### Funciona sem GPU NVIDIA?
Sim! Você pode usar CPU alterando `device: "cpu"` no `config.yaml`, mas será **muito mais lento**. GPU é altamente recomendada.

### Quanto VRAM preciso?
- **Mínimo:** 2GB (modelo small)
- **Recomendado:** 6GB+ (modelo medium/large)
- **Ideal:** 12GB+ (modelo large-v3)

### Posso usar em apps elevados (como terminal admin)?
Sim! Quando instalado como **Windows Service**, funciona em todos os apps. No modo local, use `run_local_admin.bat`.

### O TTS é obrigatório?
Não! TTS é opcional. Você pode desabilitar em `config.yaml` definindo `tts.enabled: false`.

### Como funciona o modo Voice Assistant (LLM)?
Quando habilitado (`voice.claude_mode: true`), o sistema envia a transcrição para um LLM (Ollama, Claude) e fala a resposta via TTS. Modelos Ollama são descobertos automaticamente no menu.

### Posso interromper o LLM enquanto ele fala?
Sim! Pressione a hotkey durante a fala do TTS e ele para instantaneamente (~170ms). Com VAD habilitado, basta começar a falar.

### O que são "thinking models"?
Modelos como Qwen3 e DeepSeek-R1 expõem raciocínio interno via tags `<think>`. O sistema filtra automaticamente essas tags para não falar o processo de pensamento.

### Suporta outros idiomas além de Português?
Sim! Whisper suporta 99+ idiomas. Altere `language` no config (ex: `en` para inglês, `es` para espanhol). TTS tem 56 vozes em 9 idiomas.

### Por que usar faster-whisper ao invés de openai-whisper?
faster-whisper é **4-5x mais rápido** e usa **menos VRAM** que a implementação original do OpenAI, graças ao CTranslate2.

## 🆘 Troubleshooting

### Serviço não inicia
```batch
# Verifique logs
type logs\dictator.log

# Reinstale o serviço
uninstall_service.bat  # (como Admin)
install_service.bat    # (como Admin)
```

### Transcrição muito lenta
- Verifique se está usando GPU: `device: "cuda"` no config
- Use modelo menor (tiny/small/medium)
- Verifique VRAM disponível

### Mouse button não funciona
- Verifique qual botão está configurado no menu da bandeja
- Teste com botão diferente (side2, middle)
- Tente modo keyboard

### Texto não cola automaticamente
- Verifique `paste.auto_paste: true` no config
- Aumente `paste.delay` para 1.0
- Verifique se o campo está em foco

Para mais detalhes, veja [SERVICE.md](SERVICE.md).

## 🤝 Contribuindo

Contribuições são bem-vindas! 

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Add: MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

## 📄 Licença

MIT License - Use como quiser! Veja [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [OpenAI Whisper](https://github.com/openai/whisper) - Modelo base de transcrição
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Implementação otimizada
- [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx) - TTS de alta qualidade
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Motor de inferência

---

<div align="center">

**Desenvolvido com 🎙️ para transcrições rápidas, privadas e eficientes!**

⭐ **Se este projeto foi útil, deixe uma estrela!** ⭐

</div>

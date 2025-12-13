#!/bin/bash
# Dictator Linux Installer
# Feature-focused installation script with systemd service

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/dictator"
VENV_DIR="$INSTALL_DIR/venv"
CONFIG_DIR="$HOME/.config/dictator"
SERVICE_NAME="dictator"

# Feature flags (will be set by user)
FEATURE_GPU=false
FEATURE_TTS=false
FEATURE_VAD=false
FEATURE_LLM=false
WHISPER_MODEL="large-v3"

# Functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "Este script precisa ser executado como root"
        echo "Execute com: sudo ./install.sh"
        exit 1
    fi
}

check_distro() {
    print_header "Verificando Distribuição Linux"

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_success "Distribuição: $NAME $VERSION"

        # Check if systemd is available
        if ! command -v systemctl &> /dev/null; then
            print_error "systemd não encontrado (requerido)"
            exit 1
        fi
        print_success "systemd encontrado"
    else
        print_error "Não foi possível determinar a distribuição Linux"
        exit 1
    fi
}

check_dependencies() {
    print_header "Verificando Dependências do Sistema"

    # Python 3.10+
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python 3 não encontrado"
        exit 1
    fi

    # pip
    if command -v pip3 &> /dev/null; then
        print_success "pip3 encontrado"
    else
        print_warning "pip3 não encontrado, instalando..."
        apt-get update && apt-get install -y python3-pip
    fi

    # venv
    if python3 -m venv --help &> /dev/null 2>&1; then
        print_success "python3-venv disponível"
    else
        print_warning "python3-venv não encontrado, instalando..."
        apt-get install -y python3-venv
    fi

    # portaudio (for sounddevice)
    if ldconfig -p | grep -q portaudio; then
        print_success "portaudio encontrado"
    else
        print_warning "portaudio não encontrado, instalando..."
        apt-get install -y portaudio19-dev
    fi

    # Git (for cloning)
    if command -v git &> /dev/null; then
        print_success "Git encontrado"
    else
        print_warning "Git não encontrado, instalando..."
        apt-get install -y git
    fi
}

feature_selection() {
    print_header "Seleção de Features"

    echo ""
    echo "Selecione as features que deseja instalar:"
    echo ""

    # GPU Acceleration
    echo -e "${BLUE}1. GPU Acceleration${NC}"
    echo "   Aceleração GPU para transcrição 5-10x mais rápida"
    echo "   Requer: NVIDIA GPU + CUDA drivers"
    echo "   Download: ~800 MB"
    read -p "   Habilitar GPU? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        FEATURE_GPU=true
        print_success "GPU habilitado"
    fi

    echo ""

    # TTS
    echo -e "${BLUE}2. Text-to-Speech (TTS)${NC}"
    echo "   Síntese de voz com 56 vozes em 9 idiomas"
    echo "   Download: ~338 MB (modelos ONNX)"
    read -p "   Habilitar TTS? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        FEATURE_TTS=true
        print_success "TTS habilitado"
    fi

    echo ""

    # VAD
    echo -e "${BLUE}3. Voice Activity Detection (VAD)${NC}"
    echo "   Detecção automática de fala para auto-stop"
    echo "   Recomendado com GPU"
    read -p "   Habilitar VAD? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        FEATURE_VAD=true
        print_success "VAD habilitado"
    fi

    echo ""

    # LLM Assistant
    echo -e "${BLUE}4. LLM Voice Assistant${NC}"
    echo "   Assistente de voz com LLMs locais (Ollama)"
    echo "   Requer: Ollama instalado"
    read -p "   Habilitar LLM? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        FEATURE_LLM=true
        print_success "LLM habilitado"
    fi

    echo ""

    # Whisper model
    echo -e "${BLUE}5. Modelo Whisper${NC}"
    echo "   tiny   - Mais rápido (1GB VRAM)"
    echo "   small  - Balanceado (2GB VRAM)"
    echo "   medium - Recomendado (5GB VRAM)"
    echo "   large-v3 - Melhor qualidade (10GB VRAM)"
    read -p "   Escolha [tiny/small/medium/large-v3]: " WHISPER_INPUT
    if [ ! -z "$WHISPER_INPUT" ]; then
        WHISPER_MODEL="$WHISPER_INPUT"
    fi
    print_success "Modelo: $WHISPER_MODEL"

    echo ""
    print_info "Resumo da instalação:"
    echo "  GPU: $FEATURE_GPU"
    echo "  TTS: $FEATURE_TTS"
    echo "  VAD: $FEATURE_VAD"
    echo "  LLM: $FEATURE_LLM"
    echo "  Modelo: $WHISPER_MODEL"
    echo ""
}

create_directories() {
    print_header "Criando Diretórios"

    mkdir -p "$INSTALL_DIR"
    print_success "Criado: $INSTALL_DIR"

    mkdir -p "$CONFIG_DIR"
    print_success "Criado: $CONFIG_DIR"

    mkdir -p "$INSTALL_DIR/logs"
    print_success "Criado: $INSTALL_DIR/logs"
}

create_venv() {
    print_header "Criando Virtual Environment"

    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment criado"

    # Upgrade pip
    "$VENV_DIR/bin/pip" install --upgrade pip
    print_success "pip atualizado"
}

install_dependencies() {
    print_header "Instalando Dependências Python"

    # Install requirements
    "$VENV_DIR/bin/pip" install faster-whisper sounddevice soundfile pyperclip pyautogui numpy pynput pystray pillow pyyaml aiohttp requests psutil

    print_success "Dependências core instaladas"

    # GPU packages
    if [ "$FEATURE_GPU" = true ]; then
        print_info "Instalando PyTorch com CUDA..."
        "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121
        print_success "PyTorch com CUDA instalado"
    fi

    # TTS
    if [ "$FEATURE_TTS" = true ]; then
        print_info "Instalando kokoro-onnx..."
        "$VENV_DIR/bin/pip" install "kokoro-onnx[gpu]"
        print_success "kokoro-onnx instalado"
    fi
}

download_models() {
    if [ "$FEATURE_TTS" = false ]; then
        print_info "TTS desabilitado, pulando download de modelos"
        return
    fi

    print_header "Baixando Modelos TTS"

    cd "$INSTALL_DIR"

    # Download kokoro model
    if [ ! -f "kokoro-v1.0.onnx" ]; then
        print_info "Baixando kokoro-v1.0.onnx (311 MB)..."
        curl -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx" -o kokoro-v1.0.onnx
        print_success "Modelo baixado"
    else
        print_success "Modelo já existe"
    fi

    # Download voices
    if [ ! -f "voices-v1.0.bin" ]; then
        print_info "Baixando voices-v1.0.bin (27 MB)..."
        curl -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin" -o voices-v1.0.bin
        print_success "Vozes baixadas"
    else
        print_success "Vozes já existem"
    fi
}

generate_config() {
    print_header "Gerando Configuração"

    CONFIG_FILE="$CONFIG_DIR/config.yaml"

    cat > "$CONFIG_FILE" << EOF
whisper:
  model: "$WHISPER_MODEL"
  language: "pt"
  device: "$( [ "$FEATURE_GPU" = true ] && echo "cuda" || echo "cpu" )"

hotkey:
  type: "keyboard"
  keyboard_trigger: "ctrl+alt+v"
  mode: "toggle"
  vad_enabled: $FEATURE_VAD

tts:
  enabled: $FEATURE_TTS
  engine: "kokoro-onnx"
  kokoro:
    model_path: "$INSTALL_DIR/kokoro-v1.0.onnx"
    voices_path: "$INSTALL_DIR/voices-v1.0.bin"
    voice: "pf_dora"

voice:
  claude_mode: $FEATURE_LLM

service:
  auto_start: true
  log_file: "$INSTALL_DIR/logs/dictator.log"

audio:
  sample_rate: 16000
  channels: 1
EOF

    print_success "Configuração gerada: $CONFIG_FILE"
}

install_systemd_service() {
    print_header "Instalando Serviço systemd"

    SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Dictator Voice to Text Service
After=network.target sound.target

[Service]
Type=simple
User=$SUDO_USER
Environment="PYTHONUNBUFFERED=1"
ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/src/dictator/service.py $CONFIG_DIR/config.yaml
Restart=on-failure
RestartSec=5
StandardOutput=append:$INSTALL_DIR/logs/service.log
StandardError=append:$INSTALL_DIR/logs/service_error.log

[Install]
WantedBy=multi-user.target
EOF

    print_success "Service file criado: $SERVICE_FILE"

    # Reload systemd
    systemctl daemon-reload
    print_success "systemd recarregado"

    # Enable service
    systemctl enable "$SERVICE_NAME"
    print_success "Serviço habilitado para auto-start"
}

start_service() {
    print_header "Iniciando Serviço"

    systemctl start "$SERVICE_NAME"

    sleep 2

    if systemctl is-active --quiet "$SERVICE_NAME"; then
        print_success "Serviço iniciado com sucesso"
    else
        print_error "Falha ao iniciar serviço"
        echo "Verifique os logs em: $INSTALL_DIR/logs/"
    fi
}

print_completion() {
    print_header "Instalação Concluída!"

    echo ""
    echo -e "${GREEN}✓ Dictator instalado com sucesso!${NC}"
    echo ""
    echo "Features instaladas:"
    echo "  • Core Transcription (modelo: $WHISPER_MODEL)"
    [ "$FEATURE_GPU" = true ] && echo "  • GPU Acceleration"
    [ "$FEATURE_TTS" = true ] && echo "  • Text-to-Speech"
    [ "$FEATURE_VAD" = true ] && echo "  • Voice Activity Detection"
    [ "$FEATURE_LLM" = true ] && echo "  • LLM Voice Assistant"
    echo ""
    echo "Comandos úteis:"
    echo "  • Ver status:     systemctl status $SERVICE_NAME"
    echo "  • Ver logs:       journalctl -u $SERVICE_NAME -f"
    echo "  • Parar serviço:  sudo systemctl stop $SERVICE_NAME"
    echo "  • Reiniciar:      sudo systemctl restart $SERVICE_NAME"
    echo "  • Editar config:  nano $CONFIG_DIR/config.yaml"
    echo ""
    echo "Arquivos:"
    echo "  • Instalação:     $INSTALL_DIR"
    echo "  • Configuração:   $CONFIG_DIR/config.yaml"
    echo "  • Logs:           $INSTALL_DIR/logs/"
    echo ""
}

# Main installation flow
main() {
    print_header "DICTATOR LINUX INSTALLER"

    check_root
    check_distro
    check_dependencies
    feature_selection

    echo ""
    read -p "Continuar com a instalação? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Instalação cancelada"
        exit 0
    fi

    create_directories
    create_venv
    install_dependencies
    download_models
    generate_config
    install_systemd_service
    start_service
    print_completion
}

# Run main
main

#!/bin/bash
# Dictator Linux Uninstaller

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="/opt/dictator"
CONFIG_DIR="$HOME/.config/dictator"
SERVICE_NAME="dictator"

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Este script precisa ser executado como root${NC}"
        echo "Execute com: sudo ./uninstall.sh"
        exit 1
    fi
}

echo "========================================"
echo "  DICTATOR UNINSTALLER"
echo "========================================"
echo ""
echo "Isto irá remover:"
echo "  • Serviço systemd"
echo "  • Arquivos de instalação em $INSTALL_DIR"
echo ""
echo "Será mantido (para preservar suas configurações):"
echo "  • Configuração em $CONFIG_DIR"
echo "  • Logs em $INSTALL_DIR/logs"
echo ""
read -p "Continuar com a desinstalação? [y/N]: " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Desinstalação cancelada"
    exit 0
fi

check_root

# Stop service
if systemctl is-active --quiet "$SERVICE_NAME"; then
    systemctl stop "$SERVICE_NAME"
    print_success "Serviço parado"
fi

# Disable service
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl disable "$SERVICE_NAME"
    print_success "Serviço desabilitado"
fi

# Remove service file
if [ -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
    rm "/etc/systemd/system/$SERVICE_NAME.service"
    systemctl daemon-reload
    print_success "Service file removido"
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    print_success "Diretório de instalação removido"
fi

echo ""
echo -e "${GREEN}✓ Desinstalação concluída!${NC}"
echo ""
echo "Configurações preservadas em: $CONFIG_DIR"
echo "Para remover completamente: rm -rf $CONFIG_DIR"

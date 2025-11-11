# 🎙️ Dictator - Voice to Text Windows Service

Serviço Windows para transcrição de voz em texto usando **Whisper AI localmente com GPU**.

## ✨ Características

- ✅ **Windows Service** - Inicia automaticamente com o Windows
- ✅ **Hotkey Global** - Ative de qualquer lugar (Ctrl+Alt+V padrão)
- ✅ **System Tray** - Controle fácil via ícone na bandeja
- ✅ **GPU Accelerated** - NVIDIA CUDA para transcrição rápida
- ✅ **100% Local** - Sem APIs externas, privacidade total
- ✅ **Auto-paste** - Cola texto automaticamente no campo em foco
- ✅ **Configurável** - YAML para fácil personalização
- ✅ **Poetry** - Gerenciamento moderno de dependências

## 📋 Requisitos

- **Windows 10/11**
- **Python 3.10+**
- **NVIDIA GPU** com CUDA (RTX series recomendado)
- **Poetry** (instalado automaticamente pelo installer)

## 🚀 Instalação Rápida

### Passo 0: Instalar Dependências (PRIMEIRA VEZ)

**Execute APENAS UMA VEZ antes de usar:**
```batch
setup.bat
```

Este script irá:
- Verificar/instalar Poetry
- Instalar todas as dependências Python
- Instalar PyTorch com CUDA

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
     - Verificar Python
     - Instalar Poetry (se necessário)
     - Instalar NSSM (gerenciador de serviço)
     - Instalar todas as dependências
     - Configurar o serviço Windows
     - Iniciar o serviço

2. **Procure o ícone de microfone na bandeja do sistema**

## 🎯 Como Usar

1. **Pressione** `Ctrl+Alt+V` (ou seu hotkey configurado)
2. **Fale** o que deseja transcrever
3. **Pressione** `Ctrl+Alt+V` novamente para parar
4. **Aguarde** a transcrição (alguns segundos)
5. **Texto será colado** automaticamente no campo em foco!

### Menu da Bandeja

Clique com botão direito no ícone do microfone:
- **Open Config** - Editar configurações
- **Open Logs** - Ver logs do serviço
- **Restart Service** - Reiniciar serviço
- **Exit** - Sair do serviço

## ⚙️ Configuração

Edite `config.yaml`:

```yaml
# Modelo Whisper (tiny, base, small, medium, large, large-v3)
whisper:
  model: "medium"      # Padrão recomendado
  language: "pt"       # Idioma (pt, en, es, etc.)
  device: "cuda"       # GPU (cuda) ou CPU (cpu)

# Hotkey personalizado
hotkey:
  trigger: "ctrl+alt+v"

# Auto-paste
paste:
  auto_paste: true     # false = apenas copia para clipboard
  delay: 0.5           # Delay antes de colar
```

Após editar, reinicie o serviço pelo menu da bandeja.

## 🧪 Testar Sem Instalar

Para testar antes de instalar como serviço:

```batch
run_local.bat
```

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

```batch
# Instalar dependências
poetry install

# Instalar PyTorch com CUDA
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Executar localmente
poetry run python src/dictator/tray.py config.yaml
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
│       ├── main.py          # Script original
│       ├── service.py       # Serviço Windows
│       └── tray.py          # System tray
├── config.yaml              # Configuração
├── pyproject.toml          # Poetry
├── install_service.bat     # Instalador
├── uninstall_service.bat   # Desinstalador
├── run_local.bat           # Teste local
├── SERVICE.md              # Docs completa
└── README.md               # Este arquivo
```

## 🔐 Privacidade

- ✅ Tudo roda **100% local**
- ✅ Nenhum dado enviado para nuvem
- ✅ Modelos armazenados localmente
- ✅ Sem telemetria ou tracking

## 💰 Custo

**GRATUITO!** 🎉
- Sem APIs pagas
- Sem limites de uso
- Apenas custo de hardware (GPU local)

## 🆘 Suporte

Problemas comuns? Veja [SERVICE.md](SERVICE.md) seção de Troubleshooting.

## 📄 Licença

MIT License - Use como quiser!

---

**Desenvolvido com 🎙️ para transcrições rápidas, privadas e eficientes!**

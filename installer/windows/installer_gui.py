"""
Installer GUI Wizard

Feature-focused GUI wizard for Dictator installation using tkinter.
Each screen explains features, their benefits, and required dependencies.
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.state_manager import StateManager
from installer_core import InstallationEngine
from detection import InstallationDetector, ExistingInstallation

logger = logging.getLogger("InstallerGUI")


class FeatureInfo:
    """Feature information for user display"""

    # Feature definitions with descriptions and dependencies
    FEATURES = {
        "core_transcription": {
            "name": "Core Transcription",
            "required": True,
            "description": "Transcrição de voz para texto usando Whisper AI",
            "benefits": [
                "✓ Transcrição local de alta qualidade",
                "✓ Suporte a 99+ idiomas",
                "✓ Funciona 100% offline",
                "✓ Privacidade total (nada enviado para internet)"
            ],
            "dependencies": [
                "faster-whisper (Whisper otimizado)",
                "sounddevice (captura de áudio)",
                "numpy (processamento)"
            ],
            "disk_space_mb": 500,
            "download_mb": 0
        },
        "gpu_acceleration": {
            "name": "GPU Acceleration",
            "required": False,
            "description": "Aceleração GPU para transcrição ultrarrápida",
            "benefits": [
                "✓ Transcrição 5-10x mais rápida",
                "✓ Menor latência (1-3s vs 10-30s)",
                "✓ Suporta modelos maiores e mais precisos",
                "✓ Melhor para uso frequente"
            ],
            "requirements": [
                "NVIDIA GPU (GTX/RTX series)",
                "Drivers CUDA atualizados",
                "Mínimo 4GB VRAM (8GB+ recomendado)"
            ],
            "dependencies": [
                "torch (PyTorch com CUDA)",
                "CUDA toolkit (via drivers NVIDIA)"
            ],
            "disk_space_mb": 2000,
            "download_mb": 800,
            "fallback": "Modo CPU (mais lento mas funcional)"
        },
        "tts": {
            "name": "Text-to-Speech (TTS)",
            "required": False,
            "description": "Síntese de voz local de alta qualidade",
            "benefits": [
                "✓ 56 vozes em 9 idiomas",
                "✓ Qualidade de voz natural",
                "✓ Velocidade ajustável",
                "✓ Interrupção instantânea (~170ms)",
                "✓ Útil para modo assistente de voz"
            ],
            "dependencies": [
                "kokoro-onnx (engine TTS)",
                "kokoro-v1.0.onnx (modelo 311MB)",
                "voices-v1.0.bin (vozes 27MB)"
            ],
            "disk_space_mb": 500,
            "download_mb": 338,
            "fallback": "Modo silencioso (apenas transcrição)"
        },
        "vad": {
            "name": "Voice Activity Detection (VAD)",
            "required": False,
            "description": "Detecção automática de fala e silêncio",
            "benefits": [
                "✓ Para gravação automaticamente após silêncio",
                "✓ Não precisa clicar para parar",
                "✓ Transcrição mais fluida",
                "✓ Detecta quando você parou de falar"
            ],
            "requirements": [
                "Recomendado: GPU para melhor detecção",
                "Funciona em CPU (com latência maior)"
            ],
            "dependencies": [
                "silero-vad (modelo de detecção)",
                "torch (já incluído se GPU ativado)"
            ],
            "disk_space_mb": 50,
            "download_mb": 0,
            "fallback": "Modo manual (clique para parar)"
        },
        "llm_assistant": {
            "name": "LLM Voice Assistant",
            "required": False,
            "description": "Assistente de voz com IA conversacional",
            "benefits": [
                "✓ Converse com LLMs locais (Ollama)",
                "✓ Respostas faladas via TTS",
                "✓ Contexto preservado entre conversas",
                "✓ Suporta modelos thinking (Qwen, DeepSeek)",
                "✓ 100% local ou integração com Claude"
            ],
            "requirements": [
                "Ollama instalado (para modelos locais)",
                "OU credenciais Claude API",
                "TTS habilitado (recomendado para respostas)"
            ],
            "dependencies": [
                "aiohttp (chamadas assíncronas)",
                "requests (HTTP client)"
            ],
            "disk_space_mb": 100,
            "download_mb": 0,
            "fallback": "Modo ditado simples"
        }
    }

    @staticmethod
    def get_total_disk_space(selected_features: list) -> int:
        """Get total disk space required in MB"""
        total = 0
        for feature_id in selected_features:
            if feature_id in FeatureInfo.FEATURES:
                total += FeatureInfo.FEATURES[feature_id]["disk_space_mb"]
        return total

    @staticmethod
    def get_total_download_size(selected_features: list) -> int:
        """Get total download size in MB"""
        total = 0
        for feature_id in selected_features:
            if feature_id in FeatureInfo.FEATURES:
                total += FeatureInfo.FEATURES[feature_id].get("download_mb", 0)
        return total


class InstallerWizard:
    """
    Main installer wizard GUI

    Feature-focused flow:
    1. Welcome
    2. System Check
    3. Feature Selection (with explanations)
    4. Installation Location
    5. Confirmation & Summary
    6. Installation Progress
    7. Success/Failure
    """

    def __init__(self):
        """Initialize installer wizard"""
        self.root = tk.Tk()
        self.root.title("Dictator Installer")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # State
        self.current_screen = 0
        self.selected_features = {
            "core_transcription": True,  # Always required
            "gpu_acceleration": True,  # Enabled by default
            "tts": True,  # Enabled by default
            "vad": False,  # Disabled by default
            "llm_assistant": False  # Disabled by default
        }
        self.install_dir = Path("C:/Program Files/Dictator")
        self.whisper_model = "large-v3"

        # Installation mode and existing installation detection
        self.installation_mode = "install"  # "install", "update", "repair"
        self.existing_installation: Optional[ExistingInstallation] = None

        # State manager will be initialized later
        self.state_manager: Optional[StateManager] = None
        self.installation_engine: Optional[InstallationEngine] = None

        # Navigation buttons frame (CREATE FIRST so it stays at bottom)
        self.nav_frame = tk.Frame(self.root, height=60, relief="groove", borderwidth=1)
        self.nav_frame.pack(side="bottom", fill="x", pady=0, padx=0)
        self.nav_frame.pack_propagate(False)  # Don't shrink

        self.btn_back = tk.Button(self.nav_frame, text="← Voltar", command=self.go_back, width=15, height=2)
        self.btn_back.pack(side="left", padx=10, pady=10)

        self.btn_cancel = tk.Button(self.nav_frame, text="Cancelar", command=self.cancel_installation, width=15, height=2)
        self.btn_cancel.pack(side="left", padx=5, pady=10)

        self.btn_next = tk.Button(self.nav_frame, text="Avançar →", command=self.go_next, width=15, height=2)
        self.btn_next.pack(side="right", padx=10, pady=10)

        # Main container (AFTER nav_frame so it doesn't cover buttons)
        self.container = tk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        # Create screens
        self.screens = [
            self.create_welcome_screen,
            self.create_detection_screen,  # NEW: Detect existing installation
            self.create_system_check_screen,
            self.create_feature_selection_screen,
            self.create_whisper_model_screen,
            self.create_location_screen,
            self.create_confirmation_screen,
            self.create_installation_screen,
            self.create_completion_screen
        ]

        # Show first screen
        self.show_screen(0)

    def clear_container(self):
        """Clear current screen"""
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_screen(self, screen_index: int):
        """Show specific screen"""
        if 0 <= screen_index < len(self.screens):
            self.current_screen = screen_index
            self.clear_container()
            self.screens[screen_index]()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update navigation button states"""
        # Back button
        self.btn_back.config(state="normal" if self.current_screen > 0 else "disabled")

        # Next/Install button
        # Screen count: 9 screens total
        # Index 0: Welcome
        # Index 1: Detection
        # Index 2: System Check
        # Index 3: Feature Selection
        # Index 4: Whisper Model
        # Index 5: Location
        # Index 6: Confirmation (len - 3)
        # Index 7: Installation (len - 2)
        # Index 8: Completion (len - 1)

        if self.current_screen == len(self.screens) - 1:
            # Last screen (completion)
            self.btn_next.config(text="Finalizar", state="normal")
        elif self.current_screen == len(self.screens) - 3:
            # Confirmation screen (before installation) - show "Instalar" button
            self.btn_next.config(text="Instalar", state="normal")
        elif self.current_screen == len(self.screens) - 2:
            # Installation screen (during installation) - disable button
            self.btn_next.config(text="Instalando...", state="disabled")
        else:
            self.btn_next.config(text="Avançar →", state="normal")

        # Cancel button (disable during installation)
        if self.current_screen == len(self.screens) - 2:
            self.btn_cancel.config(state="disabled")
        else:
            self.btn_cancel.config(state="normal")

    def go_next(self):
        """Go to next screen"""
        if self.current_screen == len(self.screens) - 1:
            # Last screen - close installer
            self.root.quit()
        elif self.current_screen == len(self.screens) - 3:
            # Confirmation screen - start installation
            self.start_installation()
        else:
            self.show_screen(self.current_screen + 1)

    def go_back(self):
        """Go to previous screen"""
        if self.current_screen > 0:
            self.show_screen(self.current_screen - 1)

    def cancel_installation(self):
        """Cancel installation"""
        if messagebox.askyesno("Cancelar", "Tem certeza que deseja cancelar a instalação?"):
            self.root.quit()

    # Screen implementations

    def create_welcome_screen(self):
        """Screen 1: Welcome"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        # Title
        title = tk.Label(frame, text="Bem-vindo ao Dictator Installer", font=("Arial", 24, "bold"))
        title.pack(pady=20)

        # Description
        desc = tk.Label(
            frame,
            text="Dictator é um serviço de transcrição de voz para texto que roda 100% local.\n\n"
                 "• Transcrição de alta qualidade com Whisper AI\n"
                 "• Aceleração GPU para velocidade máxima\n"
                 "• Text-to-Speech com 56 vozes\n"
                 "• Assistente de voz com LLMs locais\n"
                 "• Privacidade total - nada enviado para internet\n\n"
                 "Este instalador irá guiá-lo através da seleção de features\n"
                 "e instalação das dependências necessárias.",
            font=("Arial", 11),
            justify="left"
        )
        desc.pack(pady=20)

        # Version info
        version = tk.Label(frame, text="Versão 1.0.0", font=("Arial", 9), fg="gray")
        version.pack(side="bottom", pady=10)

    def create_detection_screen(self):
        """Screen 1.5: Detect Existing Installation"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        # Title
        title = tk.Label(frame, text="Detectando Instalação Existente", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        # Detect existing installation
        detector = InstallationDetector()
        self.existing_installation = detector.detect_existing_installation()

        if self.existing_installation is None:
            # No existing installation - proceed with clean install
            desc = tk.Label(
                frame,
                text="✓ Nenhuma instalação existente detectada.\n\n"
                     "Uma instalação limpa será realizada.",
                font=("Arial", 12),
                justify="center"
            )
            desc.pack(pady=30)

            # Set installation mode
            self.installation_mode = "install"

        else:
            # Existing installation found - show options
            # Get summary
            summary = detector.get_installation_summary(self.existing_installation)

            # Show summary in scrollable text
            summary_frame = tk.Frame(frame)
            summary_frame.pack(fill="both", expand=True, pady=10)

            summary_text = tk.Text(summary_frame, height=10, font=("Courier", 10), wrap="word")
            summary_text.pack(side="left", fill="both", expand=True)
            summary_text.insert("1.0", summary)
            summary_text.config(state="disabled")

            # Scrollbar
            scrollbar = tk.Scrollbar(summary_frame, command=summary_text.yview)
            scrollbar.pack(side="right", fill="y")
            summary_text.config(yscrollcommand=scrollbar.set)

            # Ask user what to do
            question = tk.Label(
                frame,
                text="\nO que você deseja fazer?",
                font=("Arial", 12, "bold")
            )
            question.pack(pady=10)

            # Mode selection variable
            mode_var = tk.StringVar(value="update")

            # Radio buttons for mode selection
            options_frame = tk.Frame(frame)
            options_frame.pack(pady=10)

            tk.Radiobutton(
                options_frame,
                text="Atualizar instalação existente (recomendado)",
                variable=mode_var,
                value="update",
                font=("Arial", 11)
            ).pack(anchor="w", pady=5)

            tk.Radiobutton(
                options_frame,
                text="Reparar instalação existente",
                variable=mode_var,
                value="repair",
                font=("Arial", 11)
            ).pack(anchor="w", pady=5)

            tk.Radiobutton(
                options_frame,
                text="Remover e reinstalar do zero",
                variable=mode_var,
                value="reinstall",
                font=("Arial", 11)
            ).pack(anchor="w", pady=5)

            # Store mode selection callback
            def store_mode():
                selected_mode = mode_var.get()
                if selected_mode == "reinstall":
                    self.installation_mode = "install"
                    # TODO: Add uninstall step before proceeding
                else:
                    self.installation_mode = selected_mode

                # Update install_dir to existing path
                self.install_dir = self.existing_installation.path

            # Call immediately and on radio button change
            store_mode()
            mode_var.trace_add("write", lambda *args: store_mode())

            # Info about backup
            info = tk.Label(
                frame,
                text="\n✓ Backup automático de config.yaml será realizado",
                font=("Arial", 9),
                fg="gray"
            )
            info.pack(pady=5)

    def create_system_check_screen(self):
        """Screen 2: System Requirements Check"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        title = tk.Label(frame, text="Verificação do Sistema", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        desc = tk.Label(
            frame,
            text="Verificando se seu sistema atende aos requisitos mínimos...",
            font=("Arial", 11)
        )
        desc.pack(pady=10)

        # Results frame
        results_frame = tk.Frame(frame)
        results_frame.pack(fill="both", expand=True, pady=20)

        # Import and run actual checks
        from shared.validation import SystemRequirements
        import platform
        import psutil
        import shutil

        checks = []

        # OS Check
        os_name = platform.system()
        os_version = platform.release()
        checks.append(("Sistema Operacional:", f"✓ {os_name} {os_version}", "green"))

        # Python Check
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        checks.append(("Python:", f"✓ Python {py_version}", "green"))

        # RAM Check
        ram_gb = round(psutil.virtual_memory().total / (1024**3))
        checks.append(("RAM:", f"✓ {ram_gb} GB", "green"))

        # Disk Space Check
        disk = shutil.disk_usage("C:/")
        free_gb = round(disk.free / (1024**3))
        checks.append(("Espaço em Disco:", f"✓ {free_gb} GB livres", "green"))

        # GPU Check
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(',')
                gpu_name = gpu_info[0].strip()
                # Get CUDA version
                cuda_result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=5
                )
                cuda_version = "Unknown"
                if "CUDA Version:" in cuda_result.stdout:
                    cuda_line = [line for line in cuda_result.stdout.split('\n') if "CUDA Version:" in line]
                    if cuda_line:
                        cuda_version = cuda_line[0].split("CUDA Version:")[-1].strip().split()[0]
                checks.append(("GPU:", f"✓ {gpu_name} (CUDA {cuda_version})", "green"))
            else:
                checks.append(("GPU:", "⚠ Não detectada (modo CPU disponível)", "orange"))
        except:
            checks.append(("GPU:", "⚠ Não detectada (modo CPU disponível)", "orange"))

        # Network Check
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            checks.append(("Rede:", "✓ Conectado", "green"))
        except:
            checks.append(("Rede:", "✗ Sem conexão (necessário para download)", "red"))

        # Admin Check
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if is_admin:
            checks.append(("Permissões:", "✓ Administrador", "green"))
        else:
            checks.append(("Permissões:", "✗ Execute como Administrador", "red"))

        for check_name, result, color in checks:
            row = tk.Frame(results_frame)
            row.pack(fill="x", pady=5)

            label = tk.Label(row, text=f"{check_name}:", font=("Arial", 10, "bold"), width=20, anchor="w")
            label.pack(side="left")

            value = tk.Label(row, text=result, font=("Arial", 10), fg=color)
            value.pack(side="left")

    def create_feature_selection_screen(self):
        """Screen 3: Feature Selection (Main screen)"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=30, pady=30)

        title = tk.Label(frame, text="Seleção de Features", font=("Arial", 18, "bold"))
        title.pack(pady=10)

        desc = tk.Label(
            frame,
            text="Selecione as funcionalidades que deseja instalar.\n"
                 "Cada feature exibe suas dependências e requisitos de espaço.",
            font=("Arial", 10)
        )
        desc.pack(pady=5)

        # Scrollable frame for features (reduced height to accommodate nav buttons)
        canvas = tk.Canvas(frame, height=340, highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Unbind when leaving this screen
        def cleanup_mousewheel():
            canvas.unbind_all("<MouseWheel>")

        self.container.bind("<Destroy>", lambda e: cleanup_mousewheel())

        canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scrollbar.pack(side="right", fill="y")

        # Create feature cards
        self.feature_vars = {}

        for feature_id, feature_info in FeatureInfo.FEATURES.items():
            self.create_feature_card(scrollable_frame, feature_id, feature_info)

        # Summary at bottom
        summary_frame = tk.Frame(self.container, bg="#f0f0f0")
        summary_frame.pack(side="bottom", fill="x", padx=30, pady=10)

        self.summary_label = tk.Label(
            summary_frame,
            text="",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0"
        )
        self.summary_label.pack(pady=10)

        self.update_summary()

    def create_feature_card(self, parent, feature_id: str, info: dict):
        """Create a feature selection card"""
        # Main card frame with border
        card = tk.LabelFrame(
            parent,
            text="",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=8,
            relief="groove",
            borderwidth=2
        )
        card.pack(fill="x", padx=10, pady=8)

        # Header with checkbox
        header_frame = tk.Frame(card)
        header_frame.pack(fill="x", pady=(0, 5))

        var = tk.BooleanVar(value=self.selected_features.get(feature_id, False))
        self.feature_vars[feature_id] = var

        check = tk.Checkbutton(
            header_frame,
            text=info["name"],
            variable=var,
            font=("Arial", 11, "bold"),
            state="disabled" if info.get("required", False) else "normal",
            command=lambda: self.on_feature_toggle(feature_id)
        )
        check.pack(side="left", anchor="w")

        if info.get("required", False):
            req_label = tk.Label(header_frame, text="(Obrigatório)", fg="red", font=("Arial", 9, "bold"))
            req_label.pack(side="left", padx=(10, 0))

        # Description
        desc_label = tk.Label(
            card,
            text=info["description"],
            font=("Arial", 9),
            fg="#333",
            wraplength=700,
            justify="left"
        )
        desc_label.pack(anchor="w", pady=(0, 8))

        # Content frame (2 columns)
        content_frame = tk.Frame(card)
        content_frame.pack(fill="x", pady=5)

        # Left column: Benefits and Requirements
        left_col = tk.Frame(content_frame)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Benefits
        if "benefits" in info and info["benefits"]:
            for benefit in info["benefits"]:
                label = tk.Label(left_col, text=benefit, font=("Arial", 9), fg="darkgreen", anchor="w")
                label.pack(anchor="w", pady=1)

        # Requirements
        if "requirements" in info and info["requirements"]:
            if "benefits" in info:  # Add spacing
                tk.Frame(left_col, height=5).pack()
            req_title = tk.Label(left_col, text="Requisitos:", font=("Arial", 9, "bold"), anchor="w")
            req_title.pack(anchor="w", pady=(3, 2))

            for req in info["requirements"]:
                label = tk.Label(left_col, text=f"  • {req}", font=("Arial", 8), fg="#555", anchor="w")
                label.pack(anchor="w", pady=1)

        # Right column: Dependencies and Space
        right_col = tk.Frame(content_frame)
        right_col.pack(side="right", fill="both", expand=True)

        # Dependencies
        if "dependencies" in info and info["dependencies"]:
            dep_title = tk.Label(right_col, text="Dependências instaladas:", font=("Arial", 9, "bold"), anchor="w")
            dep_title.pack(anchor="w", pady=(0, 2))

            for dep in info["dependencies"]:
                label = tk.Label(right_col, text=f"  • {dep}", font=("Arial", 8), fg="#555", anchor="w")
                label.pack(anchor="w", pady=1)

        # Space requirements at bottom
        space_frame = tk.Frame(card, bg="#f5f5f5")
        space_frame.pack(fill="x", pady=(8, 0))

        space_text = f"💾 Espaço necessário: {info['disk_space_mb']} MB"
        if info.get("download_mb", 0) > 0:
            space_text += f"  |  📥 Download: {info['download_mb']} MB"

        space_label = tk.Label(
            space_frame,
            text=space_text,
            font=("Arial", 8, "bold"),
            fg="#666",
            bg="#f5f5f5"
        )
        space_label.pack(anchor="w", pady=(5, 0))

        # Fallback option
        if "fallback" in info:
            fallback_label = tk.Label(
                card,
                text=f"Se desabilitado: {info['fallback']}",
                font=("Arial", 8, "italic"),
                fg="orange"
            )
            fallback_label.pack(anchor="w", pady=(2, 0))

    def on_feature_toggle(self, feature_id: str):
        """Handle feature toggle"""
        self.selected_features[feature_id] = self.feature_vars[feature_id].get()
        self.update_summary()

    def update_summary(self):
        """Update installation summary"""
        selected = [f for f, enabled in self.selected_features.items() if enabled]

        total_disk = FeatureInfo.get_total_disk_space(selected)
        total_download = FeatureInfo.get_total_download_size(selected)

        summary = f"Features selecionadas: {len(selected)} | "
        summary += f"Espaço necessário: ~{total_disk} MB | "
        summary += f"Download: ~{total_download} MB"

        self.summary_label.config(text=summary)

    def create_whisper_model_screen(self):
        """Screen 4: Whisper Model Selection"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        title = tk.Label(frame, text="Modelo Whisper", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        desc = tk.Label(
            frame,
            text="Escolha o modelo Whisper para transcrição.\n"
                 "Modelos maiores = mais precisão, mas mais VRAM e tempo de processamento.",
            font=("Arial", 11)
        )
        desc.pack(pady=10)

        # Model options
        models = [
            ("tiny", "Tiny - Mais rápido (1GB VRAM, ~0.5s/min, precisão básica)"),
            ("small", "Small - Balanceado (2GB VRAM, ~0.8s/min, boa precisão)"),
            ("medium", "Medium - Recomendado (5GB VRAM, ~1.5s/min, alta precisão)"),
            ("large-v3", "Large-v3 - Melhor qualidade (10GB VRAM, ~2.5s/min, precisão máxima)")
        ]

        self.model_var = tk.StringVar(value=self.whisper_model)

        for model_id, model_desc in models:
            rb = tk.Radiobutton(
                frame,
                text=model_desc,
                variable=self.model_var,
                value=model_id,
                font=("Arial", 10)
            )
            rb.pack(anchor="w", pady=5, padx=40)

        # GPU recommendation
        if self.selected_features.get("gpu_acceleration", False):
            gpu_note = tk.Label(
                frame,
                text="✓ GPU habilitado - Recomendamos 'medium' ou 'large-v3'",
                font=("Arial", 10),
                fg="green"
            )
            gpu_note.pack(pady=20)
        else:
            cpu_note = tk.Label(
                frame,
                text="⚠ Modo CPU - Recomendamos 'tiny' ou 'small' para velocidade",
                font=("Arial", 10),
                fg="orange"
            )
            cpu_note.pack(pady=20)

    def create_location_screen(self):
        """Screen 5: Installation Location"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        title = tk.Label(frame, text="Localização da Instalação", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        # Directory selection
        dir_frame = tk.Frame(frame)
        dir_frame.pack(fill="x", pady=20)

        label = tk.Label(dir_frame, text="Diretório de instalação:", font=("Arial", 11))
        label.pack(anchor="w", pady=5)

        path_frame = tk.Frame(dir_frame)
        path_frame.pack(fill="x", pady=5)

        self.path_entry = tk.Entry(path_frame, font=("Arial", 10), width=50)
        self.path_entry.insert(0, str(self.install_dir))
        self.path_entry.pack(side="left", padx=(0, 10))

        browse_btn = tk.Button(path_frame, text="Procurar...", command=self.browse_directory)
        browse_btn.pack(side="left")

        # Space info
        info_text = f"Espaço necessário: ~{FeatureInfo.get_total_disk_space([f for f, e in self.selected_features.items() if e])} MB"
        info_label = tk.Label(frame, text=info_text, font=("Arial", 10), fg="gray")
        info_label.pack(pady=10)

    def browse_directory(self):
        """Browse for installation directory"""
        directory = filedialog.askdirectory(initialdir=self.install_dir)
        if directory:
            self.install_dir = Path(directory)
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, str(self.install_dir))

    def create_confirmation_screen(self):
        """Screen 6: Confirmation & Summary"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        title = tk.Label(frame, text="Confirmação", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        desc = tk.Label(frame, text="Revise as configurações antes de instalar:", font=("Arial", 11))
        desc.pack(pady=10)

        # Summary text
        summary_frame = tk.Frame(frame, bg="#f9f9f9", relief="sunken", bd=1)
        summary_frame.pack(fill="both", expand=True, pady=20)

        text = tk.Text(summary_frame, wrap="word", font=("Courier", 9), height=20)
        text.pack(fill="both", expand=True, padx=10, pady=10)

        # Build summary
        selected = [f for f, enabled in self.selected_features.items() if enabled]
        feature_names = [FeatureInfo.FEATURES[f]["name"] for f in selected]

        summary_text = f"""
RESUMO DA INSTALAÇÃO
{'=' * 60}

Local: {self.install_dir}

Modelo Whisper: {self.model_var.get() if hasattr(self, 'model_var') else 'large-v3'}

Features Selecionadas ({len(selected)}):
{chr(10).join('  • ' + name for name in feature_names)}

Espaço em Disco: ~{FeatureInfo.get_total_disk_space(selected)} MB
Download Necessário: ~{FeatureInfo.get_total_download_size(selected)} MB

{'=' * 60}

Pressione 'Instalar' para continuar.
        """

        text.insert("1.0", summary_text.strip())
        text.config(state="disabled")

    def create_installation_screen(self):
        """Screen 7: Installation Progress"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        title = tk.Label(frame, text="Instalando...", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        # Progress bars
        self.progress_frame = tk.Frame(frame)
        self.progress_frame.pack(fill="both", expand=True, pady=20)

        # Overall progress
        overall_label = tk.Label(self.progress_frame, text="Progresso Geral:", font=("Arial", 11, "bold"))
        overall_label.pack(anchor="w", pady=(0, 5))

        self.overall_progress = ttk.Progressbar(self.progress_frame, length=600, mode="determinate")
        self.overall_progress.pack(fill="x", pady=(0, 20))

        # Current step
        self.step_label = tk.Label(self.progress_frame, text="Preparando...", font=("Arial", 10))
        self.step_label.pack(anchor="w", pady=(0, 5))

        self.step_progress = ttk.Progressbar(self.progress_frame, length=600, mode="determinate")
        self.step_progress.pack(fill="x", pady=(0, 20))

        # Log output
        log_label = tk.Label(self.progress_frame, text="Log de Instalação:", font=("Arial", 10, "bold"))
        log_label.pack(anchor="w", pady=(10, 5))

        log_frame = tk.Frame(self.progress_frame)
        log_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(log_frame, wrap="word", font=("Courier", 8), height=10, yscrollcommand=scrollbar.set)
        self.log_text.pack(fill="both", expand=True)

        scrollbar.config(command=self.log_text.yview)

    def create_completion_screen(self):
        """Screen 8: Success/Failure"""
        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        # Will be populated after installation

    def start_installation(self):
        """Start installation process"""
        self.show_screen(len(self.screens) - 2)  # Show installation screen

        # Disable navigation during installation
        self.btn_back.config(state="disabled")
        self.btn_next.config(state="disabled")
        self.btn_cancel.config(state="disabled")

        # Run installation in background thread
        thread = threading.Thread(target=self.run_installation_process)
        thread.daemon = True
        thread.start()

    def run_installation_process(self):
        """Run installation in background thread"""
        try:
            # Initialize state manager
            self.state_manager = StateManager()

            # ALWAYS reset state for clean installation (all modes)
            logger.info(f"Installation mode: {self.installation_mode} - resetting state for clean install")
            self.state_manager.reset_state()

            # Configure state with user selections
            self.state_manager.update_config({
                "install_dir": str(self.install_dir),
                "venv_dir": str(self.install_dir / "venv"),
                "config_dir": str(self.install_dir / "config"),
                "log_dir": str(self.install_dir / "logs")
            })

            # Set features
            features = {
                "gpu_enabled": self.selected_features.get("gpu_acceleration", False),
                "tts_enabled": self.selected_features.get("tts", False),
                "llm_enabled": self.selected_features.get("llm_assistant", False),
                "vad_enabled": self.selected_features.get("vad", False),
                "whisper_model": self.model_var.get() if hasattr(self, 'model_var') else "large-v3",
                "install_as_service": True,
                "auto_start_service": True
            }

            self.state_manager.update_features(features)

            # Create installation engine with mode
            self.installation_engine = InstallationEngine(
                self.state_manager,
                progress_callback=self.on_installation_progress,
                installation_mode=self.installation_mode
            )

            # Run installation
            success = self.installation_engine.run_installation()

            # Clean up state file
            if success:
                try:
                    self.state_manager.cleanup()
                    logger.info("State file cleaned up after successful installation")
                except Exception as e:
                    logger.warning(f"Failed to cleanup state file: {e}")
            elif self.installation_mode == "install":
                # Clean state after failed install to avoid confusion on next attempt
                try:
                    self.state_manager.cleanup()
                    logger.info("State file cleaned up after failed install (clean mode)")
                except Exception as e:
                    logger.warning(f"Failed to cleanup state file: {e}")

            # Show completion screen
            self.root.after(0, lambda: self.show_completion_screen(success))

        except Exception as e:
            logger.error(f"Installation failed: {e}")
            self.root.after(0, lambda: self.show_completion_screen(False, str(e)))

    def on_installation_progress(self, step: str, percent: float, message: str):
        """Handle installation progress callback"""
        def update():
            self.step_label.config(text=f"Passo atual: {step}")
            self.step_progress["value"] = percent

            # Update overall progress (rough estimate)
            total_steps = 11
            current_step_index = list(self.state_manager.completed_steps).index(step) if step in self.state_manager.completed_steps else 0
            overall = (current_step_index / total_steps) * 100
            self.overall_progress["value"] = overall

            # Add to log
            self.log_text.insert("end", f"{message}\n")
            self.log_text.see("end")

        self.root.after(0, update)

    def show_completion_screen(self, success: bool, error: str = ""):
        """Show completion screen with results"""
        self.clear_container()

        frame = tk.Frame(self.container)
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        if success:
            title = tk.Label(frame, text="✓ Instalação Concluída!", font=("Arial", 20, "bold"), fg="green")
            title.pack(pady=30)

            message = tk.Label(
                frame,
                text="Dictator foi instalado com sucesso!\n\n"
                     "O serviço foi iniciado e está rodando em segundo plano.\n"
                     "Procure o ícone do microfone na bandeja do sistema.\n\n"
                     "Próximos passos:\n"
                     "1. Clique no ícone da bandeja para configurar hotkeys\n"
                     "2. Teste a transcrição com o botão do mouse\n"
                     "3. Confira os logs em caso de problemas",
                font=("Arial", 11),
                justify="left"
            )
            message.pack(pady=20)
        else:
            title = tk.Label(frame, text="✗ Instalação Falhou", font=("Arial", 20, "bold"), fg="red")
            title.pack(pady=20)

            # Error message
            error_text = error if error else "Erro desconhecido"
            message = tk.Label(
                frame,
                text=f"A instalação encontrou um erro:\n\n{error_text}",
                font=("Arial", 11),
                justify="left",
                fg="red"
            )
            message.pack(pady=10)

            # Log file location
            import tempfile
            log_file = Path(tempfile.gettempdir()) / "dictator_installer.log"

            log_info = tk.Label(
                frame,
                text=f"\nArquivo de log:\n{log_file}",
                font=("Arial", 9),
                justify="left",
                fg="gray"
            )
            log_info.pack(pady=5)

            # Button to open log
            def open_log():
                try:
                    import subprocess
                    subprocess.Popen(["notepad.exe", str(log_file)])
                except Exception as e:
                    messagebox.showerror("Erro", f"Não foi possível abrir o log:\n{e}")

            btn_log = tk.Button(
                frame,
                text="Abrir Log",
                command=open_log,
                width=20,
                height=2
            )
            btn_log.pack(pady=10)

        # Update navigation
        self.btn_next.config(text="Finalizar", state="normal")
        self.btn_cancel.config(state="disabled")

    def run(self):
        """Run the installer wizard"""
        self.root.mainloop()


def main():
    """Main entry point"""
    # Setup logging (console and file)
    import tempfile
    log_file = Path(tempfile.gettempdir()) / "dictator_installer.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )

    logger.info(f"Installer log file: {log_file}")

    # Create and run wizard
    wizard = InstallerWizard()
    wizard.run()


if __name__ == "__main__":
    # Protect against multiprocessing spawn on Windows (PyInstaller requirement)
    import multiprocessing
    multiprocessing.freeze_support()

    main()

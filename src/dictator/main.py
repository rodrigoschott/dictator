#!/usr/bin/env python3
"""
Voice to Text - Grava áudio, transcreve com Whisper LOCAL e cola no campo em foco
Uso: python voice_to_text.py
Aperte ENTER para começar a gravar, ENTER novamente para parar.
"""

import os
import sys
import tempfile
import sounddevice as sd
import soundfile as sf
import pyperclip
import pyautogui
import whisper

# Configurações
SAMPLE_RATE = 16000  # 16kHz recomendado para Whisper
CHANNELS = 1  # Mono
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # tiny, base, small, medium, large

def gravar_audio():
    """Grava áudio do microfone até o usuário pressionar ENTER"""
    print("\n[MIC] Pressione ENTER para começar a gravar...")
    input()

    print("[REC] GRAVANDO... (Pressione ENTER para parar)")

    # Lista para armazenar os chunks de áudio
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        recording.append(indata.copy())

    # Inicia a gravação
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        input()  # Espera o usuário pressionar ENTER novamente

    print("[STOP]  Gravação finalizada!")

    # Converte para numpy array
    import numpy as np
    audio_data = np.concatenate(recording, axis=0)

    return audio_data

def salvar_audio_temporario(audio_data):
    """Salva o áudio em um arquivo temporário"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, SAMPLE_RATE)
    print(f" Áudio salvo em: {temp_file.name}")
    return temp_file.name

def transcrever_com_whisper_local(audio_path, model):
    """Transcreve o áudio usando Whisper LOCAL"""
    try:
        print(f"🤖 Transcrevendo com Whisper local (modelo: {WHISPER_MODEL})...")

        result = model.transcribe(audio_path, language="pt")
        return result["text"]

    except Exception as e:
        print(f"[ERRO] Erro ao transcrever: {e}")
        sys.exit(1)

def colar_texto(texto):
    """Cola o texto no campo em foco"""
    print(f"\n[NOTE] Texto transcrito: {texto}")

    # Copia para o clipboard
    pyperclip.copy(texto)

    # Aguarda um momento para o usuário focar no campo desejado
    print("\n⏳ Colando texto no campo em foco em 2 segundos...")
    import time
    time.sleep(2)

    # Cola o texto (Ctrl+V)
    pyautogui.hotkey('ctrl', 'v')
    print("[OK] Texto colado!")

def main():
    print("=" * 60)
    print("[MIC]  VOICE TO TEXT - Whisper LOCAL")
    print("=" * 60)

    # Carrega o modelo Whisper (apenas na primeira vez)
    print(f"[PACKAGE] Carregando modelo Whisper '{WHISPER_MODEL}'...")
    print("[TIP] Dica: Use 'tiny' ou 'base' para velocidade, 'medium' ou 'large' para precisão")

    try:
        model = whisper.load_model(WHISPER_MODEL)
        print("[OK] Modelo carregado!\n")
    except Exception as e:
        print(f"[ERRO] Erro ao carregar modelo: {e}")
        print("\n[TIP] Instale o Whisper: pip install openai-whisper")
        sys.exit(1)

    try:
        # 1. Grava áudio
        audio_data = gravar_audio()

        # 2. Salva em arquivo temporário
        audio_path = salvar_audio_temporario(audio_data)

        # 3. Transcreve
        texto = transcrever_com_whisper_local(audio_path, model)

        # 4. Cola no campo em foco
        colar_texto(texto)

        # Limpa arquivo temporário
        os.unlink(audio_path)
        print(f"  Arquivo temporário removido")

    except KeyboardInterrupt:
        print("\n\n[PAUSE]  Operação cancelada pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERRO] Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

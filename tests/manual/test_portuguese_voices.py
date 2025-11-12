#!/usr/bin/env python3
"""
Test Portuguese voices in Kokoro TTS
Play samples of all Portuguese voices to help choose the best one
"""

import time
from pathlib import Path
from kokoro_onnx import Kokoro  # type: ignore
import sounddevice as sd  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]

print("🎤 Testando vozes em Português do Kokoro TTS")
print("=" * 60)
print()

# Initialize Kokoro
print("📦 Carregando Kokoro TTS...")
kokoro = Kokoro(str(REPO_ROOT / "kokoro-v1.0.onnx"), str(REPO_ROOT / "voices-v1.0.bin"))
print("✅ Kokoro carregado!")
print()

# Portuguese voices to test
portuguese_voices = [
    ("pf_dora", "Portuguese Female - Dora"),
    ("pm_alex", "Portuguese Male - Alex"),
    ("pm_santa", "Portuguese Male - Santa"),
]

# Test text in Portuguese
test_texts = [
    "Olá! Esta é a voz em português do Kokoro.",
    "Estou testando diferentes vozes para encontrar a melhor qualidade de áudio.",
    "Como você está hoje? Espero que goste desta demonstração.",
]

print("🔊 Reproduzindo amostras de cada voz...")
print("   (Aguarde o áudio terminar antes da próxima voz)")
print()

for voice_id, voice_name in portuguese_voices:
    print(f"▶️  Testando: {voice_name} ({voice_id})")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"   Frase {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # Generate audio
            start_time = time.time()
            audio, sample_rate = kokoro.create(
                text,
                voice=voice_id,
                lang="pt-br",
                speed=1.0
            )
            gen_time = time.time() - start_time
            
            # Play audio
            sd.play(audio, sample_rate)
            sd.wait()  # Wait until audio finishes
            
            print(f"      ⚡ Gerado em {gen_time:.2f}s")
            
            # Brief pause between sentences
            time.sleep(0.5)
            
        except Exception as e:
            print(f"      ❌ Erro: {e}")
    
    print()
    print("   🎵 Reprodução completa!")
    print()
    
    # Pause between voices
    time.sleep(1.5)

print("=" * 60)
print("✅ Teste concluído!")
print()
print("📊 Resumo das vozes testadas:")
for voice_id, voice_name in portuguese_voices:
    print(f"   - {voice_id}: {voice_name}")
print()
print("💡 Para alterar a voz no Dictator, edite config.yaml:")
print("   tts.kokoro.voice: pf_dora  (ou pm_alex, pm_santa)")
print()
print("🎯 Qual voz você preferiu?")

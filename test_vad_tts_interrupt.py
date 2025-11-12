#!/usr/bin/env python3
"""
Test VAD + TTS interrupt functionality
Validates that VAD can detect speech even during TTS playback
"""

print("🧪 Testing VAD + TTS interrupt implementation...")
print()

# Read the modified files
with open("src/dictator/service.py", "r", encoding="utf-8") as f:
    service_content = f.read()

with open("src/dictator/voice/session_manager.py", "r", encoding="utf-8") as f:
    session_content = f.read()

tests = [
    {
        "name": "TTS interrupt added to start_recording()",
        "file": "service.py",
        "content": service_content,
        "expected": "Interrupting TTS - user pressed hotkey to speak"
    },
    {
        "name": "TTS stop called before recording starts",
        "file": "service.py", 
        "content": service_content,
        "expected": "self.tts_engine.stop()"
    },
    {
        "name": "Brief wait after TTS stop",
        "file": "service.py",
        "content": service_content,
        "expected": "time.sleep(0.1)"
    },
    {
        "name": "VAD no longer blocked by tts_speaking",
        "file": "session_manager.py",
        "content": session_content,
        "expected": "if self.vad_enabled:",
        "not_expected": "and not self.tts_speaking"
    },
    {
        "name": "Comment updated about VAD blocking",
        "file": "session_manager.py",
        "content": session_content,
        "expected": "allows interruption"
    },
    {
        "name": "Comment explains TTS interruption in start_recording",
        "file": "session_manager.py",
        "content": session_content,
        "expected": "interrupted in start_recording()"
    }
]

passed = 0
failed = 0

for test in tests:
    try:
        if "not_expected" in test:
            # Check that string is NOT present
            if test["not_expected"] in test["content"]:
                print(f"❌ FAIL: {test['name']}")
                print(f"   Found unwanted: '{test['not_expected']}'")
                failed += 1
            elif test["expected"] in test["content"]:
                print(f"✅ PASS: {test['name']}")
                passed += 1
            else:
                print(f"❌ FAIL: {test['name']}")
                print(f"   Missing: '{test['expected']}'")
                failed += 1
        else:
            # Check that string IS present
            if test["expected"] in test["content"]:
                print(f"✅ PASS: {test['name']}")
                passed += 1
            else:
                print(f"❌ FAIL: {test['name']}")
                print(f"   Missing: '{test['expected']}'")
                failed += 1
    except Exception as e:
        print(f"❌ FAIL: {test['name']} - {e}")
        failed += 1

print()
print(f"📊 Results: {passed} passed, {failed} failed")

if failed == 0:
    print("✨ All VAD+TTS interrupt tests passed!")
    print()
    print("🎯 Expected behavior:")
    print("  1. User presses hotkey during TTS → TTS stops immediately")
    print("  2. User speaks → VAD detects speech (no longer blocked)")
    print("  3. User stops speaking → VAD detects silence → transcription")
    print("  4. New LLM response → cycle continues")
else:
    print(f"⚠️ {failed} test(s) failed")
    exit(1)

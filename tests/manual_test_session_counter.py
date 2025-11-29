#!/usr/bin/env python3
"""
Test Phase 4.3: Session Counter System
Validates that stale TTS events from previous sessions are properly discarded.
"""

import time
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dictator.service import DictatorService
import yaml


def setup_logging():
    """Configure test logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_session_counter_system():
    """
    Test Scenario: Rapid interruptions during TTS playback

    Expected Behavior:
    - Session counter increments on each recording start
    - Stale TTS events (from old sessions) are skipped with log message
    - Current session TTS plays normally
    """

    print("\n" + "="*80)
    print("Phase 4.3 Test: Session Counter System")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))

    # Check that session counter exists and starts at 0
    assert hasattr(service, '_session_counter'), "Missing _session_counter"
    assert hasattr(service, '_current_tts_session'), "Missing _current_tts_session"
    assert service._session_counter == 0, f"Expected _session_counter=0, got {service._session_counter}"

    print("[OK] Session counter initialized correctly")

    # Simulate recording session 1
    print("\n--- Session 1: Start Recording ---")
    service.start_recording()
    time.sleep(0.1)

    # Check counter incremented
    assert service._session_counter == 1, f"Expected _session_counter=1, got {service._session_counter}"
    print(f"[OK] Session counter incremented to {service._session_counter}")

    # Stop recording (this captures session number for TTS)
    service.stop_recording()
    time.sleep(0.1)

    assert service._current_tts_session == 1, f"Expected _current_tts_session=1, got {service._current_tts_session}"
    print(f"[OK] Current TTS session captured: {service._current_tts_session}")

    # Simulate user interrupting before TTS plays (start session 2)
    print("\n--- Session 2: Interrupt before TTS ---")
    service.start_recording()
    time.sleep(0.1)

    assert service._session_counter == 2, f"Expected _session_counter=2, got {service._session_counter}"
    print(f"[OK] Session counter incremented to {service._session_counter}")

    service.stop_recording()
    time.sleep(0.1)

    # Now simulate stale TTS event from session 1 arriving
    print("\n--- Simulating Stale TTS from Session 1 ---")
    print(f"Current session: {service._session_counter}")
    print(f"Current TTS session: {service._current_tts_session}")
    print("Attempting to speak text from old session...")

    # Manually set TTS session back to 1 to simulate stale event
    service._current_tts_session = 1

    # Try to speak - should be skipped
    service.speak_text("This is stale TTS from session 1")
    time.sleep(0.5)

    print("\n--- Check logs above for 'Skipping stale TTS from session 1' ---")

    # Simulate current session TTS (session 2)
    print("\n--- Simulating Current TTS from Session 2 ---")
    service._current_tts_session = 2

    print("Attempting to speak text from current session...")
    service.speak_text("This is current TTS from session 2")
    time.sleep(2.0)  # Wait for TTS to complete

    print("\n" + "="*80)
    print("[OK] Test completed! Check logs for:")
    print("  1. 'Skipping stale TTS from session 1 (current session: 2)'")
    print("  2. TTS from session 2 should have played")
    print("="*80)


def test_rapid_interruption():
    """
    Test Scenario: User interrupts multiple times rapidly

    Expected Behavior:
    - Each interrupt increments session counter
    - All TTS from previous sessions are skipped
    - Only the last session's TTS plays
    """

    print("\n" + "="*80)
    print("Rapid Interruption Test")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))

    # Simulate 5 rapid recording cycles
    for i in range(1, 6):
        print(f"\n--- Session {i} ---")
        service.start_recording()
        time.sleep(0.05)
        service.stop_recording()
        time.sleep(0.05)

        print(f"Session counter: {service._session_counter}")
        print(f"Current TTS session: {service._current_tts_session}")

    # Now simulate TTS events arriving from all sessions
    print("\n--- Simulating TTS events from all sessions ---")

    for session_num in range(1, 6):
        service._current_tts_session = session_num
        print(f"\nTrying TTS from session {session_num} (current={service._session_counter})...")
        service.speak_text(f"TTS from session {session_num}")
        time.sleep(0.5)

    print("\n" + "="*80)
    print("[OK] Test completed! Check logs:")
    print("  - Sessions 1-4 should show 'Skipping stale TTS'")
    print("  - Session 5 should have played")
    print("="*80)


if __name__ == "__main__":
    setup_logging()

    try:
        test_session_counter_system()
        print("\n" + "="*80)
        time.sleep(1)
        test_rapid_interruption()

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

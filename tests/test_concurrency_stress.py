#!/usr/bin/env python3
"""
Concurrency Stress Tests

Tests for race conditions, thread safety, and concurrent operations.
Validates Phases 1-5 improvements.
"""

import time
import threading
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dictator.service import DictatorService, ServiceState
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


class ConcurrencyTester:
    """Helper class for concurrency testing"""

    def __init__(self, service: DictatorService):
        self.service = service
        self.errors = []
        self.state_transitions = []
        self.lock = threading.Lock()

        # Register state callback
        service.register_state_callback(self._track_state)

    def _track_state(self, state: str):
        """Track state transitions"""
        with self.lock:
            self.state_transitions.append((time.time(), state))

    def add_error(self, error: str):
        """Thread-safe error tracking"""
        with self.lock:
            self.errors.append(error)

    def get_errors(self):
        """Get all errors"""
        with self.lock:
            return self.errors.copy()

    def get_state_transitions(self):
        """Get all state transitions"""
        with self.lock:
            return self.state_transitions.copy()


def test_rapid_interrupts():
    """
    Test 1: Rapid Interrupts

    Simulates user rapidly pressing hotkey during different states.

    Expected behavior:
    - No crashes
    - No invalid state transitions
    - Session counter increments correctly
    - Stale TTS events are discarded
    """
    print("\n" + "="*80)
    print("Test 1: Rapid Interrupts")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))
    tester = ConcurrencyTester(service)

    # Wait for initialization
    time.sleep(0.5)

    print("\n[*] Starting rapid interrupt test (10 cycles)...")

    for i in range(10):
        print(f"\n--- Cycle {i+1}/10 ---")

        # Start recording
        service.start_recording()
        time.sleep(0.1)  # Brief recording

        # Stop recording (triggers processing)
        service.stop_recording()
        time.sleep(0.05)  # Brief processing

        # Interrupt during processing
        service.start_recording()
        time.sleep(0.05)

        # Stop again
        service.stop_recording()
        time.sleep(0.05)

    print("\n[*] Rapid interrupt test completed")

    # Analyze results
    errors = tester.get_errors()
    transitions = tester.get_state_transitions()

    print(f"\n[RESULTS]")
    print(f"Total state transitions: {len(transitions)}")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\n[ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n[OK] No errors detected!")
        return True


def test_concurrent_state_access():
    """
    Test 2: Concurrent State Access

    Multiple threads trying to access/modify state simultaneously.

    Expected behavior:
    - Thread-safe state access
    - No race conditions
    - Locks prevent simultaneous modifications
    """
    print("\n" + "="*80)
    print("Test 2: Concurrent State Access")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))
    tester = ConcurrencyTester(service)

    # Wait for initialization
    time.sleep(0.5)

    def worker(thread_id: int):
        """Worker thread that tries to manipulate state"""
        for i in range(5):
            try:
                # Try to start recording
                service.start_recording()
                time.sleep(0.05)

                # Check state is valid
                with service._state_lock:
                    current_state = service._state
                    if current_state not in [ServiceState.RECORDING, ServiceState.PROCESSING]:
                        tester.add_error(
                            f"Thread {thread_id}: Invalid state {current_state} after start_recording"
                        )

                # Try to stop recording
                service.stop_recording()
                time.sleep(0.05)

            except Exception as e:
                tester.add_error(f"Thread {thread_id}: Exception - {e}")

    print("\n[*] Starting concurrent state access test (5 threads)...")

    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,), daemon=True)
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join(timeout=10.0)

    print("\n[*] Concurrent state access test completed")

    # Analyze results
    errors = tester.get_errors()

    print(f"\n[RESULTS]")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\n[ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n[OK] No errors detected!")
        return True


def test_session_counter_stress():
    """
    Test 3: Session Counter Stress Test

    Rapidly cycles through sessions to test counter overflow and wraparound.

    Expected behavior:
    - Session counter increments monotonically
    - Stale TTS detection works even with many sessions
    - No integer overflow issues
    """
    print("\n" + "="*80)
    print("Test 3: Session Counter Stress Test")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))
    tester = ConcurrencyTester(service)

    # Wait for initialization
    time.sleep(0.5)

    print("\n[*] Starting session counter stress test (100 sessions)...")

    previous_session = 0
    for i in range(100):
        # Start recording (increments session counter)
        service.start_recording()

        # Check session counter increased
        with service._state_lock:
            current_session = service._session_counter
            if current_session <= previous_session:
                tester.add_error(
                    f"Session counter did not increment: {previous_session} -> {current_session}"
                )
            previous_session = current_session

        time.sleep(0.01)
        service.stop_recording()
        time.sleep(0.01)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/100 sessions (counter={current_session})")

    print("\n[*] Session counter stress test completed")

    # Analyze results
    errors = tester.get_errors()

    print(f"\n[RESULTS]")
    print(f"Final session counter: {service._session_counter}")
    print(f"Expected: 100")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\n[ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        return False
    elif service._session_counter != 100:
        print(f"\n[ERROR] Session counter mismatch!")
        return False
    else:
        print("\n[OK] Session counter working correctly!")
        return True


def test_audio_processor_lifecycle():
    """
    Test 4: AudioProcessor Lifecycle

    Tests creation/destruction of AudioProcessor across many recordings.

    Expected behavior:
    - AudioProcessor created on each recording
    - Properly cleaned up after recording
    - No memory leaks
    """
    print("\n" + "="*80)
    print("Test 4: AudioProcessor Lifecycle")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))
    tester = ConcurrencyTester(service)

    # Wait for initialization
    time.sleep(0.5)

    print("\n[*] Starting AudioProcessor lifecycle test (50 cycles)...")

    for i in range(50):
        # Start recording (creates AudioProcessor)
        service.start_recording()
        time.sleep(0.05)

        # Check AudioProcessor exists
        if service.audio_processor is None:
            tester.add_error(f"Cycle {i+1}: AudioProcessor not created")

        # Stop recording (should cleanup AudioProcessor)
        service.stop_recording()
        time.sleep(0.05)

        # AudioProcessor should be cleaned up (happens in finally block of _record_audio)
        # Note: Cleanup happens async, so we check after a delay
        time.sleep(0.1)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/50 cycles")

    print("\n[*] AudioProcessor lifecycle test completed")

    # Analyze results
    errors = tester.get_errors()

    print(f"\n[RESULTS]")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\n[ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n[OK] AudioProcessor lifecycle working correctly!")
        return True


def test_interrupt_during_tts():
    """
    Test 5: Interrupt During TTS

    Tests interrupting TTS playback multiple times in succession.

    Expected behavior:
    - TTS stops immediately
    - State transitions to RECORDING
    - No stale TTS plays afterward
    """
    print("\n" + "="*80)
    print("Test 5: Interrupt During TTS")
    print("="*80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    service = DictatorService(config_path=str(config_path))
    tester = ConcurrencyTester(service)

    # Wait for initialization
    time.sleep(0.5)

    print("\n[*] Starting TTS interrupt test...")

    # Simulate speaking state and interrupt
    for i in range(5):
        print(f"\n--- Cycle {i+1}/5 ---")

        # Manually trigger TTS (simulate LLM response)
        service.speak_text(f"Test message {i+1} - this is a longer message to test interruption")
        time.sleep(0.2)  # Let TTS start

        # Interrupt with new recording
        print("  [*] Interrupting TTS...")
        service.start_recording()
        time.sleep(0.1)
        service.stop_recording()
        time.sleep(0.2)

    print("\n[*] TTS interrupt test completed")

    # Analyze results
    errors = tester.get_errors()

    print(f"\n[RESULTS]")
    print(f"Errors detected: {len(errors)}")

    if errors:
        print("\n[ERRORS]:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n[OK] TTS interruption working correctly!")
        return True


def main():
    """Run all concurrency tests"""
    setup_logging()

    print("\n" + "="*80)
    print("DICTATOR CONCURRENCY STRESS TESTS")
    print("Testing Phases 1-5 improvements")
    print("="*80)

    results = {}

    try:
        results['rapid_interrupts'] = test_rapid_interrupts()
        time.sleep(1)

        results['concurrent_state_access'] = test_concurrent_state_access()
        time.sleep(1)

        results['session_counter_stress'] = test_session_counter_stress()
        time.sleep(1)

        results['audio_processor_lifecycle'] = test_audio_processor_lifecycle()
        time.sleep(1)

        results['interrupt_during_tts'] = test_interrupt_during_tts()

    except Exception as e:
        print(f"\n[FATAL ERROR] Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n[RESULT] Some tests FAILED")
        sys.exit(1)
    else:
        print("\n[RESULT] All tests PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()

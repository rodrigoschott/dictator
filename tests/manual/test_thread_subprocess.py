#!/usr/bin/env python3
"""
Test subprocess.run via asyncio.to_thread in secondary event loop
"""

import asyncio
import subprocess
import threading


async def call_claude_in_secondary_loop():
    """Test calling Claude via subprocess.run in asyncio.to_thread"""

    def _run_claude():
        claude_cmd = r"C:\Users\rodri\AppData\Roaming\npm\claude.CMD"
        result = subprocess.run(
            [claude_cmd, '--print', '--output-format', 'text'],
            input=b"Say hello in 5 words or less",
            capture_output=True,
            timeout=10.0,
            creationflags=subprocess.CREATE_NO_WINDOW if subprocess.os.name == 'nt' else 0
        )
        return result.stdout.decode('utf-8').strip()

    print(f"[*] Calling Claude via asyncio.to_thread...")
    response = await asyncio.to_thread(_run_claude)
    print(f"[+] Response: {response}")
    return response


def run_in_secondary_thread():
    """Run asyncio loop in secondary thread (like voice_session)"""
    print("[*] Starting secondary event loop in thread...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(call_claude_in_secondary_loop())
        print(f"[+] Success! Got: {result}")
        return True
    except Exception as e:
        print(f"[!] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        loop.close()


if __name__ == "__main__":
    print("Testing subprocess in secondary asyncio loop (like voice_session)\n")

    thread = threading.Thread(target=run_in_secondary_thread)
    thread.start()
    thread.join()

    print("\n[+] Test complete!")

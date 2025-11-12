#!/usr/bin/env python3
"""
Test Claude CLI via subprocess to debug hanging issue
"""

import asyncio
import sys


async def test_claude_subprocess():
    """Test Claude CLI with exact same code as llm_caller.py"""

    claude_cmd = r"C:\Users\rodri\AppData\Roaming\npm\claude.CMD"

    test_prompt = """You are Dictator, a concise voice assistant.

User: Hello, how are you?
Assistant:"""

    print("[*] Creating subprocess...")

    try:
        process = await asyncio.create_subprocess_exec(
            claude_cmd,
            '--print',
            '--output-format', 'text',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        print(f"[*] Process created (PID: {process.pid})")
        print(f"[*] Sending prompt ({len(test_prompt)} chars)...")

        # Try with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=test_prompt.encode('utf-8')),
                timeout=10.0
            )

            print(f"[+] Got response!")
            print(f"[+] Return code: {process.returncode}")
            print(f"[+] Stdout length: {len(stdout)} bytes")
            print(f"[+] Stderr length: {len(stderr)} bytes")

            if process.returncode == 0:
                response = stdout.decode('utf-8', errors='replace').strip()
                print(f"\n[+] Response:\n{response}\n")
            else:
                error = stderr.decode('utf-8', errors='replace')
                print(f"\n[!] Error:\n{error}\n")

        except asyncio.TimeoutError:
            print("[!] TIMEOUT after 10 seconds")
            process.kill()
            await process.wait()
            return False

    except Exception as e:
        print(f"[!] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("Testing Claude CLI subprocess behavior\n")
    success = asyncio.run(test_claude_subprocess())
    sys.exit(0 if success else 1)

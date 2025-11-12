#!/usr/bin/env python3
"""
Test Claude Code CLI Integration

Quick test to verify Claude Code CLI is accessible.
"""

import subprocess
import shutil
import sys
from pathlib import Path


def find_claude_executable():
    """Find Claude Code CLI executable"""
    # Check PATH
    claude_path = shutil.which('claude')
    if claude_path:
        return claude_path

    # Check common paths
    import os
    common_paths = [
        Path(os.environ.get('APPDATA', '')) / 'npm' / 'claude.cmd',
        Path(os.environ.get('APPDATA', '')) / 'npm' / 'claude.exe',
        Path.home() / '.local' / 'bin' / 'claude.exe',
        Path.home() / '.local' / 'bin' / 'claude',
        Path('C:/Program Files/Claude/claude.exe'),
        Path('C:/Program Files (x86)/Claude/claude.exe'),
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


def test_claude_cli():
    """Test Claude Code CLI"""
    print("[*] Searching for Claude Code CLI...")

    claude_cmd = find_claude_executable()

    if not claude_cmd:
        print("[!] Claude Code CLI not found!")
        print("\n[i] Install Claude Code:")
        print("    https://docs.anthropic.com/en/docs/claude-code")
        print("\nOr ensure 'claude' is in your PATH")
        return False

    print(f"[+] Found Claude Code CLI: {claude_cmd}")

    # Test simple query
    print("\n[*] Testing Claude Code CLI with simple query...")
    try:
        result = subprocess.run(
            [claude_cmd, '--print'],
            input=b"Hello! Please respond with just 'Hello back!' and nothing else.",
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0:
            response = result.stdout.decode('utf-8', errors='replace').strip()
            print(f"[+] Claude responded: {response[:100]}...")
            print(f"\n[+] Claude Code CLI is working!")
            return True
        else:
            error = result.stderr.decode('utf-8', errors='replace')
            print(f"[!] Claude CLI error: {error}")
            return False

    except subprocess.TimeoutExpired:
        print("[!] Claude CLI timed out (30s)")
        return False
    except Exception as e:
        print(f"[!] Error testing Claude CLI: {e}")
        return False


if __name__ == "__main__":
    success = test_claude_cli()
    sys.exit(0 if success else 1)

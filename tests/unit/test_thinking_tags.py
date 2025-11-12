#!/usr/bin/env python3
"""
Test script to validate thinking tag removal
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dictator.voice.llm_caller import remove_thinking_tags


def test_thinking_tag_removal():
    """Test various thinking tag scenarios"""
    
    print("🧪 Testing thinking tag removal...\n")
    
    test_cases = [
        {
            "name": "Simple thinking block",
            "input": "<think>Let me analyze this...</think>The answer is 42.",
            "expected": "The answer is 42."
        },
        {
            "name": "Multiline thinking (Qwen style)",
            "input": """<think>
Okay, the user is talking about using a tanking model. I need to explain what happens when that's done. First, I should define what a tanking model is.
</think>

When using a thinking model, the AI exposes its reasoning process.""",
            "expected": "When using a thinking model, the AI exposes its reasoning process."
        },
        {
            "name": "Multiple thinking blocks",
            "input": "<think>Hmm...</think>First point. <think>Wait...</think>Second point.",
            "expected": "First point. Second point."
        },
        {
            "name": "Mixed case tags",
            "input": "<THINK>Internal thought</THINK>External response.",
            "expected": "External response."
        },
        {
            "name": "No thinking tags",
            "input": "Just a normal response without any thinking.",
            "expected": "Just a normal response without any thinking."
        },
        {
            "name": "Thinking in middle of sentence",
            "input": "Here's what I found <think>after analyzing...</think> in the database.",
            "expected": "Here's what I found  in the database."
        },
        {
            "name": "Real Qwen3 response",
            "input": """<think>
Okay, the user asked, "Você só consegue pensar em inglês?" which means "Can you only think in English?" I need to respond in Portuguese since the user is using Portuguese here.

First, I should clarify that I can process and respond in multiple languages, including Portuguese.
</think>

Não, eu consigo pensar e responder em vários idiomas, incluindo português!""",
            "expected": "Não, eu consigo pensar e responder em vários idiomas, incluindo português!"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        result = remove_thinking_tags(test['input'])
        
        if result == test['expected']:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            print(f"  Expected: {repr(test['expected'])}")
            print(f"  Got:      {repr(result)}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✨ All tests passed!")
        return True
    else:
        print(f"⚠️ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_thinking_tag_removal()
    sys.exit(0 if success else 1)

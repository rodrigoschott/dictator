#!/usr/bin/env python3
"""
Test auto-restart functionality in tray menu actions
"""

print("🧪 Testing auto-restart implementation...")
print()

# Read tray.py and check for restart_service() calls
with open("src/dictator/tray.py", "r", encoding="utf-8") as f:
    content = f.read()

tests = [
    {
        "name": "toggle_claude_mode calls restart_service",
        "search": "def toggle_claude_mode",
        "expected": "self.restart_service()"
    },
    {
        "name": "set_llm_provider calls restart_service conditionally",
        "search": "def set_llm_provider",
        "expected": "self.restart_service()"
    },
    {
        "name": "set_ollama_model calls restart_service conditionally",
        "search": "def set_ollama_model",
        "expected": "self.restart_service()"
    },
    {
        "name": "toggle_vad calls restart_service conditionally",
        "search": "def toggle_vad",
        "expected": "self.restart_service()"
    }
]

passed = 0
failed = 0

for test in tests:
    # Find the function
    start_idx = content.find(test["search"])
    if start_idx == -1:
        print(f"❌ FAIL: {test['name']} - function not found")
        failed += 1
        continue
    
    # Find next function (end of current function)
    next_func_idx = content.find("\n    def ", start_idx + 1)
    if next_func_idx == -1:
        next_func_idx = len(content)
    
    function_body = content[start_idx:next_func_idx]
    
    # Check if restart_service() is called
    if test["expected"] in function_body:
        print(f"✅ PASS: {test['name']}")
        passed += 1
    else:
        print(f"❌ FAIL: {test['name']}")
        failed += 1

print()
print(f"📊 Results: {passed} passed, {failed} failed")

if failed == 0:
    print("✨ All auto-restart tests passed!")
else:
    print(f"⚠️ {failed} test(s) failed")
    exit(1)

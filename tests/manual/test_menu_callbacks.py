#!/usr/bin/env python3
"""
Test script to validate menu callbacks don't pollute config
"""

import yaml  # type: ignore
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load config
config_path = REPO_ROOT / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("🔍 Checking config.yaml structure...")
print()

def check_value(path, value, depth=0):
    """Recursively check for non-serializable values"""
    indent = "  " * depth
    
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"{indent}✓ {path}.{k}: {type(v).__name__}")
            check_value(f"{path}.{k}", v, depth + 1)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            check_value(f"{path}[{i}]", item, depth + 1)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        print(f"{indent}❌ INVALID TYPE: {path} = {type(value)}")
        return False
    
    return True

# Check all top-level keys
all_valid = True
for key, value in config.items():
    print(f"✓ {key}: {type(value).__name__}")
    if not check_value(key, value, 1):
        all_valid = False

print()
if all_valid:
    print("✅ Config is clean - no non-serializable objects!")
else:
    print("❌ Config contains non-serializable objects!")
    
# Specifically check voice.llm.ollama.model
model = config.get('voice', {}).get('llm', {}).get('ollama', {}).get('model')
print()
print(f"🦙 Ollama model: {model} (type: {type(model).__name__})")

if isinstance(model, str):
    print("✅ Model is a string - correct!")
else:
    print(f"❌ Model is {type(model).__name__} - WRONG!")

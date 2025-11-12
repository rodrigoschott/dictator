#!/usr/bin/env python3
"""
Test script to verify Ollama model discovery
"""

import requests

def test_ollama_api():
    """Test fetching models from Ollama API"""
    try:
        base_url = "http://localhost:11434"
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        response.raise_for_status()
        
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        
        print("✅ Ollama API Response:")
        print(f"   Found {len(models)} models:")
        for model in sorted(models):
            print(f"   - {model}")
        
        return models
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama - is it running?")
        return None
    except Exception as e:
        print(f"❌ Error fetching Ollama models: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Testing Ollama model discovery...")
    print()
    models = test_ollama_api()
    
    if models:
        print()
        print("✨ Dynamic model discovery is working!")
    else:
        print()
        print("⚠️ Will fallback to default models")

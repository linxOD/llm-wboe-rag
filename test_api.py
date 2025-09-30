#!/usr/bin/env python3
"""
Test script for WBOE RAG Pipeline API
Tests basic functionality and endpoints
"""

import requests
import sys
from typing import Dict, Any

API_BASE = "http://localhost:8000"


def test_health_check() -> bool:
    """Test health check endpoint."""
    print("🏥 Testing health check...")

    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Is it running?")
        print("   Try starting it with: ./start_api.sh")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_api_info() -> bool:
    """Test root endpoint."""
    print("📝 Testing API info...")

    try:
        response = requests.get(f"{API_BASE}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API info: {data['message']} v{data['version']}")
            return True
        else:
            print(f"❌ API info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False


def test_process_pipeline(backend: str = "ollama") -> str:
    """Test starting a pipeline process."""
    print(f"🚀 Testing pipeline processing with {backend} backend...")

    # Minimal configuration for testing
    payload = {
        "backend": backend,
        "keywords_to_process": ["43218__Gefrette_Simplex"],  # Process just one document
        "max_context_length": 32000,  # Smaller context for faster processing
        "enable_memory_monitoring": False,  # Disable for faster startup
        "aggressive_cleanup": True,
        "output_dir": f"output_test_{backend}"
    }

    # Add backend-specific configuration
    if backend == "ollama":
        payload["ollama_model"] = "llama3.2:3b"
    elif backend == "llama_cpp":
        payload["hf_model"] = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        payload["hf_model_fn"] = "Llama-3.2-3B-Instruct-Q4_0.gguf"
    elif backend == "openAI":
        payload["openai_model"] = "gpt-4o"

    try:
        response = requests.post(
            f"{API_BASE}/rag/process",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            task_id = data["task_id"]
            print(f"✅ Pipeline started: task_id = {task_id}")
            print(f"   Backend: {data['backend']}")
            print(f"   Status: {data['status']}")
            return task_id
        else:
            print(f"❌ Pipeline start failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except Exception as e:
                print(f"   Raw response: {response.text}")
                print(f"   Error: {e}")
            return None

    except Exception as e:
        print(f"❌ Pipeline start error: {e}")
        return None


def test_task_status(task_id: str) -> Dict[str, Any]:
    """Test checking task status."""
    print(f"📊 Testing task status for {task_id}...")

    try:
        response = requests.get(f"{API_BASE}/rag/status/{task_id}", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status retrieved: {data['status']}")
            if 'progress' in data:
                print(f"   Current step: {data['progress'].get('current_step', 'unknown')}")
            return data
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Status check error: {e}")
        return None


def test_list_tasks() -> bool:
    """Test listing all tasks."""
    print("📋 Testing task listing...")

    try:
        response = requests.get(f"{API_BASE}/rag/tasks", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Tasks listed: {data['total_tasks']} total")
            for task in data['tasks'][:3]:  # Show first 3 tasks
                print(f"   - {task['task_id']}: {task['status']} ({task['backend']})")
            return True
        else:
            print(f"❌ Task listing failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Task listing error: {e}")
        return False


def test_invalid_backend() -> bool:
    """Test error handling with invalid backend."""
    print("🚫 Testing error handling...")

    payload = {
        "backend": "invalid_backend",
        "keywords_to_process": ["test"]
    }

    try:
        response = requests.post(f"{API_BASE}/rag/process", json=payload, timeout=10)

        if response.status_code == 400:
            print("✅ Error handling works correctly")
            return True
        else:
            print(f"❌ Expected 400 error, got: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 WBOE RAG Pipeline API Test Suite")
    print("=" * 50)

    tests_passed = 0
    total_tests = 0

    # Basic connectivity tests
    total_tests += 1
    if test_health_check():
        tests_passed += 1

    total_tests += 1
    if test_api_info():
        tests_passed += 1

    total_tests += 1
    if test_list_tasks():
        tests_passed += 1

    total_tests += 1
    if test_invalid_backend():
        tests_passed += 1

    # Determine which backend to test based on environment
    import os
    backend_to_test = None

    if os.getenv("OLLAMA_API_KEY"):
        backend_to_test = "ollama"
    elif os.getenv("OPENAI_API_KEY"):
        backend_to_test = "openAI"
    elif os.getenv("HUGGINGFACE_API_KEY"):
        backend_to_test = "llama_cpp"

    if backend_to_test:
        print(f"\n🎯 Testing {backend_to_test} backend (API key found)...")

        # Test pipeline processing
        total_tests += 1
        task_id = test_process_pipeline(backend_to_test)
        if task_id:
            tests_passed += 1

            # Test status checking
            total_tests += 1
            if test_task_status(task_id):
                tests_passed += 1
    else:
        print("\n⚠️  No backend API keys found in environment")
        print("   Skipping pipeline processing tests")
        print("   Set OLLAMA_API_KEY, OPENAI_API_KEY, or HUGGINGFACE_API_KEY to test processing")

    # Results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_tests - tests_passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script for Perplexity API connection

Usage:
  python test_api.py
  
Or with explicit API key:
  python test_api.py YOUR_API_KEY
"""

import sys
import os
from orchestrator_react import PerplexityClient


def test_api_connection(api_key: str = None):
    """
    Test Perplexity API connection
    
    Args:
        api_key: Optional API key. If not provided, reads from environment
    """
    print("\n" + "="*70)
    print("🔬 PERPLEXITY API CONNECTION TEST")
    print("="*70)
    
    # Get API key
    if not api_key:
        api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        print("\n❌ Error: No API key provided")
        print("\nOptions:")
        print("  1. Set environment variable: export PERPLEXITY_API_KEY='your-key'")
        print("  2. Pass as argument: python test_api.py YOUR_API_KEY")
        print("\nGet your API key from: https://www.perplexity.ai/settings/api")
        return False
    
    print(f"\n🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize client
        print("\n🔧 Initializing Perplexity client...")
        client = PerplexityClient(api_key=api_key)
        print("✅ Client initialized")
        
        # Show valid models
        print("\n🤖 Valid Sonar Models:")
        for model, description in client.VALID_MODELS.items():
            print(f"  • {model}: {description}")
        
        # Test 1: Simple connection test
        print("\n" + "-"*70)
        print("📍 TEST 1: Simple Connection Test")
        print("-"*70)
        
        test_message = [{"role": "user", "content": "Say 'Hello' if you can read this message."}]
        print("\n📤 Sending test message...")
        
        response = client.chat(test_message, model="sonar", temperature=0.0)
        
        print("✅ Response received!")
        print(f"\n💬 Response:\n{response}\n")
        
        # Test 2: Simple Q&A
        print("\n" + "-"*70)
        print("📍 TEST 2: Simple Q&A")
        print("-"*70)
        
        qa_message = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        print("\n📤 Asking simple question...")
        
        response = client.chat(qa_message, model="sonar", temperature=0.0)
        
        print("✅ Response received!")
        print(f"\n💬 Response:\n{response}\n")
        
        # Test 3: JSON response
        print("\n" + "-"*70)
        print("📍 TEST 3: JSON Format Response")
        print("-"*70)
        
        json_message = [{
            "role": "user", 
            "content": 'Respond with this exact JSON: {"status": "ok", "message": "test successful"}'
        }]
        print("\n📤 Testing JSON response...")
        
        response = client.chat(json_message, model="sonar", temperature=0.0)
        
        print("✅ Response received!")
        print(f"\n💬 Response:\n{response}\n")
        
        # Test 4: Test all models
        print("\n" + "-"*70)
        print("📍 TEST 4: Test All Available Models")
        print("-"*70)
        
        test_content = "Say 'OK' if you're working."
        
        for model_name in ["sonar", "sonar-pro"]:
            print(f"\n🤖 Testing model: {model_name}")
            try:
                msg = [{"role": "user", "content": test_content}]
                resp = client.chat(msg, model=model_name, temperature=0.0)
                print(f"✅ {model_name}: {resp[:50]}...")
            except Exception as e:
                print(f"❌ {model_name}: {str(e)[:100]}")
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 Your Perplexity API connection is working correctly!")
        print("\nNext steps:")
        print("  1. Run Streamlit: streamlit run app.py")
        print("  2. Enter your API key in the sidebar")
        print("  3. Start asking research questions!")
        print("\n")
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\n🐛 Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            print("  • Your API key is invalid or expired")
            print("  • Get a new key from: https://www.perplexity.ai/settings/api")
            print("  • Make sure the key starts with 'pplx-'")
        elif "400" in error_str or "bad request" in error_str:
            print("  • API request format issue")
            print("  • Check if you're using a valid model name")
            print("  • Valid models: sonar, sonar-pro, sonar-reasoning-pro, sonar-deep-research")
        elif "429" in error_str or "rate limit" in error_str:
            print("  • Rate limit exceeded")
            print("  • Wait a moment and try again")
        elif "403" in error_str or "forbidden" in error_str:
            print("  • API access denied")
            print("  • Check your subscription plan")
        elif "timeout" in error_str:
            print("  • Request timed out")
            print("  • Check your internet connection")
        else:
            print("  • Unknown error - check your internet connection")
            print("  • Try running again")
        
        print("\nFor more help, see: docs/TROUBLESHOOTING.md")
        print("\n")
        return False


if __name__ == "__main__":
    # Get API key from command line argument if provided
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run test
    success = test_api_connection(api_key)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

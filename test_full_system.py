#!/usr/bin/env python3
"""
Comprehensive System Test

Tests:
1. Business Analyst initialization
2. ChromaDB vector database connection
3. Document retrieval capability
4. Full ReAct loop with specialist agents
5. Multi-iteration orchestration

Usage:
  python test_full_system.py
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test 1: Can we import all required modules?"""
    print("\n" + "="*70)
    print("📍 TEST 1: Module Imports")
    print("="*70)
    
    try:
        print("\n⏳ Importing orchestrator_react...")
        from orchestrator_react import ReActOrchestrator, PerplexityClient
        print("✅ orchestrator_react imported")
        
        print("\n⏳ Importing Business Analyst...")
        from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
        print("✅ Business Analyst imported")
        
        return True, {"ReActOrchestrator": ReActOrchestrator, "BusinessAnalystGraphAgent": BusinessAnalystGraphAgent}
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_api_connection():
    """Test 2: Perplexity API connection"""
    print("\n" + "="*70)
    print("📍 TEST 2: Perplexity API Connection")
    print("="*70)
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ No PERPLEXITY_API_KEY found")
        print("   Set with: export PERPLEXITY_API_KEY='your-key'")
        return False, None
    
    try:
        from orchestrator_react import PerplexityClient
        print(f"\n🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
        
        client = PerplexityClient(api_key=api_key)
        print("✅ Client initialized")
        
        # Test connection
        print("\n⏳ Testing API call...")
        success = client.test_connection()
        
        if success:
            print("✅ API connection successful")
            return True, client
        else:
            print("❌ API connection failed")
            return False, None
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False, None


def test_business_analyst():
    """Test 3: Business Analyst initialization and ChromaDB"""
    print("\n" + "="*70)
    print("📍 TEST 3: Business Analyst & ChromaDB")
    print("="*70)
    
    try:
        from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
        
        # Check data directory
        data_path = "./data"
        db_path = "./storage/chroma_db"
        
        print(f"\n📂 Checking paths...")
        print(f"   Data path: {data_path}")
        print(f"   DB path: {db_path}")
        
        if Path(data_path).exists():
            pdfs = list(Path(data_path).rglob("*.pdf"))
            print(f"✅ Data directory exists ({len(pdfs)} PDFs found)")
            if pdfs:
                print(f"   PDFs: {[p.name for p in pdfs[:3]]}{'...' if len(pdfs) > 3 else ''}")
        else:
            print(f"⚠️ Data directory not found")
        
        if Path(db_path).exists():
            print(f"✅ ChromaDB directory exists")
        else:
            print(f"⚠️ ChromaDB directory not found (will be created)")
        
        # Initialize Business Analyst
        print(f"\n⏳ Initializing Business Analyst...")
        analyst = BusinessAnalystGraphAgent(
            data_path=data_path,
            db_path=db_path
        )
        print("✅ Business Analyst initialized")
        
        # Test analyze method
        print(f"\n⏳ Testing analyze method...")
        test_query = "What is this company's primary business?"
        result = analyst.analyze(test_query)
        print(f"✅ Analyze method works")
        print(f"   Result length: {len(result)} characters")
        print(f"   Preview: {result[:150]}...")
        
        return True, analyst
        
    except Exception as e:
        print(f"❌ Business Analyst test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_orchestrator_init():
    """Test 4: ReAct Orchestrator initialization"""
    print("\n" + "="*70)
    print("📍 TEST 4: ReAct Orchestrator Initialization")
    print("="*70)
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ No API key")
        return False, None
    
    try:
        from orchestrator_react import ReActOrchestrator
        from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
        
        # Initialize orchestrator
        print("\n⏳ Creating orchestrator...")
        orchestrator = ReActOrchestrator(
            perplexity_api_key=api_key,
            max_iterations=5
        )
        print("✅ Orchestrator created")
        
        # Register Business Analyst
        print("\n⏳ Registering Business Analyst...")
        analyst = BusinessAnalystGraphAgent(
            data_path="./data",
            db_path="./storage/chroma_db"
        )
        orchestrator.register_specialist("business_analyst", analyst)
        print("✅ Business Analyst registered")
        
        # Check registration
        print(f"\n📋 Registered agents: {list(orchestrator.specialist_agents.keys())}")
        
        return True, orchestrator
        
    except Exception as e:
        print(f"❌ Orchestrator init failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_full_react_loop(orchestrator):
    """Test 5: Full ReAct loop with specialist agents"""
    print("\n" + "="*70)
    print("📍 TEST 5: Full ReAct Loop")
    print("="*70)
    
    if not orchestrator:
        print("❌ No orchestrator available")
        return False
    
    try:
        # Test query that should trigger Business Analyst
        test_query = "Analyze Apple Inc's key competitive risks and business model based on their 10-K filing"
        
        print(f"\n⏳ Running query: {test_query}")
        print("\nThis should:")
        print("  1. Call Business Analyst agent")
        print("  2. Retrieve from ChromaDB")
        print("  3. Run multiple iterations")
        print("  4. Synthesize comprehensive report")
        
        print("\n" + "-"*70)
        print("STARTING REACT LOOP")
        print("-"*70)
        
        # Execute research
        report = orchestrator.research(test_query)
        
        # Analyze results
        print("\n" + "-"*70)
        print("ANALYZING RESULTS")
        print("-"*70)
        
        trace = orchestrator.trace
        num_iterations = len(trace.thoughts)
        specialists_called = trace.get_specialist_calls()
        
        print(f"\n📊 Metrics:")
        print(f"   Iterations: {num_iterations}/{orchestrator.max_iterations}")
        print(f"   Specialists called: {', '.join(specialists_called) if specialists_called else 'None'}")
        print(f"   Observations: {len(trace.observations)}")
        print(f"   Report length: {len(report)} characters")
        
        # Check success criteria
        print(f"\n✅ Success Criteria:")
        
        checks = [
            (num_iterations >= 2, f"Multiple iterations: {num_iterations} >= 2"),
            ('business_analyst' in specialists_called, "Business Analyst was called"),
            (len(report) > 500, f"Comprehensive report: {len(report)} > 500 chars"),
            (len(trace.observations) > 0, f"Observations recorded: {len(trace.observations)} > 0")
        ]
        
        all_passed = True
        for check, description in checks:
            status = "✅" if check else "❌"
            print(f"   {status} {description}")
            if not check:
                all_passed = False
        
        # Show report preview
        print(f"\n📄 Report Preview:")
        print("-" * 70)
        print(report[:500] + "..." if len(report) > 500 else report)
        print("-" * 70)
        
        if all_passed:
            print("\n✅ Full ReAct loop test PASSED")
            return True
        else:
            print("\n⚠️ Some checks failed - see details above")
            return False
            
    except Exception as e:
        print(f"\n❌ ReAct loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "="*70)
    print("📦 COMPREHENSIVE SYSTEM TEST")
    print("="*70)
    print("\nThis will test:")
    print("  1. Module imports")
    print("  2. Perplexity API connection")
    print("  3. Business Analyst & ChromaDB")
    print("  4. ReAct Orchestrator initialization")
    print("  5. Full ReAct loop with specialists")
    
    results = {}
    
    # Test 1: Imports
    success, modules = test_imports()
    results['imports'] = success
    if not success:
        print("\n❌ Cannot continue without imports")
        return results
    
    # Test 2: API
    success, client = test_api_connection()
    results['api'] = success
    if not success:
        print("\n⚠️ API connection failed - ReAct loop will not work")
        print("   Set PERPLEXITY_API_KEY and try again")
    
    # Test 3: Business Analyst
    success, analyst = test_business_analyst()
    results['business_analyst'] = success
    if not success:
        print("\n⚠️ Business Analyst failed")
        print("   Check: Ollama running, models downloaded, paths correct")
    
    # Test 4: Orchestrator
    if results['api']:
        success, orchestrator = test_orchestrator_init()
        results['orchestrator'] = success
    else:
        results['orchestrator'] = False
        orchestrator = None
    
    # Test 5: Full loop (only if previous tests passed)
    if results['api'] and results['orchestrator']:
        success = test_full_react_loop(orchestrator)
        results['full_loop'] = success
    else:
        print("\n⏭️ Skipping full loop test (prerequisites failed)")
        results['full_loop'] = False
    
    # Final summary
    print("\n" + "="*70)
    print("📈 TEST SUMMARY")
    print("="*70)
    
    test_names = [
        ('imports', 'Module Imports'),
        ('api', 'Perplexity API'),
        ('business_analyst', 'Business Analyst & ChromaDB'),
        ('orchestrator', 'ReAct Orchestrator'),
        ('full_loop', 'Full ReAct Loop')
    ]
    
    for key, name in test_names:
        status = "✅ PASS" if results.get(key) else "❌ FAIL"
        print(f"   {status}  {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\n✅ Your system is fully configured and working!")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Enter your API key")
        print("  3. Ask complex research questions")
        print("  4. Watch the 5-iteration ReAct loop in action!")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("="*70)
        print("\nTroubleshooting:")
        
        if not results.get('api'):
            print("\n🔑 API Issues:")
            print("   export PERPLEXITY_API_KEY='your-key'")
            print("   Get key: https://www.perplexity.ai/settings/api")
        
        if not results.get('business_analyst'):
            print("\n🤖 Business Analyst Issues:")
            print("   1. Start Ollama: ollama serve")
            print("   2. Pull models: ollama pull qwen2.5:7b")
            print("   3. Pull embeddings: ollama pull nomic-embed-text")
            print("   4. Check paths: ./data/ and ./storage/chroma_db/")
        
        print("\nFor more help: docs/TROUBLESHOOTING.md")
    
    print("\n")
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)

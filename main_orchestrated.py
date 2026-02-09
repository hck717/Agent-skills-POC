#!/usr/bin/env python3
"""
ReAct-Based Multi-Agent Equity Research System

This script uses the ReAct (Reasoning + Acting) framework for iterative,
intelligent orchestration of specialist agents.

ReAct Loop: Think → Act → Observe → Repeat
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator_react import ReActOrchestrator
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent


def check_environment():
    """Verify all required environment variables and dependencies"""
    issues = []
    
    # Check API keys
    if not os.getenv("PERPLEXITY_API_KEY"):
        issues.append("❌ PERPLEXITY_API_KEY not set (required for ReAct orchestrator)")
    else:
        print("✅ PERPLEXITY_API_KEY found")
    
    if not os.getenv("EODHD_API_KEY"):
        print("⚠️  EODHD_API_KEY not set (optional - for market data features)")
    else:
        print("✅ EODHD_API_KEY found")
    
    # Check data directory
    if not os.path.exists("./data"):
        print("⚠️  ./data directory not found - business analyst may have limited functionality")
    else:
        print("✅ ./data directory found")
    
    # Check if ChromaDB exists
    if not os.path.exists("./storage/chroma_db"):
        print("⚠️  ./storage/chroma_db not found - you may need to ingest data first")
    else:
        print("✅ Vector database found")
    
    if issues:
        print("\n" + "="*70)
        print("ENVIRONMENT ISSUES:")
        for issue in issues:
            print(issue)
        print("\nPlease fix these issues before running.")
        print("="*70)
        return False
    
    return True


def main():
    print("\n" + "="*70)
    print("🔁 REACT-BASED MULTI-AGENT EQUITY RESEARCH SYSTEM")
    print("="*70)
    print("\n🧠 Using ReAct Framework: Thought → Action → Observation → Repeat")
    print("\nInitializing system...\n")
    
    # Check environment
    if not check_environment():
        print("\nSetup Instructions:")
        print("  1. export PERPLEXITY_API_KEY='your-key'")
        print("  2. export EODHD_API_KEY='your-key' (optional)")
        print("  3. Ensure Ollama is running: ollama serve")
        print("  4. Pull models: ollama pull qwen2.5:7b")
        return
    
    print("\n" + "-"*70)
    print("INITIALIZING REACT ORCHESTRATOR...")
    print("-"*70)
    
    # Initialize ReAct orchestrator
    try:
        orchestrator = ReActOrchestrator(max_iterations=5)
        print("✅ ReAct orchestrator initialized (max 5 iterations)")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return
    
    # Initialize and register Business Analyst
    try:
        print("\n📡 Initializing Business Analyst (Local RAG + Ollama)...")
        business_analyst = BusinessAnalystGraphAgent(
            data_path="./data",
            db_path="./storage/chroma_db"
        )
        orchestrator.register_specialist("business_analyst", business_analyst)
    except Exception as e:
        print(f"⚠️  Business Analyst initialization warning: {e}")
        print("   System will continue with placeholder agent")
    
    # Display agent status
    print("\n" + "-"*70)
    print("SPECIALIST AGENT STATUS:")
    print("-"*70)
    
    from orchestrator_react import ReActOrchestrator as RO
    for agent_name, info in RO.SPECIALIST_AGENTS.items():
        status = "✅ ACTIVE" if agent_name in orchestrator.specialist_agents else "⏳ PLANNED"
        print(f"{status:12} {agent_name:20} - {info['description'][:45]}...")
    
    # Interactive research loop
    print("\n" + "="*70)
    print("🚀 SYSTEM READY - ReAct Framework Active")
    print("="*70)
    
    print("\n🔮 ReAct Framework Features:")
    print("  • Iterative reasoning - orchestrator thinks before each action")
    print("  • Dynamic adaptation - adjusts strategy based on observations")
    print("  • Self-correction - can call additional agents if needed")
    print("  • Early stopping - finishes when sufficient info gathered")
    
    print("\n💬 Example queries:")
    print("  - What are Apple's key competitive risks and profit margins?")
    print("  - Compare Microsoft and Google's competitive positioning")
    print("  - Analyze Tesla's market position and growth trajectory")
    
    print("\nCommands:")
    print("  'ingest'    - Process new documents in ./data folder")
    print("  'trace'     - Show detailed ReAct trace for last query")
    print("  'quit'      - Exit the system")
    
    last_orchestrator = None
    
    while True:
        print("\n" + "="*70)
        user_input = input("\n💬 Your question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the ReAct Equity Research Orchestrator!")
            break
        
        if user_input.lower() == 'trace':
            if last_orchestrator:
                print("\n" + "="*70)
                print("🔁 REACT TRACE FROM LAST QUERY")
                print("="*70)
                print(last_orchestrator.get_trace_summary())
            else:
                print("⚠️  No previous query to show trace for")
            continue
        
        if user_input.lower() == 'ingest':
            if 'business_analyst' in orchestrator.specialist_agents:
                print("\n📂 Starting data ingestion...")
                try:
                    orchestrator.specialist_agents['business_analyst'].ingest_data()
                    print("✅ Data ingestion complete")
                except Exception as e:
                    print(f"❌ Ingestion error: {e}")
            else:
                print("⚠️  Business Analyst not initialized - cannot ingest data")
            continue
        
        # Execute ReAct research workflow
        try:
            print("\n🔄 Starting ReAct loop...")
            final_report = orchestrator.research(user_input)
            last_orchestrator = orchestrator
            
            print("\n" + "="*70)
            print("📄 FINAL RESEARCH REPORT")
            print("="*70)
            print(f"\n{final_report}\n")
            
            # Show iteration count
            num_iterations = len(orchestrator.trace.thoughts)
            print(f"\n📊 Completed in {num_iterations} ReAct iteration(s)")
            
            # Option to see trace
            show_trace = input("\n🔍 Show detailed ReAct trace? (y/n): ").lower()
            if show_trace == 'y':
                print("\n" + "="*70)
                print("🔁 REACT TRACE")
                print("="*70)
                print(orchestrator.get_trace_summary())
            
            # Save report option
            save = input("\n💾 Save this report? (y/n): ").lower()
            if save == 'y':
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reports/report_react_{timestamp}.md"
                
                os.makedirs("reports", exist_ok=True)
                with open(filename, 'w') as f:
                    f.write(f"# Equity Research Report (ReAct Framework)\n\n")
                    f.write(f"**Query**: {user_input}\n\n")
                    f.write(f"**Generated**: {timestamp}\n")
                    f.write(f"**ReAct Iterations**: {num_iterations}\n\n")
                    f.write("---\n\n")
                    f.write(final_report)
                    f.write("\n\n---\n\n## ReAct Trace\n\n")
                    f.write(orchestrator.get_trace_summary())
                
                print(f"✅ Report with ReAct trace saved to {filename}")
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Interrupted by user")
            continue
        except Exception as e:
            print(f"\n❌ Error during research: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nPlease try rephrasing your question or check system logs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

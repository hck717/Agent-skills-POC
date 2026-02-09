#!/usr/bin/env python3
"""
Orchestrated Multi-Agent Equity Research System

This script demonstrates the full multi-agent orchestration with the 
existing BusinessAnalystGraphAgent integrated as one of the specialists.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator import EquityResearchOrchestrator
from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent


def check_environment():
    """Verify all required environment variables and dependencies"""
    issues = []
    
    # Check API keys
    if not os.getenv("PERPLEXITY_API_KEY"):
        issues.append("❌ PERPLEXITY_API_KEY not set (required for Planner & Synthesis agents)")
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
    print("🎓 MULTI-AGENT EQUITY RESEARCH ORCHESTRATOR")
    print("="*70)
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
    print("INITIALIZING AGENTS...")
    print("-"*70)
    
    # Initialize orchestrator
    try:
        orchestrator = EquityResearchOrchestrator()
        print("✅ Orchestrator initialized")
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
    
    from orchestrator import PlannerAgent
    for agent_name, info in PlannerAgent.SPECIALIST_AGENTS.items():
        status = "✅ ACTIVE" if agent_name in orchestrator.specialist_agents else "⏳ PLANNED"
        print(f"{status:12} {agent_name:20} - {info['description'][:45]}...")
    
    # Interactive research loop
    print("\n" + "="*70)
    print("🚀 SYSTEM READY - Enter your equity research questions")
    print("="*70)
    
    print("\nExample queries:")
    print("  - What are Apple's key competitive risks in 2024?")
    print("  - Compare Microsoft and Google's profit margins")
    print("  - Analyze Tesla's market position in the EV industry")
    print("  - What are NVIDIA's main revenue drivers?")
    
    print("\nCommands:")
    print("  'ingest' - Process new documents in ./data folder")
    print("  'quit'   - Exit the system")
    
    while True:
        print("\n" + "="*70)
        user_input = input("\n💬 Your question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the Equity Research Orchestrator!")
            break
        
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
        
        # Execute research workflow
        try:
            final_report = orchestrator.research(user_input)
            
            print("\n" + "="*70)
            print("📄 FINAL RESEARCH REPORT")
            print("="*70)
            print(f"\n{final_report}\n")
            
            # Save report option
            save = input("\n💾 Save this report? (y/n): ").lower()
            if save == 'y':
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reports/report_{timestamp}.md"
                
                os.makedirs("reports", exist_ok=True)
                with open(filename, 'w') as f:
                    f.write(f"# Equity Research Report\n\n")
                    f.write(f"**Query**: {user_input}\n\n")
                    f.write(f"**Generated**: {timestamp}\n\n")
                    f.write("---\n\n")
                    f.write(final_report)
                
                print(f"✅ Report saved to {filename}")
        
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

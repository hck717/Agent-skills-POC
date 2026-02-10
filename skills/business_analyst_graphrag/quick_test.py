#!/usr/bin/env python3
"""Quick end-to-end test for GraphRAG v28.0"""

from graph_agent_graphrag_v28 import UltimateGraphRAGBusinessAnalyst
import os

print("🚀 Quick Test - Ultimate GraphRAG v28.0")
print("=" * 80)

# Initialize
print("\n📦 Step 1: Initialization")
agent = UltimateGraphRAGBusinessAnalyst(
    data_path="../../data",
    db_path="../../storage/chroma_db",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)
print("✅ Agent initialized")

# Check if data exists
print("\n📂 Step 2: Data Check")
if not os.path.exists("../../data/AAPL"):
    print("❌ No data found!")
    print("   Creating data directory...")
    os.makedirs("../../data/AAPL", exist_ok=True)
    print("   📁 Please add PDF files to ./data/AAPL/")
    print("   Then run: agent.ingest_data()")
    exit(1)

aapl_files = [f for f in os.listdir("../../data/AAPL") if f.endswith('.pdf')]
if not aapl_files:
    print("❌ No PDF files found in ./data/AAPL/")
    print("   📁 Please add PDF files and run: agent.ingest_data()")
    exit(1)

print(f"✅ Found {len(aapl_files)} PDF files")

# Check vector store
print("\n📊 Step 3: Vector Store Check")
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="../../storage/chroma_db/AAPL",
        embedding_function=embeddings,
        collection_name="AAPL"
    )
    chunk_count = vectorstore._collection.count()
    
    if chunk_count == 0:
        print("⚠️  Vector store is empty")
        response = input("Run ingestion now? (y/n): ")
        if response.lower() == 'y':
            print("\n📥 Running ingestion...")
            agent.ingest_data()
            print("✅ Ingestion complete")
        else:
            print("❌ Cannot proceed without data")
            exit(1)
    else:
        print(f"✅ Vector store ready: {chunk_count} chunks")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Running ingestion...")
    agent.ingest_data()

# Test query
print("\n🔍 Step 4: Test Query")
print("Query: 'What are Apple's main revenue sources?'")

print("\n💡 Tip: If confidence scores are low, try:")
print("   agent.confidence_threshold = 0.55")
print("   agent.entity_confidence_threshold = 0.3")

try:
    result = agent.analyze("What are Apple's main revenue sources?")
    
    print("\n📊 RESULT:")
    print("=" * 80)
    if len(result) > 500:
        print(result[:500] + "...")
        print(f"\n(Showing first 500 of {len(result)} chars)")
    else:
        print(result)
    print("=" * 80)
    
    print("\n✅ Test complete!")
    print("\n📝 Next steps:")
    print("   - Run full test suite: python test_graphrag_v28.py")
    print("   - Run diagnostics: python diagnose_graphrag.py")
    print("   - Try more queries with agent.analyze('your query')")
    
except Exception as e:
    print(f"\n❌ Error during query: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Check logs above for specific errors")
    print("   2. Try lowering thresholds:")
    print("      agent.confidence_threshold = 0.55")
    print("      agent.entity_confidence_threshold = 0.3")
    print("   3. Run diagnostics: python diagnose_graphrag.py")

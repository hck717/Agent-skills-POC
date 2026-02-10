#!/usr/bin/env python3
"""Diagnostic script for GraphRAG v28 issues"""

from graph_agent_graphrag_v28 import UltimateGraphRAGBusinessAnalyst
from neo4j import GraphDatabase
import os

print("🔍 GraphRAG v28 Diagnostics")
print("=" * 80)

# Initialize agent
print("\n📦 Initializing agent...")
agent = UltimateGraphRAGBusinessAnalyst(
    data_path="../../data",
    db_path="../../storage/chroma_db",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)
print("✅ Agent initialized\n")

# 1. Check data files
print("=" * 80)
print("📂 DATA FILES")
print("=" * 80)
data_path = "../../data/AAPL"
if os.path.exists(data_path):
    aapl_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    print(f"Found {len(aapl_files)} PDF files in AAPL folder:")
    total_size = 0
    for f in aapl_files:
        size = os.path.getsize(f"{data_path}/{f}") / 1024
        total_size += size
        print(f"  📄 {f}: {size:.1f} KB")
    print(f"\n📊 Total size: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
else:
    print(f"❌ Data path not found: {data_path}")
    print("   Please create ./data/AAPL/ and add PDF files")

# 2. Check vector store
print("\n" + "=" * 80)
print("📊 VECTOR STORE")
print("=" * 80)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
try:
    vectorstore = Chroma(
        persist_directory="../../storage/chroma_db/AAPL",
        embedding_function=embeddings,
        collection_name="AAPL"
    )
    collection = vectorstore._collection
    chunk_count = collection.count()
    print(f"Total chunks indexed: {chunk_count}")
    
    if chunk_count > 0:
        # Sample search
        print("\n🔍 Testing similarity search for 'revenue'...")
        results = vectorstore.similarity_search("revenue", k=3)
        print(f"Retrieved {len(results)} documents")
        
        for i, doc in enumerate(results[:2]):
            print(f"\n📄 Document {i+1}:")
            print(f"   Length: {len(doc.page_content)} chars")
            print(f"   Preview: {doc.page_content[:150]}...")
            print(f"   Metadata: {doc.metadata}")
    else:
        print("⚠️  Vector store is empty - run agent.ingest_data() first")
except Exception as e:
    print(f"❌ Error accessing vector store: {e}")

# 3. Check Neo4j
print("\n" + "=" * 80)
print("🕸️  NEO4J KNOWLEDGE GRAPH")
print("=" * 80)
try:
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    with driver.session() as session:
        # Count nodes
        node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
        print(f"Total nodes: {node_count}")
        
        # Count relationships
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
        print(f"Total relationships: {rel_count}")
        
        if node_count > 0:
            # Node types
            print("\n📊 Node types:")
            result = session.run("""
                MATCH (n) 
                RETURN labels(n) as labels, count(*) as count 
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['labels']}: {record['count']}")
            
            # Sample nodes
            print("\n📝 Sample nodes:")
            result = session.run("MATCH (n) RETURN n LIMIT 5")
            for i, record in enumerate(result):
                node = record['n']
                labels = list(node.labels)
                props = dict(node.items())
                print(f"   {i+1}. {labels}: {props}")
        else:
            print("\n⚠️  Graph is empty - possible causes:")
            print("   1. Entity validation threshold too high (current: 0.6)")
            print("   2. No entities extracted from documents")
            print("   3. Ingestion not completed")
    driver.close()
except Exception as e:
    print(f"❌ Error connecting to Neo4j: {e}")
    print("   Make sure Neo4j is running: docker ps | grep neo4j")

# 4. Configuration check
print("\n" + "=" * 80)
print("⚙️  CONFIGURATION")
print("=" * 80)
print(f"Confidence threshold: {agent.confidence_threshold}")
print(f"Entity confidence threshold: {agent.entity_confidence_threshold}")
print(f"Max correction attempts: {agent.max_correction_attempts}")
print(f"Neo4j enabled: {agent.neo4j_enabled}")

# 5. Recommendations
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

if chunk_count == 0:
    print("❌ Vector store is empty")
    print("   → Run: agent.ingest_data()")
elif chunk_count < 50:
    print("⚠️  Low chunk count - may need more documents")
    print(f"   Current: {chunk_count} chunks")
    print("   → Add more PDF files to ./data/AAPL/")
else:
    print(f"✅ Good chunk count: {chunk_count}")

if node_count == 0:
    print("\n❌ Knowledge graph is empty")
    print("   → Try lowering entity_confidence_threshold:")
    print("      agent.entity_confidence_threshold = 0.3")
    print("   → Then re-run: agent.ingest_data()")
elif node_count < 10:
    print("\n⚠️  Few entities in graph")
    print(f"   Current: {node_count} nodes")
    print("   → Consider lowering entity_confidence_threshold to 0.4")
else:
    print(f"\n✅ Good entity count: {node_count}")

print("\n" + "=" * 80)
print("✅ Diagnostics complete!")
print("=" * 80)
print("\nNext steps:")
print("1. If issues persist, try: agent.confidence_threshold = 0.55")
print("2. Test with simple query: agent.analyze('What is Apple?')")
print("3. Check logs for specific error messages")

#!/usr/bin/env python3
"""
Seed Neo4j with CPU-only embeddings (no GPU)
Fixes M3 Mac GPU memory issues
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Fallback to CPU if GPU fails

import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Force CPU usage (avoid M3 GPU memory issues)
device = 'cpu'

print("\n" + "="*80)
print("Seeding Neo4j with CPU-only embeddings (M3 Mac compatible)")
print("="*80 + "\n")

# Connect to Neo4j
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")

print(f"Connecting to {neo4j_uri}...")
try:
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    with driver.session() as session:
        session.run("RETURN 1")
    print("✅ Connected to Neo4j\n")
except Exception as e:
    print(f"❌ Failed to connect: {e}\n")
    exit(1)

# Initialize embedder on CPU
print("Loading embedding model (CPU mode)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"✅ Embedder loaded on {device}\n")

# Clear existing data
print("Clearing existing data...")
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
print("✅ Database cleared\n")

# Minimal test data (fewer chunks to avoid memory issues)
print("Creating MSFT test data...\n")

chunks = [
    "Microsoft cloud revenue grew 30% year-over-year driven by Azure and Office 365.",
    "Azure infrastructure services represent 40% of total cloud revenue.",
    "Key risk: Competition from AWS which holds 32% market share vs Microsoft 23%."
]

with driver.session() as session:
    # Create Company
    session.run(
        "CREATE (c:Company {ticker: 'MSFT', name: 'Microsoft Corporation'})"
    )
    print("✅ Created MSFT company node")
    
    # Create Chunks with embeddings
    for i, text in enumerate(chunks):
        print(f"  Creating chunk {i+1}/3...")
        
        # Generate embedding on CPU
        embedding = embedder.encode(text, device=device, show_progress_bar=False)
        embedding_list = embedding.tolist()
        
        # Verify embedding is not all zeros
        if sum(embedding_list) == 0:
            print(f"    ⚠️  Warning: Zero embedding for chunk {i+1}")
        else:
            print(f"    ✅ Embedding OK (sum: {sum(embedding_list):.2f})")
        
        session.run("""
            MATCH (c:Company {ticker: 'MSFT'})
            CREATE (chunk:Chunk {
                text: $text,
                embedding: $embedding,
                ticker: 'MSFT',
                chunk_id: $chunk_id
            })
            CREATE (c)-[:HAS_CHUNK]->(chunk)
        """, text=text, embedding=embedding_list, chunk_id=f"MSFT_CHUNK_{i}")
    
    # Create Strategy
    session.run("""
        MATCH (c:Company {ticker: 'MSFT'})
        CREATE (s:Strategy {
            title: 'AI-First Transformation',
            description: 'Integrate GPT-4 and Copilot across all products'
        })
        CREATE (c)-[:HAS_STRATEGY]->(s)
    """)
    print("✅ Created strategy node")
    
    # Create Risk
    session.run("""
        MATCH (c:Company {ticker: 'MSFT'})
        CREATE (r:Risk {
            title: 'Cloud Competition',
            description: 'AWS holds 32% market share vs Microsoft 23%'
        })
        CREATE (c)-[:FACES_RISK]->(r)
    """)
    print("✅ Created risk node")

print("\nCreating vector index...")
with driver.session() as session:
    try:
        session.run("""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (n:Chunk) ON (n.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 384,
              `vector.similarity_function`: 'cosine'
            }}
        """)
        print("✅ Vector index created\n")
    except Exception as e:
        print(f"⚠️  Vector index warning: {e}\n")

# Verify
print("="*80)
print("Verifying Data")
print("="*80 + "\n")

with driver.session() as session:
    result = session.run("MATCH (n:Company) RETURN count(n) as count")
    print(f"✅ Companies: {result.single()['count']}")
    
    result = session.run("MATCH (n:Chunk) RETURN count(n) as count")
    print(f"✅ Chunks: {result.single()['count']}")
    
    result = session.run("MATCH (n:Strategy) RETURN count(n) as count")
    print(f"✅ Strategies: {result.single()['count']}")
    
    result = session.run("MATCH (n:Risk) RETURN count(n) as count")
    print(f"✅ Risks: {result.single()['count']}")
    
    # Verify embeddings are not zero
    result = session.run("""
        MATCH (n:Chunk)
        RETURN n.text as text, n.embedding[0] as first_val
        LIMIT 1
    """)
    record = result.single()
    if record:
        print(f"\n✅ Sample embedding check:")
        print(f"  Text: {record['text'][:60]}...")
        print(f"  First embedding value: {record['first_val']:.6f}")
        if record['first_val'] == 0:
            print(f"  ⚠️  WARNING: Embeddings are all zeros!")

driver.close()

print("\n" + "="*80)
print("✅ Seeding Complete (CPU mode)!")
print("="*80 + "\n")
print("🚀 Ready to test Business Analyst!\n")

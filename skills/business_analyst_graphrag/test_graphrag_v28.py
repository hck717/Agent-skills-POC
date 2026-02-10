#!/usr/bin/env python3
"""Comprehensive test suite for Ultimate GraphRAG v28.0"""

from graph_agent_graphrag_v28 import UltimateGraphRAGBusinessAnalyst
from neo4j import GraphDatabase
import time
import os

print("🧪 Ultimate GraphRAG v28.0 - Test Suite")
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

# Test 1: Simple Query
print("\n" + "=" * 80)
print("TEST 1: Simple Query (Single-Entity)")
print("=" * 80)
query = "What is Apple's revenue?"
print(f"Query: {query}")
start = time.time()
result = agent.analyze(query)
elapsed = time.time() - start
print(f"\n⏱️  Time: {elapsed:.2f}s")
print(f"📝 Result length: {len(result)} chars")
print(f"📄 Preview: {result[:300]}...")

# Test 2: Medium Complexity
print("\n" + "=" * 80)
print("TEST 2: Medium Complexity (Section-Target)")
print("=" * 80)
query = "What are Apple's key risks?"
print(f"Query: {query}")
start = time.time()
result = agent.analyze(query)
elapsed = time.time() - start
print(f"\n⏱️  Time: {elapsed:.2f}s")
print(f"📝 Result length: {len(result)} chars")
print(f"📄 Preview: {result[:300]}...")

# Test 3: Abstract Query (HyDE)
print("\n" + "=" * 80)
print("TEST 3: Abstract Query (HyDE Strategy)")
print("=" * 80)
query = "Why is Apple's margin declining?"
print(f"Query: {query}")
start = time.time()
result = agent.analyze(query)
elapsed = time.time() - start
print(f"\n⏱️  Time: {elapsed:.2f}s")
print(f"📝 Result length: {len(result)} chars")
print(f"📄 Preview: {result[:300]}...")

# Test 4: Multi-Hop Query
print("\n" + "=" * 80)
print("TEST 4: Multi-Hop Query (Graph Reasoning)")
print("=" * 80)
query = "If TSMC production drops 30%, which Apple products are most at risk?"
print(f"Query: {query}")
start = time.time()
result = agent.analyze(query)
elapsed = time.time() - start
print(f"\n⏱️  Time: {elapsed:.2f}s")
print(f"📝 Result length: {len(result)} chars")
print(f"📄 Preview: {result[:300]}...")

# Test 5: Query Decomposition
print("\n" + "=" * 80)
print("TEST 5: Complex Query (Decomposition Strategy)")
print("=" * 80)
query = "Analyze Apple's competitive position, supply chain risks, and growth strategy"
print(f"Query: {query}")
start = time.time()
result = agent.analyze(query)
elapsed = time.time() - start
print(f"\n⏱️  Time: {elapsed:.2f}s")
print(f"📝 Result length: {len(result)} chars")
print(f"📄 Preview: {result[:300]}...")

print("\n" + "=" * 80)
print("✅ All tests complete!")
print("=" * 80)

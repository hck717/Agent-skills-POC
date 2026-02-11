import os
from typing import List, Dict, Any
try:
    from qdrant_client import QdrantClient
    from neo4j import GraphDatabase
except ImportError:
    print("⚠️ Qdrant or Neo4j drivers not found. Install with: pip install qdrant-client neo4j")
    QdrantClient = None
    GraphDatabase = None

class HybridRetriever:
    def __init__(self, qdrant_client, neo4j_driver):
        self.qdrant = qdrant_client
        self.neo4j = neo4j_driver
        self.collection_name = "financial_docs"

    def search(self, query: str, ticker: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute Hybrid Search: 
        1. Vector Search for textual nuance (Strategy/Sentiment)
        2. Graph Search for structural dependencies (Risks/Suppliers)
        """
        print(f"   🔍 [Hybrid] Searching for '{query}' on {ticker}...")
        
        # 1. Dense Search (Vector) - Qdrant Cloud
        vector_results = []
        if self.qdrant:
            try:
                # In a real implementation, you'd encode the query here
                # query_vector = self.encoder.encode(query)
                # results = self.qdrant.search(
                #    collection_name=self.collection_name,
                #    query_vector=query_vector,
                #    limit=top_k
                # )
                # vector_results = results
                pass
            except Exception as e:
                print(f"   ⚠️ Qdrant search failed: {e}")

        # 2. Graph Traversal (Neo4j) - Local Docker
        graph_context = []
        if self.neo4j:
            try:
                graph_context = self._search_graph_deep(query, ticker)
            except Exception as e:
                print(f"   ⚠️ Neo4j search failed: {e}")
        
        return {
            "documents": vector_results, 
            "graph_context": graph_context
        }

    def _search_graph_deep(self, query: str, ticker: str) -> List[str]:
        """
        Business Analyst Logic:
        Find Strategy/Risk nodes explicitly.
        """
        # Cypher: Find Strategy or Risk nodes related to the query terms
        # This is a "Concept Search" in the graph
        cypher = """
        MATCH (c:Company {ticker: $ticker})
        MATCH (c)-[r:HAS_STRATEGY|FACES_RISK|HAS_SEGMENT]->(n)
        WHERE toLower(n.description) CONTAINS toLower($query_term)
        RETURN type(r) as relation, n.description as description, labels(n) as type
        LIMIT 5
        """
        
        results = []
        # Simple keyword extraction from query for graph matching
        # In prod, use an LLM to extract "Entities" from query
        query_term = query.split()[0] if query else "" 
        
        with self.neo4j.session() as session:
            result = session.run(cypher, ticker=ticker, query_term=query_term)
            for record in result:
                rel = record["relation"]
                desc = record["description"]
                # Safety check for type list
                node_type = record["type"][0] if record["type"] else "Concept"
                results.append(f"GRAPH FACT: {ticker} {rel} {node_type} -> '{desc}'")
                
        return results

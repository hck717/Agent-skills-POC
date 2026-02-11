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
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[r]->(n)
        WITH type(r) AS relation, labels(n) AS labels, properties(n) AS props
        RETURN relation, labels, props
        LIMIT 15
        """
        results = []
        with self.neo4j.session() as session:
            rows = session.run(cypher, ticker=ticker)
            for row in rows:
                rel = row["relation"]
                labels = row["labels"] or []
                props = row["props"] or {}

                # Prefer readable fields if present; otherwise show keys
                text = (
                    props.get("description")
                    or props.get("title")
                    or props.get("name")
                    or str(props)
                )
                results.append(f"GRAPH FACT: {ticker} -[{rel}]-> {labels} :: {text}")
        return results

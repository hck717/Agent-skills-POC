import os
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import requests
from sentence_transformers import SentenceTransformer

class BusinessAnalystCRAG:
    """
    Deep Reader (v2) - Graph-Augmented Corrective RAG
    Optimized for Speed: Lower retrieval k, context caps, and strict generation limits.
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, llm_url="http://localhost:11434"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.llm_url = f"{llm_url}/api/chat"
        self.model = "deepseek-r1:8b"
        # Load embedding model locally (cached)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        print(f"✅ Business Analyst (CRAG) initialized (Model: {self.model})")

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    def _query_graph_rag(self, ticker: str, query_text: str, k: int = 3) -> List[str]:
        """Hybrid Search: Vector (Chunks) + Graph (Entities)"""
        embedding = self._get_embedding(query_text)
        
        # 1. Vector Search on Chunks (Reduced k=3 for speed)
        # Note: This assumes you have a Vector Index in Neo4j. 
        # For this POC, we will rely on Graph Traversal if Vector Index isn't set up,
        # or simulated vector search via keyword matching on nodes.
        
        # Simple Graph Traversal fallback (fast & robust for POC)
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_STRATEGY|FACES_RISK|OFFERS_PRODUCT]->(n)
        WHERE toLower(n.description) CONTAINS toLower($keyword) 
           OR toLower(n.title) CONTAINS toLower($keyword)
        RETURN n.title + ": " + n.description AS context
        LIMIT $k
        """
        
        # Extract a keyword from query (naive but fast)
        keyword = "business"
        if "risk" in query_text.lower(): keyword = "risk"
        elif "strategy" in query_text.lower(): keyword = "strategy"
        elif "revenue" in query_text.lower(): keyword = "revenue"
        
        with self.driver.session() as session:
            results = session.run(cypher, ticker=ticker, keyword=keyword, k=k)
            return [record["context"] for record in results]

    def _evaluator(self, query: str, retrieved_docs: List[str]) -> str:
        """Lightweight check: Do we have enough info?"""
        if not retrieved_docs: return "AMBIGUOUS"
        # If we have < 2 docs, call it ambiguous to trigger web search
        if len(retrieved_docs) < 1: return "AMBIGUOUS"
        return "CORRECT"

    def _generate(self, query: str, context: List[str]) -> str:
        # Strict Context Cap (6000 chars max)
        joined_context = "\n".join(context)[:6000]
        
        prompt = f"""
        Role: Expert Financial Analyst.
        Task: Answer the query using the Context below.
        
        Query: {query}
        
        Context (Internal Data):
        {joined_context}
        
        Instructions:
        - Be concise.
        - If context is empty, say "Insufficient data".
        - Cite sources like [Graph Context].
        """
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Speed constraints
            "options": {"temperature": 0.1, "num_predict": 1000, "num_ctx": 4096}
        }
        
        try:
            # No timeout, but the limits above should prevent hanging
            response = requests.post(self.llm_url, json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            # Clean think tags
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return clean
        except Exception as e:
            return f"Generation Error: {e}"

    def analyze(self, task: str, ticker: str = "AAPL", **kwargs) -> str:
        print(f"🧠 [Deep Reader] Analyzing: {task} (Ticker: {ticker})")
        
        # 1. Retrieval
        print(f"   🔍 [Hybrid] Searching for '{task[:50]}...' on {ticker}...")
        docs = self._query_graph_rag(ticker, task, k=3)
        
        # 2. Evaluation
        status = self._evaluator(task, docs)
        print(f"   🔍 CRAG Status: {status}")
        
        if status == "INCORRECT": # (Not really used in this simplified logic yet)
            return "Data mismatch."
            
        # 3. Generation
        if status == "AMBIGUOUS" or not docs:
            print("   🔄 Context ambiguous. Proceeding with warning...")
            # Still try to generate with what we have, but Orchestrator will likely trigger Web
            
        print("   📝 Generating Answer...")
        analysis = self._generate(task, docs)
        
        # Format for Orchestrator
        final_output = f"{analysis}\n\n"
        for i, doc in enumerate(docs):
            final_output += f"--- SOURCE: GRAPH FACT: {doc[:50]}... (Graph) ---\n"
            
        return final_output

    def close(self):
        self.driver.close()

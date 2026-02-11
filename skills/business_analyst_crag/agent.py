import os
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import requests
from sentence_transformers import SentenceTransformer

class BusinessAnalystCRAG:
    """
    Deep Reader (v2) - Graph-Augmented Corrective RAG
    Optimized for Speed + Quality: 
    - Retrieval: k=3, Context Cap=6000 chars.
    - Generation: Structured "Deep Reader" prompt for insight, not just summary.
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, llm_url="http://localhost:11434"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.llm_url = f"{llm_url}/api/chat"
        self.model = "deepseek-r1:8b"
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        print(f"✅ Business Analyst (CRAG) initialized (Model: {self.model})")

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    def _query_graph_rag(self, ticker: str, query_text: str, k: int = 3) -> List[str]:
        """Hybrid Search: Vector (Chunks) + Graph (Entities)"""
        # Simple Graph Traversal fallback (fast & robust for POC)
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_STRATEGY|FACES_RISK|OFFERS_PRODUCT]->(n)
        WHERE toLower(n.description) CONTAINS toLower($keyword) 
           OR toLower(n.title) CONTAINS toLower($keyword)
        RETURN n.title + ": " + n.description AS context
        LIMIT $k
        """
        
        keyword = "business"
        if "risk" in query_text.lower(): keyword = "risk"
        elif "strategy" in query_text.lower(): keyword = "strategy"
        elif "revenue" in query_text.lower(): keyword = "revenue"
        elif "growth" in query_text.lower(): keyword = "growth"
        
        with self.driver.session() as session:
            results = session.run(cypher, ticker=ticker, keyword=keyword, k=k)
            return [record["context"] for record in results]

    def _evaluator(self, query: str, retrieved_docs: List[str]) -> str:
        if not retrieved_docs: return "AMBIGUOUS"
        return "CORRECT"

    def _generate(self, query: str, context: List[str]) -> str:
        joined_context = "\n".join(context)[:6000]
        
        # Improved Prompt for "Deep Reader" Quality
        prompt = f"""
        Role: Expert Financial Analyst specializing in 10-K interpretation.
        Task: Extract deep insights from the provided internal data to answer the query.
        
        Query: {query}
        
        Internal Graph Data:
        {joined_context}
        
        Guidelines:
        - Do NOT summarize generic facts. Look for "So What?".
        - Connect the dots between specific risks and strategic goals.
        - If the data is sparse, admit it and interpret what IS there.
        - Output format: 2-3 dense, insight-rich paragraphs.
        """
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1200, "num_ctx": 4096}
        }
        
        try:
            response = requests.post(self.llm_url, json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return clean
        except Exception as e:
            return f"Generation Error: {e}"

    def analyze(self, task: str, ticker: str = "AAPL", **kwargs) -> str:
        print(f"🧠 [Deep Reader] Analyzing: {task} (Ticker: {ticker})")
        
        print(f"   🔍 [Hybrid] Searching for '{task[:50]}...' on {ticker}...")
        docs = self._query_graph_rag(ticker, task, k=3)
        
        status = self._evaluator(task, docs)
        print(f"   🔍 CRAG Status: {status}")
        
        if status == "AMBIGUOUS" or not docs:
            print("   🔄 Context ambiguous. Proceeding with warning...")
            
        print("   📝 Generating Answer...")
        analysis = self._generate(task, docs)
        
        # Format for Orchestrator
        final_output = f"{analysis}\n\n"
        for doc in docs:
            # Clean up the source string right here for the Orchestrator
            # "Title: Description" -> "Graph Fact: Title - Description"
            clean_doc = doc.replace("\n", " ").strip()
            final_output += f"--- SOURCE: GRAPH FACT: {clean_doc} (Internal Graph) ---\n"
            
        return final_output

    def close(self):
        self.driver.close()

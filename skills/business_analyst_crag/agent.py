import os
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

class BusinessAnalystCRAG:
    """
    Deep Reader (v3) - Fully Upgraded Architecture
    
    1. Retrieval: Hybrid (Graph Traversal + Vector Re-ranking)
    2. Evaluation: Cross-Encoder Score (CRAG)
    3. Generation: Insight-focused Prompt
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, llm_url="http://localhost:11434"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.llm_url = f"{llm_url}/api/chat"
        self.model = "deepseek-r1:8b"
        
        # 1. Bi-Encoder for fast embedding (Simulated Vector Search if index missing)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Cross-Encoder for High-Precision Reranking & Evaluation (The "Judge")
        # Lightweight model, fast on CPU
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print(f"✅ Business Analyst (Deep Reader v3) initialized")
        print(f"   - Model: {self.model}")
        print(f"   - Reranker: ms-marco-MiniLM-L-6-v2 (Loaded)")

    def _query_graph_rag(self, ticker: str, query_text: str, k: int = 3) -> List[str]:
        """
        Hybrid Retrieval Strategy:
        1. Broad Graph Traversal (Get ~20 candidates)
        2. Semantic Reranking (Narrow to Top-k)
        """
        # 1. Broad Fetch (Limit 20)
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_STRATEGY|FACES_RISK|OFFERS_PRODUCT]->(n)
        WHERE toLower(n.description) CONTAINS toLower($keyword) 
           OR toLower(n.title) CONTAINS toLower($keyword)
        RETURN n.title + ": " + n.description AS context
        LIMIT 20
        """
        
        # Simple keyword extractor for the Cypher hook
        keyword = "business"
        q_lower = query_text.lower()
        if "risk" in q_lower: keyword = "risk"
        elif "strategy" in q_lower: keyword = "strategy"
        elif "revenue" in q_lower: keyword = "revenue"
        elif "growth" in q_lower: keyword = "growth"
        elif "compet" in q_lower: keyword = "compet"
        
        candidates = []
        with self.driver.session() as session:
            results = session.run(cypher, ticker=ticker, keyword=keyword)
            candidates = [record["context"] for record in results]
            
        if not candidates:
            return []
            
        # 2. Semantic Reranking (The "Vector" simulation)
        # Pairs of (Query, Document)
        pairs = [[query_text, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by score descending
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Filter logic: Keep top K, but only if score > 0.0 (sanity check)
        top_k = [doc for doc, score in scored_docs[:k] if score > -10.0] # Logits can be negative
        
        return top_k

    def _evaluator(self, query: str, retrieved_docs: List[str]) -> str:
        """
        CRAG Evaluator Logic:
        - Score the top retrieved document against the query.
        - > 0.5: CORRECT
        - < 0.5 or Empty: INCORRECT (Trigger Web)
        """
        if not retrieved_docs:
            return "INCORRECT"
            
        # Check confidence of the BEST match
        top_doc = retrieved_docs[0]
        score = self.reranker.predict([[query, top_doc]])[0]
        
        print(f"   📊 CRAG Confidence Score: {score:.4f}")
        
        if score > 0.5:
            return "CORRECT"
        elif score > 0.0:
            return "AMBIGUOUS" # In a full system, we'd rewrite query. Here we'll treat as weak correct.
        else:
            return "INCORRECT"

    def _generate(self, query: str, context: List[str]) -> str:
        joined_context = "\n".join(context)[:6000]
        
        prompt = f"""
        Role: Expert Financial Analyst.
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
        # 1. Retrieval (Hybrid: Graph + Rerank)
        docs = self._query_graph_rag(ticker, task, k=3)
        
        # 2. Evaluation (CRAG)
        status = self._evaluator(task, docs)
        print(f"   🔍 CRAG Status: {status}")
        
        # 3. Decision Logic
        if status == "INCORRECT":
            return "CRAG_FALLBACK_REQUIRED: Insufficient internal data confidence."
            
        print("   📝 Generating Answer...")
        analysis = self._generate(task, docs)
        
        # Format for Orchestrator
        final_output = f"{analysis}\n\n"
        for doc in docs:
            # Clean formatting
            clean_doc = doc.replace("\n", " ").strip()
            final_output += f"--- SOURCE: GRAPH FACT: {clean_doc} (Internal Graph) ---\n"
            
        return final_output

    def close(self):
        self.driver.close()

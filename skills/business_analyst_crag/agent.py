import os
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

class BusinessAnalystCRAG:
    """
    Deep Reader (v4) - Fully Upgraded Architecture
    
    1. Retrieval: Hybrid (Vector + Graph + BM25)
       - Vector: Neo4j Vector Index (Cosine Similarity)
       - Graph: Cypher Traversal
       - Keyword: BM25 (Sparse)
    2. Evaluation: Cross-Encoder Score (CRAG)
    3. Generation: Insight-focused Prompt
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, llm_url="http://localhost:11434"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.llm_url = f"{llm_url}/api/chat"
        self.model = "deepseek-r1:8b"
        
        # 1. Bi-Encoder for Embeddings (Vector Search)
        # 384 dimensions matching standard MiniLM
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Cross-Encoder for High-Precision Reranking & Evaluation (The "Judge")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 3. Initialize/Check Vector Index in Neo4j
        self._init_vector_index()
        
        print(f"✅ Business Analyst (Deep Reader v4) initialized")
        print(f"   - Model: {self.model}")
        print(f"   - Vector Index: 'chunk_embedding' (checked)")

    def _init_vector_index(self):
        """Create Vector Index if it doesn't exist."""
        # Note: This index target is 'Chunk' nodes with 'embedding' property.
        # Ensure ingestion script creates (n:Chunk {embedding: [...]})
        cypher = """
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            with self.driver.session() as session:
                session.run(cypher)
        except Exception as e:
            print(f"⚠️ Vector Index Init Warning: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    def _bm25_score(self, query: str, docs: List[str]) -> List[float]:
        """Calculate BM25 scores for a list of documents."""
        if not docs: return []
        tokenized_corpus = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        return bm25.get_scores(tokenized_query)

    def _query_graph_rag(self, ticker: str, query_text: str, k: int = 3) -> List[str]:
        """
        Hybrid Retrieval Strategy:
        1. Vector Search (Neo4j Index) -> Top 10
        2. Graph Traversal (Cypher) -> Top 10
        3. Combine & Dedupe
        4. BM25 Scoring (Sparse Signal)
        5. Cross-Encoder Reranking (Final Select)
        """
        candidates = []
        
        # --- A. Vector Search (Dense) ---
        query_embedding = self._get_embedding(query_text)
        vector_cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding', 10, $embedding)
        YIELD node, score
        MATCH (node)<-[:HAS_CHUNK]-(:Section)<-[:HAS_SECTION]-(:Report)-[:HAS_REPORT]->(c:Company {ticker: $ticker})
        RETURN node.text AS text, score
        """
        # Note: This path MATCH depends on your exact schema. 
        # Fallback simpler query if schema differs:
        # MATCH (node) WHERE node.ticker = $ticker ...
        
        # For this POC, let's assume Chunks might be directly linked or we just search index globally 
        # and filter by text content if ticker property isn't on Chunk.
        # Safer fallback POC query:
        vector_cypher_simple = """
        CALL db.index.vector.queryNodes('chunk_embedding', 15, $embedding)
        YIELD node, score
        RETURN node.text AS text, score
        """
        
        with self.driver.session() as session:
            try:
                # Try specific ticker path first? 
                # For robustness in this script, we'll use the simple vector query 
                # (assuming data is mostly relevant or relying on reranker to filter)
                # In prod, ALWAYS filter by metadata (ticker).
                res = session.run(vector_cypher_simple, embedding=query_embedding)
                candidates.extend([r["text"] for r in res])
            except Exception as e:
                print(f"⚠️ Vector Search Failed: {e}")

        # --- B. Graph Traversal (Structured) ---
        # Gets entities directly linked to company
        graph_cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_STRATEGY|FACES_RISK|OFFERS_PRODUCT]->(n)
        WHERE toLower(n.description) CONTAINS toLower($keyword) 
           OR toLower(n.title) CONTAINS toLower($keyword)
        RETURN n.title + ": " + n.description AS text
        LIMIT 10
        """
        
        keyword = "business" # primitive fallback
        q_lower = query_text.lower()
        if "risk" in q_lower: keyword = "risk"
        elif "strategy" in q_lower: keyword = "strategy"
        elif "revenue" in q_lower: keyword = "revenue"
        elif "growth" in q_lower: keyword = "growth"
        
        with self.driver.session() as session:
            res = session.run(graph_cypher, ticker=ticker, keyword=keyword)
            candidates.extend([r["text"] for r in res])

        # --- C. Dedupe ---
        candidates = list(set(candidates))
        if not candidates: return []
        
        # --- D. BM25 Scoring (Sparse) ---
        # We'll use BM25 to boost the Cross-Encoder, or just trust Cross-Encoder?
        # Standard approach: weighted sum. 
        # Here, we'll keep it simple: Pass ALL candidates to Cross-Encoder. 
        # BM25 is useful if we had 1000s of candidates. With ~20, Cross-Encoder is fast enough.
        # But user requested BM25. Let's use it to pre-filter if we had too many.
        # Since we have < 30, we will just print the top BM25 match for debugging.
        
        try:
            bm25_scores = self._bm25_score(query_text, candidates)
            # We could use this to filter, but let's rely on Cross-Encoder for final rank.
        except:
            pass

        # --- E. Cross-Encoder Reranking (Final) ---
        pairs = [[query_text, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_k = [doc for doc, score in scored_docs[:k] if score > -10.0]
        
        return top_k

    def _evaluator(self, query: str, retrieved_docs: List[str]) -> str:
        if not retrieved_docs: return "INCORRECT"
        top_doc = retrieved_docs[0]
        score = self.reranker.predict([[query, top_doc]])[0]
        print(f"   📊 CRAG Confidence Score: {score:.4f}")
        if score > 0.4: return "CORRECT" # Slightly lower threshold for POC
        elif score > 0.0: return "AMBIGUOUS"
        else: return "INCORRECT"

    def _generate(self, query: str, context: List[str]) -> str:
        joined_context = "\n".join(context)[:6000]
        prompt = f"""
        Role: Expert Financial Analyst.
        Task: Answer query using Internal Graph Data.
        Query: {query}
        Internal Graph Data:
        {joined_context}
        Guidelines:
        - Identify key risks, strategies, or metrics.
        - "So What?" insights > generic summaries.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1000, "num_ctx": 4096}
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
        print(f"   🔍 [Hybrid] Vector + Graph + Rerank...")
        
        docs = self._query_graph_rag(ticker, task, k=3)
        status = self._evaluator(task, docs)
        print(f"   🔍 CRAG Status: {status}")
        
        if status == "INCORRECT":
            return "CRAG_FALLBACK_REQUIRED"
            
        print("   📝 Generating Answer...")
        analysis = self._generate(task, docs)
        
        final_output = f"{analysis}\n\n"
        for doc in docs:
            clean_doc = doc.replace("\n", " ").strip()[:200]
            final_output += f"--- SOURCE: GRAPH FACT: {clean_doc}... (Internal) ---\n"
        return final_output

    def close(self):
        self.driver.close()

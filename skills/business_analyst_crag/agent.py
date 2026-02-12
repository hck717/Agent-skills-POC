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
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, llm_url="http://localhost:11434", web_search_agent=None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.llm_url = f"{llm_url}/api/chat"
        self.model = "deepseek-r1:8b"
        self.web_search_agent = web_search_agent  # Web fallback for low-confidence retrieval
        
        # 1. Bi-Encoder for Embeddings (Vector Search)
        # 384 dimensions matching standard MiniLM
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Cross-Encoder for High-Precision Reranking & Evaluation (The "Judge")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 3. Initialize/Check Vector Index in Neo4j
        self._init_vector_index()
        
        print(f"✅ Business Analyst (Deep Reader v4.2 - Full CRAG) initialized")
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
        WHERE node.ticker CONTAINS $ticker
        RETURN node.text AS text, score
        """
        
        with self.driver.session() as session:
            try:
                # Try specific ticker path first? 
                # For robustness in this script, we'll use the simple vector query 
                # (assuming data is mostly relevant or relying on reranker to filter)
                # In prod, ALWAYS filter by metadata (ticker).
                res = session.run(vector_cypher_simple, embedding=query_embedding, ticker=ticker)
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
        
        # --- D. BM25 Scoring (Sparse) - INTEGRATED ---
        bm25_scores = [0.0] * len(candidates)
        try:
            bm25_raw = self._bm25_score(query_text, candidates)
            # Normalize BM25 to 0-1 range
            max_bm25 = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0
            bm25_scores = [s / max_bm25 for s in bm25_raw]
            print(f"   📊 BM25: Top score = {max(bm25_scores):.3f}")
        except Exception as e:
            print(f"   ⚠️ BM25 scoring failed: {e}")

        # --- E. Cross-Encoder Reranking (Final) ---
        pairs = [[query_text, doc] for doc in candidates]
        cross_scores = self.reranker.predict(pairs)
        
        # Hybrid scoring: 30% BM25 (sparse) + 70% Cross-Encoder (semantic)
        final_scores = [
            0.3 * bm25_scores[i] + 0.7 * cross_scores[i]
            for i in range(len(candidates))
        ]
        
        scored_docs = sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)
        print(f"   🎯 Hybrid Ranking: Top scores = {[f'{s:.3f}' for _, s in scored_docs[:3]]}")
        top_k = [doc for doc, score in scored_docs[:k]]
        
        return top_k

    def _evaluator(self, query: str, retrieved_docs: List[str]) -> str:
        """CRAG Evaluator with spec-compliant thresholds.
        
        Returns:
            CORRECT: score > 0.7 (use directly)
            AMBIGUOUS: 0.5 <= score <= 0.7 (rewrite query)
            INCORRECT: score < 0.5 (trigger web fallback)
        """
        if not retrieved_docs:
            print(f"   ❌ CRAG: No documents retrieved")
            return "INCORRECT"
        
        top_doc = retrieved_docs[0]
        score = self.reranker.predict([[query, top_doc]])[0]
        print(f"   📊 CRAG Confidence Score: {score:.4f}")
        
        # Spec-compliant thresholds
        if score > 0.7:
            return "CORRECT"
        elif score >= 0.5:
            return "AMBIGUOUS"
        else:
            return "INCORRECT"

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

    def _rewrite_query(self, original_query: str, context_hint: str) -> str:
        """Use LLM to rewrite ambiguous queries with context."""
        prompt = f"""You are a query optimization expert. Rewrite this query to be more specific based on the context hint.

Original Query: {original_query}
Context Hint: {context_hint[:300]}...

Rules:
1. Keep the core intent
2. Add specific keywords from context
3. Remove vague terms like "analyze", "corresponding"
4. Output ONLY the rewritten query (one line)
5. NO explanations

Rewritten Query:"""
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 100}
        }
        try:
            response = requests.post(self.llm_url, json=payload, timeout=10)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            # Clean thinking tags and extract first line
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            rewritten = clean.split('\n')[0].strip()
            return rewritten if len(rewritten) > 5 else original_query
        except Exception as e:
            print(f"   ⚠️ Query rewrite failed: {e}")
            return original_query


    def analyze(self, task: str, ticker: str = "AAPL", **kwargs) -> str:
        """Complete CRAG pipeline with spec-compliant evaluation.
        
        Pipeline:
        1. Hybrid retrieval (Vector + Graph + BM25)
        2. CRAG evaluation (CORRECT/AMBIGUOUS/INCORRECT)
        3. Adaptive handling:
           - CORRECT (>0.7): Use directly
           - AMBIGUOUS (0.5-0.7): Rewrite query & retry
           - INCORRECT (<0.5): Trigger web fallback
        """
        print(f"🧠 [Deep Reader v4.2 - Full CRAG] Analyzing: {task} (Ticker: {ticker})")
        print(f"   🔍 [Hybrid] Vector + Graph + BM25 + Cross-Encoder...")
        
        # Phase 1: Initial Retrieval
        docs = self._query_graph_rag(ticker, task, k=5)
        status = self._evaluator(task, docs)
        print(f"   🔍 CRAG Status: {status}")
        
        # Phase 2: Handle AMBIGUOUS (0.5-0.7) - Adaptive Query Rewriting
        if status == "AMBIGUOUS":
            print(f"   🔄 [CRAG] Ambiguous confidence - rewriting query with LLM...")
            context_hint = docs[0] if docs else ""
            rewritten_query = self._rewrite_query(task, context_hint)
            print(f"   🔄 Rewritten: '{rewritten_query}'")
            
            # Retry with rewritten query
            docs_retry = self._query_graph_rag(ticker, rewritten_query, k=5)
            status_retry = self._evaluator(rewritten_query, docs_retry)
            print(f"   🔍 CRAG Status (retry): {status_retry}")
            
            # Use retry if improved
            if status_retry in ["CORRECT", "AMBIGUOUS"]:
                docs = docs_retry
                status = status_retry
                task = rewritten_query  # Use rewritten for generation
        
        # Phase 3: Handle INCORRECT (<0.5) - Web Fallback
        if status == "INCORRECT":
            if self.web_search_agent:
                print(f"   🌐 [CRAG] Low confidence - triggering Web Search fallback...")
                try:
                    web_result = self.web_search_agent.analyze(
                        query=task,
                        prior_analysis="",
                        metadata={"years": [2026], "topics": ["Business Analysis"]},
                        use_hyde=True,
                        use_step_back=True,
                        top_n=3
                    )
                    print(f"   ✅ [CRAG] Web fallback successful")
                    # Prepend header to indicate web source
                    return f"## External Intelligence (Web Search Fallback)\n\n{web_result}"
                except Exception as e:
                    print(f"   ❌ [CRAG] Web fallback failed: {e}")
                    return f"## Analysis Unavailable\n\nInsufficient graph context and web fallback failed. Error: {e}"
            else:
                print(f"   ❌ [CRAG] Low confidence - no web agent configured")
                return "CRAG_FALLBACK_REQUIRED"
        
        # Phase 4: Generate Answer from Retrieved Context
        print("   📝 Generating Answer from Graph Context...")
        analysis = self._generate(task, docs)
        
        # Phase 5: Append Source Citations
        final_output = f"{analysis}\n\n"
        for i, doc in enumerate(docs, 1):
            clean_doc = doc.replace("\n", " ").strip()[:200]
            final_output += f"[{i}] GRAPH FACT: {clean_doc}...\n"
        
        return final_output

    def close(self):
        self.driver.close()

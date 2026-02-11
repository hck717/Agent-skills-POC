import os
from .ingestion import SemanticChunker
from .retrieval import HybridRetriever
from .crag_evaluator import CragEvaluator
import ollama

try:
    from qdrant_client import QdrantClient
    from neo4j import GraphDatabase
except ImportError:
    print("⚠️ Qdrant or Neo4j drivers not found.")

class BusinessAnalystCRAG:
    """
    Business Analyst v2: Graph-Augmented Corrective RAG
    Connects to Qdrant Cloud (Vectors) and Neo4j Docker (Graph).
    """
    def __init__(self, qdrant_url=None, qdrant_key=None, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_pass="password", tavily_client=None):
        self.ollama = ollama 
        
        # Initialize DB Clients
        self.qdrant = None
        self.neo4j = None
        
        # 1. Connect Qdrant (Cloud)
        if qdrant_url and qdrant_key:
            try:
                self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                print("   ✅ [CRAG] Qdrant Cloud Connected")
            except Exception as e:
                print(f"   ❌ [CRAG] Qdrant Connection Failed: {e}")
        
        # 2. Connect Neo4j (Local Docker)
        try:
            self.neo4j = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            # Test connection
            with self.neo4j.session() as session:
                session.run("RETURN 1")
            print("   ✅ [CRAG] Neo4j (Docker) Connected")
        except Exception as e:
            print(f"   ⚠️ [CRAG] Neo4j Connection Failed: {e}")

        self.retriever = HybridRetriever(self.qdrant, self.neo4j)
        self.evaluator = CragEvaluator(self.ollama)
        self.tavily = tavily_client

    def analyze(self, query: str, ticker: str = "AAPL", prior_analysis: str = "") -> str:
        print(f"🧠 [Deep Reader] Analyzing: {query}")
        
        # 1. Retrieval
        retrieval_result = self.retriever.search(query, ticker)
        docs = [d['content'] for d in retrieval_result['documents']] if retrieval_result['documents'] else []
        graph_data = retrieval_result['graph_context']
        
        combined_context = docs + graph_data
        
        if not combined_context:
             print("   ⚠️ No context found in DBs. Skipping CRAG eval.")
             status = "INCORRECT" # Force fallback
        else:
            # 2. Evaluation (CRAG)
            status = self.evaluator.evaluate(query, combined_context)
            print(f"   🔍 CRAG Status: {status}")
        
        final_context = combined_context
        
        # 3. Corrective Actions
        if status == "INCORRECT":
            print("   ⚠️ Context irrelevant/missing. Triggering Web Search Fallback...")
            if self.tavily:
                web_results = self.tavily.search(query, max_results=3)
                # Ensure we handle list/dict return types from Tavily client wrapper
                if isinstance(web_results, dict) and 'results' in web_results:
                     final_context = [f"WEB SOURCE: {r['content']}" for r in web_results['results']]
                else:
                     final_context = [str(web_results)]
            else:
                return "Insufficient data in 10-K/Graph to answer this query, and Web Search is disabled."
                
        elif status == "AMBIGUOUS":
            print("   🔄 Context ambiguous. Proceeding with warning...")
            final_context.append("NOTE: Context might be partial or ambiguous.")

        # 4. Generation
        print("   📝 Generating Answer...")
        
        context_str = "\n".join([str(c) for c in final_context])
        
        prompt = f"""
        You are a Deep Reader Business Analyst.
        Answer based on the verified context below.
        
        VERIFIED CONTEXT:
        {context_str}
        
        QUERY: {query}
        
        ANALYSIS:
        """
        
        try:
            response = self.ollama.chat(
                model="deepseek-r1:8b",
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            return response['message']['content'] if isinstance(response, dict) else response.message.content
        except Exception as e:
            return f"Error generating analysis: {e}"

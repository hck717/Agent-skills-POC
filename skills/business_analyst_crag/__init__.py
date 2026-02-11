from .ingestion import SemanticChunker
from .retrieval import HybridRetriever
from .crag_evaluator import CragEvaluator
import ollama

class BusinessAnalystCRAG:
    """
    Business Analyst v2: Graph-Augmented Corrective RAG
    """
    def __init__(self, qdrant_client=None, neo4j_driver=None, tavily_client=None):
        self.ollama = ollama # Use standard library or wrapper
        self.retriever = HybridRetriever(qdrant_client, neo4j_driver)
        self.evaluator = CragEvaluator(self.ollama)
        self.tavily = tavily_client # For fallback

    def analyze(self, query: str, ticker: str = "AAPL", prior_analysis: str = "") -> str:
        print(f"🧠 [Deep Reader] Analyzing: {query}")
        
        # 1. Retrieval
        retrieval_result = self.retriever.search(query, ticker)
        docs = [d['content'] for d in retrieval_result['documents']]
        graph_data = retrieval_result['graph_context']
        
        combined_context = docs + graph_data
        
        # 2. Evaluation (CRAG)
        status = self.evaluator.evaluate(query, combined_context)
        print(f"   🔍 CRAG Status: {status}")
        
        final_context = combined_context
        
        # 3. Corrective Actions
        if status == "INCORRECT":
            print("   ⚠️ Context irrelevant. Triggering Web Search Fallback...")
            if self.tavily:
                web_results = self.tavily.search(query, max_results=3)
                final_context = [r['content'] for r in web_results['results']]
            else:
                return "Insufficient data in 10-K to answer this query."
                
        elif status == "AMBIGUOUS":
            print("   🔄 Context ambiguous. Refining query...")
            # Simple query expansion logic could go here
            # For now, we proceed but warn the LLM
            final_context.append("NOTE: Context might be partial.")

        # 4. Generation
        print("   📝 Generating Answer...")
        
        context_str = "\n".join(final_context)
        
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

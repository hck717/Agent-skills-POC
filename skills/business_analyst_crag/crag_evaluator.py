"""
CRAG Evaluator
Scores retrieved documents to trigger Corrective actions.
"""
class CragEvaluator:
    def __init__(self, ollama_client):
        self.client = ollama_client

    def evaluate(self, query: str, retrieved_docs: list[str]) -> str:
        """
        Score the relevance of retrieved docs to the query.
        Returns: 'CORRECT', 'INCORRECT', or 'AMBIGUOUS'
        """
        context_block = "\n".join(retrieved_docs)
        
        prompt = f"""
        You are a grader assessing relevance.
        Query: {query}
        Retrieved Context:
        {context_block}
        
        Does the retrieved context contain the answer to the query?
        Output only one word: CORRECT, INCORRECT, or AMBIGUOUS.
        """
        
        try:
            # FIX: Explicitly specify model parameter to avoid Pydantic validation error
            response = self.client.chat(
                model="deepseek-r1:8b", 
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            
            content = response['message']['content'] if isinstance(response, dict) else response.message.content
            score = content.strip().upper()
            
            if "CORRECT" in score: return "CORRECT"
            if "AMBIGUOUS" in score: return "AMBIGUOUS"
            return "INCORRECT"
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return "AMBIGUOUS" # Fail safe

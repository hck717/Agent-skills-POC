"""
Semantic Chunking Module
Uses an LLM to break text into atomic propositions before vectorizing.
"""
import re
from typing import List

class SemanticChunker:
    def __init__(self, ollama_client):
        self.client = ollama_client

    def chunk_by_proposition(self, text: str) -> List[str]:
        """
        Break text into standalone atomic statements (propositions).
        This prevents context fragmentation in standard RAG.
        """
        prompt = f"""
        Break the following text into independent, atomic statements. 
        Each statement must be self-contained and factually complete.
        Remove conjunctions like "and", "but".
        
        TEXT:
        {text}
        
        OUTPUT FORMAT:
        - Statement 1
        - Statement 2
        """
        
        try:
            # In production, process in batches to save LLM calls
            # For POC, we process the raw text (assuming it's a section)
            # Adjust the call signature depending on your specific Ollama client implementation
            response = self.client.chat(
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            
            # Handle different response formats if needed (dict vs object)
            content = response['message']['content'] if isinstance(response, dict) else response.message.content
            
            # Extract bullet points
            propositions = [
                line.strip('- ').strip() 
                for line in content.split('\n') 
                if line.strip().startswith('-')
            ]
            
            return propositions if propositions else [text] # Fallback to raw text
            
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            return [text] # Fallback

"""
Semantic Chunking Module
Uses an LLM to break text into atomic propositions before vectorizing.
"""
import os
import re
import json
from typing import List

class SemanticChunker:
    """
    Splits text into atomic, standalone propositions.
    Logic: "Risk Factor 1" is not split in half. It is one clean vector.
    """
    def __init__(self, ollama_client=None):
        self.client = ollama_client
        # If OpenAI is available, we prefer it for chunking quality, 
        # otherwise we can use Ollama or a deterministic fallback.
        self.use_openai = bool(os.getenv("OPENAI_API_KEY"))

    def _fallback_sentence_split(self, text: str, max_len: int = 220) -> List[str]:
        text = re.sub(r"\s+", " ", (text or "").strip())
        if not text: return []
        
        # Split on sentence terminators
        sents = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for s in sents:
            s = s.strip()
            if not s: continue
            if len(s) > max_len:
                # Sub-split long sentences
                parts = re.split(r"[;:]\s+|,\s+", s)
                for p in parts:
                    if 15 <= len(p) <= 500: out.append(p.strip())
            else:
                if 15 <= len(s) <= 500: out.append(s)
        
        # Dedupe
        seen = set()
        dedup = []
        for x in out:
            if x.lower() not in seen:
                seen.add(x.lower())
                dedup.append(x)
        return dedup

    def chunk_by_proposition(self, text: str) -> List[str]:
        """
        Break text into standalone atomic statements (propositions).
        """
        text = (text or "").strip()
        if not text: return []
        
        prompt = f"""
        Task: Split the text below into atomic, standalone propositions.
        Each proposition must be a single, factually complete sentence that makes sense on its own.
        Remove conjunctions like "and", "but".
        
        TEXT:
        {text[:8000]}
        
        OUTPUT format: JSON array of strings. Example: ["Prop 1", "Prop 2"]
        """
        
        # 1. Try OpenAI (Best Quality)
        if self.use_openai:
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=os.getenv("PROPOSITION_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                raw = resp.choices[0].message.content
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    props = json.loads(match.group(0))
                    if isinstance(props, list) and len(props) > 0:
                        return [p for p in props if isinstance(p, str)]
            except Exception as e:
                print(f"⚠️ OpenAI Chunking failed: {e}. Falling back.")

        # 2. Try Ollama (Local)
        if self.client:
            try:
                response = self.client.chat(
                    model="deepseek-r1:8b", # Or whatever model is loaded
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.0, 'num_predict': 1000}
                )
                content = response['message']['content'] if isinstance(response, dict) else response.message.content
                
                # Try to find JSON
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    props = json.loads(match.group(0))
                    return [p for p in props if isinstance(p, str)]
                
                # Fallback to bullet point parsing if JSON fails
                propositions = [
                    line.strip('- ').strip() 
                    for line in content.split('\n') 
                    if line.strip().startswith('-') or line.strip().startswith('*')
                ]
                if propositions: return propositions
                
            except Exception as e:
                print(f"⚠️ Ollama Chunking failed: {e}")

        # 3. Deterministic Fallback
        return self._fallback_sentence_split(text)

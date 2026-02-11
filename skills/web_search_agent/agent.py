#!/usr/bin/env python3
""" 
Web Search Agent ("News Desk")

Goal: Find unknown unknowns fast.
Architecture: Step-Back Prompting + HyDE expansion + corrective reranking.
Trusted Domains: Bloomberg, Reuters, WSJ, CNBC, FT, TechCrunch.

Outputs MUST preserve SOURCE markers:
--- SOURCE: Title (https://...) ---
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import requests
import ollama
from tavily import TavilyClient


class WebSearchAgent:
    TRUSTED_DOMAINS = [
        "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com", "ft.com", 
        "techcrunch.com", "forbes.com", "marketwatch.com", "yahoo.com/finance",
        "investopedia.com", "businessinsider.com", "nytimes.com", "seekingalpha.com"
    ]

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        ollama_model: str = "deepseek-r1:8b",
        ollama_base_url: str = "http://localhost:11434",
        cohere_api_key: Optional[str] = None,
    ):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found. Set env var or pass to constructor.")

        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")

        print(f"✅ Web Search Agent initialized (News Desk, model={ollama_model})")

    def _clean_think(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _ollama_chat(self, messages: List[Dict], temperature: float = 0.0, num_predict: int = 400) -> str:
        try:
            resp = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={"temperature": temperature, "num_predict": num_predict},
            )
            content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            return self._clean_think(content)
        except Exception as e:
            print(f"   ⚠️ Ollama Error: {e}")
            return ""

    def _step_back_query(self, query: str) -> str:
        prompt = f"""Rewrite: '{query}' into a broader search query for recent market news/catalysts.
        Output ONE sentence. No quotes."""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=100)
        return out.strip().strip('"')

    def _hyde_brief(self, query: str, prior_analysis: str = "") -> str:
        prompt = f"""Write a fake 3-sentence news brief answering: {query}
        Use prior context if any: {prior_analysis[:300]}
        Plain text only."""
        return self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.2, num_predict=200)

    def _hyde_queries_from_brief(self, hyde_text: str, max_queries: int = 2) -> List[str]:
        prompt = f"""Extract {max_queries} search queries from this brief:
        {hyde_text}
        Output ONLY valid JSON list of strings."""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=150)
        try:
            match = re.search(r'\[.*\]', out, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        return [hyde_text[:80]] if hyde_text else []

    def _tavily_search(self, query: str, max_results: int = 6) -> List[Dict]:
        if not query or len(query) < 3: return []
        try:
            # Append "site:bloomberg.com OR site:reuters.com ..." to enforce domains? 
            # Better to let Tavily search broadly then filter, unless query is very specific.
            # For now, we search broadly to get max recall.
            resp = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )
            return resp.get("results", []) or []
        except Exception as e:
            print(f"   ❌ Tavily search error: {e}")
            return []

    def _is_trustworthy(self, url: str) -> bool:
        """Filter results to only allow trusted financial news domains."""
        if not url: return False
        domain = url.lower().replace("https://", "").replace("http://", "").split('/')[0]
        # Allow subdomains like www.bloomberg.com or finance.yahoo.com
        return any(trusted in domain for trusted in self.TRUSTED_DOMAINS)

    def _dedupe_and_filter(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in results:
            url = (r.get("url") or "").strip()
            if not url or not self._is_trustworthy(url): 
                continue
            if url in seen: continue
            seen.add(url)
            out.append(r)
        return out

    def _format_context(self, results: List[Dict]) -> Tuple[str, List[Dict]]:
        ctx = ""
        citations = []
        for r in results:
            title = (r.get("title") or "Unknown").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()
            if not url: continue
            
            # Formatting for the LLM
            ctx += f"\n--- SOURCE: {title} ({url}) ---\n{content[:500]}...\n" # Limit content length
            citations.append({"title": title, "url": url, "content": content})
        return ctx.strip(), citations

    def _inject_citations(self, text: str, citations: List[Dict]) -> str:
        if "--- SOURCE:" in text: return text
        if not citations: return text
        
        # Simple injection strategy: append 1 citation per paragraph
        paras = [p for p in text.split('\n\n') if len(p) > 50]
        out = []
        for i, p in enumerate(paras):
            out.append(p)
            if i < len(citations):
                c = citations[i]
                out.append(f"--- SOURCE: {c['title']} ({c['url']}) ---")
        return "\n\n".join(out)

    def _synthesize(self, query: str, context: str, citations: List[Dict], prior_analysis: str = "") -> str:
        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = f"""You are a Financial News Desk.
        Date: {current_date}
        
        Task: Write a market update based on the Web Context below.
        
        Rules:
        1. Use ONLY the provided Web Context.
        2. Write 3-4 concise paragraphs.
        3. After EVERY paragraph, add: --- SOURCE: Title (URL) ---
        4. Focus on facts, numbers, dates.
        
        Question: {query}
        
        Web Context:
        {context}
        """
        
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=800)
        
        # Fallback if LLM fails to cite
        if "--- SOURCE:" not in out:
            out = self._inject_citations(out, citations)
            
        return out

    def analyze(self, query: str, prior_analysis: str = "", metadata: Dict = {}) -> str:
        print(f"\n🌐 News Desk analyzing: '{query}'")
        
        # 1. Generate Queries
        step_back = self._step_back_query(query)
        hyde = self._hyde_brief(query, prior_analysis)
        hyde_queries = self._hyde_queries_from_brief(hyde, max_queries=2)
        
        year = datetime.now().year
        meta_years = metadata.get("years", [year])
        target_year = meta_years[0] if meta_years else year
        direct = f"{query} news {target_year}"
        
        # 2. Search
        queries = list(set([direct, step_back] + hyde_queries))
        print(f"   🔍 Queries: {queries}")
        
        all_results = []
        for q in queries:
            all_results.extend(self._tavily_search(q, max_results=5))
            
        # 3. Filter (Trustworthy Only)
        filtered_results = self._dedupe_and_filter(all_results)
        print(f"   ✅ Found {len(filtered_results)} trusted articles")
        
        # 4. Fallback if filtering removed everything
        if not filtered_results and all_results:
            print("   ⚠️ Strict filtering yielded 0 results. Relaxing filter.")
            filtered_results = all_results[:3] # Take top 3 untrusted if necessary
            
        # 5. Synthesize
        if not filtered_results:
            return "## Web Research\n\nNo recent reliable news found."
            
        context, citations = self._format_context(filtered_results)
        return self._synthesize(query, context, citations, prior_analysis)

if __name__ == "__main__":
    agent = WebSearchAgent()
    print(agent.analyze("Microsoft risks 2026"))

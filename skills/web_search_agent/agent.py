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
        """Removes <think>...</think> blocks and obvious residue from DeepSeek output."""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean = clean.replace("<think>", "").replace("</think>", "")
        # Remove some common thought-residue prefixes DeepSeek sometimes emits
        clean = re.sub(r"^(ores|okay|alright|sure)[\s\n:,-]+", "", clean.strip(), flags=re.IGNORECASE)
        return clean.strip()

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
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=120)
        return out.strip().strip('"')

    def _hyde_brief(self, query: str, prior_analysis: str = "") -> str:
        prompt = f"""Write a fake 3-sentence news brief answering: {query}
Use prior context if any: {prior_analysis[:300]}
Plain text only."""
        return self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.2, num_predict=220)

    def _hyde_queries_from_brief(self, hyde_text: str, max_queries: int = 2) -> List[str]:
        prompt = f"""Extract {max_queries} search queries from this brief:
{hyde_text}
Output ONLY valid JSON list of strings."""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=180)

        # Parse JSON list defensively
        queries: List[str] = []
        try:
            match = re.search(r'\[.*\]', out, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    queries = [q for q in parsed if isinstance(q, str)]
        except Exception:
            queries = []

        # Aggressive sanitization to avoid thought-residue / multiline junk
        cleaned: List[str] = []
        for q in queries:
            q2 = q.replace("\n", " ").strip()
            q2 = re.sub(r"\s+", " ", q2)
            # Drop garbage
            if len(q2) < 6: 
                continue
            if q2.lower().startswith("ores"):
                continue
            if "fake" in q2.lower() or "halluc" in q2.lower():
                continue
            cleaned.append(q2)

        if cleaned:
            return cleaned[:max_queries]

        return [hyde_text[:80].replace("\n", " ")] if hyde_text else []

    def _tavily_search(self, query: str, max_results: int = 8) -> List[Dict]:
        if not query or len(query) < 3:
            return []
        try:
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
        if not url:
            return False
        domain = url.lower().replace("https://", "").replace("http://", "").split('/')[0]
        return any(trusted in domain for trusted in self.TRUSTED_DOMAINS)

    def _dedupe(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in results:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append(r)
        return out

    def _dedupe_titles_fuzzy(self, results: List[Dict]) -> List[Dict]:
        """Lightweight duplicate detection using normalized titles."""
        def norm(t: str) -> str:
            t = (t or "").lower().strip()
            t = re.sub(r"[^a-z0-9\s]", " ", t)
            t = re.sub(r"\s+", " ", t)
            return t

        seen = set()
        out = []
        for r in results:
            t = norm(r.get("title") or "")
            if not t:
                out.append(r)
                continue
            key = t[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _filter_trusted(self, results: List[Dict]) -> List[Dict]:
        out = []
        for r in results:
            url = (r.get("url") or "").strip()
            if url and self._is_trustworthy(url):
                out.append(r)
        return out

    def _cohere_rerank(self, query: str, results: List[Dict], top_n: int = 3) -> List[Dict]:
        """Corrective reranking using Cohere, if API key is available."""
        if not self.cohere_api_key or not results:
            return results[:top_n]

        try:
            import cohere
            co = cohere.Client(self.cohere_api_key)
            docs = [(r.get("content") or "")[:1200] for r in results]
            reranked = co.rerank(query=query, documents=docs, top_n=min(top_n, len(docs)))
            picked = [results[item.index] for item in reranked.results]
            return picked
        except Exception as e:
            print(f"   ⚠️ Cohere rerank failed: {e}")
            return results[:top_n]

    def _format_context(self, results: List[Dict]) -> Tuple[str, List[Dict]]:
        ctx = ""
        citations = []
        for r in results:
            title = (r.get("title") or "Unknown").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()
            if not url:
                continue
            ctx += f"\n--- SOURCE: {title} ({url}) ---\n{content[:700]}...\n"
            citations.append({"title": title, "url": url, "content": content})
        return ctx.strip(), citations

    def _inject_citations(self, text: str, citations: List[Dict]) -> str:
        if "--- SOURCE:" in text:
            return text
        if not citations:
            return text
        paras = [p for p in text.split('\n\n') if len(p.strip()) > 50]
        out = []
        for i, p in enumerate(paras):
            out.append(p.strip())
            if i < len(citations):
                c = citations[i]
                out.append(f"--- SOURCE: {c['title']} ({c['url']}) ---")
        return "\n\n".join(out)

    def _synthesize(self, query: str, context: str, citations: List[Dict]) -> str:
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

        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=850)
        if "--- SOURCE:" not in out:
            out = self._inject_citations(out, citations)
        return out

    def analyze(self, query: str, prior_analysis: str = "", metadata: Dict = {}) -> str:
        print(f"\n🌐 News Desk analyzing: '{query}'")

        # 1) Step-back query
        step_back = self._step_back_query(query)

        # 2) HyDE expansion
        hyde = self._hyde_brief(query, prior_analysis)
        hyde_queries = self._hyde_queries_from_brief(hyde, max_queries=2)

        year = datetime.now().year
        meta_years = metadata.get("years", [year])
        target_year = meta_years[0] if meta_years else year

        direct = f"{query} news {target_year}"

        # 3) Search: direct mandatory, step-back + hyde optional
        queries = [direct]
        if step_back and len(step_back) > 5:
            queries.append(step_back)
        for q in hyde_queries:
            if isinstance(q, str) and len(q) > 5:
                queries.append(q)

        # Dedupe queries
        queries = list(dict.fromkeys([q.strip() for q in queries if q.strip()]))
        print(f"   🔍 Queries: {queries}")

        all_results: List[Dict] = []
        for q in queries:
            all_results.extend(self._tavily_search(q, max_results=8))

        all_results = self._dedupe(all_results)
        all_results = self._dedupe_titles_fuzzy(all_results)

        # 4) Trust filter first (precision), but keep an escape hatch
        trusted = self._filter_trusted(all_results)

        # 5) Corrective reranking (Cohere), fallback to top results if unavailable
        picked = self._cohere_rerank(query, trusted if trusted else all_results, top_n=3)

        if not picked:
            return "## Web Research\n\nNo recent reliable news found."

        context, citations = self._format_context(picked)
        return self._synthesize(query, context, citations)


if __name__ == "__main__":
    agent = WebSearchAgent()
    print(agent.analyze("Microsoft risks 2026"))

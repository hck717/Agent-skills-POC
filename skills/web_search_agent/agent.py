#!/usr/bin/env python3
""" 
Web Search Agent ("News Desk")

Goal: Find unknown unknowns fast.
Architecture: Step-Back Prompting + HyDE expansion + corrective reranking.

- Step-back: broaden the question to capture macro/sector/ticker catalysts.
- HyDE: generate a hypothetical news brief to create intent-rich search queries.
- Filtering: dedupe + rerank (Cohere rerank optional; local LLM rerank fallback).

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

    # -------------------------
    # Helper: Clean DeepSeek Output
    # -------------------------
    def _clean_think(self, text: str) -> str:
        """Removes <think>...</think> blocks from DeepSeek output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _ollama_chat(self, messages: List[Dict], temperature: float = 0.0, num_predict: int = 400) -> str:
        resp = ollama.chat(
            model=self.ollama_model,
            messages=messages,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        return self._clean_think(content)

    # -------------------------
    # Step-back + HyDE
    # -------------------------

    def _step_back_query(self, query: str) -> str:
        prompt = f"""Rewrite the user question into a broader 'step-back' web-search question that helps find recent market-moving news.

User question: {query}

Rules:
- Output ONE sentence.
- Include timeframe hints like 'today', 'this week', 'recent' when relevant.
- Avoid adding facts.
"""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=120)
        return out.strip().strip('"')

    def _hyde_brief(self, query: str, prior_analysis: str = "") -> str:
        prompt = f"""Write a hypothetical (fake) news brief (6-10 sentences) that would answer the user's question well.
This is for search expansion only.

User question: {query}

If provided, you may use this prior context only to infer what topics to look for (do not treat as facts):
{prior_analysis[:600]}

Output:
- Plain text only.
- No citations.
- Include likely keywords: company, products, catalysts, regulation, competitors, macro.
"""
        return self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.2, num_predict=350)

    def _hyde_queries_from_brief(self, hyde_text: str, max_queries: int = 2) -> List[str]:
        prompt = f"""Convert the hypothetical news brief into {max_queries} concise web search queries.

Brief:
{hyde_text}

Rules:
- Output valid JSON list of strings ONLY.
- Example: ["Microsoft cloud revenue 2026", "Azure AI strategy risks"]
- Each query <= 12 words.
- Include timeframe hint (2025, 2026, today, Q1 2026) when relevant.
"""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=200)
        
        # Robust Parsing
        try:
            # 1. Try finding JSON block
            match = re.search(r'\[.*\]', out, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return [str(x) for x in data if isinstance(x, str)][:max_queries]
            
            # 2. Fallback: Split lines if JSON fails
            lines = [l.strip().strip('- "') for l in out.splitlines() if l.strip()]
            valid_lines = [l for l in lines if len(l) > 5 and not l.startswith('[')]
            if valid_lines:
                return valid_lines[:max_queries]
            
        except Exception as e:
            print(f"   ⚠️ HyDE Parse Error: {e}")
        
        # 3. Ultimate Fallback
        return [hyde_text[:80]] if hyde_text else []

    # -------------------------
    # Search + merge
    # -------------------------

    def _tavily_search(self, query: str, max_results: int = 6) -> List[Dict]:
        if not query or len(query) < 3: return []
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

    def _dedupe_results(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in results:
            url = (r.get("url") or "").strip()
            key = url.lower() if url else (r.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    # -------------------------
    # Rerank (optional)
    # -------------------------

    def _cohere_rerank(self, query: str, results: List[Dict], top_n: int = 6) -> List[Dict]:
        if not self.cohere_api_key or not results:
            return results

        docs = []
        for r in results:
            title = r.get("title") or ""
            content = r.get("content") or ""
            docs.append(f"{title}\n{content}"[:1500])

        try:
            payload = {
                "model": "rerank-english-v3.0",
                "query": query,
                "documents": docs,
                "top_n": min(top_n, len(docs)),
            }
            headers = {
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post("https://api.cohere.com/v1/rerank", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            order = [x["index"] for x in data.get("results", [])]
            reranked = [results[i] for i in order if i < len(results)]
            return reranked
        except Exception as e:
            print(f"   ⚠️ Cohere rerank failed, falling back to LLM rerank: {e}")
            return results

    def _llm_rerank(self, query: str, results: List[Dict], top_n: int = 6) -> List[Dict]:
        if not results:
            return []

        # Keep short for speed
        candidates = results[: min(len(results), 10)]
        items = []
        for i, r in enumerate(candidates):
            items.append({
                "i": i,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:260],
            })

        prompt = f"""You are a reranker for equity research news.
Rank the following search results for answering the query.

Query: {query}

Return JSON list of indices in best-first order.
Results:
{json.dumps(items, ensure_ascii=False)}
"""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=120)
        try:
            # Clean think tags again just in case
            match = re.search(r'\[.*\]', out, re.DOTALL)
            if match:
                order = json.loads(match.group(0))
                order = [int(x) for x in order if isinstance(x, (int, float, str))]
                seen = set()
                reranked = []
                for idx in order:
                    if idx in seen or idx < 0 or idx >= len(candidates):
                        continue
                    seen.add(idx)
                    reranked.append(candidates[idx])
                # append leftovers
                for i in range(len(candidates)):
                    if i not in seen:
                        reranked.append(candidates[i])
                return reranked[:top_n]
        except Exception:
            return candidates[:top_n]
        return candidates[:top_n] # Fallback

    # -------------------------
    # Synthesis with citations
    # -------------------------

    def _format_context(self, results: List[Dict]) -> Tuple[str, List[Dict]]:
        ctx = ""
        citations = []
        for r in results:
            title = (r.get("title") or "Unknown").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()
            if not url:
                continue
            ctx += f"\n--- SOURCE: {title} ({url}) ---\n{content}\n"
            citations.append({"title": title, "url": url, "content": content})
        return ctx.strip(), citations

    def _inject_citations(self, text: str, citations: List[Dict]) -> str:
        if "--- SOURCE:" in text:
            return text
        if not citations:
            return text

        paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        out = []
        ci = 0
        for p in paras:
            out.append(p)
            if ci < len(citations):
                c = citations[ci]
                out.append(f"--- SOURCE: {c['title']} ({c['url']}) ---")
                ci += 1
        return "\n\n".join(out)

    def _synthesize(self, query: str, context: str, citations: List[Dict], prior_analysis: str = "") -> str:
        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = f"""You are the News Desk for an equity research team.
Date: {current_date}

Task: Identify unknown-unknowns and recent catalysts relevant to the user's question.

Rules:
- Use ONLY the provided web context.
- Write 4-7 short paragraphs.
- After EVERY paragraph, append exactly one SOURCE line copied from the context in the form:
  --- SOURCE: Title (URL) ---
- If the context is insufficient, say what is missing.

USER QUESTION:
{query}

WEB CONTEXT:
{context}

PRIOR (optional, treat as background, not as facts):
{prior_analysis[:600]}
"""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=900)
        out = self._inject_citations(out, citations)
        return out

    # -------------------------
    # Public API
    # -------------------------

    def analyze(self, query: str, prior_analysis: str = "", metadata: Dict = {}) -> str:
        print(f"\n🌐 News Desk analyzing: '{query}'")

        step_back = self._step_back_query(query)
        hyde = self._hyde_brief(query, prior_analysis)
        hyde_queries = self._hyde_queries_from_brief(hyde, max_queries=2)

        # Always include a time-aware direct query too
        year = datetime.now().year
        # Use Metadata if available
        meta_years = metadata.get("years", [year])
        target_year = meta_years[0] if meta_years else year
        
        direct = f"{query} latest news {target_year}"

        print(f"   🧭 Step-back: {step_back}")
        print(f"   🧪 HyDE queries: {hyde_queries}")

        all_results = []
        queries_to_run = [direct, step_back] + hyde_queries
        # Dedupe queries
        queries_to_run = list(set([q for q in queries_to_run if q and len(q) > 3]))

        for q in queries_to_run:
            all_results.extend(self._tavily_search(q, max_results=5))

        all_results = self._dedupe_results(all_results)

        # Rerank
        ranked = self._cohere_rerank(query, all_results, top_n=8)
        if ranked == all_results:
            ranked = self._llm_rerank(query, all_results, top_n=8)

        context, citations = self._format_context(ranked)
        if not citations:
            return "## Web Research\n\nNo web results found."

        analysis = self._synthesize(query, context, citations, prior_analysis)
        return analysis

    def test_connection(self) -> bool:
        try:
            self.tavily.search(query="test", max_results=1)
            ollama.chat(model=self.ollama_model, messages=[{"role": "user", "content": "test"}], options={"num_predict": 10})
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False


if __name__ == "__main__":
    agent = WebSearchAgent()
    print(agent.analyze("Why is AAPL down today?"))

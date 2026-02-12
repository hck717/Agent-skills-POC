#!/usr/bin/env python3
""" 
Web Search Agent v2 ("News Desk") - Enhanced

Improvements:
1. ✅ Streaming synthesis support
2. ✅ Async parallel search (multiple queries simultaneously)
3. ✅ Dynamic temperature tuning
4. ✅ Multi-ticker comparison support
5. ✅ Enhanced HyDE with multiple hypothetical documents
6. ✅ Improved corrective filtering with confidence scores

Architecture: Step-Back Prompting + HyDE expansion + Corrective Reranking
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Optional, Tuple, AsyncIterator, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import ollama
from tavily import TavilyClient


class WebSearchAgentV2:
    TRUSTED_DOMAINS = [
        "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com", "ft.com", 
        "techcrunch.com", "forbes.com", "marketwatch.com", "yahoo.com/finance",
        "investopedia.com", "businessinsider.com", "nytimes.com", "seekingalpha.com",
        "theverge.com", "sec.gov", "investor.com"
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
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=6)

        print(f"✅ Web Search Agent v2 initialized (News Desk, model={ollama_model})")
        print(f"   - Streaming: Enabled")
        print(f"   - Parallel Search: Enabled")
        print(f"   - Dynamic Temperature: Enabled")
        print(f"   - Multi-Ticker: Enabled")

    def _clean_think(self, text: str) -> str:
        """Removes <think>...</think> blocks from DeepSeek output."""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean = clean.replace("<think>", "").replace("</think>", "")
        clean = re.sub(r"^(ores|okay|alright|sure)[\s\n:,-]+", "", clean.strip(), flags=re.IGNORECASE)
        return clean.strip()

    def _calculate_query_complexity(self, query: str) -> float:
        """
        Calculate query complexity for dynamic temperature tuning.
        
        Returns: 0.0 (simple) to 1.0 (complex)
        """
        score = 0.0
        
        # Length
        words = len(query.split())
        score += min(words / 40.0, 0.3)
        
        # Comparative keywords
        comparative = ['compare', 'versus', 'vs', 'difference', 'better', 'contrast']
        if any(kw in query.lower() for kw in comparative):
            score += 0.3
        
        # Multiple entities (tickers)
        tickers_mentioned = len(re.findall(r'\b[A-Z]{2,5}\b', query))
        score += min(tickers_mentioned * 0.15, 0.3)
        
        # Temporal complexity
        if 'trend' in query.lower() or 'historical' in query.lower():
            score += 0.1
        
        return min(score, 1.0)

    def _adaptive_temperature(self, query: str, use_case: str = "search") -> float:
        """
        Dynamic temperature based on query complexity and use case.
        
        Use cases:
        - 'search': Query expansion (0.0-0.2)
        - 'synthesis': Final output (0.0-0.15)
        - 'hyde': Hypothetical generation (0.2-0.4)
        """
        complexity = self._calculate_query_complexity(query)
        
        if use_case == "search":
            # Query expansion: conservative
            temp = 0.0 + (complexity * 0.2)
        elif use_case == "synthesis":
            # Final output: very precise for citations
            temp = 0.0 + (complexity * 0.15)
        elif use_case == "hyde":
            # Hypothetical generation: more creative
            temp = 0.2 + (complexity * 0.2)
        else:
            temp = 0.1
        
        print(f"   🌡️ [{use_case}] Complexity: {complexity:.2f} → Temp: {temp:.2f}")
        return temp

    def _ollama_chat(self, messages: List[Dict], temperature: float = 0.0, num_predict: int = 400) -> str:
        """Synchronous Ollama chat."""
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

    async def _ollama_stream_async(self, prompt: str, temperature: float = 0.0) -> AsyncIterator[str]:
        """Async streaming generation."""
        def _stream():
            try:
                stream = ollama.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    stream=True,
                    options={"temperature": temperature, "num_predict": 1000}
                )
                
                in_think = False
                for chunk in stream:
                    token = chunk.get('response', '')
                    
                    if '<think>' in token:
                        in_think = True
                    if '</think>' in token:
                        in_think = False
                        continue
                    
                    if not in_think and token and '<think>' not in token:
                        yield token
            except Exception as e:
                yield f"\nStreaming Error: {e}"
        
        loop = asyncio.get_event_loop()
        for token in await loop.run_in_executor(self.executor, lambda: list(_stream())):
            yield token

    def _step_back_query(self, query: str, temperature: float) -> str:
        """Step-back prompting to broaden context."""
        prompt = f"""Rewrite: '{query}' into a broader search query for recent market news/catalysts.
Output ONE sentence. No quotes."""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=temperature, num_predict=120)
        return out.strip().strip('"')

    def _hyde_brief(self, query: str, prior_analysis: str, temperature: float) -> str:
        """Generate hypothetical document (HyDE)."""
        prompt = f"""Write a fake 3-sentence news brief answering: {query}
Context: {prior_analysis[:300]}
Plain text only."""
        return self._ollama_chat([{"role": "user", "content": prompt}], temperature=temperature, num_predict=250)

    def _hyde_queries_from_brief(self, hyde_text: str, num_queries: int = 2) -> List[str]:
        """Extract search queries from HyDE brief."""
        prompt = f"""Extract {num_queries} search queries from this brief:
{hyde_text}
Output ONLY valid JSON list of strings."""
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=180)

        queries: List[str] = []
        try:
            match = re.search(r'\[.*\]', out, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    queries = [q for q in parsed if isinstance(q, str)]
        except:
            pass

        # Sanitization
        cleaned = []
        for q in queries:
            q2 = q.replace("\n", " ").strip()
            q2 = re.sub(r"\s+", " ", q2)
            if 6 <= len(q2) <= 150:
                if not any(bad in q2.lower() for bad in ['fake', 'halluc', 'ores']):
                    cleaned.append(q2)

        if cleaned:
            return cleaned[:num_queries]
        return [hyde_text[:80].replace("\n", " ")] if hyde_text else []

    async def _tavily_search_async(self, query: str, max_results: int = 8) -> List[Dict]:
        """Async Tavily search."""
        if not query or len(query) < 3:
            return []
        
        def _search():
            try:
                resp = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results,
                    include_answer=False,
                )
                return resp.get("results", []) or []
            except Exception as e:
                print(f"   ❌ Tavily error for '{query}': {e}")
                return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _search)

    def _is_trustworthy(self, url: str) -> bool:
        """Check if URL is from trusted domain."""
        if not url:
            return False
        domain = url.lower().replace("https://", "").replace("http://", "").split('/')[0]
        return any(trusted in domain for trusted in self.TRUSTED_DOMAINS)

    def _dedupe_by_url(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate URLs."""
        seen = set()
        out = []
        for r in results:
            url = (r.get("url") or "").strip()
            if url and url not in seen:
                seen.add(url)
                out.append(r)
        return out

    def _dedupe_by_title_fuzzy(self, results: List[Dict]) -> List[Dict]:
        """Remove similar titles (fuzzy matching)."""
        def norm(t: str) -> str:
            t = (t or "").lower().strip()
            t = re.sub(r"[^a-z0-9\s]", " ", t)
            t = re.sub(r"\s+", " ", t)
            return t

        seen = set()
        out = []
        for r in results:
            title_norm = norm(r.get("title") or "")[:80]
            if not title_norm or title_norm in seen:
                continue
            seen.add(title_norm)
            out.append(r)
        return out

    def _cohere_rerank(self, query: str, results: List[Dict], top_n: int = 5) -> List[Tuple[Dict, float]]:
        """
        Corrective reranking with confidence scores.
        
        Returns: List of (result, confidence_score) tuples
        """
        if not self.cohere_api_key or not results:
            return [(r, 0.5) for r in results[:top_n]]

        try:
            import cohere
            co = cohere.Client(self.cohere_api_key)
            docs = [(r.get("content") or "")[:1500] for r in results]
            reranked = co.rerank(query=query, documents=docs, top_n=min(top_n, len(docs)), return_documents=False)
            
            scored_results = []
            for item in reranked.results:
                result = results[item.index]
                confidence = item.relevance_score
                scored_results.append((result, confidence))
            
            print(f"   🎯 Cohere Rerank: Top confidence = {scored_results[0][1]:.3f}")
            return scored_results
        except Exception as e:
            print(f"   ⚠️ Cohere rerank failed: {e}")
            return [(r, 0.5) for r in results[:top_n]]

    def _format_context(self, results: List[Tuple[Dict, float]]) -> Tuple[str, List[Dict]]:
        """Format search results into context with confidence annotations."""
        ctx = ""
        citations = []
        
        for result, confidence in results:
            title = (result.get("title") or "Unknown").strip()
            url = (result.get("url") or "").strip()
            content = (result.get("content") or "").strip()
            
            if not url:
                continue
            
            confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
            ctx += f"\n--- SOURCE [{confidence_label} Confidence]: {title} ({url}) ---\n{content[:800]}...\n"
            citations.append({"title": title, "url": url, "content": content, "confidence": confidence})
        
        return ctx.strip(), citations

    def _inject_citations(self, text: str, citations: List[Dict]) -> str:
        """Inject citations if LLM failed to preserve them."""
        if "--- SOURCE" in text:
            return text
        if not citations:
            return text
        
        paras = [p for p in text.split('\n\n') if len(p.strip()) > 50]
        out = []
        for i, p in enumerate(paras):
            out.append(p.strip())
            if i < len(citations):
                c = citations[i]
                conf_label = "High" if c.get('confidence', 0) > 0.7 else "Medium"
                out.append(f"--- SOURCE [{conf_label}]: {c['title']} ({c['url']}) ---")
        return "\n\n".join(out)

    async def _synthesize_streaming(
        self, 
        query: str, 
        context: str, 
        citations: List[Dict],
        temperature: float
    ) -> AsyncIterator[str]:
        """Streaming synthesis for real-time output."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""You are a Financial News Desk.
Date: {current_date}

Task: Write a market update based on Web Context below.

Rules:
1. Use ONLY the provided Web Context
2. Write 3-4 concise paragraphs
3. After EVERY paragraph, add: --- SOURCE: Title (URL) ---
4. Focus on facts, numbers, dates
5. Multi-company: Clearly separate by ticker

Question: {query}

Web Context:
{context}

Answer:"""
        
        has_citations = False
        buffer = ""
        
        async for token in self._ollama_stream_async(prompt, temperature):
            buffer += token
            if "--- SOURCE" in buffer:
                has_citations = True
            yield token
        
        # Inject citations if missing
        if not has_citations:
            yield "\n\n## Sources\n\n"
            for c in citations:
                conf = c.get('confidence', 0)
                conf_label = "High" if conf > 0.7 else "Medium" if conf > 0.4 else "Low"
                yield f"- [{conf_label}] {c['title']} ({c['url']})\n"

    def _synthesize_blocking(
        self, 
        query: str, 
        context: str, 
        citations: List[Dict],
        temperature: float
    ) -> str:
        """Blocking synthesis (fallback)."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""Financial News Desk
Date: {current_date}

Task: Market update from Web Context

Rules:
1. Use ONLY Web Context
2. 3-4 paragraphs
3. After each paragraph: --- SOURCE: Title (URL) ---
4. Facts, numbers, dates

Question: {query}

Web Context:
{context}"""
        
        out = self._ollama_chat([{"role": "user", "content": prompt}], temperature=temperature, num_predict=1000)
        
        if "--- SOURCE" not in out:
            out = self._inject_citations(out, citations)
        
        return out

    async def analyze_async(
        self, 
        query: str, 
        tickers: Union[str, List[str], None] = None,
        prior_analysis: str = "", 
        meta Dict = {},
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Async analysis with parallel search and streaming.
        
        Args:
            query: User question
            tickers: Optional ticker(s) for focused search
            prior_analysis: Context from other agents
            meta Additional context (years, etc.)
            stream: Enable streaming output
        """
        print(f"\n🌐 News Desk v2 analyzing: '{query}'")
        
        # Normalize tickers
        if isinstance(tickers, str):
            tickers = [tickers]
        elif tickers is None:
            tickers = []
        
        # Dynamic temperatures
        temp_search = self._adaptive_temperature(query, "search")
        temp_hyde = self._adaptive_temperature(query, "hyde")
        temp_synthesis = self._adaptive_temperature(query, "synthesis")
        
        # 1) Query expansion (parallel)
        step_back_task = asyncio.create_task(
            asyncio.to_thread(self._step_back_query, query, temp_search)
        )
        hyde_task = asyncio.create_task(
            asyncio.to_thread(self._hyde_brief, query, prior_analysis, temp_hyde)
        )
        
        step_back, hyde = await asyncio.gather(step_back_task, hyde_task)
        hyde_queries = self._hyde_queries_from_brief(hyde, num_queries=2)
        
        # 2) Build query list
        year = datetime.now().year
        meta_years = metadata.get("years", [year])
        target_year = meta_years[0] if meta_years else year
        
        queries = []
        
        # Direct query (mandatory)
        if tickers:
            for ticker in tickers:
                queries.append(f"{ticker} {query} news {target_year}")
        else:
            queries.append(f"{query} news {target_year}")
        
        # Step-back (optional)
        if step_back and len(step_back) > 5:
            queries.append(step_back)
        
        # HyDE queries (optional)
        for hq in hyde_queries:
            if isinstance(hq, str) and 6 <= len(hq) <= 150:
                queries.append(hq)
        
        # Dedupe
        queries = list(dict.fromkeys([q.strip() for q in queries if q.strip()]))
        print(f"   🔍 Parallel Searches: {len(queries)} queries")
        for i, q in enumerate(queries, 1):
            print(f"      {i}. {q}")
        
        # 3) Parallel search
        search_tasks = [self._tavily_search_async(q, max_results=8) for q in queries]
        all_results_list = await asyncio.gather(*search_tasks)
        
        # Flatten
        all_results = []
        for results in all_results_list:
            all_results.extend(results)
        
        # 4) Filtering pipeline
        all_results = self._dedupe_by_url(all_results)
        all_results = self._dedupe_by_title_fuzzy(all_results)
        
        print(f"   🧹 After deduplication: {len(all_results)} results")
        
        # Trust filter (precision)
        trusted = [r for r in all_results if self._is_trustworthy(r.get('url', ''))]
        print(f"   ✅ Trusted sources: {len(trusted)} results")
        
        # 5) Corrective reranking
        candidates = trusted if trusted else all_results
        scored_results = self._cohere_rerank(query, candidates, top_n=5)
        
        if not scored_results:
            return "## Web Research\n\nNo recent reliable news found."
        
        # 6) Format context
        context, citations = self._format_context(scored_results)
        
        # 7) Synthesis
        print(f"   📝 Synthesizing from {len(citations)} sources...")
        
        if stream:
            return self._synthesize_streaming(query, context, citations, temp_synthesis)
        else:
            return self._synthesize_blocking(query, context, citations, temp_synthesis)

    def analyze(
        self, 
        query: str, 
        tickers: Union[str, List[str], None] = None,
        prior_analysis: str = "", 
        meta Dict = {},
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Synchronous wrapper for analyze_async.
        
        Maintains backward compatibility.
        """
        return asyncio.run(self.analyze_async(query, tickers, prior_analysis, metadata, stream))


if __name__ == "__main__":
    # Demo
    agent = WebSearchAgentV2()
    
    # Single ticker
    result = agent.analyze("Microsoft AI risks", tickers="MSFT")
    print(result)
    
    # Multi-ticker comparison
    result = agent.analyze(
        "Compare AI strategies",
        tickers=["MSFT", "GOOGL"],
        stream=False
    )
    print(result)

#!/usr/bin/env python3
""" 
Web Search Agent - HyDE Enhanced ("News Desk") 🌐

Goal: Find "Unknown Unknowns" fast with maximum precision.

Architecture: 
1. Step-Back Prompting (broaden context)
2. HyDE (Hypothetical Document Embeddings) - search by semantic similarity
3. Corrective Reranking - filter noise with advanced reranker

Outputs MUST preserve SOURCE markers:
--- SOURCE: Title (https://...) ---
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

import ollama
from tavily import TavilyClient
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from project root
load_dotenv(Path(__file__).parent.parent.parent / '.env')


class WebSearchAgentHyDE:
    """Enhanced Web Search Agent with HyDE and Step-Back Prompting."""
    
    TRUSTED_DOMAINS = [
        "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com", "ft.com", 
        "techcrunch.com", "forbes.com", "marketwatch.com", "yahoo.com/finance",
        "investopedia.com", "businessinsider.com", "nytimes.com", "seekingalpha.com",
        "theverge.com", "sec.gov", "barrons.com", "economist.com"
    ]
    
    # Noise patterns for clickbait/low-quality detection
    NOISE_PATTERNS = [
        r"\d+\s+(shocking|amazing|incredible|unbelievable)",  # "10 shocking..."
        r"you\s+won't\s+believe",
        r"click\s+here",
        r"(must|need\s+to)\s+(see|read|watch)",
        r"\d+\s+(tips|tricks|hacks|secrets)",
        r"(doctors|experts)\s+hate\s+(this|him|her)",
    ]

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        ollama_model: str = "deepseek-r1:8b",
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        cohere_api_key: Optional[str] = None,
    ):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found. Set env var or pass to constructor.")

        self.ollama_model = ollama_model
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url
        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")

        print(f"✅ Web Search Agent (HyDE Enhanced) initialized")
        print(f"   - Analysis Model: {ollama_model}")
        print(f"   - Embedding Model: {embed_model}")
        print(f"   - Step-Back: Enabled")
        print(f"   - HyDE Expansion: Enabled")
        print(f"   - Corrective Reranking: {bool(self.cohere_api_key)}")

    def _clean_think(self, text: str) -> str:
        """Removes <think>...</think> blocks and thought residue from DeepSeek output."""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean = clean.replace("<think>", "").replace("</think>", "")
        # Remove common thought-residue prefixes
        clean = re.sub(r"^(ores|okay|alright|sure|hmm|well)[\s\n:,-]+", "", clean.strip(), flags=re.IGNORECASE)
        return clean.strip()

    def _ollama_chat(self, messages: List[Dict], temperature: float = 0.0, num_predict: int = 400) -> str:
        """Call Ollama chat API with error handling."""
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

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embeddings using Ollama."""
        try:
            resp = ollama.embeddings(model=self.embed_model, prompt=text)
            embedding = resp.get("embedding") if isinstance(resp, dict) else resp.embedding
            return np.array(embedding) if embedding else None
        except Exception as e:
            print(f"   ⚠️ Embedding Error: {e}")
            return None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # ============================================================================
    # STEP 1: Step-Back Prompting (Broaden Context)
    # ============================================================================

    def _step_back_query(self, query: str) -> str:
        """
        Transform narrow query into broader contextual search.
        Uses Qwen for better reliability on simple expansion tasks.
        
        Example:
        - Input: "Why is AAPL down today?"
        - Output: "Apple stock decline news technology sector market trends February 2026"
        """
        # Simple, direct prompt - Qwen handles this better than DeepSeek
        prompt = f"""Rewrite this search query to be broader and include related context.

Original: {query}

Broader version (one line, 5-30 words):"""
        
        try:
            # Use Qwen for simple expansion (faster and more reliable)
            resp = ollama.chat(
                model="qwen2.5:7b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 80}
            )
            out = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            out = self._clean_think(out).strip().strip('"').strip("'")
            
            # Validate length
            if 5 <= len(out) <= 300:
                print(f"   📚 Step-Back (Qwen): {out}")
                return out
            else:
                print(f"   ⚠️ Step-back invalid length ({len(out)}), using original")
                return query
                
        except Exception as e:
            print(f"   ⚠️ Step-back failed ({e}), using original query")
            return query

    # ============================================================================
    # STEP 2: HyDE (Hypothetical Document Embeddings)
    # ============================================================================

    def _generate_hyde_document(self, query: str, prior_analysis: str = "") -> str:
        """
        Generate a FAKE news article that would answer the query.
        Uses Qwen for better creative hallucination (more reliable than DeepSeek for this task).
        
        Example:
        - Query: "Why is AAPL down today?"
        - HyDE Output: "Apple Inc. shares declined 3.2% in morning trading following 
          disappointing Q1 iPhone sales figures reported by supply chain analysts. 
          The tech giant faces headwinds from weakening consumer demand in China..."
        """
        current_date = datetime.now().strftime("%B %d, %Y")
        
        context_snippet = ""
        if prior_analysis:
            context_snippet = f"\nContext: {prior_analysis[:300]}"
        
        # Simplified prompt for Qwen - more direct, less meta
        prompt = f"""Write a realistic financial news article answering this query.

Date: {current_date}
Query: {query}{context_snippet}

Write 4-5 sentences like a Bloomberg journalist. Include specific numbers, percentages, company names, dates. Be factual and professional.

Article:"""
        
        try:
            # Use Qwen for creative hallucination (better than DeepSeek)
            resp = ollama.chat(
                model="qwen2.5:7b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 300}  # Higher temp for creativity
            )
            hyde_doc = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            hyde_doc = self._clean_think(hyde_doc)
            
            if hyde_doc and len(hyde_doc) > 100:
                print(f"   🎭 HyDE Document Generated (Qwen): {len(hyde_doc)} chars")
                return hyde_doc
            
            # Retry with even simpler prompt
            print(f"   ⚠️ HyDE too short ({len(hyde_doc)} chars), retrying...")
            simple_prompt = f"Write 3 sentences about: {query}. Include specific numbers and dates. Be direct."
            resp = ollama.chat(
                model="qwen2.5:7b",
                messages=[{"role": "user", "content": simple_prompt}],
                options={"temperature": 0.8, "num_predict": 150}
            )
            hyde_doc = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
            hyde_doc = self._clean_think(hyde_doc)
            
            if hyde_doc and len(hyde_doc) > 50:
                print(f"   ✅ HyDE Retry Successful (Qwen): {len(hyde_doc)} chars")
                return hyde_doc
            else:
                print(f"   ❌ HyDE generation failed after retry, will use direct search only")
                return ""
                
        except Exception as e:
            print(f"   ❌ HyDE generation error ({e}), will use direct search only")
            return ""

    def _hyde_embedding_search(self, hyde_document: str, search_results: List[Dict]) -> List[Dict]:
        """
        Rank search results by semantic similarity to the HyDE document.
        
        This finds articles that "look like" our hypothetical answer,
        which often better matches user intent than keyword matching.
        """
        if not hyde_document or not search_results:
            return search_results
        
        # Get HyDE embedding
        hyde_embedding = self._embed_text(hyde_document)
        if hyde_embedding is None:
            print(f"   ⚠️ HyDE embedding failed, skipping similarity ranking")
            return search_results
        
        # Calculate similarity scores for each result
        scored_results = []
        for result in search_results:
            content = result.get("content", "") or result.get("title", "")
            if not content:
                continue
            
            # Embed the search result
            result_embedding = self._embed_text(content[:1500])  # Limit for speed
            if result_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(hyde_embedding, result_embedding)
            scored_results.append((similarity, result))
        
        # Sort by similarity (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"   🎯 HyDE Ranking: Top scores = {[f'{s:.3f}' for s, _ in scored_results[:3]]}")
        
        # Return ranked results
        return [result for _, result in scored_results]

    # ============================================================================
    # STEP 3: Corrective Filtering (Remove Noise)
    # ============================================================================

    def _is_clickbait(self, title: str, content: str = "") -> bool:
        """Detect clickbait/low-quality content using pattern matching."""
        text = f"{title} {content}".lower()
        
        for pattern in self.NOISE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for excessive punctuation
        if title.count('!') > 2 or title.count('?') > 2:
            return True
        
        # Check for ALL CAPS words (more than 30% of title)
        words = title.split()
        if words:
            caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
            if caps_ratio > 0.3:
                return True
        
        return False

    def _calculate_quality_score(self, result: Dict) -> float:
        """
        Assign quality score to each search result.
        
        Scoring factors:
        - Domain trust (0-40 points)
        - Content length (0-20 points)
        - Recency (0-20 points)
        - Not clickbait (0-20 points)
        """
        score = 0.0
        
        # 1. Domain trust (40 points)
        url = result.get("url", "")
        if self._is_trustworthy(url):
            score += 40.0
        else:
            score += 10.0  # Penalty for untrusted domains
        
        # 2. Content length (20 points) - longer = more substantive
        content = result.get("content", "")
        content_len = len(content)
        if content_len > 800:
            score += 20.0
        elif content_len > 400:
            score += 15.0
        elif content_len > 200:
            score += 10.0
        else:
            score += 5.0
        
        # 3. Recency (20 points) - check if date mentioned
        current_year = datetime.now().year
        recent_keywords = [str(current_year), str(current_year - 1), "today", "this week", "latest"]
        title_content = f"{result.get('title', '')} {content}".lower()
        if any(kw in title_content for kw in recent_keywords):
            score += 20.0
        else:
            score += 10.0
        
        # 4. Not clickbait (20 points)
        if not self._is_clickbait(result.get("title", ""), content):
            score += 20.0
        else:
            score -= 10.0  # Penalty for clickbait
        
        return max(0.0, score)  # Ensure non-negative

    def _filter_by_quality(self, results: List[Dict], min_score: float = 50.0) -> List[Dict]:
        """Filter results by quality score threshold."""
        scored = [(self._calculate_quality_score(r), r) for r in results]
        filtered = [(score, r) for score, r in scored if score >= min_score]
        filtered.sort(key=lambda x: x[0], reverse=True)
        
        print(f"   📊 Quality Filter: {len(filtered)}/{len(results)} passed (min={min_score})")
        if filtered:
            print(f"      Top scores: {[f'{s:.1f}' for s, _ in filtered[:3]]}")
        
        return [r for _, r in filtered]

    def _cohere_rerank(self, query: str, results: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Advanced reranking using Cohere Rerank API.
        Significantly better than simple keyword matching.
        """
        if not self.cohere_api_key or not results:
            print(f"   ⚠️ Cohere API not available, using quality scores only")
            return results[:top_n]

        try:
            import cohere
            co = cohere.Client(self.cohere_api_key)
            
            # Prepare documents for reranking
            docs = [
                f"{r.get('title', '')}: {r.get('content', '')[:1200]}" 
                for r in results
            ]
            
            # Call Cohere Rerank API
            reranked = co.rerank(
                query=query,
                documents=docs,
                top_n=min(top_n, len(docs)),
                model="rerank-english-v2.0"
            )
            
            # Return reranked results
            picked = [results[item.index] for item in reranked.results]
            scores = [item.relevance_score for item in reranked.results]
            
            print(f"   🎯 Cohere Rerank: Top {len(picked)} results (scores: {[f'{s:.3f}' for s in scores]})")
            return picked
            
        except ImportError:
            print(f"   ⚠️ Cohere library not installed: pip install cohere")
            return results[:top_n]
        except Exception as e:
            print(f"   ⚠️ Cohere rerank failed: {e}")
            return results[:top_n]

    # ============================================================================
    # Search and Deduplication Utilities
    # ============================================================================

    def _tavily_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Execute Tavily search with error handling."""
        if not query or len(query) < 3:
            return []
        try:
            resp = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )
            results = resp.get("results", []) or []
            print(f"   🔍 Tavily: Found {len(results)} results for '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"   ❌ Tavily search error: {e}")
            return []

    def _is_trustworthy(self, url: str) -> bool:
        """Check if URL is from a trusted domain."""
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
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(r)
        return out

    def _dedupe_by_title(self, results: List[Dict]) -> List[Dict]:
        """Remove near-duplicate titles using fuzzy matching."""
        def normalize(t: str) -> str:
            t = (t or "").lower().strip()
            t = re.sub(r"[^a-z0-9\s]", " ", t)
            t = re.sub(r"\s+", " ", t)
            return t

        seen = set()
        out = []
        for r in results:
            norm_title = normalize(r.get("title") or "")
            if not norm_title:
                out.append(r)
                continue
            
            # Use first 60 chars as fingerprint
            fingerprint = norm_title[:60]
            if fingerprint in seen:
                continue
            
            seen.add(fingerprint)
            out.append(r)
        
        if len(results) != len(out):
            print(f"   🗑️ Dedupe: Removed {len(results) - len(out)} duplicate titles")
        
        return out

    # ============================================================================
    # Synthesis and Citation Formatting
    # ============================================================================

    def _format_context(self, results: List[Dict]) -> Tuple[str, List[Dict]]:
        """Format search results into context string with citation markers."""
        ctx = ""
        citations = []
        
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "Unknown").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()
            
            if not url:
                continue
            
            # Add source marker
            ctx += f"\n--- SOURCE {i}: {title} ({url}) ---\n{content[:800]}...\n"
            citations.append({"title": title, "url": url, "content": content})
        
        return ctx.strip(), citations

    def _inject_citations(self, text: str, citations: List[Dict]) -> str:
        """Inject citation markers if LLM failed to include them."""
        if "--- SOURCE" in text:
            return text  # Citations already present
        
        if not citations:
            return text
        
        # Split into paragraphs
        paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
        
        if not paras:
            return text
        
        # Inject citations after each paragraph
        out = []
        for i, para in enumerate(paras):
            out.append(para)
            if i < len(citations):
                c = citations[i]
                out.append(f"--- SOURCE: {c['title']} ({c['url']}) ---")
        
        return "\n\n".join(out)

    def _synthesize(self, query: str, context: str, citations: List[Dict]) -> str:
        """
        Generate final market update from filtered, reranked sources.
        
        CRITICAL: Must preserve citation markers.
        """
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""You are a Professional Financial News Analyst.

Date: {current_date}

Task: Write a concise market intelligence briefing based on the Web Sources below.

Rules:
1. Use ONLY information from the Web Sources provided
2. Write 3-5 clear, factual paragraphs
3. After EVERY paragraph, add the source marker: --- SOURCE: Title (URL) ---
4. Focus on facts, data, dates, and numbers (avoid speculation)
5. Maintain professional financial journalism tone
6. NO introductory phrases like "Based on sources" or "According to"

Question: {query}

Web Sources:
{context}

Market Briefing:"""

        output = self._ollama_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,  # Strict adherence to sources
            num_predict=1000
        )
        
        # Ensure citations are present
        if "--- SOURCE" not in output:
            print(f"   ⚠️ Citations missing from synthesis, injecting...")
            output = self._inject_citations(output, citations)
        
        return output

    # ============================================================================
    # Main Analysis Pipeline
    # ============================================================================

    def analyze(
        self, 
        query: str, 
        prior_analysis: str = "", 
        metadata={},
        use_hyde: bool = True,
        use_step_back: bool = True,
        top_n: int = 5
    ) -> str:
        # Hard cap at 5 sources maximum
        top_n = min(top_n, 5)
        # Hard cap at 5 sources maximum
        top_n = min(top_n, 5)
        """
        Execute full HyDE + Step-Back + Corrective Filtering pipeline.
        
        Args:
            query: User's research question
            prior_analysis: Context from previous agents (e.g., business analyst)
            meta Additional metadata (years, tickers, etc.)
            use_hyde: Enable HyDE expansion (default: True)
            use_step_back: Enable step-back prompting (default: True)
            top_n: Number of final sources to include (default: 5)
        
        Returns:
            Formatted market briefing with citations
        """
        print(f"\n🌐 News Desk (HyDE Enhanced) analyzing: '{query}'")
        print(f"   - HyDE: {use_hyde}")
        print(f"   - Step-Back: {use_step_back}")
        print(f"   - Target Sources: {top_n}")
        
        # ========================================================================
        # Phase 1: Query Transformation
        # ========================================================================
        
        queries = []
        
        # Add temporal context
        year = datetime.now().year
        meta_years = metadata.get("years", [year])
        target_year = meta_years[0] if meta_years else year
        
        # Direct query (always included)
        direct_query = f"{query} news {target_year}"
        queries.append(direct_query)
        
        # Step-Back query (broadens context)
        if use_step_back:
            step_back_query = self._step_back_query(query)
            if step_back_query and len(step_back_query) > 10:
                queries.append(step_back_query)
        
        # ========================================================================
        # Phase 2: Initial Search
        # ========================================================================
        
        all_results = []
        # Limit Tavily API calls: max 5 results per query
        for q in queries:
            results = self._tavily_search(q, max_results=5)
            all_results.extend(results)
        
        if not all_results:
            return "## Web Research\n\nNo recent news found for this query."
        
        # Deduplicate
        all_results = self._dedupe_by_url(all_results)
        all_results = self._dedupe_by_title(all_results)
        
        print(f"   📦 Total unique results: {len(all_results)}")
        
        # ========================================================================
        # Phase 3: HyDE Ranking (if enabled)
        # ========================================================================
        
        if use_hyde and len(all_results) > 0:
            hyde_doc = self._generate_hyde_document(query, prior_analysis)
            if hyde_doc:
                all_results = self._hyde_embedding_search(hyde_doc, all_results)
        
        # ========================================================================
        # Phase 4: Corrective Filtering
        # ========================================================================
        
        # Step 4a: Quality scoring and filtering
        filtered_results = self._filter_by_quality(all_results, min_score=50.0)
        
        if not filtered_results:
            print(f"   ⚠️ No results passed quality filter, relaxing threshold...")
            filtered_results = self._filter_by_quality(all_results, min_score=30.0)
        
        if not filtered_results:
            return "## Web Research\n\nNo reliable news sources found after quality filtering."
        
        # Step 4b: Advanced reranking (Cohere)
        final_results = self._cohere_rerank(query, filtered_results, top_n=top_n)
        
        if not final_results:
            return "## Web Research\n\nNo relevant news found after reranking."
        
        # ========================================================================
        # Phase 5: Synthesis
        # ========================================================================
        
        context, citations = self._format_context(final_results)
        
        print(f"   ✅ Final sources: {len(final_results)}")
        
        return self._synthesize(query, context, citations)


if __name__ == "__main__":
    # Example usage
    agent = WebSearchAgentHyDE()
    
    # Test queries
    queries = [
        "Why is AAPL down today?",
        "Microsoft AI risks 2026",
        "Tesla competition challenges"
    ]
    
    for q in queries:
        print("\n" + "="*80)
        result = agent.analyze(q, use_hyde=True, use_step_back=True, top_n=3)
        print("\n" + result)

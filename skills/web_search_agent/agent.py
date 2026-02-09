#!/usr/bin/env python3
"""
Web Search Agent - Supplements Business Analyst with current web information

Uses Tavily for web search + local Ollama for synthesis
"""

import os
from typing import List, Dict, Optional
import ollama
from tavily import TavilyClient


class WebSearchAgent:
    """
    Web Search Agent that supplements document-based analysis with current web info.
    
    Role:
    - Fills gaps from document analysis (e.g., current stock prices, recent news)
    - Provides market sentiment and analyst opinions
    - Supplements with competitor intelligence
    
    Always runs AFTER Business Analyst to identify what's missing.
    """
    
    def __init__(self, tavily_api_key: str = None, ollama_model: str = "qwen2.5:7b"):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found. Set via environment variable or pass to constructor.")
        
        self.ollama_model = ollama_model
        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        
        print(f"✅ Web Search Agent initialized (Model: {ollama_model})")
    
    def _search_web(self, query: str, max_results: int = 5) -> tuple[str, List[Dict]]:
        """
        Search web using Tavily and return context + citations
        
        Returns:
            (context_text, citations_list)
        """
        try:
            print(f"   🔍 Searching web: {query[:80]}...")
            
            response = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True  # Get Tavily's AI summary too
            )
            
            # Format context with SOURCE markers (matching Business Analyst format)
            context_text = ""
            citations = []
            
            for idx, result in enumerate(response.get('results', []), 1):
                title = result.get('title', 'Unknown')
                url = result.get('url', '')
                content = result.get('content', '')
                
                # 🔥 CRITICAL: Use SOURCE marker format for consistency
                context_text += f"\n--- SOURCE: {title} ({url}) ---\n{content}\n\n"
                
                citations.append({
                    'index': idx,
                    'title': title,
                    'url': url,
                    'content': content
                })
            
            print(f"   ✅ Found {len(citations)} web sources")
            return context_text, citations
            
        except Exception as e:
            print(f"   ❌ Search error: {str(e)}")
            return f"Search Error: {str(e)}", []
    
    def _synthesize_with_llm(self, query: str, search_context: str, prior_analysis: str = "") -> str:
        """
        Use local Ollama to synthesize search results
        
        Args:
            query: User's original question
            search_context: Web search results with SOURCE markers
            prior_analysis: Optional prior analysis from Business Analyst
        """
        
        system_prompt = """
You are a Web Research Specialist supplementing document-based equity research.

Your Role:
- Fill gaps in document analysis with current web information
- Provide recent news, analyst opinions, market sentiment
- Update with latest financial data not in 10-K filings
- Identify emerging risks or opportunities

CRITICAL CITATION RULES:
1. PRESERVE all "--- SOURCE: Title (URL) ---" markers in your analysis
2. After each point, include the SOURCE marker:
   
   Example:
   ## Recent Market Developments
   Apple's stock rose 15% following Q1 earnings beat.
   --- SOURCE: Bloomberg (https://bloomberg.com/...) ---
   
   Analysts raised price targets to $225 on strong iPhone demand.
   --- SOURCE: Reuters (https://reuters.com/...) ---

3. CITE FREQUENTLY - every factual claim needs a source
4. If prior document analysis exists, note how web info supplements it

Format:
- Use Markdown headers (##, ###)
- Keep analysis concise and factual
- Highlight what's NEW vs what's in documents
        """
        
        # Build prior analysis section separately to avoid f-string backslash issue
        prior_section = ""
        if prior_analysis:
            prior_section = f"### PRIOR DOCUMENT ANALYSIS:\n{prior_analysis}\n\n"
        
        user_prompt = f"""### WEB SEARCH CONTEXT (with SOURCE markers):
{search_context}

{prior_section}### USER QUESTION:
{query}

### YOUR SUPPLEMENTAL ANALYSIS:
Provide additional insights from web sources that complement the document analysis.
PRESERVE all SOURCE markers in your response.
        """
        
        try:
            print(f"   🤖 Synthesizing with {self.ollama_model}...")
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.3,
                    'num_predict': 2000
                }
            )
            
            result = response['message']['content']
            print(f"   ✅ Synthesis complete ({len(result)} chars)")
            return result
            
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            return f"## Web Research\n\n{search_context}\n\n**Note**: Synthesis failed, showing raw results."
    
    def analyze(self, query: str, prior_analysis: str = "") -> str:
        """
        Main analysis method - searches web and synthesizes results
        
        Args:
            query: User's research question
            prior_analysis: Optional prior analysis from Business Analyst
        
        Returns:
            Synthesized analysis with SOURCE markers for web citations
        """
        print(f"\n🌐 Web Search Agent analyzing: '{query}'")
        
        # Enhance query based on what's likely missing from documents
        enhanced_query = self._enhance_query(query, prior_analysis)
        
        # Search web
        search_context, citations = self._search_web(enhanced_query)
        
        if not citations:
            return "## Web Research\n\nNo web results found. Analysis limited to document sources."
        
        # Synthesize with LLM
        analysis = self._synthesize_with_llm(query, search_context, prior_analysis)
        
        return analysis
    
    def _enhance_query(self, query: str, prior_analysis: str = "") -> str:
        """
        Enhance query to focus on information likely missing from 10-K documents
        
        10-Ks are historical (6-12 months old), so focus on:
        - Recent news (last 3 months)
        - Current stock performance
        - Analyst opinions
        - Breaking developments
        """
        query_lower = query.lower()
        
        enhancements = []
        
        # Add temporal focus
        if "recent" not in query_lower and "latest" not in query_lower:
            enhancements.append("recent")
        
        # Add specific info types based on query
        if "compet" in query_lower:
            enhancements.append("market share")
            enhancements.append("analyst ratings")
        
        if "risk" in query_lower:
            enhancements.append("breaking news")
            enhancements.append("regulatory developments")
        
        if "financial" in query_lower or "revenue" in query_lower:
            enhancements.append("latest earnings")
            enhancements.append("analyst estimates")
        
        if "product" in query_lower:
            enhancements.append("product reviews")
            enhancements.append("market reception")
        
        # Construct enhanced query
        if enhancements:
            enhanced = f"{query} {' '.join(enhancements)} 2025 2026"
        else:
            enhanced = f"{query} latest news analysis 2025 2026"
        
        print(f"   📝 Enhanced query: {enhanced}")
        return enhanced
    
    def test_connection(self) -> bool:
        """
        Test both Tavily and Ollama connections
        """
        try:
            # Test Tavily
            print("🔌 Testing Tavily API...")
            self.tavily.search(query="test", max_results=1)
            print("   ✅ Tavily connected")
            
            # Test Ollama
            print("🔌 Testing Ollama...")
            ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': 'test'}],
                options={'num_predict': 10}
            )
            print("   ✅ Ollama connected")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Connection test failed: {str(e)}")
            return False


if __name__ == "__main__":
    import sys
    
    # Quick test
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ TAVILY_API_KEY not set")
        print("\nSet it with:")
        print("  export TAVILY_API_KEY='your-key'")
        sys.exit(1)
    
    agent = WebSearchAgent(tavily_api_key=tavily_key)
    
    if agent.test_connection():
        print("\n" + "="*60)
        result = agent.analyze("What are Apple's latest competitive challenges?")
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(result)

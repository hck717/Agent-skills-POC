#!/usr/bin/env python3
"""
Web Search Agent - Supplements Business Analyst with current web information

Uses Tavily for web search + local Ollama for synthesis
"""

import os
import re
from typing import List, Dict, Optional
import ollama
from tavily import TavilyClient
from datetime import datetime


class WebSearchAgent:
    """
    Web Search Agent that supplements document-based analysis with current web info.
    
    Role:
    - Fills gaps from document analysis (e.g., current stock prices, recent news)
    - Provides market sentiment and analyst opinions
    - Supplements with competitor intelligence
    
    Always runs AFTER Business Analyst to identify what's missing.
    """
    
    def __init__(self, tavily_api_key: str = None, ollama_model: str = "deepseek-r1:8b"):
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
    
    def _inject_web_citations(self, analysis: str, citations: List[Dict]) -> str:
        """
        🔥 POST-PROCESSING FIX: Inject web citations if LLM didn't preserve them
        """
        # Check if analysis already has citations
        if '--- SOURCE:' in analysis:
            citation_count = analysis.count('--- SOURCE:')
            print(f"   ✅ LLM preserved {citation_count} web citations")
            return analysis
        
        print("   ⚠️ LLM didn't preserve web citations - injecting them automatically")
        
        if not citations:
            print("   ❌ No citations available to inject")
            return analysis
        
        print(f"   📚 Found {len(citations)} web sources to distribute")
        
        # Split analysis into sections/paragraphs
        lines = analysis.split('\n')
        result_lines = []
        citation_idx = 0
        
        for i, line in enumerate(lines):
            result_lines.append(line)
            
            # Add citation after substantial paragraphs (not headers, not empty lines)
            if (line.strip() and 
                not line.startswith('#') and 
                len(line) > 100 and 
                citation_idx < len(citations) and
                i < len(lines) - 1):  # Don't add to last line
                
                cite = citations[citation_idx]
                result_lines.append(f"--- SOURCE: {cite['title']} ({cite['url']}) ---")
                print(f"   [Web Citation {citation_idx + 1}] {cite['title'][:50]}...")
                citation_idx += 1
        
        injected_analysis = '\n'.join(result_lines)
        final_count = injected_analysis.count('--- SOURCE:')
        print(f"   ✅ Injected {final_count} web citations into analysis")
        
        return injected_analysis
    
    def _synthesize_with_llm(self, query: str, search_context: str, citations: List[Dict], prior_analysis: str = "") -> str:
        """
        Use local Ollama to synthesize search results
        
        Args:
            query: User's original question
            search_context: Web search results with SOURCE markers
            citations: List of citation dictionaries
            prior_analysis: Optional prior analysis from Business Analyst
        """
        
        # Get current date for temporal context
        current_date = datetime.now().strftime("%B %Y")
        
        system_prompt = f"""
You are a Web Research Specialist for professional equity research analysis.
Current Date: {current_date}

Your Role:
- Supplement historical 10-K data with CURRENT market developments
- Provide recent news, analyst opinions, market sentiment from 2025-2026
- Update with latest financial data not in SEC filings
- Identify emerging risks or opportunities

⚠️ CRITICAL CITATION FORMAT ⚠️

You MUST output in this EXACT format:

[Your analysis paragraph - 2-4 sentences]
--- SOURCE: Article Title (https://url.com) ---

[Next analysis paragraph]
--- SOURCE: Article Title (https://url.com) ---

EXAMPLE OUTPUT YOU MUST FOLLOW:

## Recent Market Performance (Q4 2025 - Q1 2026)
Apple stock rose 12% in Q4 2025 following stronger-than-expected iPhone sales in China. The company reported record Services revenue of $23.1B, beating analyst estimates.
--- SOURCE: Bloomberg Markets (https://bloomberg.com/...) ---

Analysts raised price targets to $245 average, citing strong ecosystem growth and AI integration momentum heading into 2026.
--- SOURCE: Reuters Business (https://reuters.com/...) ---

## Competitive Landscape Updates
Samsung launched Galaxy S25 with advanced AI features in January 2026, intensifying competition in the premium smartphone segment. Early reviews highlight comparable camera quality to iPhone 15 Pro.
--- SOURCE: The Verge (https://theverge.com/...) ---

RULES:
1. Write 2-4 sentences per point
2. Add SOURCE line immediately after EVERY point
3. Use format: --- SOURCE: Title (URL) ---
4. Emphasize TEMPORAL CONTEXT: "As of 2026", "Recent reports", "Q1 2026"
5. Distinguish from historical 10-K data: "While 10-K shows... recent developments indicate..."
6. Be SPECIFIC with dates and timeframes
7. Focus on NEW information not in SEC filings
8. Keep response concise (~1200 tokens max)

Professional Tone:
- Concise, factual, data-driven
- Use specific metrics and numbers
- Cite analyst firms, dates, percentages
- Avoid speculation without attribution
        """
        
        # Build prior analysis section
        prior_section = ""
        if prior_analysis:
            # Extract key points from prior analysis (first 500 chars)
            prior_summary = prior_analysis[:500] + "..." if len(prior_analysis) > 500 else prior_analysis
            prior_section = f"""### PRIOR 10-K ANALYSIS (Historical Data):
{prior_summary}

### YOUR TASK: 
Supplement the above historical analysis with CURRENT web information (2025-2026).
Highlight what's NEW vs what's in the 10-K.

"""
        
        user_prompt = f"""### WEB SEARCH CONTEXT (Current Sources with SOURCE markers):
{search_context}

{prior_section}### USER QUESTION:
{query}

### YOUR SUPPLEMENTAL ANALYSIS:
Provide analysis following the EXACT citation format shown above.
Emphasize temporal context and recent developments.
PRESERVE all SOURCE markers in your response.
Keep response focused and concise (target ~1200 tokens).
        """
        
        try:
            print(f"   🤖 Synthesizing with {self.ollama_model}...")
            print(f"   ⚡ Token limit: 1200 (optimized for synthesis)")
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.0,  # 🔥 CRITICAL: Deterministic for citation preservation
                    'num_predict': 1200  # 🔥 FIX 2: Reduced from 2500 (52% reduction)
                }
            )
            
            result = response['message']['content']
            print(f"   ✅ Synthesis complete ({len(result)} chars)")
            
            # 🔥 POST-PROCESS: Inject citations if LLM failed
            result = self._inject_web_citations(result, citations)
            
            return result
            
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            # Fallback: Return raw context with citations preserved
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
        analysis = self._synthesize_with_llm(query, search_context, citations, prior_analysis)
        
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
        
        # Current date context
        current_year = datetime.now().year
        current_quarter = f"Q{(datetime.now().month-1)//3 + 1}"
        
        enhancements = []
        
        # Add temporal focus emphasizing CURRENT data
        if "recent" not in query_lower and "latest" not in query_lower:
            enhancements.append("latest")
            enhancements.append(f"{current_quarter} {current_year}")
        
        # Add specific info types based on query
        if "compet" in query_lower:
            enhancements.append("market developments")
            enhancements.append("analyst ratings")
            enhancements.append("competitive moves")
        
        if "risk" in query_lower:
            enhancements.append("breaking news")
            enhancements.append("regulatory updates")
            enhancements.append("analyst warnings")
        
        if "financial" in query_lower or "revenue" in query_lower or "performance" in query_lower:
            enhancements.append("latest earnings")
            enhancements.append("analyst estimates")
            enhancements.append("guidance updates")
        
        if "product" in query_lower or "development" in query_lower:
            enhancements.append("product launches")
            enhancements.append("technology updates")
            enhancements.append("innovation news")
        
        # Construct enhanced query with strong temporal emphasis
        if enhancements:
            enhanced = f"{query} {' '.join(enhancements)} {current_year}"
        else:
            enhanced = f"{query} latest developments {current_quarter} {current_year}"
        
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

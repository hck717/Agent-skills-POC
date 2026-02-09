#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Rule-based orchestration with LOCAL LLM synthesis for professional equity research.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import requests


class ActionType(Enum):
    """Types of actions the ReAct agent can take"""
    CALL_SPECIALIST = "call_specialist"
    SYNTHESIZE = "synthesize"
    FINISH = "finish"
    REFINE_QUERY = "refine_query"


@dataclass
class Thought:
    """Represents a reasoning step"""
    content: str
    iteration: int


@dataclass
class Action:
    """Represents an action to take"""
    action_type: ActionType
    agent_name: Optional[str] = None
    task_description: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class Observation:
    """Represents the result of an action"""
    action: Action
    result: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActTrace:
    """Complete trace of the ReAct loop"""
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    
    def add_thought(self, content: str, iteration: int):
        self.thoughts.append(Thought(content=content, iteration=iteration))
    
    def add_action(self, action: Action):
        self.actions.append(action)
    
    def add_observation(self, obs: Observation):
        self.observations.append(obs)
    
    def get_history_summary(self) -> str:
        """Get a summary of the ReAct history for context"""
        summary = []
        for i, (thought, action, obs) in enumerate(zip(self.thoughts, self.actions, self.observations), 1):
            summary.append(f"\n=== Iteration {i} ===")
            summary.append(f"Thought: {thought.content}")
            summary.append(f"Action: {action.action_type.value}")
            if action.agent_name:
                summary.append(f"  Agent: {action.agent_name}")
                summary.append(f"  Task: {action.task_description}")
            summary.append(f"Observation: {obs.result[:200]}..." if len(obs.result) > 200 else f"Observation: {obs.result}")
        return "\n".join(summary)
    
    def get_specialist_calls(self) -> List[str]:
        """Get list of specialist agents that were called"""
        return [action.agent_name for action in self.actions if action.action_type == ActionType.CALL_SPECIALIST and action.agent_name]


class OllamaClient:
    """Client for local Ollama LLM (no web search)"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        self.base_url = base_url
        self.model = model
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Send chat request to Ollama API"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 3500  # Increased for comprehensive reports
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def test_connection(self) -> bool:
        try:
            print("🔌 Testing Ollama connection...")
            messages = [{"role": "user", "content": "Say 'OK'."}]
            response = self.chat(messages, temperature=0.0)
            if response:
                print("✅ Ollama connected!")
                return True
            return False
        except Exception as e:
            print(f"❌ Ollama connection failed: {str(e)}")
            return False


class ReActOrchestrator:
    """Rule-based orchestrator with LOCAL synthesis (no web interference)"""
    
    SPECIALIST_AGENTS = {
        "business_analyst": {
            "description": "Analyzes 10-K filings, financial statements, competitive positioning using RAG",
            "keywords": ["10-K", "10-Q", "filing", "risk", "competitive", "financial"],
            "priority": 1  # Always first
        },
        "web_search_agent": {
            "description": "Supplements document analysis with current web info (news, market data, analyst opinions)",
            "keywords": ["recent", "current", "latest", "news", "market", "price"],
            "priority": 2  # Always after business_analyst
        },
        "quantitative_analyst": {
            "description": "Performs quantitative analysis, ratios, valuation metrics",
            "keywords": ["ratio", "P/E", "margin", "DCF", "valuation", "CAGR"],
            "priority": 3
        },
        "market_analyst": {
            "description": "Tracks market sentiment, technical indicators, price movements",
            "keywords": ["sentiment", "price", "stock", "volume", "technical"],
            "priority": 3
        },
        "industry_analyst": {
            "description": "Provides sector insights, trends, regulatory landscape",
            "keywords": ["industry", "sector", "peers", "competitors", "market share"],
            "priority": 3
        },
    }
    
    def __init__(self, ollama_url: str = "http://localhost:11434", max_iterations: int = 3):
        self.client = OllamaClient(base_url=ollama_url)
        self.max_iterations = max_iterations
        self.specialist_agents = {}
        self.trace = ReActTrace()
        
    def register_specialist(self, agent_name: str, agent_instance):
        """Register a specialist agent implementation"""
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def test_connection(self) -> bool:
        return self.client.test_connection()
    
    def _reason_rule_based(self, user_query: str, iteration: int) -> Action:
        """
        RULE-BASED REASONING
        
        Rules:
        - Iteration 1: Call business_analyst (document analysis)
        - Iteration 2: Call web_search_agent (supplement with web data)
        - Iteration 3: Finish and synthesize
        """
        print(f"\n🧠 [THOUGHT {iteration}] Rule-based reasoning...")
        
        query_lower = user_query.lower()
        called_agents = self.trace.get_specialist_calls()
        
        # 🟢 Rule 1: ALWAYS call business_analyst first (if available)
        if "business_analyst" in self.specialist_agents and "business_analyst" not in called_agents:
            thought = "Starting with Business Analyst for document-based analysis"
            self.trace.add_thought(thought, iteration)
            print(f"   💡 {thought}")
            
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst",
                task_description=user_query,
                reasoning="Rule 1: Business Analyst provides foundational document analysis"
            )
        
        # 🟢 Rule 2: Call web_search_agent AFTER business_analyst (if available)
        if "web_search_agent" in self.specialist_agents and "web_search_agent" not in called_agents and "business_analyst" in called_agents:
            thought = "Calling Web Search Agent to supplement document analysis with current data"
            self.trace.add_thought(thought, iteration)
            print(f"   💡 {thought}")
            
            # Get business analyst's output to pass as context
            business_analyst_output = ""
            for obs in self.trace.observations:
                if obs.action.agent_name == "business_analyst":
                    business_analyst_output = obs.result
                    break
            
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="web_search_agent",
                task_description=user_query,
                reasoning="Rule 2: Web Search Agent supplements with recent info"
            )
        
        # Rule 3: Finish after both agents called (or max iterations)
        thought = f"Analysis complete with {len(called_agents)} specialists, ready to synthesize"
        self.trace.add_thought(thought, iteration)
        print(f"   💡 {thought}")
        
        return Action(
            action_type=ActionType.FINISH,
            reasoning="Rule 3: Sufficient analysis gathered"
        )
    
    def _execute_action(self, action: Action) -> Observation:
        """Execute the decided action"""
        print(f"\n⚙️ [ACTION] Executing {action.action_type.value}...")
        
        if action.action_type == ActionType.CALL_SPECIALIST:
            result = self._call_specialist(action.agent_name, action.task_description)
            return Observation(
                action=action,
                result=result,
                success=True,
                metadata={"agent": action.agent_name}
            )
        
        elif action.action_type == ActionType.FINISH:
            return Observation(
                action=action,
                result="Orchestration complete - proceeding to synthesis",
                success=True
            )
        
        else:
            return Observation(
                action=action,
                result="Unknown action type",
                success=False
            )
    
    def _call_specialist(self, agent_name: str, task: str) -> str:
        """Call a specialist agent"""
        print(f"   🤖 Calling {agent_name}...")
        
        if agent_name in self.specialist_agents:
            agent = self.specialist_agents[agent_name]
            try:
                # For web_search_agent, pass prior analysis if available
                if agent_name == "web_search_agent" and hasattr(agent, 'analyze'):
                    # Get business analyst output
                    prior_analysis = ""
                    for obs in self.trace.observations:
                        if obs.action.agent_name == "business_analyst":
                            prior_analysis = obs.result
                            break
                    
                    result = agent.analyze(task, prior_analysis=prior_analysis)
                else:
                    result = agent.analyze(task)
                
                print(f"   ✅ {agent_name} completed ({len(result)} chars)")
                return result
            except Exception as e:
                print(f"   ❌ {agent_name} error: {e}")
                import traceback
                traceback.print_exc()
                return f"Error executing {agent_name}: {str(e)}"
        else:
            print(f"   ⏳ {agent_name} not implemented")
            return f"[{agent_name.upper()} PLACEHOLDER]\n\nThis agent would analyze: {task}"
    
    def _extract_document_sources(self, specialist_outputs: List[Dict]) -> Dict[int, str]:
        """
        Extract BOTH document and web sources from specialist outputs
        
        Supports:
        - --- SOURCE: filename.pdf (Page X) ---  (document sources)
        - --- SOURCE: Title (URL) ---             (web sources)
        """
        document_sources = {}
        citation_num = 1
        
        for output in specialist_outputs:
            content = output['result']
            agent = output['agent']
            
            # Pattern 1: Document sources (filename + page)
            pattern_doc = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            doc_sources = re.findall(pattern_doc, content, re.IGNORECASE)
            
            # Pattern 2: Web sources (title + URL)
            pattern_web = r'---\s*SOURCE:\s*([^\(]+)\((https?://[^\)]+)\)\s*---'
            web_sources = re.findall(pattern_web, content, re.IGNORECASE)
            
            total_sources = len(doc_sources) + len(web_sources)
            print(f"   🔍 DEBUG: Found {total_sources} SOURCE markers in {agent} output")
            print(f"      - Document sources: {len(doc_sources)}")
            print(f"      - Web sources: {len(web_sources)}")
            
            # Add document sources
            for filename, page in doc_sources:
                filename = filename.strip()
                page = page.strip()
                source_key = f"{filename} - Page {page}"
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    print(f"      [{citation_num}] {source_key}")
                    citation_num += 1
            
            # Add web sources
            for title, url in web_sources:
                title = title.strip()
                url = url.strip()
                source_key = f"{title} - {url}"
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    print(f"      [{citation_num}] {source_key}")
                    citation_num += 1
        
        return document_sources
    
    def _synthesize(self, user_query: str) -> str:
        """
        Synthesize with LOCAL LLM - Professional Equity Research Report
        """
        specialist_calls = self.trace.get_specialist_calls()
        current_date = datetime.now().strftime("%B %d, %Y")
        
        print(f"\n📊 [SYNTHESIS] Combining insights...")
        print(f"   Specialists called: {', '.join(specialist_calls) if specialist_calls else 'None'}")
        
        specialist_outputs = []
        for obs in self.trace.observations:
            if obs.action.action_type == ActionType.CALL_SPECIALIST:
                specialist_outputs.append({
                    "agent": obs.action.agent_name,
                    "task": obs.action.task_description,
                    "result": obs.result
                })
        
        if not specialist_outputs:
            return "## Analysis Summary\n\nNo specialist analysis available."
        
        # Extract ALL sources (documents + web)
        document_sources = self._extract_document_sources(specialist_outputs)
        print(f"   📚 Found {len(document_sources)} unique sources (documents + web)")
        
        if len(document_sources) == 0:
            print("   ⚠️ WARNING: No sources found! Check if agents are preserving SOURCE markers.")
        
        # Replace SOURCE markers with citation numbers
        outputs_with_cites = []
        for output in specialist_outputs:
            content = output['result']
            
            def replace_source(match):
                # Extract either (Page X) or (URL)
                full_match = match.group(0)
                
                # Try document pattern first
                if "Page" in full_match:
                    filename = match.group(1).strip()
                    page = match.group(2).strip()
                    source_key = f"{filename} - Page {page}"
                else:
                    # Web pattern
                    title = match.group(1).strip()
                    url = match.group(2).strip()
                    source_key = f"{title} - {url}"
                
                for num, doc in document_sources.items():
                    if doc == source_key:
                        return f" [SOURCE-{num}]"
                return match.group(0)
            
            # Replace both patterns
            content_with_cites = re.sub(
                r'---\s*SOURCE:\s*([^\(]+)\(([^\)]+)\)\s*---',
                replace_source,
                content,
                flags=re.IGNORECASE
            )
            
            outputs_with_cites.append({
                'agent': output['agent'],
                'content': content_with_cites
            })
        
        # Build specialist analysis for prompt
        outputs_text = "\n\n".join([
            f"{'='*60}\nSOURCE: {output['agent'].upper()}\n{'='*60}\n{output['content']}"
            for output in outputs_with_cites
        ])
        
        # Create document reference list
        references_list = "\n".join([
            f"[SOURCE-{num}] = {doc}"
            for num, doc in sorted(document_sources.items())
        ])
        
        # 🔥 PROFESSIONAL EQUITY RESEARCH SYNTHESIS PROMPT
        prompt = f"""You are a Senior Equity Research Analyst synthesizing a professional research report.

REPORT DATE: {current_date}
CLIENT QUERY: {user_query}

DATA SOURCES (TWO TEMPORAL LAYERS):
1. HISTORICAL: 10-K/10-Q SEC filings (FY2024-FY2025 data)
2. CURRENT: Web sources (Q4 2025 - Q1 2026 developments)

SPECIALIST ANALYSIS (with SOURCE citations):
{outputs_text}

SOURCE REFERENCE MAP:
{references_list}

==========================================================================
PROFESSIONAL REPORT STRUCTURE
==========================================================================

## Executive Summary
[2-3 sentences: Key findings, investment implications, temporal context]

## Investment Thesis
[3-4 bullet points: Core reasons to consider the stock, backed by data]

## Business Overview (Historical Context)
[Per FY2025 10-K data - cite with [X] for 10-K sources]
- Business model and revenue streams
- Market position and competitive advantages
- Key financial metrics from latest filing

## Recent Developments (Current Period)
[Per Q4 2025 - Q1 2026 web sources - cite with [Y] for web sources]
- Latest product launches and strategic initiatives  
- Recent financial performance vs expectations
- Market sentiment and analyst opinion shifts
- Emerging competitive dynamics

## Risk Analysis
### Historical Risks (Per 10-K) [cite 10-K sources]
- Disclosed risk factors from SEC filings

### Emerging Risks (Current) [cite web sources]
- New competitive threats
- Regulatory developments
- Market condition changes

## Valuation Context
[If available: P/E, EV/Sales, growth rates - cite sources]
- Historical valuation metrics [10-K sources]
- Current market valuation [web sources]
- Analyst price targets [web sources]

## Conclusion
[2-3 sentences: Balanced view considering both historical fundamentals and current developments]

## Data Quality Notice
⚠️ **Temporal Distinction**:
- 10-K data: Historical (6-12 months old as of filing date)
- Web data: Current market developments (last 3 months)
- Investment decisions should consider both layers

## References
[List ALL citations with clear temporal markers]

==========================================================================
CITATION REQUIREMENTS (CRITICAL)
==========================================================================

1. TEMPORAL MARKERS (Mandatory):
   - 10-K data: "Per the FY2025 10-K [1]..."
   - Web data: "As of Q1 2026 [8]..." or "Recent reports indicate [9]..."

2. CITATION DENSITY:
   - Every factual claim = citation
   - Every metric/number = citation  
   - Every risk factor = citation
   - Aim for 2-3 citations per paragraph

3. FORMAT:
   - Replace [SOURCE-X] with [X]
   - Example: "Revenue reached $394B per FY2025 10-K [1], while Q1 2026 guidance suggests 8% YoY growth [8]."

4. PROFESSIONAL TONE:
   - Concise, data-driven, specific
   - Use exact figures: "15.2%" not "around 15%"
   - Attribution: "Goldman Sachs projects..." not "analysts think..."
   - Objective: Present data, avoid speculation

5. MIXING SOURCES:
   - Contrast historical vs current: "While FY2025 showed... [1], recent developments indicate... [7]"
   - Show progression: "Margins were 25.3% in FY2025 [2], expanding to 26.1% in Q4 2025 [9]"

==========================================================================
GENERATE REPORT NOW
==========================================================================

Provide a professional equity research report following the structure above.
Cite FREQUENTLY with proper temporal markers.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            print("   🔄 Generating professional equity research synthesis...")
            final_report = self.client.chat(messages, temperature=0.2)  # Slightly higher for better prose
            print("   ✅ Synthesis complete")
            
            # Post-process: Ensure References section exists
            if len(document_sources) > 0 and "## References" not in final_report:
                refs = "\n\n## References\n\n" + "\n".join([
                    f"[{num}] {doc}"
                    for num, doc in sorted(document_sources.items())
                ])
                final_report += refs
            
            return final_report
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            # Fallback
            return f"""## Research Report\n\n{outputs_text}\n\n---\n\n## Sources\n\n{references_list.replace('[SOURCE-', '[').replace('] =', ']')}\n\n---\n\n**Note**: Synthesis failed. Showing raw analysis."""
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop with rule-based reasoning"""
        print("\n" + "="*70)
        print("🔁 REACT EQUITY RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"🔄 Max Iterations: {self.max_iterations}")
        print(f"📊 Registered Agents: {', '.join(self.specialist_agents.keys()) if self.specialist_agents else 'None'}")
        
        self.trace = ReActTrace()
        
        for iteration in range(1, self.max_iterations + 1):
            print("\n" + "-"*70)
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print("-"*70)
            
            action = self._reason_rule_based(user_query, iteration)
            self.trace.add_action(action)
            
            observation = self._execute_action(action)
            self.trace.add_observation(observation)
            
            obs_preview = observation.result[:150] + "..." if len(observation.result) > 150 else observation.result
            print(f"\n👁️ [OBSERVATION] {obs_preview}")
            
            if action.action_type == ActionType.FINISH:
                print(f"\n🎯 Loop ending at iteration {iteration}")
                break
        
        print("\n" + "="*70)
        print("📝 FINAL SYNTHESIS")
        print("="*70)
        
        final_report = self._synthesize(user_query)
        
        print("\n" + "="*70)
        print("📈 ORCHESTRATION SUMMARY")
        print("="*70)
        print(f"Iterations: {len(self.trace.thoughts)}")
        print(f"Specialists: {', '.join(self.trace.get_specialist_calls()) or 'None'}")
        print("="*70)
        
        return final_report
    
    def get_trace_summary(self) -> str:
        return self.trace.get_history_summary()


def main():
    try:
        orchestrator = ReActOrchestrator(max_iterations=3)
        
        if not orchestrator.test_connection():
            print("\n❌ Failed to connect to Ollama")
            print("\n💡 Make sure Ollama is running: ollama serve")
            return
    except ValueError as e:
        print(f"❌ {str(e)}")
        return
    
    print("\n🚀 Professional Equity Research Orchestrator Ready")
    print("\nAvailable agents:")
    for name in ReActOrchestrator.SPECIALIST_AGENTS.keys():
        status = "✅" if name in orchestrator.specialist_agents else "⏳"
        print(f"  {status} {name}")
    
    while True:
        print("\n" + "="*70)
        user_query = input("\n💬 Research question (or 'quit'): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_query.strip():
            continue
        
        try:
            final_report = orchestrator.research(user_query)
            print("\n" + "="*70)
            print("📄 FINAL REPORT")
            print("="*70)
            print(f"\n{final_report}\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

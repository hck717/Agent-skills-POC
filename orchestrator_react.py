#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Rule-based orchestration with LOCAL LLM synthesis (no web search interference).
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
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
                "num_predict": 3000
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
        },
        "quantitative_analyst": {
            "description": "Performs quantitative analysis, ratios, valuation metrics",
            "keywords": ["ratio", "P/E", "margin", "DCF", "valuation", "CAGR"],
        },
        "market_analyst": {
            "description": "Tracks market sentiment, technical indicators, price movements",
            "keywords": ["sentiment", "price", "stock", "volume", "technical"],
        },
        "industry_analyst": {
            "description": "Provides sector insights, trends, regulatory landscape",
            "keywords": ["industry", "sector", "peers", "competitors", "market share"],
        },
    }
    
    def __init__(self, ollama_url: str = "http://localhost:11434", max_iterations: int = 2):
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
        
        Simple rules:
        - Iteration 1: Call business_analyst
        - Iteration 2: Finish and synthesize
        """
        print(f"\n🧠 [THOUGHT {iteration}] Rule-based reasoning...")
        
        query_lower = user_query.lower()
        called_agents = self.trace.get_specialist_calls()
        
        # Rule 1: Always call business_analyst first
        if "business_analyst" in self.specialist_agents and "business_analyst" not in called_agents:
            thought = "Calling Business Analyst for document-based analysis"
            self.trace.add_thought(thought, iteration)
            print(f"   💭 {thought}")
            
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst",
                task_description=user_query,
                reasoning="Rule 1: Business Analyst first"
            )
        
        # Rule 2: Finish after first call
        thought = f"Analysis complete, ready to synthesize"
        self.trace.add_thought(thought, iteration)
        print(f"   💭 {thought}")
        
        return Action(
            action_type=ActionType.FINISH,
            reasoning="Rule 2: Finish after analysis"
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
        Extract document sources from specialist outputs
        Supports: --- SOURCE: filename.pdf (Page X) ---
        """
        document_sources = {}
        citation_num = 1
        
        for output in specialist_outputs:
            content = output['result']
            
            # Pattern: --- SOURCE: filename (Page X) ---
            pattern = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            sources = re.findall(pattern, content, re.IGNORECASE)
            
            print(f"   🔍 DEBUG: Found {len(sources)} SOURCE markers in {output['agent']} output")
            
            for filename, page in sources:
                filename = filename.strip()
                page = page.strip()
                source_key = f"{filename} - Page {page}"
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    print(f"      [{citation_num}] {source_key}")
                    citation_num += 1
        
        return document_sources
    
    def _synthesize(self, user_query: str) -> str:
        """Synthesize with LOCAL LLM (no web search)"""
        specialist_calls = self.trace.get_specialist_calls()
        
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
        
        # Extract document sources
        document_sources = self._extract_document_sources(specialist_outputs)
        print(f"   📚 Found {len(document_sources)} unique document sources")
        
        if len(document_sources) == 0:
            print("   ⚠️ WARNING: No document sources found! Check if Business Analyst is preserving SOURCE markers.")
        
        # Replace SOURCE markers with citation numbers
        outputs_with_cites = []
        for output in specialist_outputs:
            content = output['result']
            
            def replace_source(match):
                filename = match.group(1).strip()
                page = match.group(2).strip()
                source_key = f"{filename} - Page {page}"
                
                for num, doc in document_sources.items():
                    if doc == source_key:
                        return f" [SOURCE-{num}]"
                return match.group(0)
            
            pattern = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            content_with_cites = re.sub(pattern, replace_source, content, flags=re.IGNORECASE)
            
            outputs_with_cites.append({
                'agent': output['agent'],
                'content': content_with_cites
            })
        
        # Build specialist analysis for prompt
        outputs_text = "\n\n".join([
            f"{'='*60}\n{output['content']}"
            for output in outputs_with_cites
        ])
        
        # Create document reference list
        references_list = "\n".join([
            f"[SOURCE-{num}] = {doc}"
            for num, doc in sorted(document_sources.items())
        ])
        
        # 🔥 ENHANCEMENT 1: Enhanced prompt for frequent citations
        prompt = f"""You are an equity research analyst synthesizing document-based analysis.

USER QUERY: {user_query}

SPECIALIST ANALYSIS (with SOURCE citations):
{outputs_text}

DOCUMENT REFERENCE MAP:
{references_list}

INSTRUCTIONS:
1. Synthesize a comprehensive research report
2. Structure: Executive Summary, Detailed Analysis, Key Findings, Risk Factors, Conclusion, References

3. 🔥 CRITICAL CITATION RULES (ENHANCED):
   - Replace [SOURCE-X] with [X] in your output (e.g., [SOURCE-1] becomes [1])
   - CITE FREQUENTLY: Every factual claim, statistic, or risk factor MUST have a citation
   - Multiple facts in one sentence = multiple citations
   - Example of GOOD citation density:
     "Apple's revenue grew 7% YoY [1], driven by iPhone sales of $201B [2]. 
      The company faces supply chain risks in China [3] and regulatory 
      pressures in Europe [4]."
   - Example of BAD citation (avoid this):
     "Apple's revenue grew, driven by iPhone sales. The company faces 
      supply chain and regulatory risks."
   - DO NOT make unsourced claims - if no SOURCE marker exists, note as "analyst assessment"
   - Aim for at least 1-2 citations per paragraph

4. When discussing financial metrics:
   - Revenue figures → cite source [X]
   - Growth rates → cite source [X]
   - Market share → cite source [X]
   - Margins → cite source [X]
   - Product mix → cite source [X]

5. When discussing risks:
   - Each risk factor → cite source [X]
   - Regulatory issues → cite source [X]
   - Competition → cite source [X]
   - Supply chain → cite source [X]

6. End with a References section listing all citations as:
   "[1] APPL 10-k Filings.pdf - Page 23"

7. If no SOURCE citations are present in specialist analysis, note: "Analysis based on general knowledge (no document citations available)"

Synthesize now with FREQUENT citations:"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            print("   🔄 Generating synthesis with local LLM...")
            final_report = self.client.chat(messages, temperature=0.3)
            print("   ✅ Synthesis complete")
            
            # Post-process: Ensure References section exists
            if len(document_sources) > 0 and "## References" not in final_report and "## 📚 References" not in final_report:
                refs = "\n\n## 📚 References\n\n" + "\n".join([
                    f"[{num}] {doc}"
                    for num, doc in sorted(document_sources.items())
                ])
                final_report += refs
            
            return final_report
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            # Fallback: return raw with sources
            return f"""## Research Report\n\n{outputs_text}\n\n---\n\n## 📚 Document Sources\n\n{references_list.replace('[SOURCE-', '[').replace('] =', ']')}\n\n---\n\n**Note**: Synthesis failed. Showing raw analysis."""
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop with rule-based reasoning"""
        print("\n" + "="*70)
        print("🔁 REACT-BASED RESEARCH ORCHESTRATOR (Local LLM)")
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
        orchestrator = ReActOrchestrator(max_iterations=2)
        
        if not orchestrator.test_connection():
            print("\n❌ Failed to connect to Ollama")
            print("\n💡 Make sure Ollama is running: ollama serve")
            return
    except ValueError as e:
        print(f"❌ {str(e)}")
        return
    
    print("\n🚀 Local LLM Orchestrator Ready")
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

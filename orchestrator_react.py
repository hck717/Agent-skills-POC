#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Simplified version using rule-based orchestration instead of Perplexity JSON responses.
This avoids Perplexity refusing to act as an orchestrator.
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


class PerplexityClient:
    """Client for Perplexity API interactions (used only for synthesis)"""
    
    VALID_MODELS = {
        "sonar": "Fast, cost-efficient search and Q&A (128k context)",
        "sonar-pro": "Deep retrieval for complex queries (200k context)",
        "sonar-reasoning": "Multi-step reasoning (128k context)",
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found. Please provide API key.")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages: List[Dict[str, str]], model: str = "sonar", temperature: float = 0.2) -> str:
        """Send chat request to Perplexity API"""
        if model not in self.VALID_MODELS:
            print(f"⚠️ Warning: Model '{model}' not in known list. Using 'sonar' instead.")
            model = "sonar"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Bad Request (400): {error_msg}")
            elif response.status_code == 401:
                raise Exception("Unauthorized (401): Invalid API key")
            elif response.status_code == 429:
                raise Exception("Rate Limit (429): Too many requests")
            
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"Perplexity API error: {str(e)}")
    
    def test_connection(self) -> bool:
        try:
            print("🔌 Testing Perplexity API connection...")
            messages = [{"role": "user", "content": "Say 'OK' if you can read this."}]
            response = self.chat(messages, model="sonar", temperature=0.0)
            if response:
                print("✅ Connection successful!")
                return True
            return False
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False


class ReActOrchestrator:
    """Rule-based orchestrator with Perplexity for synthesis only"""
    
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
    
    def __init__(self, perplexity_api_key: str = None, max_iterations: int = 3):
        try:
            self.client = PerplexityClient(perplexity_api_key)
        except ValueError as e:
            raise ValueError(f"Failed to initialize Perplexity client: {str(e)}")
        
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
        RULE-BASED REASONING (no Perplexity JSON)
        
        Simple rules:
        - Iteration 1-2: Call business_analyst if available
        - Iteration 3: Finish and synthesize
        """
        print(f"\n🧠 [THOUGHT {iteration}] Rule-based reasoning...")
        
        query_lower = user_query.lower()
        called_agents = self.trace.get_specialist_calls()
        
        # Rule 1: Always call business_analyst first if available and not yet called
        if "business_analyst" in self.specialist_agents and "business_analyst" not in called_agents:
            thought = "Business Analyst has document access - calling first for foundational analysis"
            self.trace.add_thought(thought, iteration)
            print(f"   💭 {thought}")
            
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst",
                task_description=user_query,
                reasoning="Rule 1: Business Analyst first"
            )
        
        # Rule 2: If business_analyst called once, call again with refined query
        if iteration == 2 and "business_analyst" in self.specialist_agents:
            thought = "Getting additional perspectives from Business Analyst"
            self.trace.add_thought(thought, iteration)
            print(f"   💭 {thought}")
            
            refined_query = f"Provide detailed analysis of: {user_query}"
            if "risk" in query_lower:
                refined_query += " Focus on risk factors, regulatory concerns, and vulnerabilities."
            if "compet" in query_lower:
                refined_query += " Focus on competitive dynamics, market positioning, and rivals."
            
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst",
                task_description=refined_query,
                reasoning="Rule 2: Refined analysis"
            )
        
        # Rule 3: After 2 calls or iteration 3+, finish
        thought = f"Sufficient analysis gathered, ready to synthesize (iteration {iteration})"
        self.trace.add_thought(thought, iteration)
        print(f"   💭 {thought}")
        
        return Action(
            action_type=ActionType.FINISH,
            reasoning="Rule 3: Finish after sufficient iterations"
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
        Supports multiple formats:
        - --- SOURCE: filename.pdf (Page X) ---
        - **Source:** [filename.pdf, Page X]
        """
        document_sources = {}
        citation_num = 1
        
        for output in specialist_outputs:
            content = output['result']
            
            # Pattern 1: --- SOURCE: filename (Page X) ---
            pattern1 = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            sources = re.findall(pattern1, content, re.IGNORECASE)
            
            for filename, page in sources:
                filename = filename.strip()
                page = page.strip()
                source_key = f"{filename} - Page {page}"
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    citation_num += 1
            
            # Pattern 2: **Source:** [filename, Page X]
            pattern2 = r'\*\*Source:\*\*\s*\[([^,]+),\s*Page\s*([^\]]+)\]'
            sources2 = re.findall(pattern2, content, re.IGNORECASE)
            
            for filename, page in sources2:
                filename = filename.strip()
                page = page.strip()
                source_key = f"{filename} - Page {page}"
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    citation_num += 1
            
            # If no sources found, add placeholder
            if not sources and not sources2 and "PLACEHOLDER" in content:
                agent_name = output['agent'].replace('_', ' ').title()
                document_sources[citation_num] = f"{agent_name} (Placeholder Analysis)"
                citation_num += 1
        
        return document_sources
    
    def _synthesize(self, user_query: str) -> str:
        """Synthesize with Perplexity (as synthesis tool, not orchestrator)"""
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
        
        # Build synthesis prompt
        outputs_text = "\n\n".join([
            f"{'='*60}\nSPECIALIST: {output['agent'].upper()}\n{'='*60}\n{output['result']}"
            for output in specialist_outputs
        ])
        
        references_list = "\n".join([
            f"[{num}] {doc}"
            for num, doc in sorted(document_sources.items())
        ])
        
        prompt = f"""Synthesize a comprehensive equity research report.

QUERY: {user_query}

SPECIALIST ANALYSIS:
{outputs_text}

DOCUMENT SOURCES:
{references_list}

Create a report with:
1. Executive Summary (3-4 sentences)
2. Detailed Analysis (by theme)
3. Key Findings
4. Risk Factors
5. Conclusion
6. References section

Cite facts using [1], [2], [3] matching the document sources above.
DO NOT cite specialists - cite the actual source documents.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            print("   🔄 Generating synthesis with Perplexity...")
            final_report = self.client.chat(messages, model="sonar", temperature=0.3)
            print("   ✅ Synthesis complete")
            return final_report
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            return f"""## Research Report\n\n{outputs_text}\n\n---\n\n## 📚 References\n\n{references_list}\n\n---\n\n**Note**: Synthesis failed. Showing raw analysis."""
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop with rule-based reasoning"""
        print("\n" + "="*70)
        print("🔁 REACT-BASED RESEARCH ORCHESTRATOR (Rule-Based)")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"🔄 Max Iterations: {self.max_iterations}")
        print(f"📊 Registered Agents: {', '.join(self.specialist_agents.keys()) if self.specialist_agents else 'None'}")
        
        self.trace = ReActTrace()
        
        for iteration in range(1, self.max_iterations + 1):
            print("\n" + "-"*70)
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print("-"*70)
            
            # Rule-based reasoning (no Perplexity)
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
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️ Please set PERPLEXITY_API_KEY")
        return
    
    try:
        orchestrator = ReActOrchestrator(max_iterations=3)
        
        if not orchestrator.test_connection():
            print("\n❌ Failed to connect")
            return
    except ValueError as e:
        print(f"❌ {str(e)}")
        return
    
    print("\n🚀 Rule-Based Orchestrator Ready")
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

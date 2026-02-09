#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Implements the ReAct (Reasoning + Acting) framework for iterative agent orchestration.
The orchestrator can reason about results, decide next actions, and refine its approach.

ReAct Loop:
1. Thought: Reason about current state and what to do next
2. Action: Execute specialist agent or synthesis
3. Observation: Receive and analyze results
4. Repeat until task is complete or max iterations reached
"""

import os
import json
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
    """Client for Perplexity API interactions"""
    
    # Valid Perplexity Sonar models (as of 2026)
    VALID_MODELS = {
        "sonar": "Fast, cost-efficient search and Q&A (128k context)",
        "sonar-pro": "Deep retrieval for complex queries (200k context)",
        "sonar-reasoning-pro": "Multi-step logic and chain-of-thought (128k context)",
        "sonar-deep-research": "Long-form exhaustive synthesis (128k context)"
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
    
    def chat(self, messages: List[Dict[str, str]], model: str = "sonar-pro", temperature: float = 0.2) -> str:
        """Send chat request to Perplexity API"""
        # Validate model
        if model not in self.VALID_MODELS:
            print(f"⚠️ Warning: Model '{model}' not in known list. Using 'sonar-pro' instead.")
            print(f"Valid models: {', '.join(self.VALID_MODELS.keys())}")
            model = "sonar-pro"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=120
            )
            
            # Better error handling
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                    raise Exception(f"Bad Request (400): {error_msg}")
                except:
                    raise Exception(f"Bad Request (400): Check API key and model name. Valid models: {', '.join(self.VALID_MODELS.keys())}")
            elif response.status_code == 401:
                raise Exception("Unauthorized (401): Invalid API key. Get a valid key from https://www.perplexity.ai/settings/api")
            elif response.status_code == 429:
                raise Exception("Rate Limit (429): Too many requests. Please wait and try again.")
            elif response.status_code == 403:
                raise Exception("Forbidden (403): API access denied. Check your subscription plan.")
            
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception(f"Unexpected API response format: {result}")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            raise Exception("Request timeout (120s). The query may be too complex.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            if "Perplexity API error" in str(e):
                raise
            raise Exception(f"Perplexity API error: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test if API connection works"""
        try:
            print("🔌 Testing Perplexity API connection...")
            messages = [{"role": "user", "content": "Say 'OK' if you can read this."}]
            response = self.chat(messages, model="sonar", temperature=0.0)
            if response:
                print("✅ Connection successful!")
                print(f"📝 Response: {response[:100]}")
                return True
            return False
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False


class ReActOrchestrator:
    """ReAct-based orchestrator that iteratively reasons and acts"""
    
    SPECIALIST_AGENTS = {
        "business_analyst": {
            "description": "Analyzes 10-K filings, financial statements, business models, competitive positioning, and strategic risks using RAG from local documents",
            "capabilities": ["Financial statement analysis", "Competitive intelligence", "Risk assessment", "Business model evaluation", "Document retrieval"],
            "keywords": ["10-K", "10-Q", "filing", "risk", "competitive", "business model", "MD&A", "financial statement"],
            "priority": "HIGH - Has access to proprietary documents via ChromaDB"
        },
        "quantitative_analyst": {
            "description": "Performs quantitative analysis including financial ratios, valuation metrics, growth rates, and statistical modeling",
            "capabilities": ["DCF valuation", "Ratio analysis", "Trend forecasting", "Comparative metrics"],
            "keywords": ["calculate", "ratio", "P/E", "margin", "DCF", "valuation", "CAGR", "growth"],
            "priority": "MEDIUM - Placeholder for now"
        },
        "market_analyst": {
            "description": "Tracks market sentiment, technical indicators, price movements, trading volumes, and market positioning",
            "capabilities": ["Sentiment analysis", "Technical analysis", "Market trends", "Trading patterns"],
            "keywords": ["sentiment", "price", "stock", "volume", "technical", "chart", "analyst rating"],
            "priority": "MEDIUM - Placeholder for now"
        },
        "industry_analyst": {
            "description": "Provides sector-specific insights, industry trends, regulatory landscape, and peer benchmarking",
            "capabilities": ["Industry dynamics", "Regulatory analysis", "Peer comparison", "Sector trends"],
            "keywords": ["industry", "sector", "peers", "competitors", "market share", "regulation"],
            "priority": "MEDIUM - Placeholder for now"
        },
        "esg_analyst": {
            "description": "Evaluates environmental, social, and governance factors, sustainability initiatives, and ESG risk exposure",
            "capabilities": ["ESG scoring", "Sustainability assessment", "Governance evaluation", "Climate risk"],
            "keywords": ["ESG", "sustainability", "carbon", "governance", "diversity", "environmental"],
            "priority": "LOW - Placeholder for now"
        },
        "macro_analyst": {
            "description": "Analyzes macroeconomic factors, interest rates, currency impacts, geopolitical risks affecting the company",
            "capabilities": ["Economic indicators", "Rate sensitivity", "FX exposure", "Geopolitical risk"],
            "keywords": ["interest rate", "inflation", "GDP", "currency", "FX", "geopolitical", "macro"],
            "priority": "LOW - Placeholder for now"
        }
    }
    
    def __init__(self, perplexity_api_key: str = None, max_iterations: int = 5):
        try:
            self.client = PerplexityClient(perplexity_api_key)
        except ValueError as e:
            raise ValueError(f"Failed to initialize Perplexity client: {str(e)}")
        
        self.max_iterations = max_iterations
        self.specialist_agents = {}  # Registry of implemented agents
        self.trace = ReActTrace()
        
    def register_specialist(self, agent_name: str, agent_instance):
        """Register a specialist agent implementation"""
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def test_connection(self) -> bool:
        """Test Perplexity API connection"""
        return self.client.test_connection()
    
    def _create_reasoning_prompt(self, user_query: str, iteration: int, history: str) -> str:
        """Create prompt for reasoning step"""
        
        agents_list = "\n".join([
            f"- {name}: {info['description']}\n  Priority: {info['priority']}\n  Keywords: {', '.join(info['keywords'][:5])}"
            for name, info in self.SPECIALIST_AGENTS.items()
        ])
        
        available_agents_status = "\n".join([
            f"- {name}: {'✅ AVAILABLE (Has data access)' if name in self.specialist_agents else '⏳ Placeholder'}"
            for name in self.SPECIALIST_AGENTS.keys()
        ])
        
        # Get already called agents
        called_agents = self.trace.get_specialist_calls()
        
        prompt = f"""You are a ReAct-based Research Orchestrator using iterative reasoning.

USER QUERY:
{user_query}

CURRENT ITERATION: {iteration}/{self.max_iterations}

AVAILABLE SPECIALIST AGENTS:
{agents_list}

AGENT STATUS:
{available_agents_status}

ALREADY CALLED: {', '.join(called_agents) if called_agents else 'None'}

PREVIOUS HISTORY:
{history if history else "[First iteration - no previous actions]"}

🎯 ORCHESTRATION STRATEGY:

1. **Iteration {iteration}/{self.max_iterations}**: You should generally use ALL {self.max_iterations} iterations for comprehensive analysis

2. **When to call specialists**:
   - If Business Analyst is AVAILABLE (✅), you MUST call it for document-based queries
   - Call different specialists for different perspectives
   - Each specialist provides unique insights from their domain

3. **When to finish**:
   - Only after calling at least 2-3 specialist agents
   - Only on iteration 4 or 5
   - Only when comprehensive analysis is complete

4. **This iteration ({iteration})**: 
   {'- First iteration: Start with Business Analyst if available for foundational analysis' if iteration == 1 else ''}
   {'- Middle iterations: Call additional specialists for different angles' if 1 < iteration < self.max_iterations else ''}
   {'- Final iterations: Consider if more analysis needed or ready to finish' if iteration >= self.max_iterations - 1 else ''}

AVAILABLE ACTIONS:
- call_specialist: Call a specialist agent (recommended for iterations 1-4)
- finish: Complete and synthesize (only use on iteration 4-5 after calling agents)

⚠️ IMPORTANT:
- Business Analyst has RAG access to actual 10-K documents via ChromaDB
- Don't finish early - use the full {self.max_iterations} iterations
- Each specialist provides unique value
- Comprehensive research requires multiple perspectives

RESPOND IN THIS JSON FORMAT:
{{
  "thought": "Your reasoning about what to do in iteration {iteration}",
  "action": "call_specialist" | "finish",
  "agent_name": "agent_name" (required if action is call_specialist),
  "task_description": "specific task for the agent" (required if action is call_specialist),
  "reasoning": "Why this action for iteration {iteration}/{self.max_iterations}"
}}

Provide ONLY valid JSON."""
        return prompt
    
    def _reason(self, user_query: str, iteration: int) -> Action:
        """Reasoning step: Decide what to do next"""
        print(f"\n🧠 [THOUGHT {iteration}] Reasoning about next action...")
        
        history = self.trace.get_history_summary()
        prompt = self._create_reasoning_prompt(user_query, iteration, history)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Use sonar-pro for reasoning
            response = self.client.chat(messages, model="sonar-pro", temperature=0.2)
        except Exception as e:
            print(f"   ⚠️ Error calling Perplexity API: {str(e)}")
            self.trace.add_thought(f"API error: {str(e)}", iteration)
            return Action(action_type=ActionType.FINISH)
        
        try:
            # Extract JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Store thought
            thought_content = data.get("thought", "")
            self.trace.add_thought(thought_content, iteration)
            print(f"   💭 {thought_content[:150]}..." if len(thought_content) > 150 else f"   💭 {thought_content}")
            
            # Create action
            action_type = ActionType(data["action"])
            action = Action(
                action_type=action_type,
                agent_name=data.get("agent_name"),
                task_description=data.get("task_description"),
                reasoning=data.get("reasoning", "")
            )
            
            print(f"   ⚡ Action: {action.action_type.value}")
            if action.agent_name:
                print(f"      Agent: {action.agent_name}")
                print(f"      Task: {action.task_description[:100]}..." if len(action.task_description or "") > 100 else f"      Task: {action.task_description}")
            
            return action
            
        except Exception as e:
            print(f"   ⚠️ Error parsing reasoning: {e}")
            print(f"   Raw response: {response[:300]}...")
            self.trace.add_thought(f"Parse error, proceeding to synthesis", iteration)
            return Action(action_type=ActionType.FINISH)
    
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
        
        elif action.action_type == ActionType.SYNTHESIZE:
            return Observation(
                action=action,
                result="Ready to synthesize",
                success=True
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
            # Placeholder for unimplemented agents
            print(f"   ⏳ {agent_name} not yet implemented - using placeholder")
            result = f"[{agent_name.upper()} ANALYSIS - PLACEHOLDER]\n\n"
            result += f"Task: {task}\n\n"
            result += f"This specialist would analyze: {task}\n\n"
            result += "Key insights would include:\n"
            result += "- Detailed domain-specific analysis\n"
            result += "- Data-driven recommendations\n"
            result += "- Expert interpretation\n\n"
            result += "Note: This agent will be fully implemented in future versions.\n"
            return result
    
    def _synthesize(self, user_query: str) -> str:
        """Synthesize all observations into final report"""
        num_obs = len(self.trace.observations)
        specialist_calls = self.trace.get_specialist_calls()
        
        print(f"\n📊 [SYNTHESIS] Combining insights from {num_obs} observations...")
        print(f"   Specialists called: {', '.join(specialist_calls) if specialist_calls else 'None'}")
        
        # Gather all specialist outputs
        specialist_outputs = []
        for obs in self.trace.observations:
            if obs.action.action_type == ActionType.CALL_SPECIALIST:
                specialist_outputs.append({
                    "agent": obs.action.agent_name,
                    "task": obs.action.task_description,
                    "result": obs.result
                })
        
        if not specialist_outputs:
            print("   ⚠️ No specialist outputs found - generating report from reasoning only")
            # Fallback synthesis from reasoning
            reasoning_summary = "\n".join([
                f"{i+1}. {thought.content}"
                for i, thought in enumerate(self.trace.thoughts)
            ])
            return f"""## Analysis Summary

Based on the ReAct reasoning process, here's what was determined:

{reasoning_summary}

## Note

This report was generated without calling specialist agents. For more detailed analysis:
- Ensure Business Analyst agent is registered with document access
- Try more complex queries that require deep document analysis
- Allow more iterations for comprehensive multi-agent orchestration
"""
        
        # Create synthesis prompt with specialist outputs
        outputs_text = "\n\n".join([
            f"{'='*60}\n"
            f"AGENT: {output['agent'].upper().replace('_', ' ')}\n"
            f"{'='*60}\n"
            f"Task: {output['task']}\n\n"
            f"Analysis:\n{output['result']}\n"
            for output in specialist_outputs
        ])
        
        reasoning_summary = "\n".join([
            f"{i+1}. Iteration {thought.iteration}: {thought.content[:200]}..."
            for i, thought in enumerate(self.trace.thoughts)
        ])
        
        prompt = f"""You are a Senior Equity Research Analyst synthesizing a multi-agent research report.

ORIGINAL QUERY:
{user_query}

REACT REASONING TRACE ({len(self.trace.thoughts)} iterations):
{reasoning_summary}

SPECIALIST AGENT OUTPUTS ({len(specialist_outputs)} specialists):

{outputs_text}

YOUR SYNTHESIS TASK:

1. **Integrate all specialist insights** into a coherent narrative
2. **Cross-validate findings** across different specialist perspectives  
3. **Highlight key insights** supported by multiple sources
4. **Identify any gaps or contradictions** in the analysis
5. **Provide actionable conclusions** based on the comprehensive research

REQUIRED REPORT STRUCTURE:

## Executive Summary
[3-4 sentences capturing the most important findings across all specialists]

## Detailed Analysis
[Organized by key themes, integrating insights from multiple agents. Each theme should reference which specialists provided supporting evidence]

### [Theme 1]
[Synthesis of relevant specialist insights]

### [Theme 2]
[Synthesis of relevant specialist insights]

## Key Findings & Metrics
[Bullet points of critical data, ratios, and quantitative insights with sources]

## Risk Factors & Considerations
[Consolidated risks identified by specialists]

## Conclusion & Recommendations
[Balanced final assessment with actionable insights]

---

**Important**: 
- Cite which specialist provided each insight (e.g., "Business Analyst identified...", "According to the Quantitative Analysis...")
- Don't just concatenate - synthesize and integrate across sources
- Professional equity research tone
- Be specific with data and evidence

Provide the complete synthesis report now."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            # Use sonar-deep-research for comprehensive synthesis
            print("   🔄 Generating synthesis with sonar-deep-research...")
            final_report = self.client.chat(messages, model="sonar-pro", temperature=0.3)
            print("   ✅ Synthesis complete")
            return final_report
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            # Fallback: return raw specialist outputs
            return f"""## Research Report

**Note**: Synthesis failed, presenting raw specialist outputs.

{outputs_text}

---

**Error during synthesis**: {str(e)}
"""
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop: Iteratively reason and act until task is complete"""
        print("\n" + "="*70)
        print("🔁 REACT-BASED EQUITY RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"\n🔄 Max Iterations: {self.max_iterations}")
        print(f"📊 Registered Agents: {', '.join(self.specialist_agents.keys())}")
        
        # Reset trace for new query
        self.trace = ReActTrace()
        
        # ReAct loop
        for iteration in range(1, self.max_iterations + 1):
            print("\n" + "-"*70)
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print("-"*70)
            
            # Step 1: Reason (Think)
            action = self._reason(user_query, iteration)
            self.trace.add_action(action)
            
            # Step 2: Act
            observation = self._execute_action(action)
            self.trace.add_observation(observation)
            
            # Step 3: Observe & Decide if done
            obs_preview = observation.result[:150] + "..." if len(observation.result) > 150 else observation.result
            print(f"\n👁️ [OBSERVATION] {obs_preview}")
            
            # Check if we should finish
            if action.action_type in [ActionType.FINISH, ActionType.SYNTHESIZE]:
                print(f"\n🎯 ReAct loop ending at iteration {iteration}/{self.max_iterations}")
                break
        
        # Final synthesis
        print("\n" + "="*70)
        print("📝 FINAL SYNTHESIS")
        print("="*70)
        
        final_report = self._synthesize(user_query)
        
        # Summary stats
        print("\n" + "="*70)
        print("📈 ORCHESTRATION SUMMARY")
        print("="*70)
        print(f"Iterations completed: {len(self.trace.thoughts)}")
        print(f"Specialists called: {', '.join(self.trace.get_specialist_calls()) or 'None'}")
        print(f"Total observations: {len(self.trace.observations)}")
        print("="*70)
        
        return final_report
    
    def get_trace_summary(self) -> str:
        """Get a readable summary of the ReAct trace"""
        return self.trace.get_history_summary()


def main():
    """Demo of ReAct-based orchestration"""
    
    # Check for API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️ Please set PERPLEXITY_API_KEY environment variable")
        print("   export PERPLEXITY_API_KEY='your-api-key'")
        return
    
    # Initialize ReAct orchestrator
    try:
        orchestrator = ReActOrchestrator(max_iterations=5)
        
        # Test connection
        if not orchestrator.test_connection():
            print("\n❌ Failed to connect to Perplexity API. Please check your API key.")
            return
            
    except ValueError as e:
        print(f"❌ {str(e)}")
        return
    
    # Optional: Register specialist agents
    # from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
    # business_analyst = BusinessAnalystGraphAgent()
    # orchestrator.register_specialist("business_analyst", business_analyst)
    
    print("\n🚀 ReAct-Based Orchestrator Ready")
    print("\nThis system uses iterative reasoning:")
    print("  Thought → Action → Observation → Repeat")
    print("\nAvailable specialist agents:")
    for name, info in ReActOrchestrator.SPECIALIST_AGENTS.items():
        status = "✅" if name in orchestrator.specialist_agents else "⏳"
        print(f"  {status} {name}")
    
    # Interactive loop
    while True:
        print("\n" + "="*70)
        user_query = input("\n💬 Your research question (or 'quit'): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_query.strip():
            continue
        
        try:
            # Execute ReAct research
            final_report = orchestrator.research(user_query)
            
            print("\n" + "="*70)
            print("📄 FINAL REPORT")
            print("="*70)
            print(f"\n{final_report}\n")
            
            # Option to see ReAct trace
            show_trace = input("\n🔍 Show ReAct trace? (y/n): ").lower()
            if show_trace == 'y':
                print("\n" + "="*70)
                print("🔁 REACT TRACE")
                print("="*70)
                print(orchestrator.get_trace_summary())
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

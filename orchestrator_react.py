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


class PerplexityClient:
    """Client for Perplexity API interactions"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found. Please provide API key.")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.1-sonar-small-128k-online", temperature: float = 0.2) -> str:
        """Send chat request to Perplexity API"""
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
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Bad Request (400): {error_detail}. Check API key and model name.")
            elif response.status_code == 401:
                raise Exception("Unauthorized (401): Invalid API key")
            elif response.status_code == 429:
                raise Exception("Rate Limit (429): Too many requests")
            
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception(f"Unexpected API response format: {result}")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Perplexity API error: {str(e)}")


class ReActOrchestrator:
    """ReAct-based orchestrator that iteratively reasons and acts"""
    
    SPECIALIST_AGENTS = {
        "business_analyst": {
            "description": "Analyzes 10-K filings, financial statements, business models, competitive positioning, and strategic risks",
            "capabilities": ["Financial statement analysis", "Competitive intelligence", "Risk assessment", "Business model evaluation"],
            "keywords": ["10-K", "10-Q", "filing", "risk", "competitive", "business model", "MD&A"]
        },
        "quantitative_analyst": {
            "description": "Performs quantitative analysis including financial ratios, valuation metrics, growth rates, and statistical modeling",
            "capabilities": ["DCF valuation", "Ratio analysis", "Trend forecasting", "Comparative metrics"],
            "keywords": ["calculate", "ratio", "P/E", "margin", "DCF", "valuation", "CAGR", "growth"]
        },
        "market_analyst": {
            "description": "Tracks market sentiment, technical indicators, price movements, trading volumes, and market positioning",
            "capabilities": ["Sentiment analysis", "Technical analysis", "Market trends", "Trading patterns"],
            "keywords": ["sentiment", "price", "stock", "volume", "technical", "chart", "analyst rating"]
        },
        "industry_analyst": {
            "description": "Provides sector-specific insights, industry trends, regulatory landscape, and peer benchmarking",
            "capabilities": ["Industry dynamics", "Regulatory analysis", "Peer comparison", "Sector trends"],
            "keywords": ["industry", "sector", "peers", "competitors", "market share", "regulation"]
        },
        "esg_analyst": {
            "description": "Evaluates environmental, social, and governance factors, sustainability initiatives, and ESG risk exposure",
            "capabilities": ["ESG scoring", "Sustainability assessment", "Governance evaluation", "Climate risk"],
            "keywords": ["ESG", "sustainability", "carbon", "governance", "diversity", "environmental"]
        },
        "macro_analyst": {
            "description": "Analyzes macroeconomic factors, interest rates, currency impacts, geopolitical risks affecting the company",
            "capabilities": ["Economic indicators", "Rate sensitivity", "FX exposure", "Geopolitical risk"],
            "keywords": ["interest rate", "inflation", "GDP", "currency", "FX", "geopolitical", "macro"]
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
    
    def _create_reasoning_prompt(self, user_query: str, iteration: int, history: str) -> str:
        """Create prompt for reasoning step"""
        
        agents_list = "\n".join([
            f"- {name}: {info['description']}\n  Keywords: {', '.join(info['keywords'][:5])}"
            for name, info in self.SPECIALIST_AGENTS.items()
        ])
        
        available_agents_status = "\n".join([
            f"- {name}: {'✅ Available' if name in self.specialist_agents else '⏳ Planned (will use placeholder)'}"
            for name in self.SPECIALIST_AGENTS.keys()
        ])
        
        prompt = f"""You are a ReAct-based Research Orchestrator using iterative reasoning and acting.

USER QUERY:
{user_query}

CURRENT ITERATION: {iteration}/{self.max_iterations}

AVAILABLE SPECIALIST AGENTS:
{agents_list}

AGENT AVAILABILITY:
{available_agents_status}

PREVIOUS ACTIONS & OBSERVATIONS:
{history if history else "[No previous actions - this is the first iteration]"}

REACT FRAMEWORK:
You must follow the Thought → Action → Observation loop.

Your task is to THINK about:
1. What information do we have so far?
2. What information is still missing to answer the user's query?
3. Which specialist agent(s) should we call next, if any?
4. Are we ready to synthesize a final answer?

AVAILABLE ACTIONS:
- call_specialist: Call a specialist agent with a specific task
- finish: We have enough information, proceed to synthesis

RESPOND IN THIS JSON FORMAT:
{{
  "thought": "Your reasoning about current state and what to do next",
  "action": "call_specialist" | "finish",
  "agent_name": "agent_name" (only if action is call_specialist),
  "task_description": "specific task" (only if action is call_specialist),
  "reasoning": "Why this action makes sense given the query and history"
}}

REMEMBER:
- Don't call the same agent twice for the same information
- If you have enough information from previous iterations, choose 'finish'
- Be efficient - don't over-orchestrate
- Consider query complexity when deciding number of agents

Provide ONLY valid JSON, no other text."""
        return prompt
    
    def _reason(self, user_query: str, iteration: int) -> Action:
        """Reasoning step: Decide what to do next"""
        print(f"\n🧠 [THOUGHT {iteration}] Reasoning about next action...")
        
        history = self.trace.get_history_summary()
        prompt = self._create_reasoning_prompt(user_query, iteration, history)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat(messages, temperature=0.1)
        except Exception as e:
            print(f"   ⚠️ Error calling Perplexity API: {str(e)}")
            # Fallback: finish and synthesize
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
            print(f"   💭 {thought_content}")
            
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
                print(f"      Task: {action.task_description}")
            
            return action
            
        except Exception as e:
            print(f"   ⚠️ Error parsing reasoning: {e}")
            print(f"   Raw response: {response[:200]}")
            # Fallback: finish and synthesize
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
            # This will be handled by the main loop
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
                print(f"   ✅ {agent_name} completed")
                return result
            except Exception as e:
                print(f"   ❌ {agent_name} error: {e}")
                return f"Error executing {agent_name}: {str(e)}"
        else:
            # Placeholder for unimplemented agents
            print(f"   ⏳ {agent_name} not yet implemented - using placeholder")
            result = f"[{agent_name.upper()} - PLACEHOLDER]\n"
            result += f"Would analyze: {task}\n"
            result += "This agent will be implemented in future versions."
            return result
    
    def _synthesize(self, user_query: str) -> str:
        """Synthesize all observations into final report"""
        print(f"\n📊 [SYNTHESIS] Combining insights from {len(self.trace.observations)} observations...")
        
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
            return "Unable to generate report. No specialist agent outputs available."
        
        # Create synthesis prompt
        outputs_text = "\n\n".join([
            f"=== {output['agent'].upper().replace('_', ' ')} ==="
            f"\nTask: {output['task']}"
            f"\nResult:\n{output['result']}\n"
            f"{'=' * 60}"
            for output in specialist_outputs
        ])
        
        reasoning_summary = "\n".join([
            f"{i+1}. {thought.content}"
            for i, thought in enumerate(self.trace.thoughts)
        ])
        
        prompt = f"""You are a Senior Equity Research Analyst synthesizing insights from a multi-agent research process.

ORIGINAL QUERY:
{user_query}

REACT REASONING PROCESS:
{reasoning_summary}

SPECIALIST AGENT OUTPUTS:
{outputs_text}

YOUR TASK:
1. Synthesize all specialist insights into a coherent, comprehensive report
2. Maintain citations and data provenance from source agents
3. Identify key themes and cross-validate findings
4. Provide actionable insights
5. Highlight any gaps or contradictions

REPORT STRUCTURE:
## Executive Summary
[2-3 sentences capturing key findings]

## Detailed Analysis
[Organized by theme, integrating insights from multiple agents]

## Key Metrics & Data Points
[Important quantitative findings with sources]

## Risk Factors & Considerations
[Critical risks identified]

## Conclusion
[Balanced assessment and implications]

Provide a professional, well-structured equity research report."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            final_report = self.client.chat(messages, temperature=0.3)
            print("   ✅ Synthesis complete")
            return final_report
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            return f"Error generating synthesis: {str(e)}\n\nPartial information gathered:\n{outputs_text}"
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop: Iteratively reason and act until task is complete"""
        print("\n" + "="*70)
        print("🔁 REACT-BASED EQUITY RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"\n🔄 Max Iterations: {self.max_iterations}")
        
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
            print(f"\n👁️ [OBSERVATION] {observation.result[:150]}..." if len(observation.result) > 150 else f"\n👁️ [OBSERVATION] {observation.result}")
            
            # Check if we should finish
            if action.action_type in [ActionType.FINISH, ActionType.SYNTHESIZE]:
                print("\n🎯 ReAct loop complete - proceeding to synthesis")
                break
        
        # Final synthesis
        print("\n" + "="*70)
        print("📝 FINAL SYNTHESIS")
        print("="*70)
        
        final_report = self._synthesize(user_query)
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

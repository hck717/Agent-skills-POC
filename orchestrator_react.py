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
    """Client for Perplexity API interactions"""
    
    # Valid Perplexity Sonar models (as of 2026)
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
        # Validate model
        if model not in self.VALID_MODELS:
            print(f"⚠️ Warning: Model '{model}' not in known list. Using 'sonar' instead.")
            print(f"Valid models: {', '.join(self.VALID_MODELS.keys())}")
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
            raise Exception("Request timeout (60s). The query may be too complex.")
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
        """Create SIMPLIFIED prompt for reasoning step"""
        
        available_agents_status = "\n".join([
            f"- {name}: {'AVAILABLE' if name in self.specialist_agents else 'NOT AVAILABLE'}"
            for name in self.SPECIALIST_AGENTS.keys()
        ])
        
        called_agents = self.trace.get_specialist_calls()
        
        # SIMPLIFIED PROMPT - focuses on clear JSON output
        prompt = f"""You are a research orchestrator. Decide the next action.

QUERY: {user_query}

ITERATION: {iteration}/{self.max_iterations}

AVAILABLE AGENTS:
{available_agents_status}

ALREADY CALLED: {', '.join(called_agents) if called_agents else 'None'}

RULES:
- If iteration 1 and business_analyst is AVAILABLE: call it
- If iteration 2-3: call other available agents
- If iteration 4-5: finish and synthesize

Return ONLY this JSON (no other text):
{{
  "thought": "brief reasoning",
  "action": "call_specialist" or "finish",
  "agent_name": "business_analyst",
  "task_description": "the specific task"
}}
"""
        return prompt
    
    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from response with multiple fallback strategies"""
        # Strategy 1: Direct parse
        try:
            return json.loads(response.strip())
        except:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            try:
                return json.loads(json_str)
            except:
                pass
        
        if "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(json_str)
            except:
                pass
        
        # Strategy 3: Find JSON object with regex
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        # Strategy 4: Build JSON from common patterns
        # Look for action and agent_name in text
        action_match = re.search(r'["\']action["\']\s*:\s*["\']([^"\']*)["\'']', response)
        agent_match = re.search(r'["\']agent_name["\']\s*:\s*["\']([^"\']*)["\'']', response)
        
        if action_match:
            return {
                "thought": "Extracted from response",
                "action": action_match.group(1),
                "agent_name": agent_match.group(1) if agent_match else "business_analyst",
                "task_description": "Analyze the query"
            }
        
        raise ValueError(f"Could not extract JSON from: {response[:300]}")
    
    def _reason(self, user_query: str, iteration: int) -> Action:
        """Reasoning step: Decide what to do next"""
        print(f"\n🧠 [THOUGHT {iteration}] Reasoning about next action...")
        
        history = self.trace.get_history_summary()
        prompt = self._create_reasoning_prompt(user_query, iteration, history)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Use smaller, faster sonar model
            response = self.client.chat(messages, model="sonar", temperature=0.1)
            print(f"   📥 Response length: {len(response)} chars")
            
        except Exception as e:
            print(f"   ⚠️ Error calling Perplexity API: {str(e)}")
            # Fallback: call business_analyst if available
            if iteration == 1 and "business_analyst" in self.specialist_agents:
                print(f"   ⤵️ Fallback: Calling business_analyst")
                return Action(
                    action_type=ActionType.CALL_SPECIALIST,
                    agent_name="business_analyst",
                    task_description=user_query,
                    reasoning="API error fallback"
                )
            return Action(action_type=ActionType.FINISH)
        
        try:
            # Extract JSON
            data = self._extract_json_from_response(response)
            
            # Validate
            if "action" not in data:
                raise ValueError("Missing 'action' field")
            
            thought_content = data.get("thought", "No thought provided")
            self.trace.add_thought(thought_content, iteration)
            print(f"   💭 {thought_content[:100]}")
            
            # Create action
            action_type = ActionType(data["action"])
            action = Action(
                action_type=action_type,
                agent_name=data.get("agent_name"),
                task_description=data.get("task_description", user_query),
                reasoning=data.get("reasoning", "")
            )
            
            print(f"   ⚡ Action: {action.action_type.value}")
            if action.agent_name:
                print(f"   🤖 Agent: {action.agent_name}")
                print(f"   📋 Task: {action.task_description[:80]}..." if len(action.task_description or "") > 80 else f"   📋 Task: {action.task_description}")
            
            return action
            
        except Exception as e:
            print(f"   ⚠️ JSON parse error: {e}")
            print(f"   📄 Response preview: {response[:200]}...")
            
            # Intelligent fallback based on iteration
            if iteration == 1 and "business_analyst" in self.specialist_agents:
                print(f"   ⤵️ Fallback: Calling business_analyst (iteration 1)")
                self.trace.add_thought(f"Parse error, fallback to business_analyst", iteration)
                return Action(
                    action_type=ActionType.CALL_SPECIALIST,
                    agent_name="business_analyst",
                    task_description=user_query,
                    reasoning="JSON parse error fallback"
                )
            else:
                print(f"   ⤵️ Fallback: Finishing (iteration {iteration})")
                self.trace.add_thought(f"Parse error on iteration {iteration}, finishing", iteration)
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
    
    def _extract_document_sources(self, specialist_outputs: List[Dict]) -> Dict[int, str]:
        """
        Extract actual document sources from specialist outputs
        Returns mapping of citation number to document source
        """
        document_sources = {}
        citation_num = 1
        
        for output in specialist_outputs:
            content = output['result']
            
            # Pattern to match: --- SOURCE: filename.pdf (Page X) ---
            source_pattern = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            sources = re.findall(source_pattern, content, re.IGNORECASE)
            
            for filename, page in sources:
                filename = filename.strip()
                page = page.strip()
                source_key = f"{filename} - Page {page}"
                
                # Only add if not duplicate
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    citation_num += 1
            
            # If no sources found but has content (placeholder agents), add agent as source
            if not sources and "PLACEHOLDER" in content:
                agent_name = output['agent'].replace('_', ' ').title()
                document_sources[citation_num] = f"{agent_name} (Placeholder Analysis)"
                citation_num += 1
        
        return document_sources
    
    def _synthesize(self, user_query: str) -> str:
        """Synthesize all observations into final report WITH DOCUMENT-BASED REFERENCES"""
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
            print("   ⚠️ No specialist outputs found")
            reasoning_summary = "\n".join([
                f"{i+1}. {thought.content}"
                for i, thought in enumerate(self.trace.thoughts)
            ])
            return f"""## Analysis Summary

Based on the ReAct reasoning process:

{reasoning_summary}

## Note

No specialist agents were called. Ensure:
- Business Analyst agent is registered
- Documents exist in ./data/ folder
- Check console for errors
"""
        
        # Extract actual document sources
        document_sources = self._extract_document_sources(specialist_outputs)
        
        print(f"   📚 Found {len(document_sources)} unique document sources")
        
        # Create outputs text with SOURCE markers numbered
        outputs_with_citations = []
        for i, output in enumerate(specialist_outputs):
            content = output['result']
            
            # Replace SOURCE markers with citation numbers
            source_pattern = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            
            def replace_source(match):
                filename = match.group(1).strip()
                page = match.group(2).strip()
                source_key = f"{filename} - Page {page}"
                
                for num, doc in document_sources.items():
                    if doc == source_key:
                        return f"[CITE:{num}]"
                return match.group(0)
            
            content_with_cites = re.sub(source_pattern, replace_source, content, flags=re.IGNORECASE)
            
            outputs_with_citations.append({
                'agent': output['agent'],
                'task': output['task'],
                'content': content_with_cites
            })
        
        # Format outputs for synthesis
        outputs_text = "\n\n".join([
            f"{'='*60}\n"
            f"SPECIALIST: {output['agent'].upper().replace('_', ' ')}\n"
            f"{'='*60}\n"
            f"Task: {output['task']}\n\n"
            f"Analysis:\n{output['content']}\n"
            for output in outputs_with_citations
        ])
        
        # Create document reference list
        references_list = "\n".join([
            f"[{num}] {doc}"
            for num, doc in sorted(document_sources.items())
        ])
        
        prompt = f"""Synthesize a research report from specialist outputs.

QUERY: {user_query}

SPECIALIST OUTPUTS:

{outputs_text}

DOCUMENT SOURCES:
{references_list}

Create a report with:
1. Executive Summary (3-4 sentences)
2. Detailed Analysis (organized by theme)
3. Key Findings & Metrics
4. Risk Factors
5. Conclusion
6. References section listing all document sources

Cite facts using [1], [2], [3] corresponding to document sources above.
DO NOT cite specialist agents - cite the actual documents.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            print("   🔄 Generating synthesis...")
            final_report = self.client.chat(messages, model="sonar", temperature=0.3)
            print("   ✅ Synthesis complete")
            return final_report
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            # Fallback: return raw outputs
            fallback = f"""## Research Report

{outputs_text}

---

## 📚 References

{references_list}

---

**Note**: Synthesis failed ({str(e)}). Showing raw specialist outputs.
"""
            return fallback
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop: Iteratively reason and act until task is complete"""
        print("\n" + "="*70)
        print("🔁 REACT-BASED EQUITY RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"🔄 Max Iterations: {self.max_iterations}")
        print(f"📊 Registered Agents: {', '.join(self.specialist_agents.keys()) if self.specialist_agents else 'None'}")
        
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
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️ Please set PERPLEXITY_API_KEY environment variable")
        print("   export PERPLEXITY_API_KEY='your-key'")
        return
    
    try:
        orchestrator = ReActOrchestrator(max_iterations=5)
        
        if not orchestrator.test_connection():
            print("\n❌ Failed to connect to Perplexity API")
            return
            
    except ValueError as e:
        print(f"❌ {str(e)}")
        return
    
    print("\n🚀 ReAct-Based Orchestrator Ready")
    print("\nAvailable specialist agents:")
    for name, info in ReActOrchestrator.SPECIALIST_AGENTS.items():
        status = "✅" if name in orchestrator.specialist_agents else "⏳"
        print(f"  {status} {name}")
    
    while True:
        print("\n" + "="*70)
        user_query = input("\n💬 Your research question (or 'quit'): ")
        
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

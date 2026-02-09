#!/usr/bin/env python3
"""
Simplified Multi-Agent Orchestrator using Ollama for reasoning

This orchestrator uses Ollama (local LLM) for ReAct-style reasoning
and optionally Perplexity for final synthesis.

Benefits:
- Local reasoning (free, fast, no API limits)
- Better structured output (Ollama follows JSON instructions)
- Perplexity optional for synthesis
"""

import os
import json
import requests
from typing import List, Dict, Optional


class SimpleOrchestrator:
    """Simplified orchestrator with direct specialist calls"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.specialist_agents = {}
        self.call_history = []
        
    def register_specialist(self, agent_name: str, agent_instance):
        """Register a specialist agent"""
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def research(self, user_query: str) -> str:
        """Execute research with specialist agents"""
        print("\n" + "="*70)
        print("🔬 SIMPLIFIED EQUITY RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"📊 Available Specialists: {', '.join(self.specialist_agents.keys())}")
        
        # Call all registered specialists
        results = []
        
        for agent_name, agent in self.specialist_agents.items():
            print(f"\n🤖 Calling {agent_name}...")
            try:
                result = agent.analyze(user_query)
                results.append({
                    'agent': agent_name,
                    'result': result
                })
                print(f"   ✅ {agent_name} completed ({len(result)} chars)")
            except Exception as e:
                print(f"   ❌ {agent_name} error: {e}")
                results.append({
                    'agent': agent_name,
                    'result': f"Error: {str(e)}"
                })
        
        # If we got results, return them directly
        if results:
            print("\n" + "="*70)
            print("📝 COMPILING RESULTS")
            print("="*70)
            
            report = "# Research Report\n\n"
            report += f"**Query:** {user_query}\n\n"
            report += "---\n\n"
            
            for res in results:
                report += f"## {res['agent'].replace('_', ' ').title()} Analysis\n\n"
                report += res['result']
                report += "\n\n---\n\n"
            
            print("✅ Report compiled")
            return report
        else:
            return "No specialist agents available. Please register agents first."
    
    def get_trace_summary(self) -> str:
        """Get execution history"""
        if not self.call_history:
            return "No execution history yet."
        
        summary = []
        for i, call in enumerate(self.call_history, 1):
            summary.append(f"\n=== Call {i} ===")
            summary.append(f"Agent: {call['agent']}")
            summary.append(f"Task: {call['task'][:100]}...")
            summary.append(f"Success: {call['success']}")
        
        return "\n".join(summary)


class OllamaOrchestrator:
    """Orchestrator using Ollama for ReAct reasoning"""
    
    def __init__(self, 
                 ollama_base_url: str = "http://localhost:11434",
                 ollama_model: str = "qwen2.5:7b",
                 max_iterations: int = 3):
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.max_iterations = max_iterations
        self.specialist_agents = {}
        self.trace = []
        
    def register_specialist(self, agent_name: str, agent_instance):
        """Register a specialist agent"""
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def _reason(self, user_query: str, iteration: int, history: List[str]) -> Dict:
        """Use Ollama to reason about next action"""
        
        agents_list = "\n".join([f"- {name}" for name in self.specialist_agents.keys()])
        history_text = "\n".join(history) if history else "[No previous actions]"
        
        prompt = f"""You are a research orchestrator. Decide the next action.

USER QUERY: {user_query}

ITERATION: {iteration}/{self.max_iterations}

AVAILABLE AGENTS:
{agents_list}

HISTORY:
{history_text}

Your task: Respond with JSON only:
{{
  "action": "call_agent" or "finish",
  "agent_name": "business_analyst" (if call_agent),
  "reasoning": "why this action"
}}

Rules:
- If iteration 1 and business_analyst available: call it
- If all agents called: finish
- Otherwise: call next useful agent

JSON response:"""
        
        response = self._call_ollama(prompt)
        
        # Extract JSON
        try:
            # Try to find JSON in response
            import re
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return json.loads(response)
        except:
            # Fallback: call business_analyst if available
            if "business_analyst" in self.specialist_agents and iteration == 1:
                return {
                    "action": "call_agent",
                    "agent_name": "business_analyst",
                    "reasoning": "Fallback action"
                }
            return {"action": "finish", "reasoning": "Parse error fallback"}
    
    def research(self, user_query: str) -> str:
        """Execute research with ReAct-style reasoning"""
        print("\n" + "="*70)
        print("🧠 OLLAMA-POWERED RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 Query: {user_query}")
        print(f"🔄 Max Iterations: {self.max_iterations}")
        print(f"📊 Registered Agents: {', '.join(self.specialist_agents.keys())}")
        
        self.trace = []
        results = []
        history = []
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Reason
            try:
                decision = self._reason(user_query, iteration, history)
                print(f"💭 Reasoning: {decision.get('reasoning', 'N/A')}")
                print(f"⚡ Action: {decision['action']}")
                
                if decision['action'] == 'finish':
                    print("🎯 Finishing orchestration")
                    break
                
                if decision['action'] == 'call_agent':
                    agent_name = decision.get('agent_name')
                    if agent_name in self.specialist_agents:
                        print(f"🤖 Calling {agent_name}...")
                        agent = self.specialist_agents[agent_name]
                        result = agent.analyze(user_query)
                        results.append({
                            'agent': agent_name,
                            'result': result
                        })
                        history.append(f"Called {agent_name}")
                        print(f"   ✅ Completed ({len(result)} chars)")
                    else:
                        print(f"   ⚠️ Agent {agent_name} not found")
                        break
                        
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                break
        
        # Compile results
        if results:
            print("\n" + "="*70)
            print("📝 COMPILING FINAL REPORT")
            print("="*70)
            
            report = "# Research Report\n\n"
            report += f"**Query:** {user_query}\n\n"
            report += "---\n\n"
            
            for res in results:
                report += f"## {res['agent'].replace('_', ' ').title()} Analysis\n\n"
                report += res['result']
                report += "\n\n---\n\n"
            
            print(f"✅ Report compiled - {len(results)} specialist(s) called")
            return report
        else:
            return "No analysis available. Check if agents are registered and Ollama is running."
    
    def get_trace_summary(self) -> str:
        """Get execution trace"""
        return "\n".join(self.trace) if self.trace else "No trace available"


def main():
    """Demo"""
    print("🚀 Orchestrator Demo")
    print("\nOption 1: Simple (direct calls)")
    print("Option 2: Ollama-based (ReAct reasoning)")
    
    choice = input("\nChoose (1/2): ").strip()
    
    if choice == "1":
        orch = SimpleOrchestrator()
    else:
        orch = OllamaOrchestrator()
    
    # Register Business Analyst
    from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
    ba = BusinessAnalystGraphAgent(data_path="./data", db_path="./storage/chroma_db")
    orch.register_specialist("business_analyst", ba)
    
    # Run query
    query = "What are Apple's competitive risks according to their 10-K?"
    report = orch.research(query)
    
    print("\n" + "="*70)
    print("📄 FINAL REPORT")
    print("="*70)
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Rule-based orchestration with HYBRID LOCAL LLM synthesis:
- DeepSeek-R1 8B: Deep reasoning for specialist analysis AND Synthesis (Upgraded for Quality)
- Qwen 2.5 7B: Backup / Legacy

Version: 3.6 - Senior PM Persona & Dynamic Synthesis
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import requests
from dotenv import load_dotenv

try:
    from scripts.ingest_documents_ba import DocumentIngester
except ImportError:
    DocumentIngester = None
    print("⚠️ Could not import DocumentIngester. Auto-ingestion disabled.")

load_dotenv()

try:
    from skills.business_analyst_crag import BusinessAnalystCRAG
except ImportError:
    print("⚠️ Could not import BusinessAnalystCRAG. Ensure skills/business_analyst_crag exists.")
    BusinessAnalystCRAG = None

try:
    from skills.web_search_agent import WebSearchAgent
except ImportError:
    print("⚠️ Could not import WebSearchAgent.")
    WebSearchAgent = None

class ActionType(Enum):
    CALL_SPECIALIST = "call_specialist"
    SYNTHESIZE = "synthesize"
    FINISH = "finish"
    REFINE_QUERY = "refine_query"

@dataclass
class Thought:
    content: str
    iteration: int

@dataclass
class Action:
    action_type: ActionType
    agent_name: Optional[str] = None
    task_description: Optional[str] = None
    reasoning: Optional[str] = None
    ticker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Observation:
    action: Action
    result: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReActTrace:
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    
    def add_thought(self, content: str, iteration: int):
        self.thoughts.append(Thought(content=content, iteration=iteration))
    
    def add_action(self, action: Action):
        self.actions.append(action)
    
    def add_observation(self, obs: Observation):
        self.observations.append(obs)
    
    def get_specialist_calls(self) -> List[str]:
        return [action.agent_name for action in self.actions if action.action_type == ActionType.CALL_SPECIALIST and action.agent_name]

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:8b"):
        self.base_url = base_url
        self.model = model
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, num_predict: int = 4000, timeout: int = None) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict}
        }
        try:
            # REMOVED timeout=600 to allow unlimited processing time
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def test_connection(self) -> bool:
        try:
            print("🔌 Testing Ollama connection...")
            if self.chat([{"role": "user", "content": "hi"}], num_predict=10):
                print("✅ Ollama connected!")
                return True
            return False
        except:
            return False

class ReActOrchestrator:
    ANALYSIS_MODEL = "deepseek-r1:8b" 
    SYNTHESIS_MODEL = "deepseek-r1:8b"
    
    SPECIALIST_AGENTS = {
        "business_analyst_crag": {
            "description": "Deep Reader (v2) - Graph-Augmented Corrective RAG for 10-K analysis",
            "priority": 1
        },
        "web_search_agent": {
            "description": "Supplements document analysis with current web info",
            "priority": 2
        }
    }
    
    def __init__(self, ollama_url: str = "http://localhost:11434", max_iterations: int = 3):
        self.client = OllamaClient(base_url=ollama_url, model=self.ANALYSIS_MODEL)
        self.ollama_url = ollama_url
        self.max_iterations = max_iterations
        self.specialist_agents = {}
        self.trace = ReActTrace()
        self.current_ticker = None
        self.current_metadata = {}

        # Initialize Web Search Agent first (for CRAG fallback)
        web_agent = None
        if WebSearchAgent:
            try:
                web_agent = WebSearchAgent()
                self.register_specialist("web_search_agent", web_agent)
                print("✅ Web Search Agent initialized for CRAG fallback")
            except Exception as e:
                print(f"⚠️ Web Search Agent init failed: {e}")
        
        # Initialize Business Analyst with web fallback
        if BusinessAnalystCRAG:
            try:
                self.register_specialist("business_analyst_crag", BusinessAnalystCRAG(
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                    neo4j_pass=os.getenv("NEO4J_PASSWORD", "password"),
                    web_search_agent=web_agent  # Enable CRAG fallback chain
                ))
            except Exception as e:
                print(f"⚠️ Failed to auto-register business_analyst_crag: {e}")
                
        if WebSearchAgent:
            try:
                self.register_specialist("web_search_agent", WebSearchAgent())
            except Exception as e:
                print(f"⚠️ Failed to auto-register web_search_agent: {e}")
        
    def register_specialist(self, agent_name: str, agent_instance):
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def test_connection(self) -> bool:
        return self.client.test_connection()
    
    def _extract_ticker_and_meta(self, user_query: str, callback: Optional[Callable] = None) -> Tuple[Optional[str], Dict]:
        if callback:
             callback("Pre-Processing", "Identifying company and scope...", "running")
             
        # 1. Try Simple Keyword Match First
        ticker = None
        q_lower = user_query.lower()
        if "microsoft" in q_lower or "msft" in q_lower: ticker = "MSFT"
        elif "apple" in q_lower or "aapl" in q_lower: ticker = "AAPL"
        elif "tesla" in q_lower or "tsla" in q_lower: ticker = "TSLA"
        elif "nvidia" in q_lower or "nvda" in q_lower: ticker = "NVDA"
        elif "google" in q_lower or "goog" in q_lower: ticker = "GOOGL"
        elif "amazon" in q_lower or "amzn" in q_lower: ticker = "AMZN"
        
        # 2. Use LLM for specific Metadata extraction
        prompt = f"""
        Analyze this query: '{user_query}'
        
        Extract:
        1. Ticker (e.g., MSFT, AAPL). If none, say NONE.
        2. Years (e.g., [2024, 2025]). If none, return empty list.
        3. Topics (e.g., ["Risk", "Revenue", "Strategy"]). Pick from: [Risk, Strategy, Financials, Product, Management].
        
        Output ONLY valid JSON:
        {{
            "ticker": "MSFT",
            "years": [2025],
            "topics": ["Risk"]
        }}
        """
        try:
            response = self.client.chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=150)
            clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            metadata = {}
            match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                llm_ticker = data.get("ticker", "NONE")
                
                if ticker is None and llm_ticker != "NONE":
                    ticker = re.sub(r'[^A-Z]', '', llm_ticker.upper())
                
                metadata = {
                    "years": data.get("years", []),
                    "topics": data.get("topics", [])
                }
            
            if ticker is None:
                print("   ⚠️ No ticker identified. Defaulting to AAPL for POC.")
                ticker = "AAPL" 

            print(f"   🔍 Target: {ticker} | Meta: {metadata}")
            if callback: callback("Pre-Processing", f"Identified: {ticker} ({metadata.get('years', 'All Time')})", "complete")
            return ticker, metadata
                
        except Exception as e:
            print(f"Meta extraction failed: {e}")
            return ticker if ticker else "AAPL", {}

    def _auto_seed_graph(self, ticker: str, callback: Optional[Callable] = None):
        """Auto-clean and re-ingest documents on every query"""
        if not DocumentIngester: return
        data_path = os.path.join("data", ticker)
        
        if not os.path.exists(data_path):
            print(f"   ⚠️ No local data found for {ticker} in {data_path}")
            if callback: callback("Graph Ingestion", f"No data for {ticker}", "error")
            return

        if callback: callback("Graph Ingestion", f"Cleaning & Ingesting {ticker}...", "running")
        try:
            # Create ingester
            ingester = DocumentIngester(
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_pass=os.getenv("NEO4J_PASSWORD", "password"),
                ticker=ticker
            )
            
            # Always reset=True to clean and re-ingest
            # max_chunks=20 for quick ingestion (remove for full)
            ingester.ingest(reset=True, max_chunks=20)
            
            if callback: callback("Graph Ingestion", f"✅ Graph Ready ({ticker})", "complete")
        except Exception as e:
            print(f"   ❌ Ingestion failed: {e}")
            if callback: callback("Graph Ingestion", f"Failed: {str(e)}", "error")

    def _reason_rule_based(self, user_query: str, iteration: int, callback: Optional[Callable] = None) -> Action:
        if callback: callback(f"Iteration {iteration}", "Reasoning (Rule-Based)", "running")
        print(f"\n🧠 [THOUGHT {iteration}] Rule-based reasoning...")
        
        called_agents = self.trace.get_specialist_calls()
        
        # Rule 1: Always start with Business Analyst (CRAG)
        if "business_analyst_crag" in self.specialist_agents and "business_analyst_crag" not in called_agents:
            return Action(
                ActionType.CALL_SPECIALIST, 
                "business_analyst_crag", 
                user_query, 
                "Rule 1: Deep Reader", 
                self.current_ticker,
                self.current_metadata
            )

        # Rule 2: ALWAYS call Web Search next for Hybrid Context
        if "web_search_agent" in self.specialist_agents and "web_search_agent" not in called_agents:
            print("   💡 Forced Web Search for Hybrid Report")
            return Action(
                ActionType.CALL_SPECIALIST, 
                "web_search_agent", 
                user_query, 
                "Rule 2: Forced Web Context", 
                self.current_ticker,
                self.current_metadata
            )
        
        return Action(ActionType.FINISH, reasoning="Analysis complete")
    
    def _execute_action(self, action: Action, callback: Optional[Callable] = None) -> Observation:
        print(f"\n⚙️ [ACTION] Executing {action.action_type.value}...")
        
        if action.action_type == ActionType.CALL_SPECIALIST:
            if callback: callback(f"Action: {action.agent_name}", f"Executing {action.agent_name}...", "running")
            result = self._call_specialist(action.agent_name, action.task_description, action.ticker, action.metadata)
            if callback: callback(f"Action: {action.agent_name}", f"Completed {action.agent_name}", "complete")
            return Observation(action, result, True, {"agent": action.agent_name})
        
        return Observation(action, "Orchestration complete", True)
    
    def _call_specialist(self, agent_name: str, task: str, ticker: str = None, metadata: Dict = {}) -> str:
        print(f"   🤖 Calling {agent_name} (Ticker: {ticker} | Meta: {metadata})...")
        if agent_name in self.specialist_agents:
            agent = self.specialist_agents[agent_name]
            try:
                import inspect
                sig = inspect.signature(agent.analyze)
                kwargs = {}
                if 'ticker' in sig.parameters: kwargs['ticker'] = ticker or "AAPL"
                if 'prior_analysis' in sig.parameters: kwargs['prior_analysis'] = "" 
                if 'metadata' in sig.parameters: kwargs['metadata'] = metadata 
                
                result = agent.analyze(task, **kwargs)
                print(f"   ✅ {agent_name} completed ({len(result)} chars)")
                return result
            except Exception as e:
                print(f"   ❌ {agent_name} error: {e}")
                return f"Error executing {agent_name}: {str(e)}"
        return "Agent not found"
    
    def _extract_sources(self, specialist_outputs: List[Dict]) -> Dict[int, str]:
        sources = {}
        idx = 1
        for output in specialist_outputs:
            content = output['result']
            # Web
            for title, url in re.findall(r'---\s*SOURCE:\s*([^\(]+)\((https?://[^\)]+)\)\s*---', content, re.IGNORECASE):
                key = f"{title.strip()}||{url.strip()}"
                if key not in sources.values():
                    sources[idx] = key; idx += 1
            # Local (Graph Facts)
            for src in re.findall(r'---\s*SOURCE:\s*([^\(\n]+?)(?:\([^\)]+\))?\s*---', content, re.IGNORECASE):
                src = src.strip()
                if "http" not in src and src not in sources.values():
                    # Clean up ugly Graph citations if needed
                    if "GRAPH FACT" in src:
                        # Convert "GRAPH FACT: MSFT -[HAS_SEGMENT]-> ['Cloud']" -> "Internal Graph Knowledge: Cloud Segment"
                        pass 
                    sources[idx] = src; idx += 1
        return sources

    def _synthesize(self, user_query: str, callback: Optional[Callable] = None) -> str:
        if callback: callback("Synthesis", "Synthesizing final report...", "running")
        
        outputs = []
        for obs in self.trace.observations:
            if obs.action.action_type == ActionType.CALL_SPECIALIST:
                outputs.append({"agent": obs.action.agent_name, "result": obs.result})
        
        if not outputs: return "No analysis available."
        
        sources = self._extract_sources(outputs)
        
        # Replace markers
        for out in outputs:
            def repl(m):
                full = m.group(0)
                key = None
                if "http" in full:
                    wm = re.search(r'\((https?://[^\)]+)\)', full)
                    if wm: key = wm.group(1)
                else:
                    gm = re.search(r'SOURCE:\s*([^\(\n]+)', full)
                    if gm: key = gm.group(1).strip()
                
                if key:
                    for k, v in sources.items():
                        if key in v: return f" [{k}]"
                return ""
            
            out['result'] = re.sub(r'---\s*SOURCE:.*?\s*---', repl, out['result'], flags=re.IGNORECASE)

        context = "\n\n".join([f"FROM {o['agent'].upper()}:\n{o['result']}" for o in outputs])
        
        meta_context = f"Focused Years: {self.current_metadata.get('years', 'All')}\nTopics: {self.current_metadata.get('topics', 'General')}"
        
        prompt = f"""
        Role: Senior Portfolio Manager at a top-tier Multi-Strategy Hedge Fund.
        Date: {datetime.now().strftime('%B %d, %Y')}
        Query: {user_query}
        Constraints: {meta_context}
        
        CONTEXT:
        {context}
        
        TASK:
        Synthesize a high-conviction investment memo or direct answer to the query.
        
        GUIDELINES:
        - **Think like a PM**: Focus on the "So What?", variant views, and what the market might be missing.
        - **Dynamic Structure**: Adapt to the query. If it's a specific question, answer it directly. If it's broad (e.g. "Analyze MSFT"), structure by:
          1. **The Setup**: Current market consensus vs. reality.
          2. **The Edge/Thesis**: Key drivers the market is under/overestimating.
          3. **The Bear Case/Risks**: What kills the trade?
          4. **Catalyst Path**: Specific events to watch.
        - **Tone**: Sophisticated, decisive, concise. Avoid generic corporate descriptions.
        - **Citations**: Cite EVERY claim with [X] based on the Context.
        - **No References Section**: Do not generate a list of references at the end (I will append it programmatically).
        """
        
        try:
            resp = self.client.chat([{"role": "user", "content": prompt}], temperature=0.3)
            report = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
            
            # Add References (Cleaned)
            ref_section = "\n\n## References\n\n"
            for k, v in sources.items():
                if "||" in v:
                    t, u = v.split("||")
                    ref_section += f"[{k}] **{t}** ([Link]({u}))\n"
                else:
                    # Clean up local citations
                    display_text = v
                    if "GRAPH FACT" in v:
                         # Attempt to make it readable: "GRAPH FACT: MSFT -[RISK]-> ['Competition']"
                         match = re.search(r"-> \['(.*?)'\]", v)
                         if match:
                             display_text = f"Internal Graph Knowledge: {match.group(1)}"
                    
                    ref_section += f"[{k}] **{display_text}** (System Authenticated Source)\n"
            
            return report + ref_section
        except Exception as e:
            return f"Synthesis Failed: {e}"

    def research(self, user_query: str, callback: Optional[Callable] = None) -> str:
        self.trace = ReActTrace()
        self.current_ticker = None
        self.current_metadata = {}
        
        # Combined extraction (Ticker + Metadata)
        ticker, metadata = self._extract_ticker_and_meta(user_query, callback)
        
        if ticker:
            self.current_ticker = ticker
            self.current_metadata = metadata
            self._auto_seed_graph(ticker, callback)
            
        for i in range(1, self.max_iterations + 1):
            action = self._reason_rule_based(user_query, i, callback)
            self.trace.add_action(action)
            if action.action_type == ActionType.FINISH: break
            obs = self._execute_action(action, callback)
            self.trace.add_observation(obs)
            
        final = self._synthesize(user_query, callback)
        if callback: callback("Orchestration", "Process Complete", "complete")
        return final

if __name__ == "__main__":
    r = ReActOrchestrator()
    r.research("Analyze Microsoft 2025 risks")

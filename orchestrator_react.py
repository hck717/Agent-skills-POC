#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Rule-based orchestration with HYBRID LOCAL LLM synthesis:
- DeepSeek-R1 8B: Deep reasoning for specialist analysis AND Synthesis (Upgraded for Quality)
- Qwen 2.5 7B: Backup / Legacy

Version: 3.10 - Fix Citation Links & Source Labeling
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

# Import the seeding script
try:
    from scripts.seed_neo4j_ba_graph import seed
except ImportError:
    seed = None
    print("⚠️ Could not import seed script. Auto-seeding disabled.")

# Load environment variables
load_dotenv()

# Import the new agent
try:
    from skills.business_analyst_crag import BusinessAnalystCRAG
except ImportError:
    print("⚠️ Could not import BusinessAnalystCRAG. Ensure skills/business_analyst_crag exists.")
    BusinessAnalystCRAG = None

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
    ticker: Optional[str] = None


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
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:8b"):
        self.base_url = base_url
        self.model = model
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, num_predict: int = 4000, timeout: int = 600) -> str:
        """Send chat request to Ollama API"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def test_connection(self) -> bool:
        try:
            print("🔌 Testing Ollama connection...")
            messages = [{"role": "user", "content": "Say 'OK'."}]
            response = self.chat(messages, temperature=0.0, timeout=30)
            if response:
                print("✅ Ollama connected!")
                return True
            return False
        except Exception as e:
            print(f"❌ Ollama connection failed: {str(e)}")
            return False


class ReActOrchestrator:
    """Rule-based orchestrator with HYBRID LOCAL synthesis (no web interference)"""
    
    ANALYSIS_MODEL = "deepseek-r1:8b" 
    SYNTHESIS_MODEL = "deepseek-r1:8b"
    
    SPECIALIST_AGENTS = {
        "business_analyst_crag": {
            "description": "Deep Reader (v2) - Graph-Augmented Corrective RAG for 10-K analysis",
            "keywords": ["strategy", "risk", "10-k", "deep dive"],
            "priority": 1
        },
        "business_analyst": {
            "description": "Analyzes 10-K filings, financial statements, competitive positioning using RAG",
            "keywords": ["10-K", "10-Q", "filing", "risk", "competitive", "financial"],
            "priority": 2
        },
        "web_search_agent": {
            "description": "Supplements document analysis with current web info (news, market data, analyst opinions)",
            "keywords": ["recent", "current", "latest", "news", "market", "price"],
            "priority": 3
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
        self.client = OllamaClient(base_url=ollama_url, model=self.ANALYSIS_MODEL)
        self.ollama_url = ollama_url
        self.max_iterations = max_iterations
        self.specialist_agents = {}
        self.trace = ReActTrace()
        self.current_ticker = None

        if BusinessAnalystCRAG:
            try:
                self.register_specialist("business_analyst_crag", BusinessAnalystCRAG(
                    qdrant_url=os.getenv("QDRANT_URL", None),
                    qdrant_key=os.getenv("QDRANT_API_KEY", None),
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                    neo4j_pass=os.getenv("NEO4J_PASSWORD", "password")
                ))
            except Exception as e:
                print(f"⚠️ Failed to auto-register business_analyst_crag: {e}")
        
    def register_specialist(self, agent_name: str, agent_instance):
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def test_connection(self) -> bool:
        return self.client.test_connection()
    
    def _extract_ticker(self, user_query: str, callback: Optional[Callable] = None) -> Optional[str]:
        if callback:
             callback("Pre-Processing", "Identifying target company...", "running")
             
        prompt = f"""
        User Query: "{user_query}"
        
        Extract the company ticker symbol (e.g., AAPL). 
        If no specific company, say NONE.
        Output ONLY the ticker.
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat(messages, temperature=0.0, num_predict=50)
            
            clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            ticker = "NONE"
            words = clean_response.split()
            for word in words:
                clean_word = re.sub(r'[^A-Z]', '', word.upper())
                if 2 <= len(clean_word) <= 5 and clean_word not in ["THE", "AND", "FOR", "NONE", "WHAT", "USER"]:
                    ticker = clean_word
                    break
            
            if ticker == "NONE" and "apple" in user_query.lower():
                ticker = "AAPL"
            elif ticker == "NONE" and "microsoft" in user_query.lower():
                ticker = "MSFT"
            
            if ticker == "NONE":
                if callback:
                    callback("Pre-Processing", "No specific company identified", "complete")
                return None
            
            print(f"   🔍 Identified Ticker: {ticker}")
            if callback:
                callback("Pre-Processing", f"Identified Target: {ticker}", "complete")
            return ticker
            
        except Exception as e:
            print(f"   ⚠️ Failed to extract ticker: {e}")
            return None

    def _auto_seed_graph(self, ticker: str, callback: Optional[Callable] = None):
        if not seed:
            return
            
        data_path = os.path.join("data", ticker)
        
        if not os.path.exists(data_path):
             print(f"   ⚠️ No data folder found for {ticker} at ./data/{ticker}")
             if callback:
                 callback("Graph Seeding", f"Skipped: No data found for {ticker}", "error")
             return

        if callback:
            callback("Graph Seeding", f"Resetting & Ingesting {ticker}...", "running")
            
        try:
            seed(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password"),
                ticker=ticker,
                reset=True
            )
            print(f"   ✅ Successfully seeded graph for {ticker}")
            if callback:
                callback("Graph Seeding", f"✅ Graph Ready ({ticker})", "complete")
                
        except Exception as e:
            print(f"   ❌ Seeding failed: {e}")
            if callback:
                callback("Graph Seeding", f"Failed: {str(e)}", "error")

    def _reason_rule_based(self, user_query: str, iteration: int, callback: Optional[Callable] = None) -> Action:
        if callback:
            callback(f"Iteration {iteration}", "Reasoning (Rule-Based)", "running")
            
        print(f"\n🧠 [THOUGHT {iteration}] Rule-based reasoning...")
        
        called_agents = self.trace.get_specialist_calls()
        
        if "business_analyst_crag" in self.specialist_agents and "business_analyst_crag" not in called_agents:
            thought = "Starting with Deep Reader (CRAG) for high-accuracy document analysis"
            self.trace.add_thought(thought, iteration)
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst_crag",
                task_description=user_query,
                reasoning="Rule 1: CRAG Business Analyst provides best-in-class foundation",
                ticker=self.current_ticker
            )

        if "business_analyst" in self.specialist_agents and "business_analyst" not in called_agents and "business_analyst_crag" not in called_agents:
             thought = "Starting with Standard Business Analyst"
             self.trace.add_thought(thought, iteration)
             return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst",
                task_description=user_query,
                reasoning="Fallback: Standard Business Analyst",
                ticker=self.current_ticker
            )
        
        analyst_ran = "business_analyst_crag" in called_agents or "business_analyst" in called_agents
        
        if "web_search_agent" in self.specialist_agents and "web_search_agent" not in called_agents and analyst_ran:
            thought = "Calling Web Search Agent to supplement document analysis with current data"
            self.trace.add_thought(thought, iteration)
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="web_search_agent",
                task_description=user_query,
                reasoning="Rule 2: Web Search Agent supplements with recent info",
                ticker=self.current_ticker
            )
        
        thought = f"Analysis complete with {len(called_agents)} specialists, ready to synthesize"
        self.trace.add_thought(thought, iteration)
        return Action(
            action_type=ActionType.FINISH,
            reasoning="Rule 3: Sufficient analysis gathered"
        )
    
    def _execute_action(self, action: Action, callback: Optional[Callable] = None) -> Observation:
        print(f"\n⚙️ [ACTION] Executing {action.action_type.value}...")
        
        if action.action_type == ActionType.CALL_SPECIALIST:
            if callback:
                callback(f"Action: {action.agent_name}", f"Executing {action.agent_name}...", "running")
            
            result = self._call_specialist(action.agent_name, action.task_description, action.ticker)
            
            if callback:
                callback(f"Action: {action.agent_name}", f"Completed {action.agent_name}", "complete")
            
            return Observation(
                action=action,
                result=result,
                success=True,
                metadata={"agent": action.agent_name}
            )
        
        elif action.action_type == ActionType.FINISH:
            if callback:
                callback("Action: Finish", "Orchestration complete", "complete")
                
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
    
    def _call_specialist(self, agent_name: str, task: str, ticker: str = None) -> str:
        print(f"   🤖 Calling {agent_name} (Ticker: {ticker})...")
        
        if agent_name in self.specialist_agents:
            agent = self.specialist_agents[agent_name]
            try:
                import inspect
                sig = inspect.signature(agent.analyze)
                has_ticker = 'ticker' in sig.parameters
                has_prior = 'prior_analysis' in sig.parameters
                
                if has_ticker and has_prior:
                    prior_analysis = ""
                    if agent_name == "web_search_agent":
                        for obs in self.trace.observations:
                            if obs.action.agent_name in ["business_analyst", "business_analyst_crag"]:
                                prior_analysis = obs.result
                                break
                    result = agent.analyze(task, ticker=ticker or "AAPL", prior_analysis=prior_analysis)
                elif has_ticker:
                    result = agent.analyze(task, ticker=ticker or "AAPL")
                elif has_prior:
                    result = agent.analyze(task, prior_analysis="")
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
            return f"[{agent_name.upper()} PLACEHOLDER]\n\nThis agent would analyze: {task}"
    
    def _extract_document_sources(self, specialist_outputs: List[Dict]) -> Dict[int, str]:
        """
        Extract sources and format for display
        """
        document_sources = {}
        citation_num = 1
        
        for output in specialist_outputs:
            content = output['result']
            
            # Pattern 1: Web sources (Title + URL)
            # Matches: --- SOURCE: Title (http://...) ---
            pattern_web = r'---\s*SOURCE:\s*([^\(]+)\((https?://[^\)]+)\)\s*---'
            web_sources = re.findall(pattern_web, content, re.IGNORECASE)

            # Pattern 2: Document/Graph sources
            # Matches: --- SOURCE: filename/entity ---
            # Excludes URLs
            pattern_generic = r'---\s*SOURCE:\s*([^\(\n]+?)(?:\([^\)]+\))?\s*---'
            all_sources = re.findall(pattern_generic, content, re.IGNORECASE)
            
            # Process Web Sources First (High Priority)
            for title, url in web_sources:
                title = title.strip()
                url = url.strip()
                source_key = f"{title}||{url}"
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    citation_num += 1
            
            # Process Local Sources (PDFs, Graph)
            for src in all_sources:
                src = src.strip()
                # Skip if it looks like a web title we already caught
                if any(src in v for v in document_sources.values()):
                    continue
                # Skip partial matches of URLs
                if "http" in src:
                    continue

                source_key = src
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    citation_num += 1
        
        return document_sources
    
    def _validate_citation_quality(self, report: str, total_sources: int) -> Tuple[int, List[str]]:
        """Validate citation quality"""
        warnings = []
        citations = re.findall(r'\[\d+\]', report)
        
        if not citations:
            warnings.append("❌ CRITICAL: No citations found in report")
            return 0, warnings
            
        sections = report.split('\n\n')
        uncited_paragraphs = 0
        
        for section in sections:
            if (section.strip() and 
                not section.startswith('#') and 
                'References' not in section and
                'Data Quality Notice' not in section and 
                'Temporal Distinction' not in section and
                len(section) > 100 and
                not re.search(r'\[\d+\]', section)):
                uncited_paragraphs += 1
                warnings.append(f"⚠️ Unsourced paragraph: {section[:80]}...")
        
        quality_score = max(0, 100 - (uncited_paragraphs * 10))
        return quality_score, warnings
    
    def _synthesize(self, user_query: str, callback: Optional[Callable] = None) -> str:
        if callback:
            callback("Synthesis", "Synthesizing final report...", "running")
            
        specialist_calls = self.trace.get_specialist_calls()
        current_date = datetime.now().strftime("%B %d, %Y")
        
        print(f"\n📊 [SYNTHESIS] Combining insights...")
        
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
        
        document_sources = self._extract_document_sources(specialist_outputs)
        print(f"   📚 Found {len(document_sources)} unique sources")
        
        # Replace SOURCE markers with [SOURCE-X]
        outputs_with_cites = []
        for output in specialist_outputs:
            content = output['result']
            
            def replace_source(match):
                full_match = match.group(0)
                # Try to extract the core source name/url to match our dict
                
                # Check Web
                web_match = re.search(r'---\s*SOURCE:\s*([^\(]+)\((https?://[^\)]+)\)\s*---', full_match, re.IGNORECASE)
                if web_match:
                    key = f"{web_match.group(1).strip()}||{web_match.group(2).strip()}"
                else:
                    # Generic
                    gen_match = re.search(r'---\s*SOURCE:\s*([^\(\n]+?)(?:\([^\)]+\))?\s*---', full_match, re.IGNORECASE)
                    key = gen_match.group(1).strip() if gen_match else None
                
                if key:
                    for num, val in document_sources.items():
                        if val == key:
                            return f" [SOURCE-{num}]"
                return "" # Remove if not found
            
            content_with_cites = re.sub(
                r'---\s*SOURCE:\s*(.*?)\s*---',
                replace_source,
                content,
                flags=re.IGNORECASE
            )
            outputs_with_cites.append({'agent': output['agent'], 'content': content_with_cites})
        
        outputs_text = "\n\n".join([f"SOURCE: {out['agent'].upper()}\n{out['content']}" for out in outputs_with_cites])
        
        # Construct References String for Prompt
        refs_str = "\n".join([f"[SOURCE-{k}] {v}" for k,v in document_sources.items()])
        
        prompt = f"""You are a Senior Equity Research Analyst.
REPORT DATE: {current_date}
QUERY: {user_query}

ANALYSIS:
{outputs_text}

SOURCES:
{refs_str}

GENERATE A REPORT.
RULES:
1. Cite EVERY fact with [X].
2. Structure: Executive Summary, Investment Thesis, Business Overview, Risks, Valuation, Conclusion.
3. Replace [SOURCE-X] with [X].
"""
        
        try:
            synthesis_client = OllamaClient(base_url=self.ollama_url, model=self.SYNTHESIS_MODEL)
            raw_response = synthesis_client.chat([{"role": "user", "content": prompt}], temperature=0.3, num_predict=5000, timeout=900)
            final_report = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            
            # Cleanup citations
            final_report = re.sub(r'\[SOURCE-(\d+)\]', r'[\1]', final_report)
            
            # 🔥 BUILD REFERENCE SECTION WITH LINKS
            ref_section = "\n\n## References\n\n"
            for num, src in sorted(document_sources.items()):
                if "||" in src:
                    # Web Source -> Link
                    title, url = src.split("||")
                    ref_section += f"[{num}] **{title}** ([Link]({url}))\n"
                else:
                    # Local Source -> Label
                    ref_section += f"[{num}] **{src}** (System Authenticated Source)\n"
            
            final_report += ref_section
            
            return final_report
            
        except Exception as e:
            return f"Error: {str(e)}"

    def research(self, user_query: str, callback: Optional[Callable] = None) -> str:
        self.trace = ReActTrace()
        self.current_ticker = None
        
        ticker = self._extract_ticker(user_query, callback)
        if ticker:
            self.current_ticker = ticker
            self._auto_seed_graph(ticker, callback)
        
        for iteration in range(1, self.max_iterations + 1):
            action = self._reason_rule_based(user_query, iteration, callback)
            self.trace.add_action(action)
            observation = self._execute_action(action, callback)
            self.trace.add_observation(observation)
            
            if action.action_type == ActionType.FINISH:
                break
        
        final_report = self._synthesize(user_query, callback)
        if callback:
            callback("Orchestration", "Process Complete", "complete")
        return final_report

def main():
    # Test block
    pass

if __name__ == "__main__":
    main()

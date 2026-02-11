#!/usr/bin/env python3
"""
ReAct-based Multi-Agent Orchestrator

Rule-based orchestration with HYBRID LOCAL LLM synthesis:
- DeepSeek-R1 8B: Deep reasoning for specialist analysis
- Qwen 2.5 7B: Fast synthesis for final report combining

Version: 3.0 - Atomic Citation Enforced (Target: 110/100 Quality)
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import requests
from dotenv import load_dotenv

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
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, num_predict: int = 3500, timeout: int = 300) -> str:
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
    
    # 🔥 HYBRID MODEL STRATEGY
    ANALYSIS_MODEL = "deepseek-r1:8b"   # Deep reasoning for specialist analysis
    SYNTHESIS_MODEL = "qwen2.5:7b"      # Fast synthesis for final report combining
    
    SPECIALIST_AGENTS = {
        "business_analyst_crag": {
            "description": "Deep Reader (v2) - Graph-Augmented Corrective RAG for 10-K analysis",
            "keywords": ["strategy", "risk", "10-k", "deep dive"],
            "priority": 1  # Top Priority
        },
        "business_analyst": {
            "description": "Analyzes 10-K filings, financial statements, competitive positioning using RAG",
            "keywords": ["10-K", "10-Q", "filing", "risk", "competitive", "financial"],
            "priority": 2
        },
        "web_search_agent": {
            "description": "Supplements document analysis with current web info (news, market data, analyst opinions)",
            "keywords": ["recent", "current", "latest", "news", "market", "price"],
            "priority": 3  # After Analysts
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
        # Use DeepSeek for orchestrator reasoning (not used in rule-based, but kept for compatibility)
        self.client = OllamaClient(base_url=ollama_url, model=self.ANALYSIS_MODEL)
        self.ollama_url = ollama_url
        self.max_iterations = max_iterations
        self.specialist_agents = {}
        self.trace = ReActTrace()

        # Try to register the new CRAG agent automatically if class exists
        if BusinessAnalystCRAG:
            try:
                # 🔥 REAL CONFIG: Qdrant Cloud + Local Neo4j
                # In production, these should come from os.getenv()
                self.register_specialist("business_analyst_crag", BusinessAnalystCRAG(
                    qdrant_url=os.getenv("QDRANT_URL", None), # "https://xyz.qdrant.io"
                    qdrant_key=os.getenv("QDRANT_API_KEY", None),
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"), # Local Docker
                    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                    neo4j_pass=os.getenv("NEO4J_PASSWORD", "password")
                ))
            except Exception as e:
                print(f"⚠️ Failed to auto-register business_analyst_crag: {e}")
        
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
        - Iteration 1: Call business_analyst_crag (Deep Reader)
        - Iteration 2: Call web_search_agent (supplement with web data)
        - Iteration 3: Finish and synthesize
        """
        print(f"\n🧠 [THOUGHT {iteration}] Rule-based reasoning...")
        
        query_lower = user_query.lower()
        called_agents = self.trace.get_specialist_calls()
        
        # 🟢 Rule 1: ALWAYS call business_analyst_crag first (if available)
        if "business_analyst_crag" in self.specialist_agents and "business_analyst_crag" not in called_agents:
            thought = "Starting with Deep Reader (CRAG) for high-accuracy document analysis"
            self.trace.add_thought(thought, iteration)
            print(f"   💡 {thought}")
            
            return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst_crag",
                task_description=user_query,
                reasoning="Rule 1: CRAG Business Analyst provides best-in-class foundation"
            )

        # Fallback to standard business analyst if CRAG not present
        if "business_analyst" in self.specialist_agents and "business_analyst" not in called_agents and "business_analyst_crag" not in called_agents:
             thought = "Starting with Standard Business Analyst"
             self.trace.add_thought(thought, iteration)
             return Action(
                action_type=ActionType.CALL_SPECIALIST,
                agent_name="business_analyst",
                task_description=user_query,
                reasoning="Fallback: Standard Business Analyst"
            )
        
        # 🟢 Rule 2: Call web_search_agent AFTER analysis (if available)
        # Check if ANY analyst ran
        analyst_ran = "business_analyst_crag" in called_agents or "business_analyst" in called_agents
        
        if "web_search_agent" in self.specialist_agents and "web_search_agent" not in called_agents and analyst_ran:
            thought = "Calling Web Search Agent to supplement document analysis with current data"
            self.trace.add_thought(thought, iteration)
            print(f"   💡 {thought}")
            
            # Get analyst's output to pass as context
            prior_output = ""
            for obs in self.trace.observations:
                if obs.action.agent_name in ["business_analyst", "business_analyst_crag"]:
                    prior_output = obs.result
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
                    # Get prior output
                    prior_analysis = ""
                    for obs in self.trace.observations:
                        if obs.action.agent_name in ["business_analyst", "business_analyst_crag"]:
                            prior_analysis = obs.result
                            break
                    
                    result = agent.analyze(task, prior_analysis=prior_analysis)
                else:
                    # Generic call
                    try:
                        result = agent.analyze(task, prior_analysis="")
                    except TypeError:
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
        """
        document_sources = {}
        citation_num = 1
        
        for output in specialist_outputs:
            content = output['result']
            agent = output['agent']
            
            # Pattern 1: Document sources (filename + page)
            # Matches: --- SOURCE: 10k.pdf (Page 55) ---
            pattern_doc = r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---'
            doc_sources = re.findall(pattern_doc, content, re.IGNORECASE)
            
            # Pattern 2: Web sources (title + URL)
            # Matches: --- SOURCE: Title (http://...) ---
            pattern_web = r'---\s*SOURCE:\s*([^\(]+)\((https?://[^\)]+)\)\s*---'
            web_sources = re.findall(pattern_web, content, re.IGNORECASE)

            # Pattern 3: Graph/Generic sources (Fallback)
            # Matches: --- SOURCE: Entity (local graph) --- OR --- SOURCE: Neo4j (graph) ---
            # Group 1: Entity Name
            # Group 2: Context in parens (not starting with Page or http)
            pattern_generic = r'---\s*SOURCE:\s*([^\(]+)\((?!Page|https?://)([^\)]+)\)\s*---'
            generic_sources = re.findall(pattern_generic, content, re.IGNORECASE)
            
            total_sources = len(doc_sources) + len(web_sources) + len(generic_sources)
            print(f"   🔍 DEBUG: Found {total_sources} SOURCE markers in {agent} output")
            print(f"      - Document sources: {len(doc_sources)}")
            print(f"      - Web sources: {len(web_sources)}")
            print(f"      - Generic/Graph sources: {len(generic_sources)}")
            
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
                source_key = f"{title}||{url}"  # Use || as separator for clean splitting later
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    print(f"      [{citation_num}] {title[:60]}...")
                    citation_num += 1
            
            # Add generic/graph sources
            for name, context in generic_sources:
                name = name.strip()
                context = context.strip()
                source_key = f"{name} ({context})"
                
                if source_key not in document_sources.values():
                    document_sources[citation_num] = source_key
                    print(f"      [{citation_num}] {source_key}")
                    citation_num += 1
        
        return document_sources
    
    def _validate_citation_quality(self, report: str, total_sources: int) -> Tuple[int, List[str]]:
        """
        Validate citation quality in the final report
        """
        warnings = []
        
        # Check 1: Count total citations
        citations = re.findall(r'\[\d+\]', report)
        citation_count = len(citations)
        
        if citation_count == 0:
            warnings.append("❌ CRITICAL: No citations found in report")
            return 0, warnings
        
        # Check 2: Find paragraphs without citations
        sections = report.split('\n\n')
        uncited_paragraphs = 0
        
        for section in sections:
            # Skip headers, empty lines, references section
            if (section.strip() and 
                not section.startswith('#') and 
                'References' not in section and
                len(section) > 100 and
                not re.search(r'\[\d+\]', section)):
                uncited_paragraphs += 1
                warnings.append(f"⚠️ Unsourced paragraph: {section[:80]}...")
        
        # Check 3: Ensure Investment Thesis has citations
        if '## Investment Thesis' in report:
            thesis_section = report.split('## Investment Thesis')[1].split('##')[0] if '## Investment Thesis' in report else ""
            thesis_citations = len(re.findall(r'\[\d+\]', thesis_section))
            if thesis_citations < 3:
                warnings.append("⚠️ Investment Thesis has insufficient citations (< 3)")
        
        # Check 4: Ensure valuation metrics are cited
        valuation_keywords = ['P/E', 'EV/', 'market cap', 'valuation', 'price target', '$\d+B', '\d+%']
        for keyword in valuation_keywords:
            matches = re.finditer(keyword, report, re.IGNORECASE)
            for match in matches:
                # Check if citation follows within 50 chars
                context = report[match.end():match.end()+50]
                if not re.search(r'\[\d+\]', context):
                    warnings.append(f"⚠️ Valuation metric '{keyword}' not cited: {report[match.start():match.end()+30]}")
        
        # Calculate score
        citation_density = citation_count / max(len(sections), 1)
        penalty = min(len(warnings) * 10, 50)  # Max 50% penalty
        quality_score = max(0, 100 - penalty - (100 if uncited_paragraphs > 3 else 0))
        
        return quality_score, warnings
    
    def _synthesize(self, user_query: str) -> str:
        """
        Synthesize with HYBRID LOCAL LLM - Professional Equity Research Report (10/10 Quality)
        """
        specialist_calls = self.trace.get_specialist_calls()
        current_date = datetime.now().strftime("%B %d, %Y")
        
        print(f"\n📊 [SYNTHESIS] Combining insights with Qwen 2.5 7B (fast synthesis)...")
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
        
        # Replace SOURCE markers with citation numbers
        outputs_with_cites = []
        for output in specialist_outputs:
            content = output['result']
            
            def replace_source(match):
                # Extract either (Page X) or (URL) or (Context)
                full_match = match.group(0)
                
                # Identify which pattern matched
                source_key = None
                
                # Check if it matches Page pattern
                page_match = re.search(r'---\s*SOURCE:\s*([^\(]+)\(Page\s*([^\)]+)\)\s*---', full_match, re.IGNORECASE)
                if page_match:
                    filename = page_match.group(1).strip()
                    page = page_match.group(2).strip()
                    source_key = f"{filename} - Page {page}"
                else:
                    # Check if it matches Web pattern
                    web_match = re.search(r'---\s*SOURCE:\s*([^\(]+)\((https?://[^\)]+)\)\s*---', full_match, re.IGNORECASE)
                    if web_match:
                        title = web_match.group(1).strip()
                        url = web_match.group(2).strip()
                        source_key = f"{title}||{url}"
                    else:
                        # Check generic pattern
                        gen_match = re.search(r'---\s*SOURCE:\s*([^\(]+)\((?!Page|https?://)([^\)]+)\)\s*---', full_match, re.IGNORECASE)
                        if gen_match:
                            name = gen_match.group(1).strip()
                            context = gen_match.group(2).strip()
                            source_key = f"{name} ({context})"
                
                if source_key:
                    for num, doc in document_sources.items():
                        if doc == source_key:
                            return f" [SOURCE-{num}]"
                
                return match.group(0)
            
            # Replace all patterns in one go if possible, or iterative
            # We use a broad pattern to catch the marker, then refine inside the callback
            content_with_cites = re.sub(
                r'---\s*SOURCE:\s*(.*?)\s*---',
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
        
        # Create document reference list for prompt
        references_list = "\n".join([
            f"[SOURCE-{num}] = {doc}"
            for num, doc in sorted(document_sources.items())
        ])
        
        # 🔥 110/100 QUALITY: ATOMIC CITATION ENFORCEMENT PROMPT
        prompt = f"""You are a Senior Equity Research Analyst at a top-tier investment bank synthesizing an institutional-grade research report.

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
PROFESSIONAL REPORT STRUCTURE (MANDATORY)
==========================================================================

## Executive Summary
[2-3 sentences with investment thesis and temporal context. MUST cite key metrics.]

## Investment Thesis
[3-4 SPECIFIC bullet points - EVERY point MUST have data + citations]

EXAMPLE FORMAT (FOLLOW THIS):
- **Revenue Growth Trajectory**: FY2025 revenue of $394B [1] with Q1 2026 showing 8.2% YoY growth [8], driven by Services segment margin expansion from 68.2% [2] to 71.5% per recent results [9]
- **Product Innovation Pipeline**: iPhone 15 launched [3], Vision Pro AR platform generating $2.1B initial quarter [10], foldable iPhone pipeline for H2 2026 [11] maintains competitive differentiation
- **Market Leadership**: Global smartphone market share of 23.4% [4], iOS ecosystem with 2.2B active devices [5], Services ARR growth of 15.8% YoY [9]
- **Valuation Context**: Trading at 27.5x NTM P/E [12] vs sector average of 22.1x, justified by 12.3% revenue CAGR and 43.8% gross margins [1]

## Business Overview (Historical Context - Per FY2025 10-K)
[Cite EVERY financial metric, business line, and market position claim]
- Revenue streams with exact figures and percentages [cite]
- Market position with specific market share data [cite]
- Competitive advantages with supporting evidence [cite]

## Recent Developments (Current Period - Q4 2025 to Q1 2026)
[Cite EVERY product launch, financial update, and market development]
- Product launches with dates and specifications [cite web sources]
- Financial performance vs analyst expectations [cite web sources]
- Competitive dynamics and market share shifts [cite web sources]
- Management commentary and guidance updates [cite web sources]

## Risk Analysis
### Historical Risks (Per 10-K)
[List 3-5 risks, EACH with citation]
- Risk 1: [Specific risk with impact quantification] [cite]
- Risk 2: [Specific risk with impact quantification] [cite]

### Emerging Risks (Current)
[List 2-4 new risks, EACH with citation]
- New risk 1: [Specific development] [cite web]
- New risk 2: [Specific development] [cite web]

## Valuation Context
[EVERY metric MUST be cited]
- Historical metrics: Revenue $XXB [cite], EBITDA margin XX% [cite], FCF $XXB [cite]
- Current valuation: P/E XX.Xx [cite], EV/Sales X.Xx [cite], Market cap $X.XTn [cite]
- Analyst consensus: Average price target $XXX [cite], High/Low range [cite]

## Conclusion
[2-3 sentences synthesizing risk-return profile with citations]

## Data Quality Notice
⚠️ **Temporal Distinction**:
- 10-K data: FY2025 filed October 31, 2025 (historical)
- Web data: Q4 2025 - Q1 2026 (current as of {current_date})
- Investors should verify current data before investment decisions

## References
[Will be auto-generated - DO NOT include in your response]

==========================================================================
🔥 CITATION REQUIREMENTS (EXTREME STRICTNESS) 🔥
==========================================================================

1. **ATOMIC CITATION RULE (CRITICAL)**
   - You MUST place a citation immediately after EVERY SINGLE metric, number, or fact.
   - ❌ BAD: "Revenue was $10B, margin 20%, and growth 5% [1]." (Grouped citation)
   - ✅ GOOD: "Revenue was $10B [1], margin 20% [1], and growth 5% [1]." (Atomic citation)

2. **LISTS OF DATA**
   - ❌ BAD: "P/E 20x, EV/Sales 5x, Market Cap $1T [2]"
   - ✅ GOOD: "P/E 20x [2], EV/Sales 5x [2], Market Cap $1T [2]"

3. **SENTENCE-LEVEL CITATION**
   - Do NOT group citations at the end of a paragraph. Cite immediately after the fact.
   - ✅ "Services revenue grew 15% [1], while hardware declined 2% [2]."
   - ❌ "Services revenue grew 15% while hardware declined 2% [1][2]."

4. **EVERY NUMBER MUST BE CITED**
   - ✅ "Market cap of $2.4T [9]."
   - ❌ "Market cap of $2.4T." (NO CITATION = UNACCEPTABLE)

5. **TEMPORAL MARKERS (MANDATORY)**
   - 10-K: "Per FY2025 10-K [1]" or "As disclosed in annual filing [2]"
   - Web: "As of Q1 2026 [8]" or "Recent reports indicate [9]" or "Per January 2026 analyst note [10]"

6. **INVESTMENT THESIS MUST HAVE 8+ CITATIONS**
   - Each bullet point needs 2-3 citations minimum
   - Mix historical [1-7] and current [8+] sources

7. **FORMAT**
   - Replace [SOURCE-X] with [X]
   - Space before citation: "text [1]" not "text[1]"
   - Multiple citations: "text [1][2][3]" not "text [1,2,3]"

8. **PROFESSIONAL TONE**
   - Exact figures: "15.23%" not "around 15%" or "approximately 15%"
   - Attribution: "Goldman Sachs (January 2026) projects [X]" not "analysts think"
   - Objectivity: "Data indicates" not "we believe" or "I think"
   - Temporal precision: "Q1 2026" not "recently" or "soon"

==========================================================================
⚠️ QUALITY CONTROL - YOUR REPORT WILL BE REJECTED IF:
==========================================================================
- Any metric lacks citation
- Investment Thesis has < 8 citations
- Any sentence with data lacks citation
- Generic statements without specific data
- Temporal markers missing
- Web sources not distinguished from 10-K sources

==========================================================================
GENERATE 10/10 INSTITUTIONAL-GRADE REPORT NOW
==========================================================================

Provide a report meeting EVERY requirement above.
Cite OBSESSIVELY. Every claim, every number, every statement.
"""
        
        try:
            # 🔥 HYBRID: Use Qwen 2.5 7B for fast synthesis (10x faster than DeepSeek)
            synthesis_client = OllamaClient(base_url=self.ollama_url, model=self.SYNTHESIS_MODEL)
            
            messages = [{"role": "user", "content": prompt}]
            print("   🔄 Generating 10/10 professional equity research synthesis...")
            
            # 🔥 Qwen 2.5 7B optimized parameters (faster than DeepSeek)
            final_report = synthesis_client.chat(
                messages, 
                # 🔥 FIX: Increased temperature to 0.2 to help with instruction following (too low = rigid)
                temperature=0.2,
                num_predict=3500,  # Same token count
                timeout=600        # 🔥 FIX 3: Increased from 240s to 600s (10 minutes)
            )
            print("   ✅ Synthesis complete")
            
            # 🔥 POST-PROCESSING: Clean up [SOURCE-X] to [X]
            final_report = re.sub(r'\[SOURCE-(\d+)\]', r'[\1]', final_report)
            
            # 🔥 POST-PROCESSING: Add properly formatted References
            if len(document_sources) > 0:
                # Remove any existing References section
                if "## References" in final_report:
                    final_report = final_report.split("## References")[0]
                
                refs = "\n\n## References\n\n"
                for num, doc in sorted(document_sources.items()):
                    if "||" in doc:  # Web source
                        title, url = doc.split("||", 1)
                        refs += f"[{num}] {title} - {url}\n"
                    else:  # Document or Graph source
                        refs += f"[{num}] {doc}\n"
                
                final_report += refs
            
            # 🔥 QUALITY VALIDATION
            print("\n   🔍 Running quality validation...")
            quality_score, warnings = self._validate_citation_quality(final_report, len(document_sources))
            
            print(f"   📊 Citation Quality Score: {quality_score}/100")
            
            if warnings:
                print(f"   ⚠️ Found {len(warnings)} quality issues:")
                for warning in warnings[:5]:  # Show first 5
                    print(f"      {warning}")
                if len(warnings) > 5:
                    print(f"      ... and {len(warnings)-5} more")
            else:
                print("   ✅ Perfect citation quality!")
            
            return final_report
            
        except Exception as e:
            print(f"   ❌ Synthesis error: {str(e)}")
            # Fallback
            return f"""## Research Report\n\n{outputs_text}\n\n---\n\n## Sources\n\n{references_list.replace('[SOURCE-', '[').replace('] =', ']')}\n\n---\n\n**Note**: Synthesis failed. Showing raw analysis."""
    
    def research(self, user_query: str) -> str:
        """Main ReAct loop with rule-based reasoning"""
        print("\n" + "="*70)
        print("🔁 REACT EQUITY RESEARCH ORCHESTRATOR v2.3")
        print("   10/10 Quality + Performance (Hybrid: DeepSeek + Qwen)")
        print("   Enhanced with CRAG Deep Reader")
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
            print("💡 Make sure models are installed:")
            print(f"   ollama pull {ReActOrchestrator.ANALYSIS_MODEL}")
            print(f"   ollama pull {ReActOrchestrator.SYNTHESIS_MODEL}")
            return
    except ValueError as e:
        print(f"❌ {str(e)}")
        return
    
    print("\n🚀 Professional Equity Research Orchestrator v2.3 Ready")
    print("   10/10 Quality Standard (Hybrid: DeepSeek-R1 8B + Qwen 2.5 7B)")
    print("\n🎯 Model Strategy:")
    print(f"   - Analysis: {ReActOrchestrator.ANALYSIS_MODEL} (specialists use this)")
    print(f"   - Synthesis: {ReActOrchestrator.SYNTHESIS_MODEL} (final report combining)")
    print("\n💡 Performance Optimizations:")
    print("   - Business Analyst: 2000 token limit")
    print("   - Web Search: 1200 token limit")
    print("   - Synthesis timeout: 240s (4 minutes)")
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

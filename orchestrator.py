import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import requests


@dataclass
class AgentTask:
    """Represents a task assigned to a specialist agent"""
    agent_name: str
    task_description: str
    priority: int = 1


@dataclass
class AgentOutput:
    """Represents output from a specialist agent"""
    agent_name: str
    result: str
    metadata: Dict[str, Any] = None


class PerplexityClient:
    """Client for Perplexity API interactions"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.1-sonar-large-128k-online") -> str:
        """Send chat request to Perplexity API"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Perplexity API: {str(e)}"


class PlannerAgent:
    """Orchestration agent that decides which specialist agents to deploy"""
    
    # Define the 6 specialist agents for equity research
    SPECIALIST_AGENTS = {
        "business_analyst": {
            "description": "Analyzes 10-K filings, financial statements, business models, competitive positioning, and strategic risks",
            "capabilities": ["Financial statement analysis", "Competitive intelligence", "Risk assessment", "Business model evaluation"]
        },
        "quantitative_analyst": {
            "description": "Performs quantitative analysis including financial ratios, valuation metrics, growth rates, and statistical modeling",
            "capabilities": ["DCF valuation", "Ratio analysis", "Trend forecasting", "Comparative metrics"]
        },
        "market_analyst": {
            "description": "Tracks market sentiment, technical indicators, price movements, trading volumes, and market positioning",
            "capabilities": ["Sentiment analysis", "Technical analysis", "Market trends", "Trading patterns"]
        },
        "industry_analyst": {
            "description": "Provides sector-specific insights, industry trends, regulatory landscape, and peer benchmarking",
            "capabilities": ["Industry dynamics", "Regulatory analysis", "Peer comparison", "Sector trends"]
        },
        "esg_analyst": {
            "description": "Evaluates environmental, social, and governance factors, sustainability initiatives, and ESG risk exposure",
            "capabilities": ["ESG scoring", "Sustainability assessment", "Governance evaluation", "Climate risk"]
        },
        "macro_analyst": {
            "description": "Analyzes macroeconomic factors, interest rates, currency impacts, geopolitical risks affecting the company",
            "capabilities": ["Economic indicators", "Rate sensitivity", "FX exposure", "Geopolitical risk"]
        }
    }
    
    def __init__(self, perplexity_client: PerplexityClient, use_detailed_specs: bool = True):
        self.client = perplexity_client
        self.use_detailed_specs = use_detailed_specs
        self.detailed_specs = self._load_detailed_specs() if use_detailed_specs else None
    
    def _load_detailed_specs(self) -> str:
        """Load detailed agent specifications from SPECIALIST_AGENTS.md"""
        try:
            spec_path = os.path.join(os.getcwd(), "SPECIALIST_AGENTS.md")
            if os.path.exists(spec_path):
                with open(spec_path, 'r') as f:
                    content = f.read()
                    print("📚 Loaded detailed agent specifications for enhanced planning")
                    return content
        except Exception as e:
            print(f"⚠️  Could not load SPECIALIST_AGENTS.md: {e}")
        return None
    
    def create_planning_prompt(self, user_query: str, available_agents: List[str]) -> str:
        """Generate the planning prompt for agent selection"""
        
        # Use detailed specs if available, otherwise use basic descriptions
        if self.detailed_specs:
            # Extract relevant sections for available agents only
            agent_context = "\n".join([
                f"- {name}: {info['description']}"
                for name, info in self.SPECIALIST_AGENTS.items()
                if name in available_agents
            ])
            
            prompt = f"""You are an AI Research Orchestrator for equity research. Your role is to analyze user queries and determine which specialist agents should be deployed to provide comprehensive analysis.

REFERENCE: Detailed specialist agent specifications are available in SPECIALIST_AGENTS.md.
Key decision criteria:
- Business Analyst: Best for 10-K/10-Q filings, competitive analysis, risk factors
- Quantitative Analyst: Best for calculations, ratios, valuation models, trend analysis
- Market Analyst: Best for sentiment, technical indicators, current market data
- Industry Analyst: Best for sector trends, peer benchmarking, regulatory landscape
- ESG Analyst: Best for sustainability, governance, environmental impact
- Macro Analyst: Best for economic factors, interest rates, geopolitical risks

AVAILABLE SPECIALIST AGENTS:
{agent_context}

USER QUERY:
{user_query}

TASK:
1. Analyze the user's query keywords and intent
2. Select 1-4 most relevant specialist agents based on:
   - Core capabilities match
   - Data source requirements
   - Query complexity
3. Define specific, actionable tasks for each selected agent
4. Prioritize agents (1=highest priority, execute first)

SELECTION GUIDELINES:
- Use 1 agent for focused, single-domain queries
- Use 2-3 agents when combining qualitative + quantitative analysis
- Use 3-4 agents for comprehensive research requiring multiple perspectives

RESPOND IN THIS JSON FORMAT ONLY:
{{
  "reasoning": "Brief explanation of agent selection strategy and why these agents fit the query",
  "selected_agents": [
    {{
      "agent_name": "business_analyst",
      "task_description": "Specific task for this agent",
      "priority": 1
    }}
  ]
}}

Provide only valid JSON without any markdown formatting or additional text."""
        else:
            # Fallback to basic descriptions
            agents_info = "\n".join([
                f"- {name}: {info['description']}\n  Capabilities: {', '.join(info['capabilities'])}"
                for name, info in self.SPECIALIST_AGENTS.items()
                if name in available_agents
            ])
            
            prompt = f"""You are an AI Research Orchestrator for equity research. Your role is to analyze user queries and determine which specialist agents should be deployed to provide comprehensive analysis.

AVAILABLE SPECIALIST AGENTS:
{agents_info}

USER QUERY:
{user_query}

TASK:
1. Analyze the user's query to understand what type of analysis is needed
2. Select the most relevant specialist agents (1-4 agents recommended)
3. For each selected agent, define a specific task/question they should address
4. Prioritize the agents (1=highest priority)

RESPOND IN THIS JSON FORMAT ONLY:
{{
  "reasoning": "Brief explanation of your agent selection strategy",
  "selected_agents": [
    {{
      "agent_name": "business_analyst",
      "task_description": "Specific task for this agent",
      "priority": 1
    }}
  ]
}}

Provide only valid JSON without any markdown formatting or additional text."""
        
        return prompt
    
    def plan(self, user_query: str, available_agents: List[str] = None) -> List[AgentTask]:
        """Create execution plan by selecting and tasking specialist agents"""
        if available_agents is None:
            available_agents = list(self.SPECIALIST_AGENTS.keys())
        
        print("\n🎯 [PLANNER AGENT] Analyzing query and creating execution plan...")
        
        planning_prompt = self.create_planning_prompt(user_query, available_agents)
        messages = [{"role": "user", "content": planning_prompt}]
        
        response = self.client.chat(messages)
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            plan_data = json.loads(json_str)
            
            print(f"\n💡 Reasoning: {plan_data['reasoning']}")
            print(f"\n📋 Selected Agents: {len(plan_data['selected_agents'])}")
            
            tasks = []
            for agent_spec in plan_data["selected_agents"]:
                task = AgentTask(
                    agent_name=agent_spec["agent_name"],
                    task_description=agent_spec["task_description"],
                    priority=agent_spec.get("priority", 1)
                )
                tasks.append(task)
                print(f"  - {task.agent_name} (Priority {task.priority}): {task.task_description}")
            
            # Sort by priority
            tasks.sort(key=lambda x: x.priority)
            return tasks
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️ Failed to parse planning response as JSON: {e}")
            print(f"Raw response: {response}")
            # Fallback: use business analyst for general queries
            return [AgentTask(
                agent_name="business_analyst",
                task_description=user_query,
                priority=1
            )]


class SynthesisAgent:
    """Agent that combines outputs from multiple specialist agents into coherent final report"""
    
    def __init__(self, perplexity_client: PerplexityClient):
        self.client = perplexity_client
    
    def create_synthesis_prompt(self, user_query: str, agent_outputs: List[AgentOutput]) -> str:
        """Generate synthesis prompt combining all agent outputs"""
        
        outputs_text = "\n\n".join([
            f"=== OUTPUT FROM {output.agent_name.upper().replace('_', ' ')} ==="
            f"\n{output.result}\n"
            f"{'=' * 50}"
            for output in agent_outputs
        ])
        
        prompt = f"""You are a Senior Equity Research Analyst responsible for synthesizing insights from multiple specialist analysts into a comprehensive, actionable investment research report.

ORIGINAL USER QUERY:
{user_query}

SPECIALIST ANALYST REPORTS:
{outputs_text}

YOUR TASK:
1. Synthesize insights from all specialist reports into a coherent narrative
2. Identify key themes, agreements, and contradictions across reports
3. Provide a balanced view that integrates quantitative and qualitative analysis
4. Structure your response for clarity and actionability
5. Maintain all important data points, metrics, and citations from source reports
6. Highlight any critical risks or opportunities identified

REPORT STRUCTURE:
## Executive Summary
[2-3 sentence overview of key findings]

## Key Insights
[Synthesized findings organized by theme]

## Quantitative Highlights
[Important metrics, ratios, valuations from specialist reports]

## Risk Factors
[Critical risks identified across reports]

## Conclusion
[Balanced final assessment]

Provide a professional, well-structured equity research report."""
        return prompt
    
    def synthesize(self, user_query: str, agent_outputs: List[AgentOutput]) -> str:
        """Combine multiple agent outputs into final comprehensive report"""
        print("\n🔬 [SYNTHESIS AGENT] Combining specialist insights...")
        
        if not agent_outputs:
            return "No agent outputs available for synthesis."
        
        if len(agent_outputs) == 1:
            # If only one agent, return its output directly
            print("   (Single agent output - minimal synthesis)")
            return agent_outputs[0].result
        
        synthesis_prompt = self.create_synthesis_prompt(user_query, agent_outputs)
        messages = [{"role": "user", "content": synthesis_prompt}]
        
        final_report = self.client.chat(messages)
        
        print("   ✅ Synthesis complete")
        return final_report


class EquityResearchOrchestrator:
    """Main orchestrator coordinating planner, specialist agents, and synthesis"""
    
    def __init__(self, perplexity_api_key: str = None, use_detailed_specs: bool = True):
        self.perplexity = PerplexityClient(perplexity_api_key)
        self.planner = PlannerAgent(self.perplexity, use_detailed_specs=use_detailed_specs)
        self.synthesizer = SynthesisAgent(self.perplexity)
        
        # Registry of available specialist agents
        # For now, we'll simulate specialist agents with placeholders
        self.specialist_agents = {}
    
    def register_specialist(self, agent_name: str, agent_instance):
        """Register a specialist agent implementation"""
        self.specialist_agents[agent_name] = agent_instance
        print(f"✅ Registered specialist: {agent_name}")
    
    def execute_task(self, task: AgentTask) -> AgentOutput:
        """Execute a task with the appropriate specialist agent"""
        print(f"\n🤖 Executing: {task.agent_name}")
        print(f"   Task: {task.task_description}")
        
        # Check if specialist agent is registered
        if task.agent_name in self.specialist_agents:
            agent = self.specialist_agents[task.agent_name]
            # Assuming specialist agents have an 'analyze' method
            result = agent.analyze(task.task_description)
            return AgentOutput(agent_name=task.agent_name, result=result)
        else:
            # Placeholder for unimplemented specialists
            placeholder_result = f"[{task.agent_name.upper()} AGENT - NOT YET IMPLEMENTED]\n"
            placeholder_result += f"Would analyze: {task.task_description}\n"
            placeholder_result += "This specialist agent will be integrated in future versions."
            
            print(f"   ⚠️ Agent not yet implemented - using placeholder")
            return AgentOutput(agent_name=task.agent_name, result=placeholder_result)
    
    def research(self, user_query: str) -> str:
        """Main research workflow: Plan → Execute → Synthesize"""
        print("\n" + "="*70)
        print("🎓 EQUITY RESEARCH ORCHESTRATOR")
        print("="*70)
        print(f"\n📥 User Query: {user_query}")
        
        # Step 1: Planning
        available_agents = list(self.specialist_agents.keys()) if self.specialist_agents else None
        tasks = self.planner.plan(user_query, available_agents)
        
        if not tasks:
            return "Unable to create execution plan. Please refine your query."
        
        # Step 2: Execute tasks with specialist agents
        print("\n" + "-"*70)
        print("⚙️ EXECUTING SPECIALIST AGENTS")
        print("-"*70)
        
        agent_outputs = []
        for task in tasks:
            output = self.execute_task(task)
            agent_outputs.append(output)
        
        # Step 3: Synthesis
        print("\n" + "-"*70)
        print("📊 SYNTHESIS PHASE")
        print("-"*70)
        
        final_report = self.synthesizer.synthesize(user_query, agent_outputs)
        
        return final_report


def main():
    """Demo of the orchestration system"""
    
    # Check for API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️ Please set PERPLEXITY_API_KEY environment variable")
        print("   export PERPLEXITY_API_KEY='your-api-key'")
        return
    
    # Initialize orchestrator
    orchestrator = EquityResearchOrchestrator()
    
    # Optional: Register your existing BusinessAnalystGraphAgent
    # from skills.business_analyst.graph_agent import BusinessAnalystGraphAgent
    # business_analyst = BusinessAnalystGraphAgent()
    # orchestrator.register_specialist("business_analyst", business_analyst)
    
    print("\n🚀 Equity Research Orchestrator Ready")
    print("\nThis system coordinates 6 specialist agents:")
    for name, info in PlannerAgent.SPECIALIST_AGENTS.items():
        status = "✅" if name in orchestrator.specialist_agents else "⏳"
        print(f"  {status} {name}: {info['description'][:60]}...")
    
    # Interactive loop
    while True:
        print("\n" + "="*70)
        user_query = input("\n💬 Enter your research question (or 'quit' to exit): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_query.strip():
            continue
        
        # Execute research workflow
        try:
            final_report = orchestrator.research(user_query)
            
            print("\n" + "="*70)
            print("📄 FINAL RESEARCH REPORT")
            print("="*70)
            print(f"\n{final_report}")
            
        except Exception as e:
            print(f"\n❌ Error during research: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

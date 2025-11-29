# app/agents/orchestrator.py

from typing import Dict, List, Any, Optional
from app.agents.base_agent import BaseAgent, AgentRequest, AgentResponse
from app.core.api_clients import google_client
from app.config import get_settings
from loguru import logger
import json

settings = get_settings()

class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator that interprets queries and routes to appropriate agents.
    Uses Mode Detection to prevent unnecessary agent firing.
    """
    
    AGENT_CAPABILITIES = {
        "data_harvester": "Ingests, cleans, and preprocesses data. Runs for new uploads OR when Deep Dive/Optimization requires guaranteed data quality.",
        "visualizer": "Creates charts and graphs.",
        "trend_analyst": "Identifies trends. Uses internal data (Deep Dive) or Google Trends (Cold Start).",
        "forecaster": "Predicts future values.",
        "mcts_optimizer": "Optimizes inventory/decisions. Requires internal data.",
        "order_manager": "Drafts orders. Used ONLY when user explicitly wants to buy/order.",
        "notifier": "Sends alerts. Used ONLY after an order is created."
    }
    
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            model=settings.ORCHESTRATOR_MODEL,
            api_client=google_client
        )
    
    def get_system_prompt(self) -> str:
        """System prompt with mode-aware routing rules"""
        capabilities = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.AGENT_CAPABILITIES.items()
        ])
        
        return f"""You are the Orchestrator Agent for an MSME supply chain platform.

Your role:
1. Detect the OPERATIONAL MODE (Cold Start, Deep Dive, or Ad-Hoc).
2. Create an execution plan.
3. STRICTLY follow agent constraints.

Available agents:
{capabilities}

### OPERATIONAL MODES:

**MODE 1: COLD START** (No 'dataset_id' in context)
- User asks generic market questions (e.g., "Sneaker trends").
- ALLOWED: trend_analyst (for external data), visualizer.
- BANNED: data_harvester, mcts_optimizer, forecaster (requires history), order_manager, notifier.

**MODE 2: DEEP DIVE** ('dataset_id' present + analysis/optimization request)
- User wants full optimization, forecasting, or deep analysis on their data.
- PIPELINE: data_harvester -> trend_analyst -> forecaster -> mcts_optimizer.
- **CRITICAL**: You MUST run `data_harvester` first to ensure data is clean, even if the file is already uploaded.
- Optional: order_manager (only if asked), notifier (only if order created).

**MODE 3: AD-HOC QUERY** ('dataset_id' present + specific simple question)
- User asks specific Q (e.g., "Show sales for March").
- ACTION: Call ONLY the relevant agent (e.g., visualizer OR trend_analyst).
- BANNED: Full pipeline, notifier.

### CRITICAL RULES:
1. **Notifier**: NEVER call unless `order_manager` is also called.
2. **Data Harvester Strategy**: 
   - IF Mode is "Deep Dive" or "Optimization": ALWAYS include `data_harvester` as the first step.
   - IF Mode is "Ad-Hoc": Skip `data_harvester` (assume data is already clean enough).
3. **Dependency Check**: NEVER list an agent in "depends_on" if that agent is not in your "agents" list.
   - If `data_harvester` is skipped, the first agent (e.g., `trend_analyst`) must have `depends_on: []`.

Respond in JSON:
{{
    "mode": "cold_start | deep_dive | ad_hoc",
    "reasoning": "Why you chose this mode and agents",
    "agents": ["list", "of", "agent_names"],
    "execution_plan": [
        {{
            "agent": "agent_name",
            "task": "Specific instruction for this agent",
            "parameters": {{}},
            "depends_on": ["previous_agent_name"]
        }}
    ]
}}
"""

    def _detect_mode(self, request: AgentRequest) -> str:
        """Programmatically detect the mode to enforce guardrails"""
        has_dataset = "dataset_id" in request.context or "dataset" in request.context
        query = request.query.lower()
        
        # Keywords
        order_keywords = ["order", "buy", "purchase", "procure"]
        deep_keywords = ["optimize", "full analysis", "deep dive", "strategy", "forecast", "predict", "bullwhip", "inventory"]
        
        # 1. Cold Start: No data available
        if not has_dataset:
            return "cold_start"
            
        # 2. Deep Dive: Explicit complex request
        if any(k in query for k in deep_keywords) or any(k in query for k in order_keywords):
            return "deep_dive"
            
        # 3. Ad-Hoc: Default for specific questions on existing data
        return "ad_hoc"

    async def process(self, request: AgentRequest) -> AgentResponse:
        try:
            # 1. Detect Mode Programmatically
            detected_mode = self._detect_mode(request)
            
            prompt = f"""{self.get_system_prompt()}

CURRENT CONTEXT:
- Detected Mode: {detected_mode.upper()}
- Has Dataset: {"Yes" if "dataset_id" in request.context else "No"}
- User Query: {request.query}

Generate the execution plan based on the Detected Mode constraints."""
            
            # 2. Call LLM
            response = await self.api_client.generate_content(
                model_name=self.model,
                prompt=prompt,
                temperature=0.1,
                max_tokens=2000
            )
            
            # 3. Parse Response
            content = response.get("text", "{}")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            try:
                plan = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Orchestrator JSON. Raw: {content}")
                return AgentResponse(agent_name=self.name, success=False, error="Failed to generate plan")

            # 4. ENFORCE GUARDRAILS
            agents = plan.get("agents", [])
            
            if detected_mode == "cold_start":
                forbidden = ["data_harvester", "mcts_optimizer", "order_manager", "notifier", "forecaster"]
                agents = [a for a in agents if a not in forbidden]
                
            elif detected_mode == "ad_hoc":
                if "notifier" in agents: agents.remove("notifier")
                
            # Update the plan with the filtered list
            plan["agents"] = agents
            plan["mode"] = detected_mode

            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={"plan": plan}
            )
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return AgentResponse(agent_name=self.name, success=False, error=str(e))

    async def route_to_agents(
        self,
        execution_plan: Dict[str, Any],
        request: AgentRequest
    ) -> List[AgentResponse]:
        """Execute the agents sequentially based on the plan"""
        responses = []
        
        # Lazy imports
        from app.agents.data_harvester import DataHarvesterAgent
        from app.agents.visualizer import VisualizerAgent
        from app.agents.trend_analyst import TrendAnalystAgent
        from app.agents.forecaster import ForecasterAgent
        from app.agents.mcts_optimizer import MCTSOptimizerAgent
        from app.agents.order_manager import OrderManagerAgent
        from app.agents.notifier import NotifierAgent

        agent_registry = {
            "data_harvester": DataHarvesterAgent(),
            "visualizer": VisualizerAgent(),
            "trend_analyst": TrendAnalystAgent(),
            "forecaster": ForecasterAgent(),
            "mcts_optimizer": MCTSOptimizerAgent(),
            "order_manager": OrderManagerAgent(),
            "notifier": NotifierAgent(),
        }
        
        # --- HELPER: Aggressive Normalization ---
        def normalize(name: str) -> str:
            """Converts 'Data Harvester', 'Data_Harvester', 'dataharvester' -> 'dataharvester'"""
            return name.lower().replace("_", "").replace(" ", "").replace("-", "")

        # Create lookup: {'dataharvester': 'data_harvester', ...}
        registry_lookup = {normalize(k): k for k in agent_registry.keys()}
        
        completed_agents = set() # Stores normalized names of completed agents

        for step in execution_plan.get("execution_plan", []):
            raw_name = step["agent"]
            normalized_name = normalize(raw_name)
            
            # 1. Resolve Agent Key
            if normalized_name not in registry_lookup:
                logger.warning(f"Skipping unknown agent: {raw_name}")
                continue
            
            registry_key = registry_lookup[normalized_name] # e.g. "data_harvester"
            
            # 2. Guardrails Check
            allowed_agents = {normalize(a) for a in execution_plan.get("agents", [])}
            if normalized_name not in allowed_agents:
                continue

            # 3. Check Dependencies (ROBUST FIX)
            raw_deps = step.get("depends_on", [])
            relevant_deps = [d for d in raw_deps if normalize(d) in allowed_agents]
            missing_deps = [d for d in relevant_deps if normalize(d) not in completed_agents]
            
            if missing_deps:
                logger.warning(f"Skipping {registry_key} due to missing dependencies: {missing_deps}")
                continue

            # 4. Execute
            agent = agent_registry[registry_key]
            
            agent_req = AgentRequest(
                query=step.get("task", request.query),
                context=request.context,
                session_id=request.session_id,
                user_id=request.user_id,
                parameters=step.get("parameters", {})
            )
            
            response = await agent.execute_with_observability(agent_req)
            
            # --- CRITICAL UI FIX: OVERWRITE NAME ---
            # Force the response name to match the registry key (snake_case)
            # e.g., Change "DataHarvester" -> "data_harvester"
            response.agent_name = registry_key 
            # ---------------------------------------
            
            responses.append(response)
            
            if response.success:
                completed_agents.add(normalized_name)
                if response.data:
                    request.context[f"{registry_key}_output"] = response.data
                    if registry_key == "order_manager":
                        request.context["order_manager_output"] = response.data

        return responses

# Singleton
orchestrator = OrchestratorAgent()
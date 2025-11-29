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
    Uses Claude Sonnet 4.5 for superior reasoning.
    """
    
    AGENT_CAPABILITIES = {
        "data_harvester": "Ingests, cleans, and preprocesses any tabular data. Handles CSV, Excel, JSON.",
        "visualizer": "Creates charts, graphs, and visual representations of data.",
        "trend_analyst": "Identifies patterns, trends, and anomalies in time series data.",
        "forecaster": "Predicts future values using statistical and ML models.",
        "mcts_optimizer": "Optimizes complex decisions using Monte Carlo Tree Search.",
        "order_manager": "Manages order processing, inventory, and fulfillment workflows.",
        "notifier": "Sends notifications via email, SMS, or webhooks."
    }
    
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            model=settings.ORCHESTRATOR_MODEL,
            api_client=google_client
        )
    
    def get_system_prompt(self) -> str:
        """System prompt for orchestrator"""
        capabilities = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.AGENT_CAPABILITIES.items()
        ])
        
        return f"""You are the Orchestrator Agent for an MSME intelligence platform.

Your role is to:
1. Understand user queries about business data analysis
2. Determine which specialized agent(s) should handle the request
3. Create execution plan with correct agent sequence
4. Extract parameters for each agent

Available agents:
{capabilities}

IMPORTANT: When creating execution plans:
- data_harvester has NO dependencies (can run first)
- trend_analyst depends on data_harvester
- forecaster depends on trend_analyst
- visualizer can depend on ANY agent that produces data
- mcts_optimizer depends on forecaster

Respond in JSON format:
{{
    "reasoning": "explanation of your analysis",
    "agents": ["agent1", "agent2"],
    "execution_plan": [
        {{
            "agent": "agent_name",
            "task": "specific task",
            "parameters": {{}},
            "depends_on": []
        }}
    ],
    "requires_data": true/false
}}

Examples:
- "Show me sales trends" → data_harvester (depends_on: []), trend_analyst (depends_on: ["data_harvester"])
- "Predict next quarter sales" → data_harvester, trend_analyst, forecaster
- "Optimize inventory levels" → data_harvester, trend_analyst, forecaster, mcts_optimizer
"""
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        try:
            # Skip orchestration for SESSION_INIT
            if request.query == "SESSION_INIT":
                return AgentResponse(
                    agent_name=self.name,
                    success=True,
                    data={
                        "plan": {
                            "reasoning": "Session initialization - no agents needed",
                            "agents": [],
                            "execution_plan": [],
                            "requires_data": False
                        }
                    }
                )
            
            # CRITICAL FIX: Don't serialize the entire dataset to prompt (too large + timestamp issues)
            # Just send metadata about the dataset
            context_summary = {}
            if "dataset" in request.context:
                dataset = request.context["dataset"]
                context_summary["dataset_info"] = {
                    "available": True,
                    "rows": len(dataset),
                    "sample": dataset[:2] if len(dataset) > 0 else [],
                    "columns": list(dataset[0].keys()) if len(dataset) > 0 else []
                }
            
            # Include other context (but not the full dataset)
            for key, value in request.context.items():
                if key != "dataset":
                    context_summary[key] = value
            
            prompt = f"""{self.get_system_prompt()}

User Query: {request.query}

Context: {json.dumps(context_summary)}

Provide the execution plan in JSON format. Make sure to always include at least one agent."""
            
            # Call Gemini
            response = await self.api_client.generate_content(
                model_name=self.model,
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            content = response.get("text", "{}")
            
            logger.info(f"Raw orchestrator response: {content[:500]}")
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            content = content.strip()
            if content.startswith("json"):
                content = content[4:].strip()
            
            try:
                plan = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse: {content}")
                return AgentResponse(
                    agent_name=self.name,
                    success=False,
                    error=f"Invalid JSON response: {str(e)}"
                )
            
            if not plan.get("agents"):
                return AgentResponse(
                    agent_name=self.name,
                    success=False,
                    error="No agents selected"
                )
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={"plan": plan}
            )
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e)
            )
    
    async def route_to_agents(
        self,
        execution_plan: Dict[str, Any],
        request: AgentRequest
    ) -> List[AgentResponse]:
        """
        Execute the plan by routing to appropriate agents.
        FIX: Properly resolve dependencies and pass data between agents.
        """
        responses = []
        
        # Import agents
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
        
        # Track completed agents
        completed_agents = set()
        agent_outputs = {}
        
        for step in execution_plan.get("execution_plan", []):
            agent_name = step["agent"]
            depends_on = step.get("depends_on", [])
            
            # Check dependencies
            missing_deps = [dep for dep in depends_on if dep not in completed_agents]
            if missing_deps:
                logger.warning(f"Skipping {agent_name} due to missing dependencies: {missing_deps}")
                continue
            
            if agent_name not in agent_registry:
                logger.warning(f"Unknown agent: {agent_name}")
                continue
            
            agent = agent_registry[agent_name]
            
            # Build context with outputs from dependencies
            enriched_context = dict(request.context)
            for dep in depends_on:
                if dep in agent_outputs:
                    enriched_context[f"{dep}_output"] = agent_outputs[dep]
            
            # CRITICAL FIX: Ensure dataset is passed correctly
            if "dataset" in request.context and "dataset" not in enriched_context:
                enriched_context["dataset"] = request.context["dataset"]
            
            # Create agent-specific request
            agent_request = AgentRequest(
                query=step.get("task", request.query),
                context=enriched_context,
                session_id=request.session_id,
                user_id=request.user_id,
                parameters=step.get("parameters", {})
            )
            
            logger.info(f"Executing agent: {agent_name} with context keys: {list(enriched_context.keys())}")
            
            # Execute agent
            try:
                response = await agent.execute_with_observability(agent_request)
                responses.append(response)
                
                # Track completion and output
                if response.success:
                    completed_agents.add(agent_name)
                    if response.data:
                        agent_outputs[agent_name] = response.data
                        # Also update main context
                        request.context[f"{agent_name}_output"] = response.data
                else:
                    logger.error(f"Agent {agent_name} failed: {response.error}")
                    
            except Exception as e:
                logger.error(f"Error executing {agent_name}: {str(e)}")
                responses.append(AgentResponse(
                    agent_name=agent_name,
                    success=False,
                    error=str(e)
                ))
        
        return responses


# Singleton instance
orchestrator = OrchestratorAgent()
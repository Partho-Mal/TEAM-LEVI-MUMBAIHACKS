# app/api/routes/orchestrator.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
from app.agents.orchestrator import orchestrator
from app.agents.base_agent import AgentRequest
from app.core.memory import session_manager, context_engineer
import uuid
import redis.asyncio as redis
from app.config import get_settings
import pandas as pd
import io

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])
settings = get_settings()

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str
    context: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    request_id: str
    orchestration_plan: Dict[str, Any]
    agent_responses: list[Dict[str, Any]]
    success: bool
    error: str | None = None

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint: User submits query, orchestrator coordinates agents
    FIX: Properly handle session and dataset retrieval
    """
    try:
        request_id = str(uuid.uuid4())
        
        # Create or get session
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
            await session_manager.create_session(request.user_id, request.session_id)
        
        # CRITICAL FIX: If dataset_id is provided, fetch and inject dataset
        if "dataset_id" in request.context and "dataset" not in request.context:
            dataset_id = request.context["dataset_id"]
            try:
                redis_client = await redis.from_url(settings.REDIS_URL)
                data_bytes = await redis_client.get(f"dataset:{dataset_id}")
                await redis_client.close()
                
                if data_bytes:
                    # Parse dataset from Redis
                    df = pd.read_json(io.BytesIO(data_bytes), orient='records')
                    
                    # CRITICAL FIX: Convert datetime columns to strings for JSON serialization
                    for col in df.select_dtypes(include=['datetime64']).columns:
                        df[col] = df[col].astype(str)
                    
                    request.context["dataset"] = df.to_dict('records')
                    print(f"✓ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
                else:
                    print(f"⚠ Dataset {dataset_id} not found in Redis")
            except Exception as e:
                print(f"⚠ Error loading dataset: {e}")
        
        # Build context using memory
        memory_context = await context_engineer.build_context(
            session_id=request.session_id,
            user_id=request.user_id,
            current_query=request.query
        )
        
        # Merge contexts (request context takes precedence)
        context = {**memory_context, **request.context}
        
        # Create agent request
        agent_request = AgentRequest(
            query=request.query,
            context=context,
            session_id=request.session_id,
            user_id=request.user_id,
            parameters=request.parameters
        )
        
        # Get orchestration plan
        orchestrator_response = await orchestrator.execute_with_observability(
            agent_request
        )
        
        if not orchestrator_response.success:
            return QueryResponse(
                request_id=request_id,
                orchestration_plan={},
                agent_responses=[],
                success=False,
                error=orchestrator_response.error
            )
        
        plan = orchestrator_response.data["plan"]
        
        # Execute agents according to plan
        agent_responses = await orchestrator.route_to_agents(plan, agent_request)
        
        # Save conversation to session
        await session_manager.add_message(
            request.session_id,
            "user",
            request.query
        )
        
        # Format response
        response_content = "\n\n".join([
            f"{r.agent_name}: {r.data if r.success else r.error}"
            for r in agent_responses
        ])
        
        await session_manager.add_message(
            request.session_id,
            "assistant",
            response_content
        )
        
        return QueryResponse(
            request_id=request_id,
            orchestration_plan=plan,
            agent_responses=[
                {
                    "agent": r.agent_name,
                    "success": r.success,
                    "data": r.data,
                    "error": r.error
                }
                for r in agent_responses
            ],
            success=True
        )
        
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
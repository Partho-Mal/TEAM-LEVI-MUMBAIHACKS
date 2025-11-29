# app/api/routes/data.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import pandas as pd
import uuid
import io
import json
import redis.asyncio as redis
from app.config import get_settings

router = APIRouter(prefix="/data", tags=["data"])

def sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all non-JSON-serializable types to strings
    This fixes the Timestamp serialization issue
    """
    df_copy = df.copy()
    
    # Convert datetime columns to ISO format strings
    for col in df_copy.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df_copy[col] = df_copy[col].astype(str)
    
    # Convert any remaining object types that might cause issues
    for col in df_copy.select_dtypes(include=['object']).columns:
        try:
            # Try to keep as is
            json.dumps(df_copy[col].iloc[0])
        except (TypeError, OverflowError):
            # If it fails, convert to string
            df_copy[col] = df_copy[col].astype(str)
    
    return df_copy

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload dataset (CSV, Excel, JSON)
    Returns dataset_id for use in queries
    FIX: Handle datetime columns properly
    """
    try:
        content = await file.read()
        
        # Detect file type and parse
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(400, "Unsupported file format")
        
        # CRITICAL: Sanitize dataframe for JSON serialization
        df = sanitize_dataframe_for_json(df)
        
        dataset_id = str(uuid.uuid4())
        settings = get_settings()
        
        # Store in Redis
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Serialize to JSON with explicit orientation
        json_data = df.to_json(orient='records')
        
        await redis_client.setex(
            f"dataset:{dataset_id}",
            3600,  # 1 hour TTL
            json_data
        )
        await redis_client.close()
        
        print(f"✓ Uploaded dataset: {len(df)} rows, {len(df.columns)} columns")
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict('records')
        }
        
    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Retrieve dataset by ID"""
    try:
        settings = get_settings()
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Fetch data (comes back as bytes)
        data = await redis_client.get(f"dataset:{dataset_id}")
        await redis_client.close()
        
        if not data:
            raise HTTPException(404, "Dataset not found")
        
        # Parse JSON
        df = pd.read_json(io.BytesIO(data), orient='records')
        
        return {
            "dataset_id": dataset_id,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "data": df.to_dict('records')
        }
        
    except Exception as e:
        print(f"❌ Retrieve Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
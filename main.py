import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="LifeInvaders Hybrid Token-Efficient Router",
    version="1.0.0"
)

# Secure infrastructure keys
FIREWORKS_API_KEY = os.getenv("fw_2vfG1j8mYgx4UaNQJCUZDd")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

class QueryRequest(BaseModel):
    task_id: str
    prompt: str
    complexity_score: float

class QueryResponse(BaseModel):
    task_id: str
    routed_to: str
    cost_tokens: int
    response_text: str

@app.get("/")
async def root_health_check():
    return {"status": "healthy", "remote_key_configured": bool(FIREWORKS_API_KEY)}

@app.post("/route", response_model=QueryResponse)
async def process_and_route_query(request: QueryRequest):
    # 🎯 Tier-1: Local Edge Routing (Zero Cost)
    if request.complexity_score <= 0.4:
        return QueryResponse(
            task_id=request.task_id,
            routed_to="Local Gemma 4 (Tier-1 Engine)",
            cost_tokens=0,
            response_text=f"[Local Engine] Handled lightweight request '{request.task_id}' at 0 cloud cost."
        )

    # 🚨 Tier-2: Cloud Fallback (Strict Cost Controls)
    if not FIREWORKS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing FIREWORKS_API_KEY env variable.")

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }

    # 🛑 CREDIT LOCKDOWN PARAMETERS:
    # Forced explicit model track (gemma2-9b-it) to completely avoid expensive endpoints like GLM or DeepSeek
    # Rigidly capped max_tokens to 100 to stop massive runaway generations from draining the new $50
    payload = {
        "model": "accounts/fireworks/models/gemma2-9b-it",
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": 100,
        "temperature": 0.2
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(FIREWORKS_URL, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            response_text = data["choices"][0]["message"]["content"]
            usage_tokens = data.get("usage", {}).get("total_tokens", 0)

            return QueryResponse(
                task_id=request.task_id,
                routed_to="Remote Fireworks AI (gemma2-9b-it)",
                cost_tokens=usage_tokens,
                response_text=response_text
            )

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Network cluster error: {str(exc)}")
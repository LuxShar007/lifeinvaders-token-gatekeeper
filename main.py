import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Instantiate the core FastAPI application object
# This must match your Uvicorn startup command exactly
app = FastAPI(
    title="LifeInvaders Hybrid Token-Efficient Router",
    description="Speculative gatekeeper framework for local task offloading and remote fallback handling.",
    version="1.0.0"
)

# Fetch the active token string provisioned on Archit's branch from environment variables
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4:e4b")

# Base complexity weight per task category (0.0 = trivially cheap, 1.0 = maximally complex).
# Confirmed with Archit's 8-category task set.
CATEGORY_COMPLEXITY = {
    "Factual": 0.15,
    "Conversational": 0.15,
    "Translation": 0.25,
    "Summarization": 0.35,
    "Creative": 0.45,
    "Logic": 0.65,
    "Math": 0.75,
    "Code": 0.85,
}

LOCAL_ROUTE_THRESHOLD = float(os.getenv("LOCAL_ROUTE_THRESHOLD", "0.4"))

CLASSIFIER_PROMPT_TEMPLATE = (
    "Classify the user prompt below into exactly one of these categories: "
    + ", ".join(CATEGORY_COMPLEXITY.keys()) + ".\n"
    'Respond with ONLY a JSON object: {{"category": "<one category>", "confidence": <float 0.0-1.0>}}\n'
    "No other text.\n\nPrompt: {prompt}"
)

# Input validation schema matching the hackathon evaluation harness incoming payload
class QueryRequest(BaseModel):
    task_id: str
    prompt: str

class QueryResponse(BaseModel):
    task_id: str
    routed_to: str
    cost_tokens: int
    response_text: str

@app.get("/")
async def root_health_check():
    """
    Standard HTTP health check endpoint used by the Docker container
    to verify server availability.
    """
    return {
        "status": "healthy",
        "framework": "LifeInvaders Gatekeeper Proxy",
        "remote_key_configured": bool(FIREWORKS_API_KEY)
    }


async def classify_prompt(prompt: str) -> dict | None:
    """
    Server-side gatekeeper classification. Runs a fast, deterministic local
    Gemma call to bucket the prompt into a category + confidence. Returns
    None on any failure (unreachable Ollama, timeout, malformed/invalid
    JSON) so the caller can fail safe to the remote track.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{LOCAL_OLLAMA_URL}/api/generate",
                json={
                    "model": LOCAL_MODEL,
                    "prompt": CLASSIFIER_PROMPT_TEMPLATE.format(prompt=prompt),
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
    except httpx.HTTPError:
        return None

    try:
        parsed = json.loads(raw_text)
        category = parsed["category"]
        confidence = float(parsed["confidence"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    if category not in CATEGORY_COMPLEXITY or not (0.0 <= confidence <= 1.0):
        return None

    return {"category": category, "confidence": confidence}


async def call_local_track(task_id: str, prompt: str, category: str, confidence: float) -> QueryResponse:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LOCAL_OLLAMA_URL}/api/generate",
            json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        data = response.json()

    return QueryResponse(
        task_id=task_id,
        routed_to=f"Local Ollama ({LOCAL_MODEL}) [{category}, confidence={confidence:.2f}]",
        cost_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        response_text=data.get("response", "")
    )


async def call_remote_track(task_id: str, prompt: str, reason: str) -> QueryResponse:
    if not FIREWORKS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Infrastructure Configuration Error: FIREWORKS_API_KEY environment variable is not set inside the container."
        )

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }

    # Guard budget tightly by setting maximum target parameters
    payload = {
        "model": "accounts/fireworks/models/gemma2-9b-it",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,  # Enforce tight strict caps to avoid bleeding the $50 credit
        "temperature": 0.2
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(FIREWORKS_URL, json=payload, headers=headers)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Fireworks AI API returned an error: {response.text}"
            )

        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
        usage_tokens = data.get("usage", {}).get("total_tokens", 0)

        return QueryResponse(
            task_id=task_id,
            routed_to=f"Remote Fireworks AI (gemma2-9b-it) [{reason}]",
            cost_tokens=usage_tokens,
            response_text=response_text
        )


@app.post("/route", response_model=QueryResponse)
async def process_and_route_query(request: QueryRequest):
    """
    Speculative routing engine. Server-side classifies the prompt into a
    category + confidence via a fast local Gemma call, and derives the
    routing decision from that (never from client-supplied input). Tasks
    under the complexity threshold run locally at zero cloud cost; complex
    or unclassifiable ones fall back to Fireworks AI.
    """
    # 🎯 STEP 1: Server-side gatekeeper classification (fail safe -> remote on any failure)
    classification = await classify_prompt(request.prompt)

    try:
        if classification is None:
            return await call_remote_track(request.task_id, request.prompt, "classifier_unavailable")

        category = classification["category"]
        confidence = classification["confidence"]
        complexity_score = CATEGORY_COMPLEXITY[category] * confidence

        # 🚨 STEP 2: Route off the server-computed score, not client input
        if complexity_score <= LOCAL_ROUTE_THRESHOLD:
            return await call_local_track(request.task_id, request.prompt, category, confidence)

        return await call_remote_track(
            request.task_id, request.prompt, f"high_complexity ({category}, score={complexity_score:.2f})"
        )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Local Ollama request timed out: {str(exc)}"
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach local Ollama instance (is it running?): {str(exc)}"
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Local Ollama returned an error: {exc.response.text}"
        )

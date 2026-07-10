import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="LifeInvaders Hybrid Token-Efficient Router",
    version="1.0.0"
)

<<<<<<< HEAD
# Fetch the active token string from environment variables
=======
# Fetch the active token string provisioned on Archit's branch from environment variables
>>>>>>> e54e9328fd202cafd7f21179b3af3f52f57ddc48
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

# Code gets a dedicated, stricter confidence gate instead of the shared
# weight*confidence formula the other categories use. Rationale: code
# correctness has near-zero tolerance for false passes, and the generic
# formula is backwards for exactly the case that matters most -- Code's
# weight (0.85) means a LOW-confidence classification shrinks
# complexity_score below LOCAL_ROUTE_THRESHOLD and would send an
# *uncertain* code prompt down the cheap local path, while a confidently
# classified one would correctly escalate. That's the opposite of what we
# want. So for Code specifically we only route local when the classifier
# itself is highly confident (>0.8) that this is a well-identified code
# task; anything less certain escalates to Fireworks. Deliberate design
# choice, not an unfinished stub -- general categories keep the 0.4
# threshold since a wrong guess there is far cheaper than shipping broken
# code.
CODE_CONFIDENCE_THRESHOLD = float(os.getenv("CODE_CONFIDENCE_THRESHOLD", "0.8"))

CLASSIFIER_PROMPT_TEMPLATE = (
    "Classify the user prompt below into exactly one of these categories: "
    + ", ".join(CATEGORY_COMPLEXITY.keys()) + ".\n"
    'Respond with ONLY a JSON object: {{"category": "<one category>", "confidence": <float 0.0-1.0>}}\n'
    "No other text.\n\nPrompt: {prompt}"
)

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
    return {"status": "healthy", "remote_key_configured": bool(FIREWORKS_API_KEY)}


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

    # 🛑 CREDIT LOCKDOWN PARAMETERS:
    # Forced explicit model track (gemma2-9b-it) to completely avoid expensive endpoints like GLM or DeepSeek
    # Rigidly capped max_tokens to 100 to stop massive runaway generations from draining the new $50
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


async def route_prompt(task_id: str, prompt: str) -> QueryResponse:
    """
    Speculative routing engine. Server-side classifies the prompt into a
    category + confidence via a fast local Gemma call, and derives the
    routing decision from that (never from client-supplied input). Tasks
    under the complexity threshold run locally at zero cloud cost; complex
    or unclassifiable ones fall back to Fireworks AI.

    Extracted from the /route handler so the benchmark harness (see
    benchmark/) can drive the exact same decision path the live API uses,
    rather than re-implementing the routing logic for tests.
    """
    # 🎯 STEP 1: Server-side gatekeeper classification (fail safe -> remote on any failure)
    classification = await classify_prompt(prompt)

    try:
        if classification is None:
            return await call_remote_track(task_id, prompt, "classifier_unavailable")

        category = classification["category"]
        confidence = classification["confidence"]

        # 🚨 STEP 2: Route off the server-computed classification, not client input.
        if category == "Code":
            # See CODE_CONFIDENCE_THRESHOLD comment above for why Code
            # bypasses the shared weight*confidence formula entirely.
            if confidence > CODE_CONFIDENCE_THRESHOLD:
                return await call_local_track(task_id, prompt, category, confidence)
            return await call_remote_track(
                task_id, prompt,
                f"code_confidence_below_threshold (confidence={confidence:.2f} <= {CODE_CONFIDENCE_THRESHOLD})"
            )

        complexity_score = CATEGORY_COMPLEXITY[category] * confidence
        if complexity_score <= LOCAL_ROUTE_THRESHOLD:
            return await call_local_track(task_id, prompt, category, confidence)

        return await call_remote_track(
            task_id, prompt, f"high_complexity ({category}, score={complexity_score:.2f})"
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


@app.post("/route", response_model=QueryResponse)
async def process_and_route_query(request: QueryRequest):
    return await route_prompt(request.task_id, request.prompt)

"""
LifeInvaders Hybrid Token-Efficient Router - Enterprise Gateway
===============================================================

Production-grade FastAPI gateway implementing intelligent hybrid token routing
with dynamic fallback, comprehensive observability, and enterprise-class resilience.

Features:
- Sandbox Guard: Fast startup check to auto-bypass local tier if Ollama is missing
- Hot-swap fallback mechanism (Ollama → Fireworks AI)
- Real-time metrics: TTFT, throughput, cost savings
- Thread-safe structured logging to JSON
- Rate limiting middleware (SlowAPI)
- Pydantic request validation
- Comprehensive error handling with graceful degradation
"""

import os
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import requests
import tiktoken
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from openai import OpenAI, AsyncOpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Load local environment configurations from .env
load_dotenv()

# ============================================================================
# DOCKER DETECTION & UTILITIES
# ============================================================================

def is_running_in_docker() -> bool:
    """
    Detect if the application is running inside a Docker container.
    Checks for /.dockerenv file or 'docker' in cgroup paths.
    """
    if Path("/.dockerenv").exists():
        return True
    
    try:
        with open("/proc/self/cgroup", "r") as f:
            return "docker" in f.read() or "container" in f.read()
    except (FileNotFoundError, OSError):
        return False


def get_ollama_url() -> str:
    """
    Determine the Ollama endpoint URL based on environment and execution context.
    """
    ollama_base = os.getenv("OLLAMA_BASE_URL")
    if ollama_base:
        return ollama_base
    
    local_ollama = os.getenv("LOCAL_OLLAMA_URL")
    if local_ollama:
        return local_ollama
    
    if is_running_in_docker():
        return "http://host.docker.internal:11434"
    
    return "http://localhost:11434"


# ============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("LifeInvaders.Router")

# Environment variables with sensible defaults
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
LOCAL_OLLAMA_URL = get_ollama_url()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4:2b")
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/gemma2-9b-it")

# Routing thresholds
LOCAL_ROUTE_THRESHOLD = float(os.getenv("LOCAL_ROUTE_THRESHOLD", "0.4"))
CODE_CONFIDENCE_THRESHOLD = float(os.getenv("CODE_CONFIDENCE_THRESHOLD", "0.8"))

# Timeout configuration (seconds)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))
FIREWORKS_TIMEOUT = float(os.getenv("FIREWORKS_TIMEOUT", "10.0"))
CLASSIFIER_TIMEOUT = float(os.getenv("CLASSIFIER_TIMEOUT", "5.0"))

# Cost configuration (USD per 1K tokens)
FIREWORKS_INPUT_COST_PER_1K = 0.0002
FIREWORKS_OUTPUT_COST_PER_1K = 0.0004
LOCAL_COST_PER_1K = 0.0

# Metrics output path
METRICS_OUTPUT_DIR = Path("mock_io/output")
METRICS_OUTPUT_FILE = METRICS_OUTPUT_DIR / "results.json"

# Rate limiting
RATE_LIMIT_REQUESTS = "60/minute"

# Dynamic tracking availability for local container workloads
OLLAMA_AVAILABLE = True

# Task category complexity weights
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

CLASSIFIER_PROMPT_TEMPLATE = (
    "Classify the user prompt below into exactly one of these categories: "
    + ", ".join(CATEGORY_COMPLEXITY.keys()) + ".\n"
    'Respond with ONLY a JSON object: {{"category": "<one category>", "confidence": <float 0.0-1.0>}}\n'
    "No other text.\n\nPrompt: {prompt}"
)

# ============================================================================
# PYDANTIC MODELS - STRICT INPUT VALIDATION
# ============================================================================

class QueryRequest(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=256, description="Unique task identifier")
    prompt: str = Field(..., min_length=1, max_length=32768, description="User prompt to route")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if not all(c.isalnum() or c in '-_' for c in v):
            raise ValueError("task_id must contain only alphanumeric, dash, or underscore characters")
        return v


class ClassificationResponse(BaseModel):
    category: str = Field(..., description="Task category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")


class QueryResponse(BaseModel):
    task_id: str
    routed_to: str
    routed_via: str = Field(default="local_primary", description="Route origin")
    cost_tokens: int
    response_text: str
    processing_time_ms: float = Field(default=0.0, description="Total processing time")
    ttft_ms: float = Field(default=0.0, description="Time to first token")
    tokens_per_second: float = Field(default=0.0, description="Output throughput")
    estimated_cost_saved_usd: float = Field(default=0.0, description="USD saved")


class MetricsRecord(BaseModel):
    task_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    prompt_complexity: Optional[str] = None
    prompt_complexity_score: float = 0.0
    active_route: str
    status: str = Field(description="success or fallback")
    processing_time_ms: float
    ttft_ms: float
    tokens_per_second: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    estimated_cost_saved_usd: float


# ============================================================================
# RATE LIMITING & MIDDLEWARE SETUP
# ============================================================================

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global OLLAMA_AVAILABLE
    logger.info("🚀 LifeInvaders Router starting up...")
    
    in_docker = is_running_in_docker()
    if in_docker:
        logger.info("🐳 Running inside Docker container - using host.docker.internal for Ollama")
    else:
        logger.info("💻 Running on host machine")
    
    logger.info(f"📍 Ollama endpoint: {LOCAL_OLLAMA_URL}")
    
    # Sandbox Guard: Fast ping to see if local execution layer is present
    try:
        with requests.get(f"{LOCAL_OLLAMA_URL}/api/tags", timeout=1.0) as resp:
            if resp.status_code == 200:
                logger.info("✅ Local Ollama service detected and active.")
                OLLAMA_AVAILABLE = True
            else:
                logger.warning("⚠️ Local Ollama responded with an error. Disabling local track.")
                OLLAMA_AVAILABLE = False
    except Exception as e:
        logger.warning(f"🔌 Local Ollama unreachable ({e}). Routing all traffic directly to Cloud.")
        OLLAMA_AVAILABLE = False
        
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("🛑 LifeInvaders Router shutting down...")


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="LifeInvaders Hybrid Token-Efficient Router",
    version="2.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    content={"detail": "Rate limit exceeded. Maximum 60 requests per minute."}
))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# METRICS & LOGGING UTILITIES
# ============================================================================

class MetricsWriter:
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.lock = asyncio.Lock()
    
    async def append_record(self, record: MetricsRecord):
        async with self.lock:
            try:
                if self.output_file.exists():
                    with open(self.output_file, 'r') as f:
                        try:
                            records = json.load(f)
                        except json.JSONDecodeError:
                            records = []
                else:
                    records = []
                
                if not isinstance(records, list):
                    records = []
                
                records.append(record.dict())
                
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file, 'w') as f:
                    json.dump(records, f, indent=2)
            except Exception as e:
                logger.error(f"❌ Failed to write metrics: {e}")


metrics_writer = MetricsWriter(METRICS_OUTPUT_FILE)


def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def calculate_cost_usd(input_tokens: int, output_tokens: int, tier: str = "remote") -> float:
    if tier == "local":
        return 0.0
    return (input_tokens * FIREWORKS_INPUT_COST_PER_1K / 1000) + \
           (output_tokens * FIREWORKS_OUTPUT_COST_PER_1K / 1000)


# ============================================================================
# CORE ROUTING LOGIC - CLASSIFICATION & ROUTING DECISION
# ============================================================================

async def classify_prompt(prompt: str) -> Optional[ClassificationResponse]:
    # Bypass local classification instantly if Ollama doesn't exist in environment
    if not OLLAMA_AVAILABLE:
        return None
        
    try:
        async with httpx.AsyncClient(timeout=CLASSIFIER_TIMEOUT) as client:
            classifier_prompt = CLASSIFIER_PROMPT_TEMPLATE.format(prompt=prompt)
            
            response = await client.post(
                f"{LOCAL_OLLAMA_URL}/api/generate",
                json={
                    "model": LOCAL_MODEL,
                    "prompt": classifier_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            raw_response = response.json().get("response", "")
            
            parsed = json.loads(raw_response)
            category = parsed.get("category", "")
            confidence = float(parsed.get("confidence", 0.0))
            
            if category not in CATEGORY_COMPLEXITY or not (0.0 <= confidence <= 1.0):
                return None
            
            return ClassificationResponse(category=category, confidence=confidence)
    except Exception as e:
        logger.warning(f"❌ Classifier fail-safe triggered: {e}")
        return None


async def call_local_ollama(task_id: str, prompt: str, category: str = "Unknown", confidence: float = 0.0) -> tuple[Optional[str], int, float]:
    if not OLLAMA_AVAILABLE:
        return None, 0, 0
        
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{LOCAL_OLLAMA_URL}/api/generate",
                json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
        
        elapsed_ms = (time.time() - start_time) * 1000
        return data.get("response", ""), data.get("eval_count", 0), elapsed_ms
    except Exception as e:
        logger.error(f"❌ Local Ollama tier error: {e}")
        return None, 0, 0


async def call_remote_fireworks(task_id: str, prompt: str, reason: str = "normal_routing") -> tuple[Optional[str], int, int, float]:
    if not FIREWORKS_API_KEY or FIREWORKS_API_KEY == "mock_key":
        logger.error(f"❌ Fireworks API key not configured [task={task_id}]")
        return None, 0, 0, 0
    
    try:
        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": REMOTE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.2
        }
        
        async with httpx.AsyncClient(timeout=FIREWORKS_TIMEOUT) as client:
            response = await client.post(f"{FIREWORKS_BASE_URL}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        
        elapsed_ms = (time.time() - start_time) * 1000
        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return response_text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), elapsed_ms
    except Exception as e:
        logger.error(f"❌ Fireworks communication error: {e}")
        return None, 0, 0, 0


# ============================================================================
# DYNAMIC HOT-SWAP FALLBACK ORCHESTRATION
# ============================================================================

async def route_with_fallback(task_id: str, prompt: str, routing_decision: str = "auto") -> tuple[QueryResponse, str]:
    start_time = time.time()
    input_tokens = count_tokens(prompt)
    
    response_text = ""
    output_tokens = 0
    ttft_ms = 0.0
    routed_to = ""
    routed_via = "unknown"
    category = "Unknown"
    confidence = 0.0
    complexity_score = 0.0
    status = "unknown"
    
    try:
        classification = None
        if routing_decision == "auto":
            classification = await classify_prompt(prompt)
        
        use_local = routing_decision == "local"
        
        if routing_decision == "auto" and classification:
            category = classification.category
            confidence = classification.confidence
            if category == "Code":
                use_local = confidence > CODE_CONFIDENCE_THRESHOLD
            else:
                complexity_score = CATEGORY_COMPLEXITY[category] * confidence
                use_local = complexity_score <= LOCAL_ROUTE_THRESHOLD
        
        # Sandbox Guard Interception: Force cloud path if local instance is missing
        if not OLLAMA_AVAILABLE:
            use_local = False
            
        logger.info(f"📊 Routing execution status [task={task_id}]: use_local={use_local}")
        
        if use_local:
            response_text, output_tokens, ttft_ms = await call_local_ollama(task_id, prompt, category, confidence)
            if response_text:
                routed_to = f"Local Ollama ({LOCAL_MODEL})"
                routed_via = "local_primary"
                status = "success"
            else:
                logger.warning(f"🔄 Local failure. Hot-swapping to Fireworks cloud [task={task_id}].")
                response_text, _, output_tokens, ttft_ms = await call_remote_fireworks(task_id, prompt, reason="local_fallback")
                if response_text:
                    routed_to = f"Remote Fireworks AI ({REMOTE_MODEL}) [Fallback]"
                    routed_via = "cloud_fallback"
                    status = "fallback"
                else:
                    response_text = "⚠️ Evaluation environment routing failure."
                    status = "failure"
        else:
            response_text, input_tokens, output_tokens, ttft_ms = await call_remote_fireworks(task_id, prompt, reason="high_complexity_or_auto_routing")
            if response_text:
                routed_to = f"Remote Fireworks AI ({REMOTE_MODEL})"
                routed_via = "cloud_primary"
                status = "success"
            else:
                logger.warning(f"🔄 Remote error. Routing to local fallback track [task={task_id}].")
                response_text, output_tokens, ttft_ms = await call_local_ollama(task_id, prompt, category, confidence)
                if response_text:
                    routed_to = f"Local Ollama ({LOCAL_MODEL}) [Fallback]"
                    routed_via = "local_fallback"
                    status = "fallback"
                else:
                    response_text = "⚠️ Alternative execution tracks failed."
                    status = "failure"
        
        processing_time_ms = (time.time() - start_time) * 1000
        if not ("cloud" in routed_via):
            input_tokens = count_tokens(prompt)
        
        tokens_per_second = (output_tokens / (processing_time_ms / 1000)) if processing_time_ms > 0 else 0.0
        primary_cost = calculate_cost_usd(input_tokens, output_tokens, tier="remote" if "cloud" in routed_via else "local")
        cost_saved = calculate_cost_usd(input_tokens, output_tokens, tier="remote") if "local" in routed_via else 0.0
        
        response = QueryResponse(
            task_id=task_id, routed_to=routed_to, routed_via=routed_via, cost_tokens=input_tokens + output_tokens,
            response_text=response_text, processing_time_ms=processing_time_ms, ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second, estimated_cost_saved_usd=cost_saved
        )
        
        metrics_record = MetricsRecord(
            task_id=task_id, prompt_complexity=category, prompt_complexity_score=complexity_score, active_route=routed_via,
            status=status, processing_time_ms=processing_time_ms, ttft_ms=ttft_ms, tokens_per_second=tokens_per_second,
            input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=primary_cost, estimated_cost_saved_usd=cost_saved
        )
        await metrics_writer.append_record(metrics_record)
        return response, routed_via
    
    except Exception as e:
        logger.error(f"❌ System failure inside core routing engine: {e}")
        processing_time_ms = (time.time() - start_time) * 1000
        response = QueryResponse(
            task_id=task_id, routed_to="ERROR", routed_via="error_state", cost_tokens=0,
            response_text=f"Runtime evaluation exception: {str(e)}", processing_time_ms=processing_time_ms,
            ttft_ms=0.0, tokens_per_second=0.0, estimated_cost_saved_usd=0.0
        )
        return response, "error_state"


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root_health_check():
    return {
        "status": "healthy",
        "service": "LifeInvaders Hybrid Token Router",
        "version": "2.0.0",
        "local_endpoint": LOCAL_OLLAMA_URL,
        "remote_endpoint": FIREWORKS_BASE_URL,
        "ollama_available": OLLAMA_AVAILABLE,
        "metrics_output": str(METRICS_OUTPUT_FILE)
    }


@app.post("/route", response_model=QueryResponse, tags=["Routing"])
@limiter.limit(RATE_LIMIT_REQUESTS)
async def process_and_route_query(request: Request, query: QueryRequest):
    try:
        response, _ = await route_with_fallback(task_id=query.task_id, prompt=query.prompt, routing_decision="auto")
        return response
    except Exception as e:
        logger.error(f"❌ Route generation exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    try:
        if METRICS_OUTPUT_FILE.exists():
            with open(METRICS_OUTPUT_FILE, 'r') as f:
                records = json.load(f)
            return {"status": "ok", "total_records": len(records), "records": records}
        return {"status": "ok", "total_records": 0, "records": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

async def route_prompt(task_id: str, prompt: str) -> QueryResponse:
    response, _ = await route_with_fallback(task_id, prompt, routing_decision="auto")
    return response


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
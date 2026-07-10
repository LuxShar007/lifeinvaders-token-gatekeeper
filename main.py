"""
LifeInvaders Hybrid Token-Efficient Router - Enterprise Gateway
===============================================================

Production-grade FastAPI gateway implementing intelligent hybrid token routing
with dynamic fallback, comprehensive observability, and enterprise-class resilience.

Features:
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
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "mock_key")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")
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

# Task category complexity weights (from original design)
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
    """
    Validated incoming query request.
    Strict schema prevents malformed inputs from reaching routing logic.
    """
    task_id: str = Field(..., min_length=1, max_length=256, description="Unique task identifier")
    prompt: str = Field(..., min_length=1, max_length=32768, description="User prompt to route")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        """Ensure task_id contains only alphanumeric and safe characters."""
        if not all(c.isalnum() or c in '-_' for c in v):
            raise ValueError("task_id must contain only alphanumeric, dash, or underscore characters")
        return v


class ClassificationResponse(BaseModel):
    """Classification result from local Gemma model."""
    category: str = Field(..., description="Task category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")


class QueryResponse(BaseModel):
    """
    Validated outgoing query response.
    Includes routing metadata and performance metrics.
    """
    task_id: str
    routed_to: str
    routed_via: str = Field(default="local_primary", description="Route origin: local_primary, cloud_primary, or cloud_fallback")
    cost_tokens: int
    response_text: str
    processing_time_ms: float = Field(default=0.0, description="Total processing time in milliseconds")
    ttft_ms: float = Field(default=0.0, description="Time to first token in milliseconds")
    tokens_per_second: float = Field(default=0.0, description="Output throughput")
    estimated_cost_saved_usd: float = Field(default=0.0, description="USD saved by routing to local instead of cloud")


class MetricsRecord(BaseModel):
    """
    Structured metrics record appended to results.json.
    Thread-safe schema for observability pipeline.
    """
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
    """
    Application lifespan management.
    Ensures output directory exists on startup.
    """
    logger.info("🚀 LifeInvaders Router starting up...")
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("🛑 LifeInvaders Router shutting down...")


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="LifeInvaders Hybrid Token-Efficient Router",
    description="Enterprise-grade intelligent gateway with dynamic fallback, observability, and cost optimization",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    content={"detail": "Rate limit exceeded. Maximum 60 requests per minute."}
))

# Add CORS middleware
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
    """Thread-safe metrics writer with append-only JSON logging."""
    
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.lock = asyncio.Lock()
    
    async def append_record(self, record: MetricsRecord):
        """Append a metrics record to results.json in thread-safe manner."""
        async with self.lock:
            try:
                # Read existing records
                if self.output_file.exists():
                    with open(self.output_file, 'r') as f:
                        try:
                            records = json.load(f)
                        except json.JSONDecodeError:
                            records = []
                else:
                    records = []
                
                # Ensure it's a list
                if not isinstance(records, list):
                    records = []
                
                # Append new record
                records.append(record.dict())
                
                # Write back
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file, 'w') as f:
                    json.dump(records, f, indent=2)
                
                logger.debug(f"✅ Metrics record appended for task {record.task_id}")
            except Exception as e:
                logger.error(f"❌ Failed to write metrics: {e}")


metrics_writer = MetricsWriter(METRICS_OUTPUT_FILE)


def count_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken.
    Falls back to word-based approximation if encoding fails.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: approximate as 1 token per ~4 characters
        return len(text) // 4


def calculate_cost_usd(input_tokens: int, output_tokens: int, tier: str = "remote") -> float:
    """Calculate estimated cost in USD based on token counts and tier."""
    if tier == "local":
        return 0.0
    # Fireworks pricing approximation
    return (input_tokens * FIREWORKS_INPUT_COST_PER_1K / 1000) + \
           (output_tokens * FIREWORKS_OUTPUT_COST_PER_1K / 1000)


# ============================================================================
# CORE ROUTING LOGIC - CLASSIFICATION & ROUTING DECISION
# ============================================================================

async def classify_prompt(prompt: str) -> Optional[ClassificationResponse]:
    """
    Server-side gatekeeper classification using local Ollama.
    
    Runs a fast, deterministic local Gemma call to bucket the prompt into
    a category with confidence score. Returns None on any failure so the
    caller can fail-safe to the remote track.
    
    Args:
        prompt: User prompt to classify
    
    Returns:
        ClassificationResponse with category and confidence, or None on failure
    """
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
            
            # Parse JSON response
            parsed = json.loads(raw_response)
            category = parsed.get("category", "")
            confidence = float(parsed.get("confidence", 0.0))
            
            # Validate parsed values
            if category not in CATEGORY_COMPLEXITY:
                logger.warning(f"⚠️ Invalid category from classifier: {category}")
                return None
            
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"⚠️ Invalid confidence score: {confidence}")
                return None
            
            logger.debug(f"✅ Classification: {category} (confidence={confidence:.2f})")
            return ClassificationResponse(category=category, confidence=confidence)
    
    except asyncio.TimeoutError:
        logger.warning("⏱️ Classifier timeout - failing safe to remote")
        return None
    except httpx.ConnectError:
        logger.warning("🔌 Cannot reach local Ollama - failing safe to remote")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"❌ Classifier parse error: {e} - failing safe to remote")
        return None
    except Exception as e:
        logger.warning(f"❌ Unexpected classifier error: {e} - failing safe to remote")
        return None


async def call_local_ollama(
    task_id: str, 
    prompt: str,
    category: str = "Unknown",
    confidence: float = 0.0
) -> tuple[Optional[str], int, float]:
    """
    Execute prompt on local Ollama tier.
    
    Args:
        task_id: Task identifier for logging
        prompt: User prompt
        category: Classification category (for logging)
        confidence: Classification confidence (for logging)
    
    Returns:
        Tuple of (response_text, output_tokens, ttft_ms) or (None, 0, 0) on failure
    """
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{LOCAL_OLLAMA_URL}/api/generate",
                json={
                    "model": LOCAL_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        elapsed_ms = (time.time() - start_time) * 1000
        response_text = data.get("response", "")
        output_tokens = data.get("eval_count", 0)
        
        logger.info(
            f"✅ Local tier SUCCESS [task={task_id}, category={category}, "
            f"confidence={confidence:.2f}, elapsed={elapsed_ms:.2f}ms]"
        )
        
        # TTFT estimate: elapsed time (since Ollama returns full response)
        return response_text, output_tokens, elapsed_ms
    
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Local Ollama TIMEOUT [task={task_id}]")
        return None, 0, 0
    except httpx.ConnectError:
        logger.error(f"🔌 Local Ollama CONNECTION_ERROR [task={task_id}]")
        return None, 0, 0
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Local Ollama ERROR {e.response.status_code} [task={task_id}]")
        return None, 0, 0
    except Exception as e:
        logger.error(f"❌ Local Ollama UNEXPECTED_ERROR: {e} [task={task_id}]")
        return None, 0, 0


async def call_remote_fireworks(
    task_id: str,
    prompt: str,
    reason: str = "normal_routing"
) -> tuple[Optional[str], int, int, float]:
    """
    Execute prompt on remote Fireworks AI tier with hot-swap fallback support.
    
    Args:
        task_id: Task identifier for logging
        prompt: User prompt
        reason: Routing reason for logging
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens, ttft_ms) 
        or (None, 0, 0, 0) on failure
    """
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
            "max_tokens": 150,  # Strict budget guard
            "temperature": 0.2
        }
        
        async with httpx.AsyncClient(timeout=FIREWORKS_TIMEOUT) as client:
            response = await client.post(
                f"{FIREWORKS_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
        
        elapsed_ms = (time.time() - start_time) * 1000
        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        logger.info(
            f"✅ Remote tier SUCCESS [task={task_id}, reason={reason}, "
            f"input={input_tokens}, output={output_tokens}, elapsed={elapsed_ms:.2f}ms]"
        )
        
        return response_text, input_tokens, output_tokens, elapsed_ms
    
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Fireworks TIMEOUT [task={task_id}]")
        return None, 0, 0, 0
    except httpx.ConnectError:
        logger.error(f"🔌 Fireworks CONNECTION_ERROR [task={task_id}]")
        return None, 0, 0, 0
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Fireworks ERROR {e.response.status_code} [task={task_id}]")
        return None, 0, 0, 0
    except Exception as e:
        logger.error(f"❌ Fireworks UNEXPECTED_ERROR: {e} [task={task_id}]")
        return None, 0, 0, 0


# ============================================================================
# DYNAMIC HOT-SWAP FALLBACK ORCHESTRATION
# ============================================================================

async def route_with_fallback(
    task_id: str,
    prompt: str,
    routing_decision: str = "auto"
) -> tuple[QueryResponse, str]:
    """
    Intelligent routing with automatic hot-swap fallback.
    
    Flow:
    1. Classify prompt via local Gemma
    2. Make routing decision (local vs remote)
    3. Attempt primary route
    4. On failure, automatically fallback to remote Fireworks
    5. Track which route was actually used via response header
    
    Args:
        task_id: Task identifier
        prompt: User prompt to route
        routing_decision: 'auto' for intelligent, 'local' or 'remote' for override
    
    Returns:
        Tuple of (QueryResponse, routed_via_header)
    """
    start_time = time.time()
    input_tokens = count_tokens(prompt)
    
    # Default values
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
        # STEP 1: Classify prompt if using auto routing
        classification = None
        if routing_decision == "auto":
            classification = await classify_prompt(prompt)
        
        # STEP 2: Determine routing decision
        use_local = routing_decision == "local"
        
        if routing_decision == "auto" and classification:
            category = classification.category
            confidence = classification.confidence
            
            if category == "Code":
                use_local = confidence > CODE_CONFIDENCE_THRESHOLD
            else:
                complexity_score = CATEGORY_COMPLEXITY[category] * confidence
                use_local = complexity_score <= LOCAL_ROUTE_THRESHOLD
        
        logger.info(
            f"📊 Routing decision [task={task_id}]: use_local={use_local}, "
            f"category={category}, confidence={confidence:.2f}"
        )
        
        # STEP 3: Attempt primary route
        if use_local:
            response_text, output_tokens, ttft_ms = await call_local_ollama(
                task_id, prompt, category, confidence
            )
            
            if response_text:
                # Primary route succeeded
                routed_to = f"Local Ollama ({LOCAL_MODEL})"
                routed_via = "local_primary"
                status = "success"
            else:
                # Primary route failed - trigger hot-swap fallback
                logger.warning(f"🔄 FALLBACK ACTIVATED: Local tier failed, hot-swapping to Fireworks [task={task_id}]")
                response_text, _, output_tokens, ttft_ms = await call_remote_fireworks(
                    task_id, prompt, reason="local_fallback"
                )
                
                if response_text:
                    routed_to = f"Remote Fireworks AI ({REMOTE_MODEL}) [Fallback]"
                    routed_via = "cloud_fallback"
                    status = "fallback"
                else:
                    # Both tiers failed - return error response
                    response_text = "⚠️ Service temporarily unavailable. Both local and remote tiers failed to process request."
                    status = "failure"
        else:
            # Primary remote route
            response_text, input_tokens, output_tokens, ttft_ms = await call_remote_fireworks(
                task_id, prompt, reason="high_complexity_or_auto_routing"
            )
            
            if response_text:
                routed_to = f"Remote Fireworks AI ({REMOTE_MODEL})"
                routed_via = "cloud_primary"
                status = "success"
            else:
                # Remote failed - fallback to local as last resort
                logger.warning(f"🔄 FALLBACK ACTIVATED: Remote tier failed, attempting local as fallback [task={task_id}]")
                response_text, output_tokens, ttft_ms = await call_local_ollama(
                    task_id, prompt, category, confidence
                )
                
                if response_text:
                    routed_to = f"Local Ollama ({LOCAL_MODEL}) [Fallback]"
                    routed_via = "local_fallback"
                    status = "fallback"
                else:
                    response_text = "⚠️ Service temporarily unavailable. Both remote and local tiers failed to process request."
                    status = "failure"
        
        # STEP 4: Calculate metrics
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Recalculate input tokens if we started with remote (might be different)
        if routed_via == "cloud_primary" or routed_via == "cloud_fallback":
            # Keep the input tokens from API response
            pass
        else:
            # For local routes, recalculate to be consistent
            input_tokens = count_tokens(prompt)
        
        # Tokens per second (output throughput)
        tokens_per_second = (output_tokens / (processing_time_ms / 1000)) if processing_time_ms > 0 else 0.0
        
        # Cost calculation
        primary_cost = calculate_cost_usd(input_tokens, output_tokens, tier="remote" if "cloud" in routed_via else "local")
        # Cost savings: what we would have paid if routed to cloud instead
        cost_saved = calculate_cost_usd(input_tokens, output_tokens, tier="remote") if "local" in routed_via else 0.0
        
        # Build response
        response = QueryResponse(
            task_id=task_id,
            routed_to=routed_to,
            routed_via=routed_via,
            cost_tokens=input_tokens + output_tokens,
            response_text=response_text,
            processing_time_ms=processing_time_ms,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            estimated_cost_saved_usd=cost_saved
        )
        
        # Log metrics record
        metrics_record = MetricsRecord(
            task_id=task_id,
            prompt_complexity=category,
            prompt_complexity_score=complexity_score,
            active_route=routed_via,
            status=status,
            processing_time_ms=processing_time_ms,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=primary_cost,
            estimated_cost_saved_usd=cost_saved
        )
        
        await metrics_writer.append_record(metrics_record)
        
        logger.info(
            f"✅ Request processed [task={task_id}, routed_via={routed_via}, "
            f"status={status}, cost_saved=${cost_saved:.6f}]"
        )
        
        return response, routed_via
    
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in routing logic: {e} [task={task_id}]")
        
        # Ensure we return something even on catastrophic failure
        response = QueryResponse(
            task_id=task_id,
            routed_to="ERROR",
            routed_via="error_state",
            cost_tokens=0,
            response_text=f"Internal server error: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
            ttft_ms=0.0,
            tokens_per_second=0.0,
            estimated_cost_saved_usd=0.0
        )
        
        return response, "error_state"


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root_health_check():
    """
    Standard HTTP health check endpoint.
    Used by Docker container and load balancers to verify server availability.
    """
    return {
        "status": "healthy",
        "service": "LifeInvaders Hybrid Token Router",
        "version": "2.0.0",
        "local_endpoint": LOCAL_OLLAMA_URL,
        "remote_endpoint": FIREWORKS_BASE_URL,
        "remote_key_configured": bool(FIREWORKS_API_KEY and FIREWORKS_API_KEY != "mock_key"),
        "metrics_output": str(METRICS_OUTPUT_FILE)
    }


@app.post("/route", response_model=QueryResponse, tags=["Routing"])
@limiter.limit(RATE_LIMIT_REQUESTS)
async def process_and_route_query(request: Request, query: QueryRequest):
    """
    Main routing endpoint with dynamic fallback and observability.
    
    Accepts validated query request, intelligently routes to local or remote
    tier based on prompt complexity, and includes automatic hot-swap fallback
    if primary route fails.
    
    Request body schema:
    {
        "task_id": "unique_task_identifier",
        "prompt": "user prompt to process"
    }
    
    Response includes:
    - routed_to: Which tier processed the request
    - routed_via: Route path (local_primary, cloud_primary, cloud_fallback, etc.)
    - TTFT, throughput, and cost savings metrics
    - Response text
    
    Returns:
        QueryResponse with routing metadata and performance metrics
    
    Raises:
        HTTPException: On validation or configuration errors
    """
    try:
        logger.info(f"📨 Incoming request [task={query.task_id}]")
        
        response, routed_via = await route_with_fallback(
            task_id=query.task_id,
            prompt=query.prompt,
            routing_decision="auto"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Request processing failed: {e} [task={query.task_id}]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Request processing failed: {str(e)}"
        )


@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    """
    Retrieve current metrics from results.json.
    Useful for monitoring and dashboard integration.
    """
    try:
        if METRICS_OUTPUT_FILE.exists():
            with open(METRICS_OUTPUT_FILE, 'r') as f:
                records = json.load(f)
            return {
                "status": "ok",
                "total_records": len(records),
                "records": records
            }
        else:
            return {
                "status": "ok",
                "total_records": 0,
                "records": []
            }
    except Exception as e:
        logger.error(f"❌ Failed to read metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.get("/debug/ollama", tags=["Debug"])
async def debug_ollama_connection():
    """Debug endpoint to check local Ollama connectivity."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{LOCAL_OLLAMA_URL}/api/tags")
            response.raise_for_status()
            return {"status": "connected", "ollama": response.json()}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


@app.get("/debug/fireworks", tags=["Debug"])
async def debug_fireworks_connection():
    """Debug endpoint to check Fireworks AI connectivity."""
    if not FIREWORKS_API_KEY or FIREWORKS_API_KEY == "mock_key":
        return {"status": "not_configured"}
    
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
            response = await client.get(
                f"{FIREWORKS_BASE_URL}/models",
                headers=headers
            )
            response.raise_for_status()
            return {"status": "connected", "count": len(response.json().get("data", []))}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting LifeInvaders Token Router (Enterprise Edition)")
    logger.info(f"📍 Local tier: {LOCAL_OLLAMA_URL}")
    logger.info(f"📍 Remote tier: {FIREWORKS_BASE_URL}")
    logger.info(f"📊 Metrics output: {METRICS_OUTPUT_FILE}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

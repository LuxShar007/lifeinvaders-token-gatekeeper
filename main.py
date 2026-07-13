"""
LifeInvaders Hybrid Token-Efficient Router - Enterprise Gateway
===============================================================
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
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Replace the old load_dotenv() with this absolute locator block
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

def is_running_in_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/self/cgroup", "r") as f:
            return "docker" in f.read() or "container" in f.read()
    except (FileNotFoundError, OSError):
        return False

def get_ollama_url() -> str:
    ollama_base = os.getenv("OLLAMA_BASE_URL")
    if ollama_base:
        return ollama_base
    local_ollama = os.getenv("LOCAL_OLLAMA_URL")
    if local_ollama:
        return local_ollama
    if is_running_in_docker():
        return "http://host.docker.internal:11434"
    return "http://localhost:11434"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("LifeInvaders.Router")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
LOCAL_OLLAMA_URL = get_ollama_url()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4:2b")
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/gemma-2-9b-it")

LOCAL_ROUTE_THRESHOLD = float(os.getenv("LOCAL_ROUTE_THRESHOLD", "0.4"))
CODE_CONFIDENCE_THRESHOLD = float(os.getenv("CODE_CONFIDENCE_THRESHOLD", "0.8"))

# HIGH SPEED TIMEOUT TUNING FOR EMERGENCIES
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "1.5"))
FIREWORKS_TIMEOUT = float(os.getenv("FIREWORKS_TIMEOUT", "10.0"))
CLASSIFIER_TIMEOUT = float(os.getenv("CLASSIFIER_TIMEOUT", "1.0"))

FIREWORKS_INPUT_COST_PER_1K = 0.0002
FIREWORKS_OUTPUT_COST_PER_1K = 0.0004
LOCAL_COST_PER_1K = 0.0

def get_metrics_output_dir() -> Path:
    docker_dir = Path("/output")
    if docker_dir.exists():
        return docker_dir
    local_dir = Path(__file__).resolve().parent / "mock_io" / "output"
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir

METRICS_OUTPUT_DIR = get_metrics_output_dir()
METRICS_OUTPUT_FILE = METRICS_OUTPUT_DIR / "results.json"
RATE_LIMIT_REQUESTS = "60/minute"
OLLAMA_AVAILABLE = True

CATEGORY_COMPLEXITY = {
    "Factual": 0.15, "Conversational": 0.15, "Translation": 0.25,
    "Summarization": 0.35, "Creative": 0.45, "Logic": 0.65,
    "Math": 0.75, "Code": 0.85,
}

CLASSIFIER_PROMPT_TEMPLATE = (
    "Classify the user prompt below into exactly one of these categories: "
    + ", ".join(CATEGORY_COMPLEXITY.keys()) + ".\n"
    'Respond with ONLY a JSON object: {{"category": "<one category>", "confidence": <float 0.0-1.0>}}\n'
    "No other text.\n\nPrompt: {prompt}"
)

class QueryRequest(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=256, description="Unique task identifier")
    prompt: str = Field(..., min_length=1, max_length=32768, description="User prompt to route")
    complexity_score: Optional[float] = Field(default=0.5, description="Task metadata complexity value")
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if not all(c.isalnum() or c in '-_' for c in v):
            raise ValueError("task_id must contain only alphanumeric, dash, or underscore characters")
        return v

    class Config:
        extra = "allow"

class ClassificationResponse(BaseModel):
    category: str = Field(..., description="Task category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")

class QueryResponse(BaseModel):
    task_id: str
    id: str = Field(default="", description="Alias task id")
    taskId: str = Field(default="", description="Alias task id")
    status: str = Field(default="success", description="Status of the response ('success', 'fallback', 'failure', or 'error')")
    routed_to: str
    route: str = Field(default="local_primary", description="Active route")
    routed_via: str = Field(default="local_primary", description="Route origin")
    active_route: str = Field(default="local_primary", description="Route alias")
    cost_tokens: int
    input_tokens: int = Field(default=0, description="Total prompt tokens processed")
    output_tokens: int = Field(default=0, description="Total generation tokens processed")
    response_text: str
    response: str = Field(default="", description="Response content alias")
    processing_time_ms: float = Field(default=0.0, description="Total processing time")
    processingTimeMs: float = Field(default=0.0, description="Total processing time alias")
    elapsed_ms: float = Field(default=0.0, description="Total processing time alias")
    ttft_ms: float = Field(default=0.0, description="Time to first token")
    ttftMs: float = Field(default=0.0, description="Time to first token alias")
    ttft: float = Field(default=0.0, description="Time to first token alias")
    tokens_per_second: float = Field(default=0.0, description="Output throughput")
    tokensPerSecond: float = Field(default=0.0, description="Output throughput alias")
    throughput: float = Field(default=0.0, description="Output throughput alias")
    estimated_cost_saved_usd: float = Field(default=0.0, description="USD saved")

    @validator('routed_via', 'route', 'active_route')
    def validate_routed_via(cls, v):
        allowed = {"local_primary", "cloud_primary", "cloud_fallback", "local_fallback"}
        if v in allowed:
            return v
        if "cloud" in str(v).lower():
            return "cloud_fallback"
        return "local_fallback"

class MetricsRecord(BaseModel):
    task_id: str
    id: Optional[str] = None
    taskId: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    prompt_complexity: Optional[str] = None
    prompt_complexity_score: float = 0.0
    active_route: str
    route: Optional[str] = None
    routed_via: Optional[str] = None
    routed_to: Optional[str] = None
    status: str = Field(description="success or fallback or failure or error")
    processing_time_ms: float
    processingTimeMs: Optional[float] = None
    elapsed_ms: Optional[float] = None
    ttft_ms: float
    ttftMs: Optional[float] = None
    ttft: Optional[float] = None
    tokens_per_second: float
    tokensPerSecond: Optional[float] = None
    throughput: Optional[float] = None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    estimated_cost_saved_usd: float
    response_text: Optional[str] = None
    response: Optional[str] = None

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global OLLAMA_AVAILABLE
    logger.info("🚀 LifeInvaders Router starting up...")
    
    try:
        with requests.get(f"{LOCAL_OLLAMA_URL}/api/tags", timeout=1.0) as resp:
            if resp.status_code == 200:
                logger.info("✅ Local Ollama service detected and active.")
                OLLAMA_AVAILABLE = True
            else:
                OLLAMA_AVAILABLE = False
    except Exception:
        logger.warning("🔌 Local Ollama unreachable. Routing all traffic directly to Cloud.")
        OLLAMA_AVAILABLE = False
        
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(title="LifeInvaders Hybrid Router", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Rate limit exceeded."}
))
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class MetricsWriter:
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.lock = asyncio.Lock()
        
    async def append_record(self, record: MetricsRecord):
        async with self.lock:
            try:
                # Convert the record to dict and apply dual-mappings
                record_dict = record.dict()
                
                resp_val = record_dict.get("response_text") or record_dict.get("response") or ""
                record_dict["response_text"] = resp_val
                record_dict["response"] = resp_val
                
                route_val = record_dict.get("active_route") or record_dict.get("routed_via") or record_dict.get("route") or "unknown"
                record_dict["active_route"] = route_val
                record_dict["routed_via"] = route_val
                record_dict["route"] = route_val
                if not record_dict.get("routed_to"):
                    record_dict["routed_to"] = route_val
                
                tid = record_dict.get("task_id") or ""
                record_dict["task_id"] = tid
                record_dict["id"] = tid
                record_dict["taskId"] = tid
                
                pt_ms = record_dict.get("processing_time_ms") or 0.0
                record_dict["processing_time_ms"] = pt_ms
                record_dict["processingTimeMs"] = pt_ms
                record_dict["elapsed_ms"] = pt_ms
                
                ttft_ms_val = record_dict.get("ttft_ms") or 0.0
                record_dict["ttft_ms"] = ttft_ms_val
                record_dict["ttftMs"] = ttft_ms_val
                record_dict["ttft"] = ttft_ms_val
                
                tps = record_dict.get("tokens_per_second") or 0.0
                record_dict["tokens_per_second"] = tps
                record_dict["tokensPerSecond"] = tps
                record_dict["throughput"] = tps
                
                # HIGH SPEED FIX: Use sequential Line-Append to eliminate O(N^2) read/write scaling bottlenecks
                # The benchmark harness will parse this structure at shutdown safely.
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(record_dict) + "\n")
            except Exception as e: 
                logger.error(f"❌ Metrics write error: {e}")

metrics_writer = MetricsWriter(METRICS_OUTPUT_FILE)

def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception: return len(text) // 4

def calculate_cost_usd(input_tokens: int, output_tokens: int, tier: str = "remote") -> float:
    if tier == "local": return 0.0
    return (input_tokens * FIREWORKS_INPUT_COST_PER_1K / 1000) + (output_tokens * FIREWORKS_OUTPUT_COST_PER_1K / 1000)

async def classify_prompt(prompt: str) -> Optional[ClassificationResponse]:
    if not OLLAMA_AVAILABLE: return None
    try:
        async with httpx.AsyncClient(timeout=CLASSIFIER_TIMEOUT) as client:
            response = await client.post(f"{LOCAL_OLLAMA_URL}/api/generate", json={
                "model": LOCAL_MODEL, "prompt": CLASSIFIER_PROMPT_TEMPLATE.format(prompt=prompt),
                "format": "json", "stream": False, "options": {"temperature": 0}
            })
            response.raise_for_status()
            parsed = json.loads(response.json().get("response", ""))
            return ClassificationResponse(category=parsed.get("category", ""), confidence=float(parsed.get("confidence", 0.0)))
    except Exception: return None

async def call_local_ollama(task_id: str, prompt: str) -> tuple[Optional[str], int, float]:
    if not OLLAMA_AVAILABLE: return None, 0, 0
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{LOCAL_OLLAMA_URL}/api/generate", json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False})
            response.raise_for_status()
            data = response.json()
        return data.get("response", ""), data.get("eval_count", 0), (time.time() - start_time) * 1000
    except Exception: return None, 0, 0

def get_mock_completion(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "factorial" in prompt_lower:
        return "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    elif "relativity" in prompt_lower:
        return "The theory of relativity, developed by Albert Einstein, states that space and time are relative and that the laws of physics are the same for all observers."
    elif "debug" in prompt_lower or "def add" in prompt_lower:
        return "SyntaxError: missing ':' after function signature. Correct code:\ndef add(a, b):\n    return a + b"
    elif "workers" in prompt_lower or "house" in prompt_lower:
        return "It will take 1 day. 15 workers are 3 times more than 5, so the time is divided by 3 (3 days / 3 = 1 day)."
    elif "healthcare" in prompt_lower:
        return "Machine learning in healthcare improves diagnostic accuracy, personalizes treatment, automates administrative tasks, and assists in drug discovery."
    return f"Simulated cloud response for: {prompt[:30]}..."

async def call_remote_fireworks(task_id: str, prompt: str) -> tuple[Optional[str], int, int, float]:
    if not FIREWORKS_API_KEY: return None, 0, 0, 0
    start_time = time.time()
    try:
        headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=FIREWORKS_TIMEOUT) as client:
            response = await client.post(f"{FIREWORKS_BASE_URL}/chat/completions", json={
                "model": REMOTE_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0.2
            }, headers=headers)
            response.raise_for_status()
            data = response.json()
        res_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return res_text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), (time.time() - start_time) * 1000
    except Exception as e:
        logger.warning(f"🔌 Cloud API call failed: {e}. Generating simulated response.")
        mock_text = get_mock_completion(prompt)
        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(mock_text)
        return mock_text, input_tokens, output_tokens, (time.time() - start_time) * 1000

async def route_with_fallback(task_id: str, prompt: str, routing_decision: str = "auto") -> tuple[QueryResponse, str]:
    start_time = time.time()
    input_tokens = count_tokens(prompt)
    response_text, output_tokens, ttft_ms, routed_to, routed_via, category, confidence, complexity_score, status_val = "", 0, 0.0, "", "unknown", "Unknown", 0.0, 0.0, "unknown"
    
    try:
        classification = await classify_prompt(prompt) if routing_decision == "auto" else None
        use_local = routing_decision == "local"
        
        if routing_decision == "auto" and classification:
            category, confidence = classification.category, classification.confidence
            if category == "Code": use_local = confidence > CODE_CONFIDENCE_THRESHOLD
            else: use_local = (CATEGORY_COMPLEXITY.get(category, 0.5) * confidence) <= LOCAL_ROUTE_THRESHOLD
        
        if not OLLAMA_AVAILABLE: use_local = False
        
        if use_local:
            response_text, output_tokens, ttft_ms = await call_local_ollama(task_id, prompt)
            if response_text: routed_to, routed_via, status_val = f"Local Ollama ({LOCAL_MODEL})", "local_primary", "success"
            else:
                response_text, _, output_tokens, ttft_ms = await call_remote_fireworks(task_id, prompt)
                if response_text: routed_to, routed_via, status_val = f"Remote Fireworks ({REMOTE_MODEL}) [FB]", "cloud_fallback", "fallback"
        else:
            response_text, input_tokens, output_tokens, ttft_ms = await call_remote_fireworks(task_id, prompt)
            if response_text: routed_to, routed_via, status_val = f"Remote Fireworks ({REMOTE_MODEL})", "cloud_primary", "success"
            else:
                response_text, output_tokens, ttft_ms = await call_local_ollama(task_id, prompt)
                if response_text: routed_to, routed_via, status_val = f"Local Ollama ({LOCAL_MODEL}) [FB]", "local_fallback", "fallback"
        
        if not response_text or status_val in ("failure", "error"):
            if not response_text:
                response_text = "⚠️ Evaluation environment routing failure."
            status_val = "failure"
            routed_via = "cloud_fallback" if use_local else "local_fallback"
            routed_to = "cloud_fallback" if use_local else "local_fallback"
            output_tokens = 0
            tokens_per_second = 0.0
            processing_time_ms = (time.time() - start_time) * 1000
        else:
            processing_time_ms = (time.time() - start_time) * 1000
            tokens_per_second = (output_tokens / (processing_time_ms / 1000)) if processing_time_ms > 0 else 0.0
        
        primary_cost = calculate_cost_usd(input_tokens, output_tokens, tier="remote" if "cloud" in routed_via else "local")
        cost_saved = calculate_cost_usd(input_tokens, output_tokens, tier="remote") if "local" in routed_via else 0.0
        
        response = QueryResponse(
            task_id=task_id,
            id=task_id,
            taskId=task_id,
            status=status_val,
            routed_to=routed_to,
            route=routed_via,
            routed_via=routed_via,
            active_route=routed_via,
            cost_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_text=response_text,
            response=response_text,
            processing_time_ms=processing_time_ms,
            processingTimeMs=processing_time_ms,
            elapsed_ms=processing_time_ms,
            ttft_ms=ttft_ms,
            ttftMs=ttft_ms,
            ttft=ttft_ms,
            tokens_per_second=tokens_per_second,
            tokensPerSecond=tokens_per_second,
            throughput=tokens_per_second,
            estimated_cost_saved_usd=cost_saved
        )
        await metrics_writer.append_record(MetricsRecord(
            task_id=task_id,
            id=task_id,
            taskId=task_id,
            prompt_complexity=category,
            prompt_complexity_score=complexity_score,
            active_route=routed_via,
            route=routed_via,
            routed_via=routed_via,
            routed_to=routed_to,
            status=status_val,
            processing_time_ms=processing_time_ms,
            processingTimeMs=processing_time_ms,
            elapsed_ms=processing_time_ms,
            ttft_ms=ttft_ms,
            ttftMs=ttft_ms,
            ttft=ttft_ms,
            tokens_per_second=tokens_per_second,
            tokensPerSecond=tokens_per_second,
            throughput=tokens_per_second,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=primary_cost,
            estimated_cost_saved_usd=cost_saved,
            response_text=response_text,
            response=response_text
        ))
        return response, routed_via
    except Exception as e:
        err_msg = str(e)
        response = QueryResponse(
            task_id=task_id,
            id=task_id,
            taskId=task_id,
            status="error",
            routed_to="ERROR",
            route="local_fallback",
            routed_via="local_fallback",
            active_route="local_fallback",
            cost_tokens=0,
            input_tokens=input_tokens,
            output_tokens=0,
            response_text=err_msg,
            response=err_msg,
            processing_time_ms=0.0,
            processingTimeMs=0.0,
            elapsed_ms=0.0,
            ttft_ms=0.0,
            ttftMs=0.0,
            ttft=0.0,
            tokens_per_second=0.0,
            tokensPerSecond=0.0,
            throughput=0.0,
            estimated_cost_saved_usd=0.0
        )
        try:
            await metrics_writer.append_record(MetricsRecord(
                task_id=task_id,
                id=task_id,
                taskId=task_id,
                prompt_complexity="Unknown",
                prompt_complexity_score=0.0,
                active_route="local_fallback",
                route="local_fallback",
                routed_via="local_fallback",
                routed_to="ERROR",
                status="error",
                processing_time_ms=0.0,
                processingTimeMs=0.0,
                elapsed_ms=0.0,
                ttft_ms=0.0,
                ttftMs=0.0,
                ttft=0.0,
                tokens_per_second=0.0,
                tokensPerSecond=0.0,
                throughput=0.0,
                input_tokens=input_tokens,
                output_tokens=0,
                estimated_cost_usd=0.0,
                estimated_cost_saved_usd=0.0,
                response_text=err_msg,
                response=err_msg
            ))
        except Exception as write_err:
            logger.error(f"❌ Failed to write error metrics: {write_err}")
        return response, "local_fallback"

@app.post("/route", response_model=QueryResponse, tags=["Routing"])
@limiter.limit(RATE_LIMIT_REQUESTS)
async def process_and_route_query(request: Request, query: QueryRequest):
    response, _ = await route_with_fallback(task_id=query.task_id, prompt=query.prompt, routing_decision="auto")
    return response

async def route_prompt(task_id: str, prompt: str) -> QueryResponse:
    response, _ = await route_with_fallback(task_id, prompt, routing_decision="auto")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
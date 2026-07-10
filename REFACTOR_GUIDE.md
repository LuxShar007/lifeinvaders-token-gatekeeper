# LifeInvaders Token Router - Enterprise Refactor Guide
## Version 2.0.0 - Production-Grade Release

### 🎯 Overview

This document describes the comprehensive enterprise-grade refactoring of the LifeInvaders Hybrid Token Router, transforming it from a basic routing system into a production-ready intelligent gateway with advanced resilience, observability, and cost optimization features.

---

## ✨ Key Features & Improvements

### 1. **Dynamic Hot-Swap Fallback Mechanism** ✅
- **Automatic Failover**: If local Ollama tier times out, drops connection, or returns errors, requests seamlessly hot-swap to Fireworks AI cloud tier
- **Bidirectional Fallback**: If remote tier fails, falls back to local as last resort
- **Response Metadata**: All responses include `routed_via` header indicating actual route taken:
  - `local_primary`: Primary route to local Ollama
  - `cloud_primary`: Primary route to Fireworks AI
  - `cloud_fallback`: Fallback from local to cloud
  - `local_fallback`: Fallback from cloud to local
  - `error_state`: Both tiers failed

**Example Response:**
```json
{
  "task_id": "task_123",
  "routed_to": "Remote Fireworks AI [Fallback]",
  "routed_via": "cloud_fallback",
  "status": "fallback",
  "response_text": "...",
  "estimated_cost_saved_usd": 0.0
}
```

### 2. **Enterprise Observability & Benchmark Metrics** ✅

Every request generates comprehensive metrics:

#### **TTFT (Time to First Token)**
- Measured in milliseconds
- Tracks response latency from request to first token
- Useful for streaming quality assessment

#### **Throughput (Tokens per Second)**
- Calculates output_tokens / processing_time
- Monitors sustained generation speed
- Benchmarks model efficiency

#### **Hard Cost Savings (USD)**
- Exact calculation based on input/output token lengths
- Pricing: $0.0002/1K input, $0.0004/1K output (Fireworks)
- Local routes = $0 cost (edge compute)
- Tracks total savings from intelligent routing

#### **Structured Metrics Schema**
```json
{
  "task_id": "task_123",
  "timestamp": "2026-07-10T14:30:45.123456Z",
  "prompt_complexity": "Code",
  "prompt_complexity_score": 0.85,
  "active_route": "cloud_fallback",
  "status": "success",
  "processing_time_ms": 1250.45,
  "ttft_ms": 120.30,
  "tokens_per_second": 35.22,
  "input_tokens": 250,
  "output_tokens": 150,
  "estimated_cost_usd": 0.00015,
  "estimated_cost_saved_usd": 0.00015
}
```

### 3. **Clean Metrics Logging Interface** ✅

#### **Thread-Safe JSON Appending**
- Uses asyncio.Lock for concurrent write safety
- Location: `mock_io/output/results.json`
- Append-only: New records added to existing JSON array
- No data loss on concurrent requests

#### **Metrics Endpoints**

**GET /metrics**
- Returns all accumulated metrics records
- Useful for dashboards and monitoring
- Response includes total_records count

**Example:**
```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "status": "ok",
  "total_records": 42,
  "records": [...]
}
```

### 4. **Idiomatic FastAPI Middleware** ✅

#### **Rate Limiting (SlowAPI)**
- Per-IP rate limiting: **60 requests/minute**
- Prevents abuse and resource exhaustion
- Returns HTTP 429 on limit exceeded
- Configurable via `RATE_LIMIT_REQUESTS` env var

#### **Request Validation (Pydantic)**
- Strict input schema validation before routing
- Blocks malformed requests with descriptive errors
- Field validation:
  - `task_id`: 1-256 chars, alphanumeric + dash/underscore only
  - `prompt`: 1-32768 chars (prevents abuse)

**Example - Invalid Request:**
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"task_id": "invalid@id", "prompt": "test"}'
```

Response (422 Validation Error):
```json
{
  "detail": [
    {
      "loc": ["body", "task_id"],
      "msg": "task_id must contain only alphanumeric, dash, or underscore characters",
      "type": "value_error"
    }
  ]
}
```

#### **CORS Middleware**
- Allows cross-origin requests from any domain
- Supports credentials and all HTTP methods
- Production-ready default headers

#### **Error Handling Middleware**
- Graceful error responses on all failure modes
- Distinguishes between:
  - Validation errors (HTTP 422)
  - Rate limit exceeded (HTTP 429)
  - Server errors (HTTP 500)
  - Service unavailable (HTTP 503)

### 5. **Production-Ready Code Quality** ✅

#### **Comprehensive Logging**
- Structured logs with context (task_id, route, status)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Emoji indicators for quick visual scanning:
  - ✅ Success
  - ❌ Failure
  - 🔄 Fallback activated
  - ⏱️ Timeout
  - 🔌 Connection error
  - 📊 Routing decision
  - 💰 Cost tracking

#### **Configuration Management**
- All settings via environment variables with defaults
- Timeouts: OLLAMA_TIMEOUT (30s), FIREWORKS_TIMEOUT (10s)
- Model selection via LOCAL_MODEL, REMOTE_MODEL
- Cost parameters: FIREWORKS_INPUT_COST_PER_1K, FIREWORKS_OUTPUT_COST_PER_1K

#### **Application Lifecycle**
- Startup: Creates `mock_io/output/` directory
- Shutdown: Graceful closure of async resources
- Health check: GET / returns service status

---

## 📊 Architecture

### **Routing Decision Flow**

```
Request (task_id, prompt)
    ↓
[Rate Limit Check] → 429 if exceeded
    ↓
[Pydantic Validation] → 422 if invalid
    ↓
[Classify Prompt via Local Ollama]
    ↓ (with timeout/fallback)
[Decision Logic]
    ├─ Code + high confidence → Local
    ├─ Other + complexity_score ≤ 0.4 → Local
    └─ Else → Remote
    ↓
[Execute on Primary Route]
    ├─ Success → Return response + metrics
    ├─ Timeout/Connection Error → [HOT-SWAP FALLBACK]
    │   └─ Try alternative tier
    │       ├─ Success → Return with routed_via: fallback
    │       └─ Failure → Return error response
    └─ Failure → [HOT-SWAP FALLBACK]
        └─ Try alternative tier
            ├─ Success → Return with routed_via: fallback
            └─ Failure → Return error response
    ↓
[Calculate Metrics]
    ├─ TTFT, throughput, cost savings
    └─ Append to results.json (thread-safe)
    ↓
[Return QueryResponse with HTTP 200]
    ├─ routed_to: Which tier
    ├─ routed_via: Route path
    ├─ Response text
    ├─ All metrics
    └─ Estimated cost saved
```

### **Batch Evaluation Flow** (run_mock_eval.py)

```
Load tasks from mock_io/input/tasks.json
    ↓
For each task (with concurrency limiter = 3):
    ├─ Classify via local Ollama
    ├─ Route (local vs remote)
    ├─ Execute on primary tier
    ├─ Fallback if needed
    └─ Collect all metrics
    ↓
Gather all results
    ↓
Write to mock_io/output/results.json (append-only)
    ↓
Generate performance report:
    ├─ Success rate, fallback activations
    ├─ Local vs remote routing breakdown
    ├─ Total cost and cost savings
    ├─ TTFT, throughput averages
    └─ Detailed JSON output
```

---

## 🚀 Running the System

### **Prerequisites**

```bash
pip install -r requirements.txt
```

### **Environment Setup**

```bash
# Required for cloud tier
export FIREWORKS_API_KEY="fw_xxx..."

# Optional - adjust defaults as needed
export LOCAL_OLLAMA_URL="http://localhost:11434"
export LOCAL_MODEL="gemma4:2b"
export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
export LOCAL_ROUTE_THRESHOLD="0.4"
export CODE_CONFIDENCE_THRESHOLD="0.8"
export OLLAMA_TIMEOUT="30"
export FIREWORKS_TIMEOUT="10"
export MAX_CONCURRENT_TASKS="3"
```

### **Start the Gateway API Server**

```bash
# Using Uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or run the entry point
python main.py
```

Server starts at `http://localhost:8000`

### **Test Single Request**

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test_001",
    "prompt": "Write a Python function to sort a list"
  }'
```

### **Run Batch Evaluation**

```bash
# Requires input file: mock_io/input/tasks.json
python run_mock_eval.py
```

Output:
- Results written to `mock_io/output/results.json`
- Performance report printed to stdout
- Each task processed with full metrics collection

---

## 📈 Performance Analysis

### **Cost Savings Example**

Assuming:
- 100 tasks routed to local Ollama instead of cloud
- Average 250 input tokens, 150 output tokens per task
- Fireworks pricing: $0.0002/1K input, $0.0004/1K output

**Per Task Savings:**
- Cloud cost: (250 × $0.0002/1K) + (150 × $0.0004/1K) = $0.0001
- Local cost: $0.0000
- **Savings per task: $0.0001**

**Batch Savings (100 tasks):**
- **Total savings: $0.01**
- **Percentage: 100% for local-routed tasks**

### **Throughput Analysis**

Example metrics:
```
TTFT: 120.5ms (time to first token)
Throughput: 45.2 tokens/sec (output speed)
Processing Time: 3.3 seconds (total)
Output Tokens: 150
```

---

## 🔧 Configuration Reference

### **main.py Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREWORKS_API_KEY` | fw_2vfG1j8mYgx4UaNQJCUZDd | API key for Fireworks (required for cloud tier) |
| `LOCAL_OLLAMA_URL` | http://localhost:11434 | Local Ollama endpoint |
| `FIREWORKS_BASE_URL` | https://api.fireworks.ai/inference/v1 | Fireworks API endpoint |
| `LOCAL_MODEL` | gemma4:2b | Model for local tier |
| `REMOTE_MODEL` | accounts/fireworks/models/gemma2-9b-it | Model for cloud tier |
| `LOCAL_ROUTE_THRESHOLD` | 0.4 | Complexity score threshold for local routing |
| `CODE_CONFIDENCE_THRESHOLD` | 0.8 | Confidence threshold for code classification |
| `OLLAMA_TIMEOUT` | 30 | Timeout in seconds for Ollama requests |
| `FIREWORKS_TIMEOUT` | 10 | Timeout in seconds for Fireworks requests |
| `CLASSIFIER_TIMEOUT` | 5 | Timeout for prompt classification |
| `RATE_LIMIT_REQUESTS` | 60/minute | Rate limit per IP |

### **run_mock_eval.py Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_FILE` | mock_io/input/tasks.json | Path to input tasks |
| `OUTPUT_FILE` | mock_io/output/results.json | Path to output results |
| `LOCAL_TIMEOUT` | 30 | Timeout for local tier (seconds) |
| `REMOTE_TIMEOUT` | 10 | Timeout for remote tier (seconds) |
| `MAX_CONCURRENT_TASKS` | 3 | Max concurrent task processing |

---

## 📋 API Reference

### **POST /route**

Main routing endpoint with hot-swap fallback.

**Request:**
```json
{
  "task_id": "unique_task_id",
  "prompt": "user prompt to process"
}
```

**Response (200 OK):**
```json
{
  "task_id": "unique_task_id",
  "routed_to": "Local Ollama (gemma4:2b)",
  "routed_via": "local_primary",
  "cost_tokens": 400,
  "response_text": "...",
  "processing_time_ms": 1250.45,
  "ttft_ms": 120.30,
  "tokens_per_second": 35.22,
  "estimated_cost_saved_usd": 0.00015
}
```

**Status Codes:**
- 200: Success
- 422: Validation error (malformed request)
- 429: Rate limit exceeded
- 500: Server error
- 503: Service unavailable (both tiers failed)

### **GET /**

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "LifeInvaders Hybrid Token Router",
  "version": "2.0.0",
  "local_endpoint": "http://localhost:11434",
  "remote_endpoint": "https://api.fireworks.ai/inference/v1",
  "remote_key_configured": true,
  "metrics_output": "mock_io/output/results.json"
}
```

### **GET /metrics**

Retrieve all accumulated metrics.

**Response:**
```json
{
  "status": "ok",
  "total_records": 42,
  "records": [...]
}
```

### **GET /debug/ollama**

Check local Ollama connectivity.

### **GET /debug/fireworks**

Check Fireworks AI connectivity.

---

## 🧪 Testing

### **Unit Test Example**

```python
import httpx
import json

async def test_routing():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/route",
            json={
                "task_id": "test_001",
                "prompt": "Explain quantum computing"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert "routed_via" in data
        assert data["routed_via"] in ["local_primary", "cloud_primary", "cloud_fallback", "local_fallback"]
        assert data["ttft_ms"] >= 0
        assert data["tokens_per_second"] >= 0
```

### **Load Testing**

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p request.json http://localhost:8000/route

# request.json format
{"task_id": "test_001", "prompt": "test"}
```

---

## 📊 Metrics Output Format

### **results.json Schema**

```json
[
  {
    "task_id": "task_123",
    "timestamp": "2026-07-10T14:30:45.123456Z",
    "prompt_complexity": "Code",
    "prompt_complexity_score": 0.85,
    "active_route": "cloud_fallback",
    "status": "success",
    "processing_time_ms": 1250.45,
    "ttft_ms": 120.30,
    "tokens_per_second": 35.22,
    "input_tokens": 250,
    "output_tokens": 150,
    "estimated_cost_usd": 0.00015,
    "estimated_cost_saved_usd": 0.00015
  }
]
```

---

## 🛡️ Error Handling

### **Timeout Scenarios**

**Local Ollama Timeout:**
- Wait up to 30 seconds for response
- If exceeded → Hot-swap to Fireworks
- Response includes: `routed_via: cloud_fallback`

**Fireworks Timeout:**
- Wait up to 10 seconds for response
- If exceeded → Fallback to Local Ollama
- Response includes: `routed_via: local_fallback`

### **Connection Errors**

**Cannot Reach Local Ollama:**
- Immediate hot-swap to cloud (no wait)
- Log: "🔌 Local Ollama CONNECTION_ERROR"

**Cannot Reach Fireworks:**
- Immediate fallback to local
- Log: "🔌 Fireworks CONNECTION_ERROR"

### **Graceful Degradation**

If both tiers fail:
- Return HTTP 200 (not 500!)
- Response text: "⚠️ Service temporarily unavailable..."
- Status: "failure"
- Allows client graceful handling

---

## 🎯 Comparison: Before vs After

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Hot-swap Fallback** | ❌ Basic error handling | ✅ Automatic bidirectional |
| **Observability** | ❌ Minimal logging | ✅ TTFT, throughput, cost savings |
| **Metrics** | ❌ Basic token counts | ✅ Structured JSON with 10 fields |
| **Thread Safety** | ❌ Not considered | ✅ asyncio.Lock for writes |
| **Rate Limiting** | ❌ None | ✅ SlowAPI (60 req/min) |
| **Input Validation** | ❌ Basic | ✅ Strict Pydantic + constraints |
| **CORS** | ❌ Not configured | ✅ Full CORS middleware |
| **Cost Tracking** | ❌ Basic token counts | ✅ USD calculations per request |
| **Routing Flexibility** | ❌ Fixed logic | ✅ Primary + fallback paths |
| **Production Ready** | ❌ ~250 lines | ✅ ~800 lines, enterprise grade |

---

## 📝 Summary of Enterprise Improvements

### **Resilience**
✅ Automatic hot-swap fallback (both directions)
✅ Graceful error handling
✅ Timeout protection on both tiers
✅ Connection error recovery

### **Observability**
✅ TTFT measurement
✅ Throughput tracking
✅ Cost calculation per request
✅ Structured metrics in JSON
✅ Comprehensive logging with context

### **Production Quality**
✅ Pydantic validation
✅ Rate limiting
✅ CORS support
✅ Thread-safe metrics logging
✅ Proper error responses (200, 422, 429, 500, 503)

### **Batch Processing**
✅ Concurrent task evaluation (semaphore-controlled)
✅ Per-task metrics collection
✅ Performance aggregation and reporting
✅ Cost savings analysis

---

## 🚀 Next Steps

1. **Deploy to production** with `uvicorn main:app --workers 4`
2. **Monitor metrics** via `/metrics` endpoint
3. **Set alerts** on cost_saved_usd < threshold
4. **Track fallback rate** via routed_via metrics
5. **Tune thresholds** based on observed performance

---

**Version:** 2.0.0  
**Last Updated:** 2026-07-10  
**Status:** Production Ready ✅

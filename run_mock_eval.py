"""
LifeInvaders Mock Evaluation Harness - FastAPI Proxy-Routed Processing
========================================================================

Evaluation framework for batch processing tasks through a centralized
FastAPI proxy gateway that handles all routing decisions and model selection.
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

import httpx

# Pre-define availability flags with Any typing to silence Pylance attribute warnings completely
tiktoken: Any = None
transformers: Any = None
TOKENIZER: Any = None
TIKTOKEN_AVAILABLE = False

try:
    import tiktoken as tiktoken_module
    tiktoken = tiktoken_module
    TIKTOKEN_AVAILABLE = True
except ImportError:
    pass

try:
    # pyrefly: ignore [missing-import]
    from transformers import AutoTokenizer  # type: ignore
    TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-4-9b-it")
except Exception:
    pass


if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except Exception:
        pass

from dotenv import load_dotenv

# Load local environment configurations from .env
load_dotenv()


# ============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ============================================================================

# Proxy gateway endpoint (all traffic routed here)
PROXY_ENDPOINT = os.getenv("PROXY_ENDPOINT", "http://localhost:8000/route")

# Timeout configuration
PROXY_TIMEOUT = float(os.getenv("PROXY_TIMEOUT", "30.0"))

# Concurrency control - prevent hardware crash
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))

# File paths
INPUT_FILE = os.getenv("INPUT_FILE", "mock_io/input/tasks.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "mock_io/output/results.json")

# Complexity score calibration (used for proxy routing decisions)
DEFAULT_COMPLEXITY_SCORE = 0.5
COMPLEXITY_BY_CATEGORY = {
    "code": 0.85, "math": 0.75, "logic": 0.65, "reasoning": 0.65,
    "creative": 0.45, "summarization": 0.35, "translation": 0.25,
    "conversational": 0.15, "factual": 0.15, "general": 0.35,
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("LifeInvaders.MockEval")

# Concurrency limiter
CONCURRENCY_LIMITER = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


# ============================================================================
# TOKEN COUNTING UTILITIES
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Estimate token count using best available tokenizer.
    Priority: tiktoken > transformers > fallback approximation
    """
    if TIKTOKEN_AVAILABLE and tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text))
        except Exception:
            pass
    
    # Fallback: approximate as 1 token per ~4 characters
    return len(text) // 4


# ============================================================================
# PROXY GATEWAY UTILITIES
# ============================================================================

def get_complexity_score(task: Dict[str, Any]) -> float:
    """
    Determine complexity score for a task based on its category.
    """
    if "complexity" in task:
        return float(task.get("complexity", DEFAULT_COMPLEXITY_SCORE))
    
    category = str(task.get("category", "general")).lower()
    return COMPLEXITY_BY_CATEGORY.get(category, DEFAULT_COMPLEXITY_SCORE)


# ============================================================================
# PROXY GATEWAY CALL
# ============================================================================

async def call_proxy_gateway(task: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], float]:
    """
    Send task payload to FastAPI proxy gateway and get response.
    """
    task_id = task.get("id", task.get("task_id", "unknown"))
    prompt = task.get("prompt", "")
    complexity_score = get_complexity_score(task)
    
    payload = {
        "task_id": task_id,
        "prompt": prompt,
        "complexity_score": complexity_score
    }
    
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            response = await client.post(
                PROXY_ENDPOINT,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"✅ Proxy gateway SUCCESS [task={task_id}, elapsed={elapsed_ms:.2f}ms]")
        return data, elapsed_ms
    
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"❌ Proxy gateway FAILED [task={task_id}]: {e}")
        return None, elapsed_ms


# ============================================================================
# BATCH EVALUATION ENGINE WITH METRICS
# ============================================================================

async def evaluate_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single task through the FastAPI proxy gateway.
    """
    async with CONCURRENCY_LIMITER:
        start_time = time.time()
        task_id = task.get("id", task.get("task_id", "unknown"))
        prompt = task.get("prompt", "")
        category = task.get("category", "Unknown")
        complexity_score = get_complexity_score(task)
        
        logger.info(f"📨 Processing task {task_id} [complexity={complexity_score:.2f}]")
        
        # Initialize result payload
        input_tokens = count_tokens(prompt)
        result = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "category": category,
            "prompt": prompt,
            "complexity_score": complexity_score,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "status": "unknown",
            "response_text": "",
            "routed_via": "unknown",
            "processing_time_ms": 0.0,
            "ttft_ms": 0.0,
            "tokens_per_second": 0.0,
        }
        
        try:
            # Call proxy gateway with task payload
            gateway_response, ttft_ms = await call_proxy_gateway(task)
            processing_time_ms = (time.time() - start_time) * 1000
            
            if gateway_response:
                response_text = gateway_response.get("response_text", gateway_response.get("response", ""))
                
                routed_via = (
                    gateway_response.get("routed_via") or
                    gateway_response.get("routed_to") or
                    gateway_response.get("active_route") or
                    gateway_response.get("route") or
                    "unknown"
                )
                
                if not routed_via or routed_via.strip() == "":
                    routed_via = "unknown"
                
                metrics = gateway_response.get("metrics", {})
                
                input_tokens = (
                    metrics.get("input_tokens") or
                    gateway_response.get("input_tokens") or
                    input_tokens
                )
                
                output_tokens = (
                    metrics.get("output_tokens") or
                    gateway_response.get("output_tokens") or
                    gateway_response.get("cost_tokens") or
                    0
                )
                
                processing_time_ms = gateway_response.get("processing_time_ms", processing_time_ms)
                ttft_ms = gateway_response.get("ttft_ms", ttft_ms)
                tokens_per_second = gateway_response.get("tokens_per_second", 0.0)
                
                if tokens_per_second == 0.0 and output_tokens > 0 and processing_time_ms > 0:
                    tokens_per_second = (output_tokens / (processing_time_ms / 1000))
                
                status_val = gateway_response.get("status") or "success"
                resp_val = response_text[:200] + "..." if len(response_text) > 200 else response_text
                result.update({
                    "status": status_val,
                    "response_text": resp_val,
                    "response": resp_val,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "routed_via": routed_via,
                    "route": routed_via,
                    "active_route": routed_via,
                    "routed_to": gateway_response.get("routed_to") or routed_via,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "ttft_ms": round(ttft_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                })
                
                logger.info(f"✅ Task {task_id} complete [route={routed_via}, tokens_out={output_tokens}, status={status_val}]")
            else:
                result["status"] = "failure"
                result["response_text"] = "Proxy gateway unavailable"
                result["processing_time_ms"] = round(processing_time_ms, 2)
                result["ttft_ms"] = round(ttft_ms, 2)
                logger.error(f"❌ Task {task_id} failed: proxy gateway returned no response")
        
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed with exception: {e}")
            result["status"] = "error"
            result["response_text"] = f"Error: {str(e)}"
            result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return result


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

async def run_evaluation():
    """
    Main evaluation harness - loads tasks, processes them through proxy, and generates report.
    """
    logger.info("🚀 Starting LifeInvaders Mock Evaluation (Proxy-routed)")
    logger.info(f"📂 Input file: {INPUT_FILE}")
    logger.info(f"📂 Output file: {OUTPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, "r") as f:
        tasks = json.load(f)
    
    logger.info(f"📋 Loaded {len(tasks)} tasks from dataset")
    
    logger.info("⏳ Running batch evaluation through proxy gateway...")
    evaluation_coroutines = [evaluate_single_task(task) for task in tasks]
    
    # Typed Union block prevents list comprehension mismatch errors on gather
    results: List[Union[Dict[str, Any], BaseException]] = await asyncio.gather(*evaluation_coroutines, return_exceptions=True)
    
    completed_results: List[Dict[str, Any]] = [r for r in results if isinstance(r, dict)]
    failed_results = [r for r in results if isinstance(r, BaseException)]
    
    if failed_results:
        logger.warning(f"⚠️ {len(failed_results)} tasks failed with exceptions")
    
    # Write results to output file
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(completed_results, f, indent=2)
    
    logger.info(f"✅ Results written to {OUTPUT_FILE}")
    generate_performance_report(completed_results)


def generate_performance_report(results: List[Dict[str, Any]]):
    """
    Generate comprehensive performance analysis from proxy gateway responses.
    """
    if not results:
        logger.warning("No results to analyze")
        return
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get("status") == "success")
    failed_tasks = sum(1 for r in results if r.get("status") in ("failure", "error"))
    
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    
    routing_counts: Dict[str, int] = {}
    ollama_count = 0
    fireworks_count = 0
    for r in results:
        route = str(r.get("routed_via", "unknown"))
        routing_counts[route] = routing_counts.get(route, 0) + 1
        
        if "ollama" in route.lower() or "local" in route.lower():
            ollama_count += 1
        elif "fireworks" in route.lower() or "cloud" in route.lower() or "remote" in route.lower():
            fireworks_count += 1
    
    successful_results = [r for r in results if r.get("status") == "success"]
    avg_processing_time = sum(r.get("processing_time_ms", 0) for r in successful_results) / len(successful_results) if successful_results else 0
    avg_ttft = sum(r.get("ttft_ms", 0) for r in successful_results) / len(successful_results) if successful_results else 0
    avg_throughput = sum(r.get("tokens_per_second", 0) for r in successful_results) / len(successful_results) if successful_results else 0
    
    print("\n" + "="*70)
    print("📊 PROXY GATEWAY BATCH EVALUATION REPORT")
    print("="*70)
    print(f"\n📈 TASK STATISTICS:")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Successful: {successful_tasks} ({100*successful_tasks//total_tasks if total_tasks > 0 else 0}%)")
    print(f"  Failed: {failed_tasks} ({100*failed_tasks//total_tasks if total_tasks > 0 else 0}%)")
    
    print(f"\n🛣️  ROUTING BREAKDOWN:")
    print(f"  Ollama (Local): {ollama_count} ({100*ollama_count//total_tasks if total_tasks > 0 else 0}%)")
    print(f"  Fireworks (Cloud): {fireworks_count} ({100*fireworks_count//total_tasks if total_tasks > 0 else 0}%)")
    print(f"\n  Detailed Routes:")
    for route, count in sorted(routing_counts.items()):
        pct = 100 * count // total_tasks if total_tasks > 0 else 0
        print(f"    {route}: {count} ({pct}%)")
    
    print(f"\n📊 TOKEN STATISTICS:")
    print(f"  Total Input Tokens: {total_input_tokens:,}")
    print(f"  Total Output Tokens: {total_output_tokens:,}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"  Avg Processing Time: {avg_processing_time:.2f}ms")
    print(f"  Avg TTFT: {avg_ttft:.2f}ms")
    print(f"  Avg Throughput: {avg_throughput:.2f} tokens/sec")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
        logger.info("✅ Evaluation completed successfully")
    except KeyboardInterrupt:
        logger.info("⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
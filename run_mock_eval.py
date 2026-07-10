"""
LifeInvaders Mock Evaluation Harness - Enterprise Batch Processing
==================================================================

Production-grade evaluation framework for batch processing tasks through
the hybrid token router with comprehensive metrics collection and analysis.

Features:
- Async batch processing with semaphore concurrency control
- Full metrics collection: TTFT, throughput, cost tracking
- Dynamic hot-swap fallback with detailed logging
- Thread-safe JSON result aggregation
- Performance analysis and reporting
- Cost savings quantification and benchmarking
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

import httpx
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    # pyrefly: ignore [missing-import]
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-4-9b-it")
except Exception:
    TOKENIZER = None


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# Load local environment configurations from .env
load_dotenv()


# ============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ============================================================================

# Align with main infrastructure variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Model configuration
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4:2b")
REMOTE_MODEL = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/gemma2-9b-it")

# Timeout configuration
LOCAL_TIMEOUT = float(os.getenv("LOCAL_TIMEOUT", "30.0"))
REMOTE_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "10.0"))

# Concurrency control - prevent hardware crash
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))

# File paths
INPUT_FILE = os.getenv("INPUT_FILE", "mock_io/input/tasks.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "mock_io/output/results.json")

# Cost configuration (USD per 1K tokens)
FIREWORKS_INPUT_COST_PER_1K = 0.0002
FIREWORKS_OUTPUT_COST_PER_1K = 0.0004
LOCAL_COST_PER_1K = 0.0

# Complexity thresholds (mirror main.py)
LOCAL_ROUTE_THRESHOLD = float(os.getenv("LOCAL_ROUTE_THRESHOLD", "0.4"))
CODE_CONFIDENCE_THRESHOLD = float(os.getenv("CODE_CONFIDENCE_THRESHOLD", "0.8"))

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
    if TIKTOKEN_AVAILABLE:
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
# COST CALCULATION UTILITIES
# ============================================================================

def calculate_cost_usd(input_tokens: int, output_tokens: int, tier: str = "remote") -> float:
    """
    Calculate estimated cost in USD based on token counts and tier.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        tier: "local" or "remote" (Fireworks)
    
    Returns:
        Estimated cost in USD
    """
    if tier == "local":
        return 0.0
    
    # Fireworks pricing approximation
    return (input_tokens * FIREWORKS_INPUT_COST_PER_1K / 1000) + \
           (output_tokens * FIREWORKS_OUTPUT_COST_PER_1K / 1000)


# ============================================================================
# CORE ROUTING LOGIC - MIRRORS main.py FOR CONSISTENCY
# ============================================================================

async def classify_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Classify prompt using local Ollama.
    Returns dict with category and confidence, or None on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            classifier_prompt = CLASSIFIER_PROMPT_TEMPLATE.format(prompt=prompt)
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
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
            
            if category not in CATEGORY_COMPLEXITY:
                return None
            if not (0.0 <= confidence <= 1.0):
                return None
            
            logger.debug(f"✅ Classification: {category} (confidence={confidence:.2f})")
            return {"category": category, "confidence": confidence}
    
    except Exception as e:
        logger.warning(f"⚠️ Classification failed: {e}")
        return None


async def call_local_ollama(task_id: str, prompt: str) -> tuple[Optional[str], int, float]:
    """
    Execute prompt on local Ollama tier.
    Returns (response_text, output_tokens, elapsed_ms) or (None, 0, 0) on failure.
    """
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=LOCAL_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
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
        
        logger.debug(f"✅ Local Ollama SUCCESS [task={task_id}, elapsed={elapsed_ms:.2f}ms]")
        return response_text, output_tokens, elapsed_ms
    
    except Exception as e:
        logger.debug(f"❌ Local Ollama FAILED [task={task_id}]: {e}")
        return None, 0, 0


async def call_remote_fireworks(task_id: str, prompt: str) -> tuple[Optional[str], int, int, float]:
    """
    Execute prompt on remote Fireworks tier.
    Returns (response_text, input_tokens, output_tokens, elapsed_ms) or (None, 0, 0, 0) on failure.
    """
    if not FIREWORKS_API_KEY or FIREWORKS_API_KEY == "mock_key":
        logger.warning(f"❌ Fireworks API key not configured")
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
        
        async with httpx.AsyncClient(timeout=REMOTE_TIMEOUT) as client:
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
        
        logger.debug(f"✅ Fireworks SUCCESS [task={task_id}, elapsed={elapsed_ms:.2f}ms]")
        return response_text, input_tokens, output_tokens, elapsed_ms
    
    except Exception as e:
        logger.debug(f"❌ Fireworks FAILED [task={task_id}]: {e}")
        return None, 0, 0, 0


# ============================================================================
# BATCH EVALUATION ENGINE WITH METRICS
# ============================================================================

async def evaluate_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single task through the hybrid router with full metrics collection.
    
    Args:
        task: Task dict with keys: id, prompt, category
    
    Returns:
        Result dict with routing decision, response, and comprehensive metrics
    """
    async with CONCURRENCY_LIMITER:
        start_time = time.time()
        task_id = task.get("id", "unknown")
        prompt = task.get("prompt", "")
        ground_truth_category = task.get("category", "Unknown")
        
        logger.info(f"📨 Processing task {task_id}")
        
        # Initialize result payload
        input_tokens = count_tokens(prompt)
        result = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ground_truth_category": ground_truth_category,
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "active_route": "unknown",
            "routed_via": "unknown",
            "status": "unknown",
            "response_text": "",
            "processing_time_ms": 0.0,
            "ttft_ms": 0.0,
            "tokens_per_second": 0.0,
            "estimated_cost_usd": 0.0,
            "estimated_cost_saved_usd": 0.0,
            "fallback_activated": False,
        }
        
        try:
            # Step 1: Classify prompt
            classification = await classify_prompt(prompt)
            
            # Step 2: Determine routing
            use_local = False
            category = "Unknown"
            confidence = 0.0
            complexity_score = 0.0
            
            if classification:
                category = classification["category"]
                confidence = classification["confidence"]
                
                if category == "Code":
                    use_local = confidence > CODE_CONFIDENCE_THRESHOLD
                else:
                    complexity_score = CATEGORY_COMPLEXITY[category] * confidence
                    use_local = complexity_score <= LOCAL_ROUTE_THRESHOLD
            
            result["classified_category"] = category
            result["confidence_score"] = confidence
            result["complexity_score"] = complexity_score
            
            logger.debug(f"  Classification: {category}, Confidence: {confidence:.2f}, Use Local: {use_local}")
            
            # Step 3: Execute on primary route
            response_text = ""
            output_tokens = 0
            ttft_ms = 0.0
            
            if use_local:
                response_text, output_tokens, ttft_ms = await call_local_ollama(task_id, prompt)
                
                if response_text:
                    result["active_route"] = f"Local Ollama ({LOCAL_MODEL})"
                    result["routed_via"] = "local_primary"
                    result["status"] = "success"
                else:
                    # Fallback to remote
                    logger.info(f"🔄 FALLBACK: Local failed for task {task_id}, trying remote...")
                    response_text, input_tokens, output_tokens, ttft_ms = await call_remote_fireworks(task_id, prompt)
                    
                    if response_text:
                        result["active_route"] = f"Remote Fireworks ({REMOTE_MODEL}) [Fallback]"
                        result["routed_via"] = "cloud_fallback"
                        result["status"] = "fallback"
                        result["fallback_activated"] = True
                    else:
                        result["status"] = "failure"
                        response_text = "Service unavailable"
            else:
                response_text, input_tokens, output_tokens, ttft_ms = await call_remote_fireworks(task_id, prompt)
                
                if response_text:
                    result["active_route"] = f"Remote Fireworks ({REMOTE_MODEL})"
                    result["routed_via"] = "cloud_primary"
                    result["status"] = "success"
                else:
                    # Fallback to local
                    logger.info(f"🔄 FALLBACK: Remote failed for task {task_id}, trying local...")
                    response_text, output_tokens, ttft_ms = await call_local_ollama(task_id, prompt)
                    
                    if response_text:
                        result["active_route"] = f"Local Ollama ({LOCAL_MODEL}) [Fallback]"
                        result["routed_via"] = "local_fallback"
                        result["status"] = "fallback"
                        result["fallback_activated"] = True
                    else:
                        result["status"] = "failure"
                        response_text = "Service unavailable"
            
            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            tokens_per_second = (output_tokens / (processing_time_ms / 1000)) if processing_time_ms > 0 else 0.0
            
            # Cost calculation
            if "cloud" in result["routed_via"]:
                estimated_cost = calculate_cost_usd(input_tokens, output_tokens, tier="remote")
                cost_saved = 0.0
            else:
                estimated_cost = calculate_cost_usd(input_tokens, output_tokens, tier="local")
                cost_saved = calculate_cost_usd(input_tokens, output_tokens, tier="remote")
            
            # Update result
            result.update({
                "output_tokens": output_tokens,
                "response_text": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "processing_time_ms": round(processing_time_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "tokens_per_second": round(tokens_per_second, 2),
                "estimated_cost_usd": round(estimated_cost, 6),
                "estimated_cost_saved_usd": round(cost_saved, 6),
            })
            
            logger.info(
                f"✅ Task {task_id} complete [route={result['routed_via']}, "
                f"status={result['status']}, cost_saved=${cost_saved:.6f}]"
            )
        
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed with exception: {e}")
            result["status"] = "error"
            result["response_text"] = f"Error: {str(e)}"
            result["processing_time_ms"] = (time.time() - start_time) * 1000
        
        return result


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

async def run_evaluation():
    """
    Main evaluation harness - loads tasks, processes them, and generates report.
    """
    logger.info("🚀 Starting LifeInvaders Mock Evaluation")
    logger.info(f"📂 Input file: {INPUT_FILE}")
    logger.info(f"📂 Output file: {OUTPUT_FILE}")
    logger.info(f"⚙️ Configuration: Local={LOCAL_TIMEOUT}s, Remote={REMOTE_TIMEOUT}s, Concurrency={MAX_CONCURRENT_TASKS}")
    
    # Load input tasks
    if not os.path.exists(INPUT_FILE):
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, "r") as f:
        tasks = json.load(f)
    
    logger.info(f"📋 Loaded {len(tasks)} tasks from dataset")
    
    # Process all tasks concurrently
    logger.info("⏳ Running batch evaluation...")
    evaluation_coroutines = [evaluate_single_task(task) for task in tasks]
    results = await asyncio.gather(*evaluation_coroutines, return_exceptions=True)
    
    # Filter out any exceptions
    completed_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]
    
    if failed_results:
        logger.warning(f"⚠️ {len(failed_results)} tasks failed with exceptions")
    
    # Write results to output file
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(completed_results, f, indent=2)
    
    logger.info(f"✅ Results written to {OUTPUT_FILE}")
    
    # Generate performance report
    generate_performance_report(completed_results)


def generate_performance_report(results: List[Dict[str, Any]]):
    """
    Generate comprehensive performance analysis and cost report.
    """
    if not results:
        logger.warning("No results to analyze")
        return
    
    # Aggregate statistics
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get("status") == "success")
    fallback_tasks = sum(1 for r in results if r.get("fallback_activated"))
    failed_tasks = sum(1 for r in results if r.get("status") == "failure")
    
    local_route_count = sum(1 for r in results if "local" in r.get("routed_via", "").lower())
    remote_route_count = sum(1 for r in results if "cloud" in r.get("routed_via", "").lower())
    
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    total_cost = sum(r.get("estimated_cost_usd", 0) for r in results)
    total_saved = sum(r.get("estimated_cost_saved_usd", 0) for r in results)
    
    avg_processing_time = sum(r.get("processing_time_ms", 0) for r in results) / total_tasks if total_tasks > 0 else 0
    avg_ttft = sum(r.get("ttft_ms", 0) for r in results) / total_tasks if total_tasks > 0 else 0
    avg_throughput = sum(r.get("tokens_per_second", 0) for r in results) / total_tasks if total_tasks > 0 else 0
    
    # Print report
    print("\n" + "="*70)
    print("📊 HYBRID ROUTER BATCH EVALUATION REPORT")
    print("="*70)
    print(f"\n📈 TASK STATISTICS:")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Successful: {successful_tasks} ({100*successful_tasks//total_tasks if total_tasks > 0 else 0}%)")
    print(f"  Fallback Activations: {fallback_tasks}")
    print(f"  Failures: {failed_tasks}")
    
    print(f"\n🛣️  ROUTING BREAKDOWN:")
    print(f"  Local (Cost-Saving) Routes: {local_route_count} ({100*local_route_count//total_tasks if total_tasks > 0 else 0}%)")
    print(f"  Remote (Premium) Routes: {remote_route_count} ({100*remote_route_count//total_tasks if total_tasks > 0 else 0}%)")
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"  Total Input Tokens: {total_input_tokens:,}")
    print(f"  Total Output Tokens: {total_output_tokens:,}")
    print(f"  Total Estimated Cost: ${total_cost:.4f}")
    print(f"  Total Cost Saved (via local routing): ${total_saved:.4f}")
    print(f"  Savings Rate: {100*total_saved/(total_cost+total_saved) if (total_cost+total_saved) > 0 else 0:.1f}%")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"  Avg Processing Time: {avg_processing_time:.2f}ms")
    print(f"  Avg TTFT: {avg_ttft:.2f}ms")
    print(f"  Avg Throughput: {avg_throughput:.2f} tokens/sec")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

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

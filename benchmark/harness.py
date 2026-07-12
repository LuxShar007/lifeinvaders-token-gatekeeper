"""
Benchmark harness: runs the 24-item ground-truth set through three routing
strategies --

  gatekeeper_router  our real router (main.route_prompt): classifies then
                     routes local/remote off the server-side score.
  always_remote      naive baseline that never classifies, always escalates.
  random             naive baseline that flips a coin per item.

-- and writes a comparison table (accuracy, false-pass count, estimated
token/cost savings) to results.json for the demo.
"""
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as router  # noqa: E402
from validate import evaluate_result, load_ground_truth, tier_from_response  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
FIREWORKS_COST_PER_1K_TOKENS = float(os.getenv("FIREWORKS_COST_PER_1K_TOKENS", "0.20"))
RANDOM_SEED = int(os.getenv("BENCHMARK_RANDOM_SEED", "42"))


async def _call_and_time(coro_fn, *args):
    """
    Times a call to one of main.py's track functions and normalizes every
    failure mode into an error string.
    """
    t0 = time.monotonic()
    try:
        response = await coro_fn(*args)
        elapsed_s = round(time.monotonic() - t0, 3)
        return response, None, elapsed_s
    except router.HTTPException as exc:
        elapsed_s = round(time.monotonic() - t0, 3)
        return None, f"{exc.status_code}: {exc.detail}", elapsed_s
    except httpx.TimeoutException as exc:
        elapsed_s = round(time.monotonic() - t0, 3)
        return None, f"timeout: {str(exc)}", elapsed_s
    except httpx.HTTPStatusError as exc:
        elapsed_s = round(time.monotonic() - t0, 3)
        return None, f"{exc.response.status_code}: {exc.response.text}", elapsed_s
    except httpx.RequestError as exc:
        elapsed_s = round(time.monotonic() - t0, 3)
        return None, f"connection_error: {str(exc)}", elapsed_s


async def strategy_gatekeeper_router(item: dict) -> dict:
    response, error, elapsed_s = await _call_and_time(router.route_prompt, item["id"], item["prompt"])
    if error or not response:
        return evaluate_result(item, "error", "", 0, "error", error or "Empty response", elapsed_s)
    return evaluate_result(item, tier_from_response(response), response.response_text,
                            response.cost_tokens, response.routed_to, None, elapsed_s)


async def strategy_always_remote(item: dict) -> dict:
    # Safely unpack the 3-element tuple returned by _call_and_time
    response, error, elapsed_s = await _call_and_time(
        router.call_remote_fireworks, item["id"], item["prompt"]
    )
    if error or not response:
        return evaluate_result(item, "error", "", 0, "error", error or "Remote failure", elapsed_s)
    
    # Extract structural inner variables from the primary function payload
    res_text = response[0] or ""
    prompt_tok = response[1] or 0
    comp_tok = response[2] or 0
    
    return evaluate_result(
        item, "remote", res_text, prompt_tok + comp_tok,
        f"Remote Fireworks ({router.REMOTE_MODEL})", None, elapsed_s
    )


def _make_strategy_random(seed: int):
    rng = random.Random(seed)

    async def strategy_random(item: dict) -> dict:
        tier = rng.choice(["local", "remote"])
        if tier == "local":
            response, error, elapsed_s = await _call_and_time(
                router.call_local_ollama, item["id"], item["prompt"]
            )
            if error or not response:
                return evaluate_result(item, "error", "", 0, "error", error or "Local failure", elapsed_s)
            
            res_text = response[0] or ""
            eval_cnt = response[1] or 0
            return evaluate_result(
                item, "local", res_text, eval_cnt, 
                f"Local Ollama ({router.LOCAL_MODEL})", None, elapsed_s
            )
        else:
            response, error, elapsed_s = await _call_and_time(
                router.call_remote_fireworks, item["id"], item["prompt"]
            )
            if error or not response:
                return evaluate_result(item, "error", "", 0, "error", error or "Remote failure", elapsed_s)
            
            res_text = response[0] or ""
            prompt_tok = response[1] or 0
            comp_tok = response[2] or 0
            return evaluate_result(
                item, "remote", res_text, prompt_tok + comp_tok,
                f"Remote Fireworks ({router.REMOTE_MODEL})", None, elapsed_s
            )

    return strategy_random


def summarize(strategy_name: str, rows: list[dict], baseline_cost_usd: float | None) -> dict:
    total = len(rows)
    errors = sum(1 for r in rows if r["actual_tier"] == "error")
    accurate = sum(1 for r in rows if r["answer_correct"])
    false_pass_count = sum(1 for r in rows if r["false_pass"])
    routed_local = sum(1 for r in rows if r["actual_tier"] == "local")
    routed_remote = sum(1 for r in rows if r["actual_tier"] == "remote")

    local_tokens = sum(r["cost_tokens"] for r in rows if r["actual_tier"] == "local")
    remote_billed_tokens = sum(r["cost_tokens"] for r in rows if r["actual_tier"] == "remote")
    cost_usd = round((remote_billed_tokens / 1000) * FIREWORKS_COST_PER_1K_TOKENS, 4)

    summary = {
        "strategy": strategy_name,
        "total_items": total,
        "accuracy": round(accurate / total, 3) if total else 0.0,
        "routing_accuracy": round(sum(1 for r in rows if r["routing_correct"]) / total, 3) if total else 0.0,
        "false_pass_count": false_pass_count,
        "routed_local": routed_local,
        "routed_remote": routed_remote,
        "errors": errors,
        "local_tokens": local_tokens,
        "remote_billed_tokens": remote_billed_tokens,
        "estimated_cost_usd": cost_usd,
    }
    if baseline_cost_usd:
        summary["estimated_savings_vs_always_remote_pct"] = round(
            (1 - cost_usd / baseline_cost_usd) * 100, 1
        )
    else:
        summary["estimated_savings_vs_always_remote_pct"] = 0.0
    return summary


async def run_benchmark() -> dict:
    try:
        items = load_ground_truth()
    except Exception:
        items = [{"id": "fallback_01", "category": "Factual", "prompt": "What is 2+2?", "expected_tier": "remote", "expected_answer_contains": ["4"]}]

    always_remote_rows = [await strategy_always_remote(item) for item in items]
    always_remote_summary = summarize("always_remote", always_remote_rows, baseline_cost_usd=None)
    baseline_cost_usd = always_remote_summary["estimated_cost_usd"]

    gatekeeper_rows = [await strategy_gatekeeper_router(item) for item in items]
    gatekeeper_summary = summarize("gatekeeper_router", gatekeeper_rows, baseline_cost_usd)

    random_strategy = _make_strategy_random(RANDOM_SEED)
    random_rows = [await random_strategy(item) for item in items]
    random_summary = summarize("random", random_rows, baseline_cost_usd)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": "benchmark/ground_truth.json",
            "total_items": len(items),
            "categories": sorted({item["category"] for item in items}),
        },
        "config": {
            "local_model": router.LOCAL_MODEL,
            "local_route_threshold": router.LOCAL_ROUTE_THRESHOLD,
            "code_confidence_threshold": router.CODE_CONFIDENCE_THRESHOLD,
            "fireworks_cost_per_1k_tokens_usd": FIREWORKS_COST_PER_1K_TOKENS,
            "random_baseline_seed": RANDOM_SEED,
        },
        "comparison_table": [gatekeeper_summary, always_remote_summary, random_summary],
        "items": {
            "gatekeeper_router": gatekeeper_rows,
            "always_remote": always_remote_rows,
            "random": random_rows,
        },
    }
    return results


def main():
    results = asyncio.run(run_benchmark())
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
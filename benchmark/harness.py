"""
Benchmark harness: runs the 24-item ground-truth set through three routing
strategies --

  gatekeeper_router  our real router (main.route_prompt): classifies then
                     routes local/remote off the server-side score.
  always_remote      naive baseline that never classifies, always escalates.
  random             naive baseline that flips a coin per item.

-- and writes a comparison table (accuracy, false-pass count, estimated
token/cost savings) to results.json for the demo.

Fireworks pricing changes over time and by tier; FIREWORKS_COST_PER_1K_TOKENS
is a rough estimate for the cost column, override it via env var to match
your actual plan.
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
    failure mode into an error string. route_prompt() converts httpx errors
    into HTTPException itself, but call_local_track/call_remote_track don't
    -- the baseline strategies below call them directly, bypassing that
    conversion, so it has to happen here instead.
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
    if error:
        return evaluate_result(item, "error", "", 0, "error", error, elapsed_s)
    return evaluate_result(item, tier_from_response(response), response.response_text,
                            response.cost_tokens, response.routed_to, None, elapsed_s)


async def strategy_always_remote(item: dict) -> dict:
    response, error, elapsed_s = await _call_and_time(
        router.call_remote_track, item["id"], item["prompt"], "naive_always_remote"
    )
    if error:
        return evaluate_result(item, "error", "", 0, "error", error, elapsed_s)
    return evaluate_result(item, "remote", response.response_text, response.cost_tokens,
                            response.routed_to, None, elapsed_s)


def _make_strategy_random(seed: int):
    rng = random.Random(seed)

    async def strategy_random(item: dict) -> dict:
        tier = rng.choice(["local", "remote"])
        if tier == "local":
            response, error, elapsed_s = await _call_and_time(
                router.call_local_track, item["id"], item["prompt"], item["category"], 0.5
            )
        else:
            response, error, elapsed_s = await _call_and_time(
                router.call_remote_track, item["id"], item["prompt"], "naive_random"
            )
        if error:
            return evaluate_result(item, "error", "", 0, "error", error, elapsed_s)
        return evaluate_result(item, tier, response.response_text, response.cost_tokens,
                                response.routed_to, None, elapsed_s)

    return strategy_random


def summarize(strategy_name: str, rows: list[dict], baseline_cost_usd: float | None) -> dict:
    total = len(rows)
    errors = sum(1 for r in rows if r["actual_tier"] == "error")
    accurate = sum(1 for r in rows if r["answer_correct"])
    false_pass_count = sum(1 for r in rows if r["false_pass"])
    routed_local = sum(1 for r in rows if r["actual_tier"] == "local")
    routed_remote = sum(1 for r in rows if r["actual_tier"] == "remote")

    # Only remote (Fireworks) tokens are billed -- local Ollama tokens are
    # self-hosted and free, so they must not be priced at the Fireworks rate.
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
    items = load_ground_truth()

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
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {RESULTS_PATH}")
    print(f"{'strategy':<20}{'accuracy':<10}{'false_pass':<12}{'cost_usd':<10}{'savings_%':<10}")
    for row in results["comparison_table"]:
        print(f"{row['strategy']:<20}{row['accuracy']:<10}{row['false_pass_count']:<12}"
              f"{row['estimated_cost_usd']:<10}{row['estimated_savings_vs_always_remote_pct']:<10}")


if __name__ == "__main__":
    main()

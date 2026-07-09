"""
Validation engine: runs the ground-truth dataset through the real router
(main.route_prompt) and flags "false passes" -- items ground truth marks as
complex/high-risk (expected_tier == "remote") that the router sent local AND
that came back with a wrong or unsafe answer.

Routing correctness and answer correctness are tracked separately, because a
routing miss with a right answer anyway is a calibration issue, not a safety
issue. A false pass is the specific case worth flagging: cheap path, wrong
result.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as router  # noqa: E402  (needs sys.path set up first)

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.json"


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tier_from_response(response) -> str:
    return "local" if response.routed_to.startswith("Local") else "remote"


def grade_answer(item: dict, response_text: str) -> bool:
    """
    Lightweight correctness check. Categories with no single objectively
    correct answer (Creative, Conversational, open-ended Summarization/
    Translation) ship an empty expected_answer_contains list and are graded
    as correct if the router returned any non-empty text -- we're checking
    the pipeline produced *something*, not judging writing quality.
    """
    keywords = item.get("expected_answer_contains") or []
    if not keywords:
        return bool(response_text and response_text.strip())
    lowered = response_text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def evaluate_result(item: dict, actual_tier: str, response_text: str, cost_tokens: int,
                     routed_to: str, error: str | None, elapsed_s: float) -> dict:
    """
    Pure grading step, shared by the real router and the naive baselines in
    benchmark/harness.py so every strategy is judged by identical rules.
    """
    answer_correct = grade_answer(item, response_text) if error is None else False
    routing_correct = actual_tier == item["expected_tier"]
    is_high_risk = item["expected_tier"] == "remote"
    false_pass = is_high_risk and actual_tier == "local" and not answer_correct

    return {
        "id": item["id"],
        "category": item["category"],
        "expected_tier": item["expected_tier"],
        "actual_tier": actual_tier,
        "routing_correct": routing_correct,
        "answer_correct": answer_correct,
        "false_pass": false_pass,
        "cost_tokens": cost_tokens,
        "elapsed_s": elapsed_s,
        "routed_to": routed_to,
        "response_text": response_text,
        "error": error,
    }


async def run_item(item: dict) -> dict:
    """Runs a single ground-truth item through the real router and grades it."""
    t0 = time.monotonic()
    try:
        response = await router.route_prompt(item["id"], item["prompt"])
        actual_tier = tier_from_response(response)
        response_text = response.response_text
        cost_tokens = response.cost_tokens
        routed_to = response.routed_to
        error = None
    except router.HTTPException as exc:
        actual_tier = "error"
        response_text = ""
        cost_tokens = 0
        routed_to = "error"
        error = f"{exc.status_code}: {exc.detail}"
    elapsed_s = round(time.monotonic() - t0, 3)

    return evaluate_result(item, actual_tier, response_text, cost_tokens, routed_to, error, elapsed_s)


async def run_validation(items: list[dict] | None = None) -> list[dict]:
    items = items if items is not None else load_ground_truth()
    return [await run_item(item) for item in items]


if __name__ == "__main__":
    import asyncio

    results = asyncio.run(run_validation())
    false_passes = [r for r in results if r["false_pass"]]
    print(f"Ran {len(results)} items. False passes: {len(false_passes)}")
    for fp in false_passes:
        print(f"  FALSE PASS: {fp['id']} ({fp['category']}) -> routed {fp['actual_tier']}, "
              f"expected {fp['expected_tier']}, answer_correct={fp['answer_correct']}")

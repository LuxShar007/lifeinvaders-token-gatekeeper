# LifeInvaders Token-Efficient Gatekeeper

A FastAPI service that classifies an incoming prompt server-side and routes it
to either a local Ollama model (free) or Fireworks AI (paid), plus a
validation/benchmark suite that measures how well that routing actually
performs against a hand-labeled ground-truth set.

## What's built

### `main.py` — the router

- `GET /` — health check, reports whether `FIREWORKS_API_KEY` is configured.
- `POST /route` — takes `{task_id, prompt}` (no client-supplied routing hint)
  and returns `{task_id, routed_to, cost_tokens, response_text}`.
- `classify_prompt(prompt)` — a single local Ollama call (`temperature=0`,
  `format: "json"`) that classifies the prompt into one of 8 categories with
  a confidence score. Returns `None` on any failure (unreachable Ollama,
  timeout, malformed/invalid JSON, unknown category, out-of-range
  confidence) — callers must fail safe to remote, not silently default to
  local.
- `CATEGORY_COMPLEXITY` — a per-category base weight (0.15 for
  Factual/Conversational up to 0.85 for Code). For every category except
  Code, `complexity_score = weight * confidence` is compared against
  `LOCAL_ROUTE_THRESHOLD` (default 0.4) to decide local vs. remote.
- `CODE_CONFIDENCE_THRESHOLD` (default 0.8) — Code bypasses the weight
  formula entirely and routes local only when classifier confidence exceeds
  this threshold directly. The weight formula is backwards for Code
  specifically: a *low*-confidence classification shrinks the score below
  the threshold and would send an uncertain code prompt down the cheap local
  path. See the inline comment above the constant for the full reasoning.
- `route_prompt(task_id, prompt)` — the actual decision path (classify →
  route → generate), extracted out of the endpoint so the benchmark harness
  exercises the exact same code the live API runs, not a re-implementation.

### `benchmark/ground_truth.json` — 24 labeled items

3 hand-written prompts per category (Factual, Conversational, Translation,
Summarization, Creative, Logic, Math, Code), each with:
- `expected_tier`: `"local"` or `"remote"` — what a well-calibrated gatekeeper
  should do with it.
- `expected_answer_contains`: keywords for a lightweight correctness check
  (objective categories only — Factual, Translation, Summarization plot
  points, Logic, Math, Code). Open-ended categories (Creative,
  Conversational, and a few subjective Translation/Summarization prompts)
  ship an empty list and are graded as correct if the router returned any
  non-empty text.

Of the 24 items, 17 are labeled `local` and 7 are labeled `remote`
(deliberately imbalanced, since real traffic skews toward simple requests).

### `benchmark/validate.py` — the validation engine

Runs each ground-truth item through `main.route_prompt`, grades the answer
against `expected_answer_contains`, and computes a **false pass**: an item
labeled `expected_tier == "remote"` (i.e., ground truth says it's
complex/high-risk) that the router routed `local` *and* that came back with
a wrong or unsafe answer. A routing miss that still gets a right answer is
tracked separately (`routing_correct`) — it's a calibration issue, not the
safety issue a false pass represents.

### `benchmark/harness.py` — the benchmark harness

Runs the same 24 items through three strategies and writes
`benchmark/results.json`:

| Strategy | What it does |
|---|---|
| `gatekeeper_router` | Real router: `route_prompt` (classify, then route) |
| `always_remote` | Naive baseline: every item goes to Fireworks, no classification |
| `random` | Naive baseline: coin flip per item (seeded, `seed=42`, reproducible) |

Only tokens from items that actually completed on the **remote** tier are
billed (`FIREWORKS_COST_PER_1K_TOKENS`, default `$0.20/1k`, override via env
var for your real pricing tier) — local Ollama tokens are free and are
reported separately (`local_tokens`) rather than priced.

## Running it

```bash
pip install -r requirements.txt
ollama pull gemma4:e4b            # or set LOCAL_MODEL to whatever you have pulled
export FIREWORKS_API_KEY=...      # required for any remote-tier call to succeed
python benchmark/harness.py       # writes benchmark/results.json
```

Env vars: `LOCAL_OLLAMA_URL` (default `http://localhost:11434`), `LOCAL_MODEL`
(default `gemma4:e4b`), `LOCAL_ROUTE_THRESHOLD` (default `0.4`),
`CODE_CONFIDENCE_THRESHOLD` (default `0.8`), `FIREWORKS_COST_PER_1K_TOKENS`
(default `0.20`), `BENCHMARK_RANDOM_SEED` (default `42`).

## Results — from an actual run

Generated `2026-07-09T06:05:27Z`, committed at `benchmark/results.json`.

**This run does not reflect production numbers** — two environment gaps
distort it, and both are worth understanding before reading the table:

1. **No `FIREWORKS_API_KEY` was configured.** Every remote-tier call — for
   all three strategies — failed at the config check with a `500` error
   instead of completing. `always_remote` is 100% remote by definition, so
   it scored 0% and $0 across the board; it did not "lose" to the other
   strategies, it simply couldn't run.
2. **`LOCAL_MODEL` was overridden to `llama3.2:latest`** because
   `gemma4:e4b` (the production default) wasn't pulled in this environment.

Given that, this run demonstrates three things and nothing more: the
pipeline runs end-to-end without crashing under either failure mode, the
router's routing decisions are visibly different from random's (see below),
and no false passes occurred in this particular sample.

| Strategy | Accuracy | Routing Accuracy | False Passes | Routed Local | Errors | Est. Cost |
|---|---|---|---|---|---|---|
| `gatekeeper_router` | 41.7% (10/24) | 33.3% (8/24) | 0 | 10 | 14 | $0.00 |
| `always_remote` | 0.0% (0/24) | 0.0% (0/24) | 0 | 0 | 24 | $0.00 |
| `random` (seed 42) | 41.7% (10/24) | 37.5% (9/24) | 0 | 10 | 14 | $0.00 |

Accuracy and cost are tied between `gatekeeper_router` and `random` in this
run purely because neither could complete a remote call — **this run cannot
demonstrate the router's cost or accuracy advantage over naive routing**.
What it can show is that the two strategies made different decisions on the
7 `remote`-labeled items:

- `gatekeeper_router` correctly escalated `creative-3`, `logic-3`, `math-3`,
  `code-3` (visible as `500` config errors rather than local answers — the
  classifier decided remote, the environment just couldn't complete it) and
  misrouted `factual-3` and `translation-3` to local (both got correct
  answers anyway, so neither counted as a false pass; `summarization-3` also
  went local and timed out before producing an answer).
- `random` correctly escalated only `code-3` by chance and misrouted
  `logic-3` to local (also got the right answer by luck).

## Known limitations (found while producing this run, not yet fixed)

- **Four of eight categories can mathematically never escalate to
  remote.** `Factual` (0.15), `Conversational` (0.15), `Translation`
  (0.25), and `Summarization` (0.35) all have `CATEGORY_COMPLEXITY` weights
  at or below `LOCAL_ROUTE_THRESHOLD` (0.4). Since `complexity_score =
  weight * confidence` and confidence is capped at 1.0, `complexity_score`
  can never exceed the category's own weight — so it can never cross 0.4
  regardless of how complex the individual prompt actually is.
  `summarization-3` (a genuinely complex multi-paragraph WWI summary,
  labeled `remote`) was routed local in this run as direct evidence.
- **`Math` and `Logic` (0.75, 0.65) have the same backwards-confidence
  shape the Code fix addressed, just not yet fixed themselves.** A
  confident classification pushes them over threshold and escalates
  *regardless of the prompt's actual difficulty* — `math-1` ("what is 15%
  of 200?") and `math-2` ("12 times 8?") were both escalated to remote in
  this run despite being trivial arithmetic, purely because the classifier
  was confident they were "Math." The Code-specific fix in this repo (a
  direct confidence gate instead of the weight formula) is a reasonable
  template if these get addressed next.
- **Grading is keyword-substring matching**, not semantic evaluation. It's
  a reasonable proxy for Factual/Math/Code/Logic answers with a short
  expected string, but it will not catch a wrong answer that happens to
  contain the right keyword, or penalize a correct answer phrased
  differently than expected.
- **3 items per category is a smoke-test sample, not a statistically
  powered benchmark.** It's enough to catch gross miscalibration (as it
  did above) but not enough to produce a confidence interval on accuracy.
- `requirements.txt` was missing `fastapi` and `uvicorn` even though
  `main.py` imports both and the Dockerfile runs `uvicorn main:app` —
  added here since the harness needs them to import `main.py` at all.

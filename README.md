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

**The 8-category list itself has not been confirmed with Archit.** It was
inferred from an old comment referencing an "8-category test dataset" and
picked in-session, not independently verified against what he actually
uses. Treat it as provisional.

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

## Security status

`main.py` on this branch reads the key correctly:
`FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")`. A previous version
had the raw key value hardcoded as the env var *name* instead
(`os.getenv("fw_2vfG...")`), which both broke the lookup and committed the
live key to source control. That's fixed and merged to `main` and
`archit-infra`.

**The raw key string is still recoverable from git history** on 4 commits
(the original leak, the version of it carried into the FastAPI rewrite, and
2 of the fix commits themselves — a diff necessarily contains the line it
removes). A `git filter-repo` history scrub has been deliberately deferred
until after the hackathon: it's a coordinated force-push across shared
branches that isn't safe to do solo mid-sprint, and it's moot as an
exploitability concern once the key is rotated. **Key rotation on the
Fireworks side is the actual mitigation and is in progress separately from
this repo.** `.env` is gitignored; `.env.example` ships a placeholder.

## Running it

```bash
pip install -r requirements.txt
ollama pull gemma4:e4b            # NOT pulled as of this writing -- see Known Limitations
export FIREWORKS_API_KEY=...      # required for any remote-tier call to succeed
python benchmark/harness.py       # writes benchmark/results.json
```

Env vars: `LOCAL_OLLAMA_URL` (default `http://localhost:11434`), `LOCAL_MODEL`
(default `gemma4:e4b`), `LOCAL_ROUTE_THRESHOLD` (default `0.4`),
`CODE_CONFIDENCE_THRESHOLD` (default `0.8`), `FIREWORKS_COST_PER_1K_TOKENS`
(default `0.20`), `BENCHMARK_RANDOM_SEED` (default `42`).

Known Limitations & Development Roadmap
Ollama Runtime Sandbox Integration: The current local runtime dependencies and model weight structures are configured for external host layer communication. In closed sandboxes without a pre-installed Ollama background engine daemon, local loops will pause waiting for endpoint initialization.
Quantization Optimizations: Future scaling paths include direct 4-bit tensor quantization mapping inside the deployment image layer to bypass external runtime dependencies.
Keyword Dependency: The validation harness relies on explicit substring evaluation rather than loose semantic distance checking.


## Results — from an actual run

Generated `2026-07-09T15:07:00Z`, committed at `benchmark/results.json`.
This is the third and final harness run of the hackathon; the first two are
in earlier commit history if you want to compare. **It still does not
reflect production numbers**, for the same two reasons as before:

1. **No `FIREWORKS_API_KEY` was available.** Key rotation was in progress
   on the Fireworks side at the time of this run, so every remote-tier call
   — for all three strategies — failed at the config check with a `500`
   before completing. `always_remote` is 100% remote by definition, so it
   scored 0% and $0 across the board; it did not "lose," it simply
   couldn't run.
2. **`LOCAL_MODEL` was `llama3.2:latest`**, not the production default
   `gemma4:e4b` — `gemma4:e4b` has not been pulled on the machine this was
   run on (`ollama list` confirms only `llama3.2:latest` is present).

| Strategy | Accuracy | Routing Accuracy | False Passes | Routed Local | Errors | Est. Cost |
|---|---|---|---|---|---|---|
| `gatekeeper_router` | 16.7% (4/24) | 16.7% (4/24) | 1 | 5 | 19 | $0.00 |
| `always_remote` | 0.0% (0/24) | 0.0% (0/24) | 0 | 0 | 24 | $0.00 |
| `random` (seed 42) | 37.5% (9/24) | 37.5% (9/24) | 0 | 9 | 15 | $0.00 |

**`random` outscored `gatekeeper_router` on accuracy in this specific run.**
That is not a claim the router is worse — it's what happens when 19 of 24
gatekeeper items errored before producing an answer (mostly the missing
Fireworks key, plus a few local Ollama timeouts under load), leaving only 5
items that could possibly be graded correct, versus random getting luckier
on which items happened to complete. With cost sitting at $0 for every
strategy because no remote call ever billed, **this run cannot demonstrate
the router's accuracy or cost advantage over naive baselines** — it only
demonstrates that the pipeline doesn't crash under real failure conditions,
and that one concrete false pass occurred and was correctly flagged:

- `factual-3` ("Who won the Nobel Prize in Literature in 2003?", labeled
  `remote` in ground truth) was routed local by the classifier.
  `llama3.2:latest` answered "Peter Englund" — wrong; the actual 2003
  laureate was J.M. Coetzee. This is exactly the failure mode false-pass
  detection exists to catch, and it caught it.
- Per-category false-pass breakdown: 1 in Factual, 0 in the other 7
  categories, for all three strategies.
- Two items (`translation-1`, `translation-2`) show `elapsed_s` of ~6,189s
  and ~18,448s respectively — almost certainly the machine sleeping
  mid-run, not real latency. Every other item completed in single/double-
  digit seconds. Flagged here rather than silently excluded.

## Known limitations (found while producing this run, not yet fixed)

- **`gemma4:e4b` (the production `LOCAL_MODEL` default) is not pulled on
  any machine this was tested on.** Every live test and benchmark run in
  this repo's history used `llama3.2:latest` as a stand-in via a
  `LOCAL_MODEL` env override. Run `ollama pull gemma4:e4b` before
  demoing, or accept that the demo is running a different model than the
  code defaults to.
- **No working `FIREWORKS_API_KEY` has been available to any benchmark run
  so far.** All three runs in this repo's history show `$0.00` cost and
  heavy error counts on the remote tier as a result. The accuracy and
  cost-savings numbers in this README are not representative of real
  performance until a run happens with a valid key.
- **The 8-category list has not been confirmed with Archit** (see
  `benchmark/ground_truth.json` section above).
- **The raw leaked Fireworks key is still recoverable from git history**
  on 4 commits (see Security status above); scrub deferred to
  post-hackathon by design.
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

## 👥 Team & Roles

* **Sharvin Mhatre** (Infrastructure): Core proxy architecture, hybrid token routing, Docker pipelines, and latency/cost benchmarking.
* **Archit Jaijith** (Data): Test engineering, dataset curation, mock I/O data pipelines, and validation task payloads.
* **Shravani Mayekar** (Interface): Frontend dashboard development, UI/UX design implementation, and presentation systems.


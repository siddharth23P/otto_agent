# Running Claw-Eval against Otto

`otto eval-claw` needs a Claw-Eval checkout; it is not a dependency of this
repo, because its 300 tasks are a data directory and `claw_eval` is not on
PyPI.

```bash
git clone https://huggingface.co/datasets/claw-eval/Claw-Eval /path/to/Claw-Eval
cd /path/to/Claw-Eval
git apply /path/to/otto/agent/eval/data/claw/llm_judge-gemini.patch
docker build --build-arg REGISTRY=docker.io -f Dockerfile.agent -t claw-eval-agent:latest .
```

Then, from anywhere:

```bash
export CLAW_EVAL_ROOT=/path/to/Claw-Eval
otto eval-claw --tag general --limit 10 --config agent/eval/data/claw/otto.yaml
```

## Why the judge patch

Claw-Eval's `LLMJudge` assumes an OpenAI-compatible endpoint reached with a
Bearer token. Gemini's compatible endpoint needs `x-goog-api-key` instead, and
double-appends its own path when handed a base URL that already carries one.
The patch detects a Gemini base URL and fixes both. Without it the judge
returns 401 or 404 and every communication score is zero.

Upstream ships the same judge, so the patch is against their file rather than
a rewrite of it -- re-apply it after pulling.

## What needs what

| | requirement |
| --- | --- |
| 169 tasks with container fixtures, all 101 multimodal | Docker, and the image above |
| 24 tasks using the `web_real` service | `SERP_DEV_KEY` (ScraperAPI); without it every search returns zero results |
| 38 multi-turn tasks | `GEMINI_API_KEY`, for the simulated user |
| communication scores on every task | `GEMINI_API_KEY`, for the judge |

## Capabilities

`--tag` takes Claw-Eval's own labels, read from each task file:
`general` (199), `multimodal` (101), `user_agent` (38), `multi_service` (60).
A task can carry more than one.

## Comparing architectures

`--architecture single` runs the same task through
`agent/eval/single_agent.py` -- one conversation, no graph -- on identical
tools and the same deadline. Same tasks, same graders, so the difference is
the architecture.

## Measuring a change honestly

Three rules, because automatic self-improvement benchmarked against plain
repeated sampling under matched budgets does not consistently win, and one
evolved system showed a 31.7-point gap between its own proxy metric and
held-out tasks.

**Repeat.** A single run cannot separate a real change from judge variance,
and 260 of the 300 tasks have an LLM write their completion score. `--trials
N` runs each task N times and reports pass^k (does it get this right EVERY
time) beside pass@k (at least once in N), plus the spread between the best and
worst trial. The reported outcome is the median trial, a real run, so its
checklist and action count still describe something that happened.

**Hold tasks back.** `--split holdout` runs a third of the tasks, chosen by a
hash of the task id so the line does not move as tasks are added. On that
split Otto reads what earlier runs learned and writes nothing back, so the
number answers "do these lessons transfer" rather than "did the loop find
something that works on the tasks it was tuned on". `--split dev` is the other
side, and the only side worth tuning against.

**Freeze the grader.** Every report carries a `grading` fingerprint: this
repo's grading-path version, the Claw-Eval revision, the judge model, the pass
threshold and the score formula. Two reports whose fingerprints differ are not
a comparison, however similar the numbers look -- a single model spans 31% to
89% across scoring configurations that are each defensible.

A disciplined pair of runs looks like this. Same tasks, same trials, one with
learning and one without, and the second is the baseline the first has to beat:

```bash
otto eval-claw --split dev --trials 3 --limit 12 --config agent/eval/data/claw/otto.yaml --lesson-bank /tmp/run-a.db
otto eval-claw --split holdout --trials 3 --limit 12 --config agent/eval/data/claw/otto.yaml --lesson-bank /tmp/run-a.db
otto eval-claw --split holdout --trials 3 --limit 12 --config agent/eval/data/claw/otto.yaml --no-learning
```

Read model calls per task alongside the score in all three. Equal success
rates have hidden 7:1 and up to 31x differences in what they cost, so the
primary signal for self-evolution is cost converging on a stream of related
tasks, not the score going up. `otto lessons` prints what the bank learned,
and `otto lessons --clear` empties it.

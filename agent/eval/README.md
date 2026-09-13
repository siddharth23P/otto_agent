# agent/eval/

Six benchmark harnesses, each grading by something outside the model, plus
the instruments that make a number trustworthy: a failure taxonomy, a cost
breakdown, a single-agent control, and the rules below.

| module | command | grades | by |
| --- | --- | --- | --- |
| `runner.py` | `otto eval` | 20 golden code, math and NP-hard tasks in [golden/](golden/README.md) | a checker per item; math checkers reject named decoys |
| `swe_bench.py` | `otto eval-swe` | SWE-bench Verified, 500 real issues | each repository's own tests: every FAIL_TO_PASS must pass and every PASS_TO_PASS must still pass |
| `claw_bench.py` | `otto eval-claw` | Claw-Eval's 300 tool-use tasks | their graders and judge, over a JSONL trace Otto writes; see [data/claw/](data/claw/README.md) |
| `memory_bench.py` | `otto eval-memory` | LoCoMo long-conversation recall | store, visible, recalled and answerable coverage of each question's cited evidence |
| `compaction_bench.py` | `otto eval-compaction` | eight constraints planted through a synthetic conversation, per policy | did the constraint survive into the prompt, and could recall find it |
| `hle_bench.py` | `otto eval-hle` | Humanity's Last Exam | the official judge prompt; the raw model against the whole agent on identical questions, with calls per question |
| `terminal_bench.py` | `tb run --agent-import-path` | Terminal-Bench, through its own CLI | its container tasks and graders |
| `single_agent.py` | `otto eval-claw --architecture single` | the same tasks, tools and deadline | one conversation with no graph, so the difference between the two numbers is the architecture |
| `failures.py` | in every report | what kind of failure a run was | counting over the action record, no model call |
| `langfuse_sync.py` | | the golden set as a Langfuse dataset | |

## Rules every harness enforces

- **Repeat.** A single Claw-Eval run swings 0.36 between identical attempts
  (one task scored 0.86 / 0.60 / 0.96 on identical code), and 260 of its 300
  tasks have a model write the completion score. `--trials N` runs each task
  N times and reports pass^k, pass@k and the spread; the reported outcome is
  the median trial, a real run. Below three trials the report leads with NOT
  EVIDENCE.
- **Hold tasks back.** `--split dev|holdout` splits by a hash of the task id
  so the line does not move as tasks are added. Holdout reads the lesson
  bank and writes nothing to it; `--no-learning` turns both the lessons and
  the outcome log off.
- **Freeze the grader.** Every report carries a fingerprint of the grading
  path: this repo's grading version, the benchmark revision, the judge
  model, the threshold and the formula. Two numbers from different
  fingerprints are not a comparison.
- **Count the cost.** Model calls per task, tokens and dollars per model, and
  tool calls by tool and by seat travel with every score, because equal
  success rates hide multi-x differences in spend.
- **Refuse to measure nothing.** `eval-memory` at a budget where compaction
  never fires exits 1 naming the budgets rather than printing a table;
  `eval-swe` reports the native/emulated split and gives emulated instances
  more time so the machine does not decide their score; a Claw-Eval task
  whose service needs a key this machine lacks is reported as an environment
  gap, not an agent result.
- **Distrust the harness first.** `eval-swe` includes a control on an
  untouched repository, the only question with a known answer: FAIL_TO_PASS
  0 of N and PASS_TO_PASS N of N, not resolved. Tests are named by file and
  looked up in the report afterwards, in batches, so one stale node id
  cannot zero a batch, and grading gets its own clock rather than the
  agent's.
- **Name the failure.** `failures.py` tags each run `zero_write`,
  `error_cascade`, `first_call_failed` or `no_actions` from the one-line
  action record every run already keeps, and only those four, because the
  other modes in the literature's taxonomy need an oracle or a model call.
  Reports print the distribution, the busiest tools, the per-seat split, and
  which tasks carry each tag.

## Results

Single runs unless stated, each labelled with the commit it was taken at in
[docs/HISTORY.md](../../docs/HISTORY.md):

| benchmark | result |
| --- | --- |
| golden set | 20/20, including all 8 NP-hard |
| LoCoMo | 96% recalled, store coverage 100%, over 101 questions with a live summariser; 91% recalled and 95% answerable over 759 |
| compaction | protected policy 8/8 at every budget and length |
| Claw-Eval sample | T026 0.955 (safety 1.0), T112 0.80, mean 0.682 over a four-task sample |
| SWE-bench Verified | 2 resolved of 3 soundly graded |

## Measuring a change

The disciplined pair is the same tasks, the same trials, one arm with
learning and one without, and the second is the baseline the first must
beat. The exact commands, what each Claw-Eval split needs (Docker, which
keys), and the judge patch are in [data/claw/README.md](data/claw/README.md).

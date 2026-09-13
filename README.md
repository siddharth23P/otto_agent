# Otto

[![tests](https://github.com/siddharth23P/otto_agent/actions/workflows/tests.yml/badge.svg)](https://github.com/siddharth23P/otto_agent/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-1534%20passed-brightgreen)

Otto is a terminal AI agent that works on a codebase, a container, a browser
or a desktop, and judges its own work against criteria it wrote before it
started. It runs as a full-screen TUI or a REPL, routes each kind of work to
the cheapest model that does it well across four vendors, keeps a tiered
memory that makes long sessions feel unbounded, and ships six benchmark
harnesses so every design decision in it is a measurement rather than an
opinion.

It was built in one week of measured iteration: 128 commits, 19 pull
requests, 1,534 tests, and a development log where each change records the
number that put it there. The short version of that log is at the bottom of
this page; the full one is in [docs/HISTORY.md](docs/HISTORY.md).

## Highlights

| what | measured |
| --- | --- |
| Collapsed a seven-node LangGraph pipeline into one agent loop with modes | node boundaries were 45–69% of wall time and 3 of every 5 model calls; mean score 0.54 → 0.62; a one-tool task 5 calls → 3 |
| Rubric-first evaluator: criteria written from the task before any attempt exists | judge went from spending 15 of 24 calls investigating to approving first time; same task 24 calls / 106 s → 7 calls / 31 s |
| Mutation gate held once before an irreversible action | Claw-Eval T026 (three contacts matched, agent sent to the first) 0.00 → 0.955, safety 0.0 → 1.0 |
| Type-aware compaction: what the person said is never summarised | 8/8 planted constraints kept at 120 and 400 turns, against 3/8 and 0/8 type-blind |
| Two-stage semantic recall over a content-addressed store | LoCoMo recall 10% → 96% at 2,600 tokens per query, store coverage 100% |
| Chat fast path: a greeting has no criteria, so it skips the loop | "hi otto!" 13 model calls / ~5 min → 2 calls |
| Evaluator retry path had no bound | one task 68 requests / 190 s and a crash → 5 calls / 33 s |
| Document tasks routed to a sectioned research workflow | 300 words in 23 s (approved) → 42,000 words, every section over the asked length |
| `exercise`: use what was built and show the judge a code-written report | five runs in a row (a CLI, an API, a curses app among them) wrote a harness, passed it, and finished without using the thing; with the hold, all five walked through |
| Golden set: 20 code, math and NP-hard tasks with real checkers | 20/20 |
| SWE-bench Verified, graded by each repo's own tests | 2 resolved of 3 soundly graded; three harness bugs found by control runs on untouched repos |

## Features

**Agent core**
- One conversation, one `ACTION:` / `CODE:` text protocol, one tool call per reply. Programmatic calling beats JSON tool schemas in 11 of 14 models and holds where JSON collapses under fan-out.
- Four modes (`solve`, `plan`, `summarize`, `find`) that swap the model underneath without a node boundary. Escalating restarts from the task and criteria; de-escalating carries the whole conversation.
- A rubric written from the task alone, before any attempt, that both the loop and the judge work against. The judge separates "not met" from "blocked by the environment".
- A mutation gate that holds an irreversible tool once per target and asks for the evidence that identifies the target.
- An evidence gate that holds an answer which changed code and ran nothing, once, with "there is nothing to run" accepted. Two more once-only holds: a page that was edited and never loaded, and a run that finished on its own tests without using what it built.
- Emit-path validation: a tool call is checked against its own schema and the tool name resolved from prose, backticks, bold or a typo before a round trip is spent.
- Bounded delegation to a different mode's model: a contract goes down, a report comes back, the child's trajectory is discarded.
- A run budget counted in model requests including retries, an ask-the-user budget, and a per-turn call ceiling.

**Memory**
- A tiered queue per session: recent turns verbatim, older turns compacted into cited bullets, every raw item kept as an embedded chunk in SQLite that `recall_memory` searches.
- Compaction is two-way: evicted tool output goes to the store and is searchable again.
- A lesson bank: at most three transferable lessons distilled per run, exactly one read back when relevant, off in every baseline arm.
- Every session is written to disk as it happens and can be listed, resumed, renamed, exported and imported.

**Routing**
- Nine task seats (`chat_fast`, `reason`, `evaluate`, `plan`, `summarize`, `vision`, `web`, `code_complete`, `code_edit`) over Inception, OpenAI, Anthropic and Gemini, plus any OpenAI-compatible endpoint.
- Fallback chains reordered by what each seat has achieved, with three restraints against noise and one run in ten exploring so the order cannot freeze.
- A per-model rate-limit cooldown, a per-provider circuit breaker, retired-model detection at call time, and a learned list of models that refuse `temperature`.
- Per-model temperature policy read from each vendor's published ceilings.

**Tools (18)**
`execute_bash`, `execute_python`, `read_file`, `write_file`, `edit_file`, `list_files`, `code_map`, `view_image`, `browse`, `browse_act`, `exercise`, `look`, `look_act`, `web_search`, `rag`, `recall_memory`, `complete_code`, `predict_edit`. Every shell and file tool runs either on the host workspace or inside a task container through one command-runner seam, so the tools never learn what a container is.

**Front ends**
- A Textual TUI with a setup screen (providers, models, per-seat pins), a directory browser, live progress with streaming answers, a token and dollar ledger per model, themes, and drag-to-copy that works in macOS Terminal.app.
- A REPL with the same pipeline, slash commands, and session management.
- `otto doctor`, `otto models`, `otto route <task>`, `otto lessons`, `otto sessions`.

**Evaluation**
- Six harnesses: a golden set, SWE-bench Verified, Claw-Eval, LoCoMo, a compaction benchmark, and Humanity's Last Exam.
- `--trials N` reports pass^k and the spread; `--split holdout` reads lessons and writes none; every report carries a fingerprint of the grading path; an eval that measured nothing refuses to print a number.
- A failure taxonomy computed from the action record with no model call, and a per-tool and per-seat cost breakdown.

## Quick start

```bash
git clone https://github.com/siddharth23P/otto_agent.git
cd otto_agent
uv sync
```

Put keys in `.env` at the repository root. `INCEPTION_API_KEY` is the one
Otto cannot run without (it alone serves the fill-in-the-middle and edit
endpoints); `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are
optional and each unlocks the seats routed to that vendor. `otto tui` opens
its setup screen on first start when nothing is configured.

```bash
uv run otto doctor      # which providers and seats resolve, and why not
uv run otto tui         # full-screen front end, opens on the current directory
uv run otto chat        # the same pipeline at a prompt
```

| command | what it does |
| --- | --- |
| `otto tui` / `otto chat` | interactive sessions; `--workspace PATH`, `--no-workspace`, `--resume <id\|prefix\|last>` |
| `otto sessions` | list, `--delete`, `--rename`, `--export`, `--import`, `--prune` |
| `otto doctor` | provider and route health, exit 2 on a missing required key |
| `otto models` | every model each configured vendor lists, with detected capabilities |
| `otto route <task>` | the fallback chain for a seat, pins starred, observed outcomes shown |
| `otto lessons` | print, clear, `--export`, `--import` the lesson bank |
| `otto eval` | the golden set |
| `otto eval-swe` | SWE-bench Verified |
| `otto eval-claw` | Claw-Eval (needs a checkout; see [agent/eval/data/claw/README.md](agent/eval/data/claw/README.md)) |
| `otto eval-memory` | LoCoMo recall |
| `otto eval-compaction` | what each compaction policy loses |
| `otto eval-hle` | Humanity's Last Exam, raw model vs. the agent |

Useful environment variables: `OTTO_MAX_MODEL_CALLS` (per-turn ceiling),
`OTTO_COMMAND_TIMEOUT` (120 s with a workspace, 10 s without),
`OTTO_EMBEDDING_MODEL` (`provider:model`, local BGE is the floor),
`OTTO_MODEL_PRICES` (a JSON file that overrides the price table),
`OTTO_IGNORE_ROUTES=1` (use the shipped routing table untouched; evals do),
`OTTO_NO_ANIMATION=1`, `OTTO_THEME`.

Otto ships no browser. To let it load the pages and terminal programs it
writes (see "Letting it see what it built" below), install Playwright and
`pyte` into any Python once and point `OTTO_BROWSER_PYTHON` at that
interpreter, in the environment or in `.env`:

```bash
python3 -m venv ~/.otto/browser && ~/.otto/browser/bin/pip install playwright pyte && ~/.otto/browser/bin/playwright install chromium-headless-shell
```

Without it the browser tools are simply not offered.

What the workspace boundary is worth, stated honestly: the file tools cannot
touch anything outside the root, symlinks and `..` included, and that is
enforced and tested. `execute_bash` cannot be confined the same way, so run
Otto against a repository you have committed.

## Architecture

One agent, one evaluator. The agent works the task end to end in a single
conversation and changes mode when the kind of work changes. A mode is a
model and a way of thinking, not a separate node: switching swaps the model
underneath while the conversation, the tools and everything learned so far
carry over.

```mermaid
flowchart TD
    start([request]) --> rubric[write the criteria<br/>from the task alone]
    rubric -->|no task in it| chat[answer it<br/>one cheap call] --> done
    rubric -->|a document| research[outline, then one section<br/>at a time with a continuity ledger]
    research -->|report on the file| evaluator
    rubric -->|criteria| agent

    agent{{agent}} -->|ACTION| tools
    tools -->|result| agent
    agent -->|switch_mode| agent
    agent -->|delegate| child[bounded sub-agent<br/>contract down, report up]
    child -->|report| agent

    agent -->|FINAL| gate{code changed<br/>with nothing run?}
    gate -->|yes, once| agent
    gate -->|no| evaluator

    evaluator{{evaluator}} -->|rejected + why| agent
    evaluator -->|approved| learn[distil at most<br/>three lessons]
    learn --> done([final answer])

    agent -->|ask_user| pause([paused for a question])

    subgraph tools [18 tools]
        direction LR
        shell_and_python
        files
        browser
        exercise
        screen
        code_map
        recall_memory
        web_search
    end
```

- **Criteria first.** What a correct answer must contain is written from the
  task before any attempt exists, in its own call. This is the only
  information in the whole judgment the actor did not produce. A verifier
  that re-reads the actor's own output measures at approximately nothing;
  with an external checklist the same models go from around 0% to 90–98%
  (RefineBench). Writing the criteria while looking at an answer produces
  criteria the answer happens to meet, which is why it is a separate phase.
- **Not every message is a task.** The same call says so. A greeting has no
  criteria, so it is answered in one further call on the cheapest seat: two
  calls for the turn, no loop, no judge, no lesson.
- **A document is a workflow, not a loop.** A long written deliverable goes
  to `agent/pipeline/research.py`: an outline on the plan seat, one plain
  prose call per section on the reason seat, a JSON continuity ledger
  carried section to section, word counts and required headings checked in
  code, assembly into `otto_research/<task>/document.md` (docx, pdf or xlsx
  when asked), and a report computed from the files for the judge.
- **The agent loop.** Text protocol, one tool call per reply, a mutation
  gate before the first irreversible call against each target. Mutating
  actions are 14–18% of steps and one mutating mistake cuts success odds by
  55–96%, so the gate is cheap and precisely aimed (SABER). It offers a way
  forward rather than a refusal, because an agent that is only blocked
  routes around the block (The Verifier Tax).
- **Modes.** Escalating to a deeper mode restarts from the task and the
  criteria; de-escalating carries the conversation. Handing a stronger model
  a weaker one's trajectory recovers under half the gain at 4–6x the cost,
  and discarding it moves recovery from 47% to 64% (The Handoff Tax).
- **Delegate.** One bounded subtask, one level, eight exchanges, on a
  different mode's model. Where every agent shares a model a single agent
  matches or beats the multi-agent version at lower cost, so delegating to
  your own mode is refused.
- **The evidence gate.** A ledger of what the run proved, no model call.
  Prose is exempt, it asks once, and "nothing to run" is an answer.
- **The evaluator.** Scores against the criteria written at the start,
  separates "not met" from "blocked", may check one thing with a tool, and
  is capped at two exchanges. Rejections are capped at two; past that the
  answer stands, said plainly as unverified.
- **Memory.** Compaction is type-aware: the task, the checklist, the mode,
  gathered context and a held mutating call are never compacted, and what
  the person said is never handed to the summariser. Evicted tool output is
  stored and searchable. The context tier is never re-abstracted, because
  consolidating a model's own distillations made one model fail 54% of
  problems it had solved before.
- **Routing.** A task-to-model table reordered by outcomes once twelve runs
  back a pair, a ten-point margin to overtake, only measured candidates trade
  places, and one run in ten explores. A 429 cools one model; three
  transport failures cool the provider; cooldowns are a preference and never
  leave a task without a route.

### Letting it see what it built

A run with a workspace and no container had no browser, so "a playable chess
game" was checked the only way it could be: the move logic in a script and
`node --check` on the source. Both passed on a page whose script threw at
load and never drew the board, and the judge, with nothing that could load a
page, approved it. Now `browse open index.html` loads a workspace page in a
real headless browser and fails the call when the page throws, `look` puts a
screenshot of it in front of the vision model, and a turn that edited a page
and tries to finish without loading it is held once.

Loading is not using. `exercise` runs a whole sequence, one step per line,
and its first step says what kind of thing is being used:

| first step | drives |
| --- | --- |
| `open index.html` | a page from the workspace in a real browser: `click`, `type`, `press`, `expect [not]`, `count css … = N`, `changed`, `wait` |
| `serve npm run dev` | an app behind a dev server started, waited for, used and stopped; loopback allowed here and nowhere else |
| `run mytool --count 3` | commands in a shell, with `expect` on what they printed and `exit = N` |
| `serve uvicorn app:app` + `request GET /health` | an API: `status = 200`, `expect ok`, `request POST <url> {json}` |
| `tty python3 app.py` | a program in a pseudo-terminal of 100 by 30: `expect` on the screen, `press ctrl+c`, `screen` |
| `android`, `ios`, `mac`, `linux`, `windows …` | an app on a device or desktop through adb, simctl and idb, System Events, xdotool and AT-SPI, or pywinauto, each driving only the app it launched and watching its crash log underneath |

Every kind stops at the first step that does not hold and reports each step
as the machine saw it. That report is written by code, so the judge reads it
under its own heading as the evidence the thing works. A turn that changed
code and tries to finish having only run its own tests is held once and
asked to `exercise` the thing; "nothing a person runs" is an accepted
answer. Measured before that hold, five runs in a row each wrote a harness,
passed it, and finished without using what they built; with it, all five
walked through. The macOS driver was driven for real; Android, iOS, Linux and
Windows are covered with faked commands and tracked in issues #38–#41.

### Project layout

```
agent/
  cli/          typer commands, the REPL (chat.py), the TUI (tui.py + modals,
                setup_screen, usage_panel, art), sessions, doctor, eval entry points
  pipeline/     the graph: nodes.py (agent loop, evaluator, prompts), modes,
                tools, toolkit (run-scoped tools), workspace, execution (the
                container seam), evidence, budget, usage, pricing, progress,
                research (the document workflow), codemap, browsing, screen, vision,
                walkthrough and native (the `exercise` kinds)
  memory/       TieredQueue, MemoryStore (SQLite), embeddings, retrieval,
                lessons, sessions, wiring into the CLI
  router/       Task seats, the routing table (mapping.py), provider adapters,
                health (cooldowns and breakers), outcomes (learned ordering),
                overrides (pins), automap, retired and temperature policies
  eval/         golden runner, swe_bench, claw_bench, memory_bench,
                compaction_bench, hle_bench, terminal_bench, single_agent
                control, failures (taxonomy), langfuse sync
containers/     the throwaway desktop image (Xvfb, fluxbox, xdotool)
docs/           HISTORY.md (development log), design/tiered-memory.md
tests/          1,534 tests, none of which need a key or the network
```

## Memory

The engine is a queue with two tiers. `X` holds recent items verbatim; when
its token budget overflows, its whole content moves to `Y`. When `Y`
overflows, every raw item is flushed to a content-addressed SQLite `chunks`
table keyed by sha256, and the whole of `Y` is summarised into a cited bullet
list. Each bullet's hash references resolve to real stored text no matter how
many generations later, and every item the summariser failed to cite gets its
own bullet, because a real model omitted about 30% of items and one omission
severed the citation chain for everything behind it (recall fell to 2–3%).

Recall is two-stage: compile the candidate chunks from every live bullet,
rank the chunks by their own embedding, return the best twenty with a window
of two neighbours under a 3,000-token ceiling. Ranking bullets first was
measured and rejected (5% coverage against 90%), as was BM25 blending (cost 14
points on paraphrased questions). The BGE query prefix alone was worth four
points. Measured on LoCoMo, the shipped defaults reach 96% recall at about
2,600 tokens against 99% for returning the entire conversation.

The embedding backend is swappable and every vector records the model that
made it, so a store never ranks two embedding spaces against each other.
Default is `gemini-embedding-001` when a Gemini key is present (97.3% against
93.7% for local BGE on 221 held-out questions) and the local model otherwise.
Full design and measurements: [docs/design/tiered-memory.md](docs/design/tiered-memory.md).

## Evaluation

| command | grades | by |
| --- | --- | --- |
| `otto eval` | 20 golden code, math and NP-hard tasks | a real checker per item; math checkers reject named decoys (the first-fit bin count, the greedy set cover, the target read off a subset-sum question) |
| `otto eval-swe` | SWE-bench Verified, 500 issues | each repository's own tests; FAIL_TO_PASS must pass and PASS_TO_PASS must still pass; arm64 images native, x86_64 under emulation with extra budget and the split reported |
| `otto eval-claw` | Claw-Eval, 300 tool-use tasks | their graders and judge, over a trace Otto writes; `--architecture single` runs a no-graph control on the same tools |
| `otto eval-memory` | LoCoMo long-conversation recall | store, visible, recalled and answerable coverage; refuses to print a table if compaction never fired |
| `otto eval-compaction` | eight planted constraints replayed through each policy | did the constraint survive into the prompt, and could recall find it |
| `otto eval-hle` | Humanity's Last Exam | the official judge prompt, raw model against the full agent on identical questions, calls per question reported |

Rules the harnesses enforce, because each was learned the hard way:

- **Repeat.** A single Claw-Eval run swings 0.36 between identical attempts
  (T093 scored 0.86 / 0.60 / 0.96 on identical code). Below three trials the
  report leads with NOT EVIDENCE.
- **Hold tasks back.** `--split holdout` is chosen by a hash of the task id,
  reads the lesson bank and writes nothing to it.
- **Freeze the grader.** Every report carries a fingerprint of the grading
  path, the Claw-Eval revision, the judge model, the threshold and the
  formula. Two numbers from different fingerprints are not a comparison.
- **Count the cost.** Model calls, tokens and dollars per model, per tool and
  per seat travel with every score; equal success rates hide multi-x
  differences in spend.
- **Distrust the harness first.** Three of eight SWE-bench scores were
  harness bugs, each caught by asking what the grader says about a repository
  nobody has touched. That control is now part of the harness.
- **Name the failure.** `agent/eval/failures.py` tags each run (`zero_write`,
  `error_cascade`, `first_call_failed`, `no_actions`) from the action record
  alone, so a regression says what broke rather than that something did.

Results recorded at the end of the core rebuild (PR #6), single runs unless
stated:

| benchmark | result |
| --- | --- |
| golden set | 20/20, including all 8 NP-hard |
| LoCoMo | 96% recalled, store coverage 100%, over 101 questions; 91% / 95% answerable over 759 |
| Claw-Eval sample | T026 0.00 → 0.955, T112 0.80, C01 recovered from a crash to a verified answer |
| SWE-bench Verified | 2 resolved of 3 soundly graded |
| compaction | protected policy 8/8 at every budget; type-blind 3/8 → 0/8 |

## Testing

- 1,534 tests pass and 12 skip on macOS, Linux and Windows, in about two
  minutes, with no API keys and no network. CI runs the full matrix on every
  push and pull request with a read-only token.
- Offline and live tests are split. `tests/conftest.py` sets placeholder keys
  with `setdefault`, keeps the suite out of `~/.otto`, and gives every test an
  empty temperature-learning store. Live tests are marked and skip without a
  real key.
- Behaviour, not output. Regression tests assert on the messages handed to the
  model, on the exact bodies that crashed real runs (the C01 path, the
  Terminal-Bench chat-template tokens, the truncated outline JSON), and on
  call counts against the real compiled graph with a scripted model.
- Library behaviour the fixes rest on is asserted directly against the
  installed version: Textual's `exclusive=True` not stopping a thread worker,
  `RichLog` deferring writes until sized, LangGraph's stream tuple shape.
- Every TUI test waits on a condition with a deadline, never on an iteration
  count, verified by running the file four times concurrently under CPU load.
- Invariants are tests: every dispatchable tool appears in every prompt, the
  agent prompt stays under a measured character cap, every irreversible tool
  is behind the gate, and no directory in the repo shadows a site package.
- Suite growth is recorded in the log: 236 tests on Sep 10, 845 at PR #6,
  1,441 at PR #36, 1,534 at PR #37.

## Research this design draws on

Each paper is listed with the finding that changed Otto and where it landed.
The numbers are the papers' own, as recorded in the module docstrings.

| paper | finding applied | where |
| --- | --- | --- |
| [RefineBench: Evaluating Refinement Capability of Language Models via Checklists](https://arxiv.org/abs/2511.22173) (Lee et al., ICLR 2026) | self-refinement over five turns is 31.3% for the best model and near 0% for most; the same models reach 90–98% given an external checklist | the rubric-first evaluator, `agent/pipeline/nodes.py` |
| [The Art of Building Verifiers for Computer Use Agents](https://arxiv.org/abs/2604.06240) (2026) | a verifier structured as criteria-then-judgment reaches human-level agreement (κ 0.64, inside the 0.53–0.57 human band) with a 0.01 false-positive rate, and the gain is architectural | two-phase judging, non-overlapping checkable criteria |
| [SABER: Small Actions, Big Errors](https://arxiv.org/abs/2512.07850) (2025) | mutating actions are 14–18% of steps and a single mutating deviation cuts success odds by up to 92–96% | the mutation gate, `TOOL_TIERS` |
| [The Verifier Tax: Horizon-Dependent Safety–Success Tradeoffs in Tool-Using LLM Agents](https://arxiv.org/abs/2603.19328) (2026) | enforcement that blocked 94% of non-compliant actions left safe task completion under 5%, because the actor fabricated a way around the block | the gate and the evidence gate offer a way forward; the judge treats "blocked" as not "failed" |
| [The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents](https://arxiv.org/abs/2608.24358) (2026) | across 58,000 runs, handing a stronger model a weaker one's trajectory recovers under half the gain; discarding it moves recovery from 47% to 64%, and the reverse hurts | escalate by restarting, de-escalate by carrying |
| [The Compaction Cliff in Long-Running AI Agent Memory](https://arxiv.org/abs/2608.22752) (2026) | type-blind compaction keeps 53% of constraints at 50% compression and 24% at 10%, falling to 10% over five rounds; type-aware holds 96% | type-aware compaction; `otto eval-compaction` |
| [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978) (Zhang et al., 2026) | consolidating its own memories made a model fail 54% of problems it had solved; retaining raw episodes doubles accuracy | no re-abstraction on the context tier; lessons distilled from raw trajectories, never from other lessons; the abstract-once arm measured |
| [Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?](https://arxiv.org/abs/2511.13646) (2025) | asking the agent after each step whether it should build itself a tool took 62% → 76%, and the system to 77.4% on SWE-bench Verified | the periodic tool-building reminder |
| [The Bitter Lesson of Tool Calling](https://arxiv.org/abs/2608.06370) (2026) | programmatic tool calling matches or beats JSON in 11 of 14 models and holds under fan-out where JSON collapses | the `ACTION:` / `CODE:` text protocol |
| [Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline](https://arxiv.org/abs/2601.12307) (2026) | a single agent reusing its KV cache matches homogeneous multi-agent workflows at lower cost; only genuine model heterogeneity justifies sub-agents | one loop with modes; `delegate` refuses your own mode |
| [Can Small Agents Collaborate to Beat a Single Large Language Model?](https://arxiv.org/abs/2601.11327) (2026) | reasoning at the orchestrator was worth +18.2 (GAIA) and +36.7 (AIME) at 8% latency; sub-agent size was flat (23.0 / 23.0 / 23.6) | the delegate child is thin and bounded |
| [CoAct-1: Computer-using Agents with Coding as Actions](https://arxiv.org/abs/2508.03923) (2025) | routing subtasks to code or GUI and preferring code reaches 60.76% on OSWorld in 10.15 steps against ~15 for GUI-only | the desktop image keeps a shell; `look`/`look_act` are the fallback |
| [Beyond Browsing: API-Based Web Agents](https://arxiv.org/abs/2410.16464) (ACL Findings 2025) | API plus browser beats browsing alone by 24 absolute points | `browse` is the fallback to `execute_bash` |
| [AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents](https://arxiv.org/abs/2410.13825) (ICLR 2025) | refining only the observation and action space beat every scaffolding trick by +9.8 points (+29.4%) | pages come back as a digest, never raw DOM; a small action vocabulary |
| [Building Effective AI Coding Agents for the Terminal](https://arxiv.org/abs/2603.05344) (OpenDev, 2026) | long runs suffer instruction fade-out; event-driven reminders in the conversation work where rewriting the system prompt does not | periodic checklist and tool-building reminders on tool results already being sent |
| [Recursive Experiential–Working Memory Evolution for Long-Horizon Agent Harnesses](https://arxiv.org/abs/2608.24876) (Recuris, 2026) | a structured trace localises a fault 64.8% of the time against 13.0% from the outcome alone; a named failure-mode taxonomy | `agent/eval/failures.py`, the three tags decidable from the action record |
| [Agent-as-a-Router: Agentic Model Routing for Coding Tasks](https://arxiv.org/abs/2606.22902) (2026) | routing on logged per-task outcome statistics is worth +15.3% relative | outcome-based reordering of fallback chains, `agent/router/outcomes.py` |
| [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/abs/2607.12227) (2026) | harness evolution does not consistently beat repeated sampling under matched budgets | measure self-evolution before building it; `--trials`, matched baseline arms |
| [DarwinX: Evolving Agent Harnesses Through Natural Selection](https://arxiv.org/abs/2608.07545) (2026) | a 31.7-point gap between the proxy the search maximised and held-out truth | `--split holdout`, read-only lessons on the held-out split |
| [SEA-Eval: Evaluating Self-Evolving Agents Beyond Episodic Assessment](https://arxiv.org/abs/2604.08988) (2026) | identical success rates hide up to 31x differences in token cost on a stream of related tasks | cost recorded beside every score; cost convergence as the primary self-evolution signal |
| [Stop Comparing LLM Agents Without Disclosing the Harness](https://arxiv.org/abs/2605.23950) (2026) | the same model swings by tens of points across scaffolds and scoring configurations | the grading fingerprint on every report |
| [Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents](https://arxiv.org/abs/2604.06132) (2026) | 300 human-verified tasks with trajectory-aware grading; pass^k over trials | `otto eval-claw`, run against Otto's own agent rather than a bare model |
| [SWE-bench](https://arxiv.org/abs/2310.06770) (Jimenez et al., ICLR 2024) and the Verified subset | grade a diff by the maintainers' tests, not an answer by a judge | `otto eval-swe` |
| [LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) (Maharana et al., 2024) | real long conversations with evidence-cited QA | `otto eval-memory` |
| [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) (Phan et al., 2025) | expert questions with a strict official judge | `otto eval-hle`, raw model against the agent |
| [OSWorld](https://arxiv.org/abs/2404.07972) (Xie et al., 2024) | the benchmark behind the code-vs-GUI numbers above | the desktop tools' design |
| [C-Pack / BGE embeddings](https://arxiv.org/abs/2309.07597) (Xiao et al., 2023) | `bge-small-en-v1.5` is asymmetric: queries need the search instruction prefix | `embed_query()`, worth 3.9 points on LoCoMo |

Open-source projects whose mechanisms were adopted, each credited in the
module that uses it: **OmniRoute** (10% exploration in learned routing, the
provider circuit breaker and per-model cooldown), **hermes-agent** (the
passive evidence ledger, the compaction policy matrix), **graphify**
(`code_map`: the code half of a knowledge graph needs no model), **ponytail**
(the over-build ladder, measured at 54% fewer lines and 22% fewer tokens with
safety held at 100%), **Terminal-Bench** and **Claw-Eval** (the container
harnesses).

## How it was built

Every step was measured against the one before it, and several were
reverted on the number. The full log with the measurement behind each commit
is in [docs/HISTORY.md](docs/HISTORY.md).

| day | what landed |
| --- | --- |
| Sep 7 | Skeleton: task seats, a capability-checked routing table, a router with fallback chains, provider adapters. |
| Sep 8 | Typer CLI with a diffusion view for Mercury models, friendly provider errors, Langfuse tracing, CI, provider warm-up and key caching. |
| Sep 9 | A swarm: several agents per role voting on one answer. |
| Sep 10 | Replaced the swarm with a routed specialist graph (16/16 golden, up from 9/10), then an overseer and executable plans. Conversation history reached the graph; `ask_user` pauses via LangGraph interrupts. The tiered memory engine, its wiring, and the LoCoMo benchmark, which found that uncited items were unreachable (recall 10% → 99%) and then that recall was a dump (→ 96% at a tenth of the text). A real workspace, the container execution seam, a Terminal-Bench adapter, and a `reasoning_effort` fix that took 40 calls for 2 tool calls to 1:1. |
| Sep 11 | Debugging habits added one at a time and measured; the fifth erased the first four and was reverted. Temperature clamped to what Inception honours. HLE benchmark. All four vendors restored and each role routed to the cheapest that does it well; vision, real web search and workspace RAG; retired-model and per-model temperature policy; swappable embeddings with Gemini measured best. Claw-Eval run against Otto's agent. Then the loop rewrite: the seven-node graph collapsed into one agent with modes, the evaluator capped, the rubric-first judge, the mutation gate (T026 0 → 0.955), a checklist in state, type-aware compaction, escalate-by-restart, the tool-building reminder, schema validation, delegation. |
| Sep 12 | Browser and desktop tools inside the container. Self-evolution measured before it was built, then the lesson bank; outcome-based routing with exploration; two-way compaction; the over-build ladder; the evidence gate; the circuit breaker; `otto eval-compaction`, which showed compaction destroying constraints and fixed it (8/8); `code_map`; SWE-bench Verified and NP-hard math; three harness bugs found by control runs; Windows CI, an SSRF fix and six other review findings. PR #6 merged 72 commits. Then a day of live-use fixes: concurrent TUI turns, interactive workspaces, `ask_user` answers reaching the loop, a question cap, a token and dollar ledger, judging against what was actually asked, three evals refusing to report meaningless numbers, the failure taxonomy, the chat fast path, live progress and the unbounded-retry bug (68 calls → 5). |
| Sep 13 | The TUI setup screen, directory browser, per-seat pins and a redesign; learned temperature refusals; the document research workflow (300 → 42,000 words); sessions saved as they happen with resume, export and import. Then a local browser for workspace pages and `exercise`, one tool with nine kinds of walkthrough reported step by step as the machine saw it, plus two holds so a run cannot finish on its own tests or on a launch alone. |

## Limitations

- The shell is not sandboxed on the host; only the file tools are confined.
- `code_map` covers Python only, by reading `ast`; it refuses other languages by name rather than answering partially.
- `browse` and `browse_act` each drive a fresh page and restore cookies and storage from disk; in-page state that never touches storage does not survive between calls. `exercise` exists for sequences that need it.
- The Android, iOS, Linux and Windows `exercise` drivers are covered by faked commands only; macOS was driven for real.
- Screen grounding is a description plus coordinates, as with every GUI agent; expect look, act, look again.
- Benchmark results above are mostly single runs on small samples, and the harnesses say so. A single Claw-Eval run is not evidence.
- Inception's Mercury models are deliberately absent from the price table, so a turn on them shows an unpriced marker rather than a guess.

## License

MIT. See [LICENSE](LICENSE).

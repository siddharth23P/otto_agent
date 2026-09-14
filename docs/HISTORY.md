# Development log

Otto was built between 7 and 13 September 2026 in 128 commits.
This is the commit-by-commit record of what changed and, wherever a
number was taken, what the number said. Test counts are the suite's size at
that commit. Hashes are short hashes on `main`.

The through-line: build the instrument first, change one thing, read the
number, keep or revert. Several changes below were reverted on the number,
and several "fixes" turned out to be harness bugs rather than agent
behaviour. Both kinds are recorded, because a log that only keeps the wins is
not a log.

## Day 1 — 7 Sep: skeleton and routing

| commit | change |
| --- | --- |
| `7457e8d` | Project skeleton. |
| `13c5e03` | `.gitignore`. |
| `b731179` | CLI moved into `agent/cli/`. |
| `073ef46` | `agent/router/mapping.py` as pure data: task seats, candidate chains, per-route params. |
| `7eb924b` | The router and its first tests: resolve a seat to the first usable candidate. |
| `cc5d646` | Router construction and policy: what "usable" means for a provider. |
| `8eac27e` | Model catalogue and hard-pinned routes. |

## Day 2 — 8 Sep: a CLI, observability, CI

| commit | change |
| --- | --- |
| `703e7a1` | First README. |
| `30324f6` | GitHub Actions: pytest on every push and PR. |
| `6addd2a` | Basic CLI, a diffusion view for Mercury (Inception) models, provider imports moved under `agent/`. |
| `ee1ba46` | CI fix. |
| `07206ad` | CLI branch merged. |
| `9ebc3d5` | Provider errors rendered as messages instead of tracebacks (three provider-error issues). |
| `2e115f3`, `a297520` | Langfuse observability: every model call traced. |
| `c3d59de` | Warm-up and caching: six small changes so the first fan-out does not stampede providers, and a key that changes mid-process is noticed. |

## Day 3 — 9 Sep: the swarm

| commit | change |
| --- | --- |
| `8428ff5` | "The hive": several agents per role, one vote, one live session. The architecture the next two days replaced, and the baseline the first golden-set number was taken against (9/10). |

## Day 4 — 10 Sep: a routed graph, conversation memory, workspaces, containers

**The specialist graph.**

- `8cdfeb9` Route once, work once, judge once. The orchestrator/worker/consensus/synthesise swarm became a router dispatching exactly one of planner/solver/summarizer/finder per round, judged by one evaluator. Inception-only at this point; `complete_code` and `predict_edit` added on Mercury's FIM and edit endpoints. **Golden set 16/16, up from 9/10.**
- `c4c96c3` The router became an overseer re-invoked after every node, deciding the single next action from everything accumulated. No cap on retries; the LangGraph recursion limit as pure insurance.
- `3b0976c` Accept a `NODE:` reply that dropped its own label (the model did this almost always when the answer was "evaluator"), which had been silently defaulting to solver and wasting a round.
- `b81ebe9` Plans became executable: parsed into steps with `route_to` assigned one at a time by the overseer, with no model call once every step has output. Also fixed a real bug: a literal JSON example in the planner prompt broke `str.format`.
- `36a83b8` Retry the overseer's own unparseable reply in place (up to a small cap, with corrective feedback) before defaulting to solver. A cheap router retry is far less than the extra specialist round it used to cost.
- `5ab9ac1` Provider and network failures (an `httpx.ReadTimeout` mid-stream had crashed the whole graph) now translate to `ProviderError` inside the stream iterator and route back to the planner instead of unwinding.

**The front end learns to converse.**

- `6aa74d6` TUI: thinking separated from the result and collapsed by default.
- `9230681` TUI: copy the last answer via OSC 52 rather than mouse-selecting box borders.
- `5d233cd` Conversation history carried into the graph, not just onto the screen. Live-tested: "Solve N Queens" / "improve above solution" — the third turn had no idea what "above solution" was.
- `b92676a` `ask_user`: a stuck node pauses the run through LangGraph's checkpointed `interrupt()` and asks a real question; REPL prompts inline, TUI shows a modal. 24 new tests including one that drives the real compiled graph. **260 tests.**

**Tiered memory.**

- `7aba0e6` The X/Y tiered short-term-memory engine, standalone: a small verbatim tier overflowing wholesale into a larger one; on overflow, raw text flushed to a sha256-keyed SQLite store and the tier summarised into cited bullets whose hash references stay resolvable across generations; local `fastembed` recall with graceful fallback. Budget 40% of Mercury's 260K window. 50 new tests. **310 tests.**
- `2d5b0c0` Wired into cross-turn history: the session's history became a `TieredQueue`, `recall_memory` became a tool in every prompt. **333 tests.**
- `c0a9f0a` `otto eval-memory`: LoCoMo (arXiv:2402.17753) replayed through the queue, scoring store, visible-verbatim, recall and answerable coverage. Real LoCoMo conversations (11–24K tokens) never exceed the production budget, so budget overrides exist to force compaction; at a tiny budget recall rose 0% → ~39% with store coverage held at 100%.
- `288c50b` Per-question evidence beside recalled text, so a coverage number can be inspected rather than trusted.
- `f30eb63` Detect unparsed-summary fallback bullets: a live sweep's recall fell 80% → 32% → 0% → partial recovery, a shape that fits one generation's summary failing to parse, not retrieval degrading.
- `89da621` **Every compacted item stays reachable.** A live sweep collapsed recall to ~10%: the summariser cited ~70% of its items and one uncited prior bullet severed the whole citation chain. Replaying 250 turns over 19 generations: 9–11 of 249 chunks reachable, recall 2–3%. Every uncited item now gets its own bullet. **Recall ~10% → 99%.**
- `03ff787` **Recall finds the answer instead of returning everything.** The 99% was a dump: every call returned 38,033 characters. Recall became two-stage (compile candidates from every live bullet, rank the raw chunks by their own embedding); ranking bullets first was measured and rejected (5% vs 90%); neighbours either side of a hit; the BGE query prefix (worth ~4 points); BM25 blending measured and rejected (−14 points). **96% coverage at ~2,600 tokens** (single-hop 96, temporal 92, multi-hop 88, open-domain 100).

**A workspace and a container.**

- `fc4c9cf` A workspace the agent can actually edit. `execute_python` and `execute_bash` had each run in their own throwaway temp directory, so two calls shared nothing. `read_file`, `write_file`, `edit_file` (exact-once match), `list_files`; `..` normalised before symlink resolution; the `WORKSPACE` tool tier; the tool menu derived from the dispatch table instead of hand-written five times.
- `fedb493` Tools act inside a container through one command-runner seam (`agent/pipeline/execution.py`); content crosses base64-encoded rather than in heredocs; the edit contract expressed once as a script the container runs.
- `992b8a6` A Terminal-Bench adapter running Otto's real graph, with working directory tracked across `docker exec` calls. Passes hello-world.
- `3843d4a` Three harness bugs from the first Terminal-Bench batch (1/20, mostly ours): a multi-call reply executed as one command (`ACTION:: command not found`), file tools with no format hint (the model wrote a file named `{`), an empty provider stream killing the run. Plus a two-stage deadline read from the task's own budget, and a transcript file, because two of the first three failures were Otto reporting a fix it had not made.
- `ce87289` **The `reasoning_effort` setting that was starving the agent.** 39 of 40 tool-loop turns got an empty reply. Measured six trials per setting: `high` produced usable replies 1–2 of 6 and was slowest; `medium` 6/6 and fastest. **40 model calls for 2 tool calls → 192 for 186, 20:1 down to 1:1.** Plus a dead-reply cap and a `TimeoutExpired` bytes bug.

## Day 5 — 11 Sep: habits, vendors, benchmarks, and the loop rewrite

**Teaching the solver to debug, one habit at a time.**

- `5b88f90` Stop redrafting a file never once run: 166 writes and nothing produced. An `actions` channel (one line per tool call, kept across the run) and a note when three consecutive calls hit the same target with nothing run between. Before: ten writes, zero compiles; after: write, compile, read the error, write again.
- `b2fabf0` Strip chat-template tokens (`<|tool_call_start|>`) that occasionally leak into a command body.
- `9ca6ea8` No detached commands (`&`, `nohup`, `setsid`); long output keeps both ends; the evaluator told to run something that would fail if the work were not done.
- `85c9680` Two habits: look at the system, not just the broken thing; never repeat an unchanged failure. From a transcript with 99 commands, 28 re-running the one that was broken, and no `ps` or `crontab`.
- `6ce54f7` A single-agent control (same model, tools, container; one conversation that never resets) to test whether the graph was the bottleneck. On cron-broken-network the graph explored twice as much and was the only one to inspect processes. Not the bottleneck.
- `857ee10` Two more habits, measured across five revisions on the same task: system-inspection commands 0 → 0 → 0 → 2 → 17. Still failing, but for a different reason.
- `4876f6c` **Reverted a fifth habit.** It did not merely fail to take: system inspection went 17 → 0 and commands fell by a third. Past some prompt length the model acts on none of it.
- `c1d7005` Temperature clamped to the 0.5–1.0 range Inception honours: every "careful" node asking for 0.0–0.4 had been silently served 1.0.
- `674edb1` Prompts trimmed 16–28% (the protocol block assembled once from data); `MOST_DETERMINISTIC` named for what it can actually get.

**More vendors, more senses.**

- `9817e7c` `otto eval-hle`: Humanity's Last Exam in two modes over identical questions, raw model vs. the whole graph, with calls per question. Official judge prompt; verdict parsing fails closed; gated dataset respected.
- `3e01d21` Fix the HLE command printing through Rich consoles as if they were callables.
- `a337b7a` Every configured provider usable; OpenAI, Anthropic and Gemini recovered from history. The measurements had said Inception-only had run its course: 0/101 on Claw-Eval multimodal, 5.3% multi-turn, 0/25 HLE.
- `44d586c` Exceptions from LangChain chat models Otto does not own translated to `ProviderError` by duck-typing, before any route moved off Inception.
- `c5aa001` Each role routed to the cheapest vendor that does it well: Anthropic judges and plans, OpenAI solves, Gemini summarises and sees, Inception keeps chat-fast and FIM/edit. `EVALUATE` split from `REASON`. Every chain ends at Mercury so a missing key degrades cleanly; vision and web have no floor on purpose. Temperature moved into the table.
- `39d95bc` `view_image`: a vision model answers a question about a file and returns words, so the graph and the memory stay text-only. `read_file` refuses images by name. Verified on a real PNG.
- `9fe3e3e` `web_search` (Anthropic server-side search; the tool version the cheap tier accepts settled against the live API) and `rag` (the memory store's chunk search over workspace files) made real. Both had been stubs advertised in every prompt.
- `237f1a8` Retired-model tracking (two pinned Gemini ids answered 404 "no longer available") and temperature decided per model: Inception resets, OpenAI's o-series rejects, others clamp. Verified: o4-mini at 0.0 used to 400.
- `fac6782` Vision gets a capability-query fallback, and the catalogue filters models that cannot chat (TTS, transcription, "only supports Interactions API"): Gemini 53 → 26, OpenAI 136 → 78.
- `8b0a6f5` Each model's own published temperature ceiling wins over the per-vendor default (39 Gemini models cap at 2, 7 at 1).
- `25b44a6` Swappable embedding backend; every vector records its model, and retrieval refuses to rank across spaces (same dimension from a different model raises nothing and ranks nonsense). Three backends verified at 384, 1536 and 3072 dims.
- `e64867f` Default embedding measured on 221 LoCoMo questions: local BGE 93.7%, OpenAI 96.0% (conv 1 only), **Gemini 97.3%**, with the gain larger on the held-out conversations. Gemini is the default when its key is present; local is the floor. Recorded cost: ~520 ms per query vs 17 ms.

**Claw-Eval against Otto, not against a model.**

- `7f39779` Claw-Eval's own CLI measures one model in their loop. Otto now runs the task itself, writes a conforming JSONL trace, and their graders score it. Run-scoped tools (`agent/pipeline/toolkit.py`) bound per task. Verified live: T112 pass 0.80, M001 0.61 in a container. **521 tests.**
- `0d49a24` The run made reproducible: the judge config, a Gemini judge patch as a diff, and a runbook naming what needs Docker, which key, and which 24 tasks return empty without `SERP_DEV_KEY`.
- `b6d3c16` Don't bind a task tool that answers every call with 404 (38 `Bash` tools with no endpoint).
- `aaa3553` Offline coverage of the harness decisions made before Claw-Eval's code runs. **543 tests.**
- `a1d25d0` `--architecture single` documented.
- `3a9e2fb` `--max-seconds` caps a sample's per-task budget, recorded in the report because it lowers scores by construction.

**The loop rewrite** (three steps, then everything that followed).

- `a777507` Step 1: a run budget counted at `_call` including retries (tool execution was 0.1–0.4 s of runs lasting 119–946 s, so a tool-bound deadline watched the one part that cost nothing; C01 ran 1096 s against a 900 s budget) and a mode table. Inert. **582 tests.**
- `7d6ed2f` Step 2: the loop's conversation carried in state (`transcript`), `model_calls` replacing dispatch rounds, `mode_swaps` scored, and a custom stream so the CLI shows tool calls as they happen. Inert. **588 tests.**
- `ec68bcb` **Step 3: the graph collapsed into one agent loop that switches mode.** Node boundaries were 45–69% of wall time at 20–126 s each. Asserted against the real compiled graph: one tool call and an answer 5 calls → 3; ten tool calls ~17 → 12; a rejection and retry 7 → 4. The evaluator stopped judging blind; rejections capped at two; `run_pipeline` salvages the best answer on exception, interrupt or non-approval. `nodes.py` 1810 → 1653 lines despite gaining the loop.
- `926fc07` C01 lost a verified answer to an unhandled `OSError(ENAMETOOLONG)` from prose where a path goes; now a clean refusal. T026 (the only zero) sent mail to the first of three matching contacts; the ask-user guidance became about the action, not the mood; mutating tools named from the registry. **567 tests.**
- `39f0e80` Compaction of the loop's own conversation, for free: old tool results shrunk in place past a threshold; transcript size stops growing with the number of tool calls (a test asserts it).
- `c5ecd16` **The evaluator capped.** Given evidence, it investigated instead of judging: 24 calls and 106 s on "write fib.py and run it", fifteen of them the judge. Two iterations. **7 calls, 31 s, approved first time.** 3.4x on one constant.
- `1426ac6` Stop the bleeding: the suite did not collect on a clean checkout (a module-scope `Router()` raised without a key), `otto chat` had no budget at all, the evaluator could poison its own conversation with an empty `AIMessage`, and it was told it had five tool exchanges and given two. **580 tests, and for the first time they run with no keys.**
- `510244e` **The rubric before the answer.** RefineBench: self-refinement −2.5% to 31.3%; the same models 90–98% with an external checklist. Judging became two phases: criteria from the task alone, then a score against them, with a continuous share met and "blocked by the environment" separated from "wrong". One extra call per judgment. **589 tests.**
- `64e22ac` **The mutation gate.** SABER: mutating actions are 14–18% of steps and one mutating deviation cuts success 55–96%. The first call to a mutating tool against a target is held once and asked for the evidence identifying the target; it offers a way forward, because enforcement that only blocks (94% intercepted, under 5% safe success) gets routed around. Run-scoped tools classified by word tokens ("widget" ends in "get"). **600 tests.**
- `786b2f5` **T026 0.00 → 0.955.** The gate held `gmail_send_message`; the agent found three matches and asked. Safety 0.0 → 1.0. A pause nobody can answer now returns the question instead of nothing: completion 0.30 → 0.944.
- `b7e6791` A checklist in state, written once from the task before any attempt, that the loop and the judge share. Two corrections found by measuring: the gate had been firing on `write_file` (recoverable in a workspace; a two-file task 6 → 9 calls), and the judge rejected a correct answer twice on criteria about steps it could not verify. Same task: baseline 7 calls / 31.0 s, rubric at judgment 6 / 33.4 s, **checklist in state 5 / 25.8 s**. **609 tests.**
- `3c6fcca` Type-aware compaction (the task, checklist, mode, context and a held call are never compacted), summarise-instead-of-truncate at no cost (the `actions` line replaces the evicted result), and two read policies over one store (narrow `top_k` for mid-task recall). The re-abstraction gate deliberately not changed without a measurement. **612 tests.**
- `d894356` `edit_file` matches through a cascade (exact, trailing whitespace, indentation, first-and-last-line anchor), each pass still demanding a unique hit; the container path shares the implementation. 181 of Claw-Eval's 300 tasks run in a container. **623 tests.**
- `ba2499c` **Escalate by restarting, de-escalate by carrying.** The Handoff Tax: discarding the weaker trajectory moves recovery from 47% to 64%; removing a strong trajectory before handing down hurts. The checklist survives the restart because it is state, not conversation. **636 tests.**
- `a7f0374` Two periodic reminders on tool results already being sent: should you build yourself a tool (Live-SWE-agent: 62% → 76%, system 77.4% on SWE-bench Verified), and what is still open on the checklist (against instruction fade-out). Periodic, because said every turn they become wallpaper. One bug found by a test: the reminder read a checklist that was `None` on a first run. **642 tests.**
- `9f6d2f1` Coverage is part of the answer: T136 went 0.78 → 0.42 after the rubric fix overshot into criteria a thin answer satisfies. The fix moved the behaviour (one note read → all of them) but not the score, and the real finding was that the criteria were never in the trace. They are now. **643 tests.**
- `80e9f93` A call checked against its declared schema before dispatch (required fields, types, enums), with the error stated in the call's own terms. `bool` is an `int` in Python. **652 tests.**
- `8f8c7bc` `delegate`: one bounded job on a different mode's model, one level, eight exchanges; contract down, report up, trajectory discarded. Refused for your own mode (a single agent with a shared model matches multi-agent at lower cost). The child is thin because orchestrator reasoning was worth +18.2/+36.7 points and sub-agent size was flat. **660 tests.**

## Day 6 — 12 Sep: new surfaces, self-evolution, hardening, and a day of live use

**Browser, desktop, lessons.**

- `0d1af1c` A browser in the container already being driven, through a Playwright driver script. Two tools: `browse` reads, `browse_act` acts behind the gate. The observation is a digest, never raw DOM (AgentOccam +9.8). Stated as the fallback to an API (API+browser beats browsing alone by 24 points). The tier invariant became "anything irreversible is held", instead of "the mutating tier is empty". **677 tests.**
- `80b3edd` A throwaway desktop (Xvfb, fluxbox, xdotool) in a container rather than on a machine; `look` answers a question about the screen through the vision model, `look_act` clicks and types. The image keeps a shell because code beats GUI (CoAct-1: 60.76% in 10.15 steps vs ~15). Two bugs only a live run showed: `base64` wraps at 76 columns; `xdotool type` needs `--clearmodifiers`. **692 tests.**
- `3149dd8` **Self-evolution measured before it was built.** Three rules first: `--trials N` (pass^k, pass@k, spread; median trial reported), `--split dev|holdout` by task-id hash (holdout reads lessons, writes none), model calls beside every score, and a grading fingerprint. Then the lesson bank: at most three distilled per run, exactly one read, adjudicated at write time, never re-abstracted. One `import docker` shadowing bug (a `docker/` directory became a namespace package) silently disabled 169 container tasks; a test keeps the next one out. **717 tests.**
- `fb05465` Three Claw-Eval files the docs point at had been gitignored; the measurement protocol (dev with learning, holdout reading, holdout with none) written down.
- `995ad2b` Trial service ports spaced apart, because a stride of 1 rebinds into `TIME_WAIT`.
- `7f9c1d6` The distiller moved off the cheapest seat: on a real 29-message trajectory it returned an empty array every time; the same body one seat up produced three usable lessons.
- `87d23dc` The distiller writes lessons about method, not notes about the task ("rank client requests above deadlines" is a note).
- `42aef44` Validate on the emit path: five real shapes (`ACTION: execute_bash to list the files`, backticks, bold, a full stop, a typo) each cost a call to be told the tool did not exist; all resolve now. An empty `CODE:` is refused by name. Whichever of `ACTION`/`FINAL` comes first wins, so an answer followed by a suggestion is no longer discarded. **731 tests.**

**Routing that learns, memory that gives back.**

- `09de7f0` Routing reordered by what each seat achieved (+15.3% in the literature; costs no model call). Three restraints: twelve runs before a number counts, only measured candidates trade places, a ten-point margin to overtake. Multi-mode runs record nothing; the judge's seat is never credited with its own verdict. `otto route` shows the evidence. **747 tests.**
- `9fa4a0a` Compaction made two-way: evicted tool output stored under a context kind and searched by `recall_memory`, with no bullet layer (no re-abstraction). The stub says "searchable" only when a store is bound. **758 tests.**
- `1dbe046` Exploration: without it a demoted candidate froze at exactly 12 runs forever (simulated over 200 more). One run in ten goes to the trusted candidate with the fewest runs; the loser reaches 49. Off when the log is read-only. **763 tests, run five times.**
- `adf34cd` The over-build ladder from ponytail (54% fewer lines, 22% fewer tokens, safety held at 100% in their twelve-ticket measurement), as its own block rather than a fifth habit. The prompt cap held: 631 characters came back at 398.
- `daddc80` The evidence gate from hermes-agent's ledger idea: an answer that changed code with nothing passing since is held once; prose exempt; "nothing to run" accepted. No model call. **781 tests.**
- `133ed87` A circuit breaker from OmniRoute's separation: a 429 cools one model, three transport failures cool the provider, `Retry-After` honoured, windows double to five minutes, cooldowns never leave a task without a route. One import-binding bug had tests measuring the process-wide instance. **794 tests.**
- `776e0f5` **`otto eval-compaction`, and what it found.** Eight planted constraints replayed through each policy with a deterministic summariser: type-blind 3/8 at 120 turns and 0/8 at 400; protected 8/8 at every budget. The fix is one rule: what the person said is never handed to the summariser. The abstract-once mode the plan called for was built and measured at 8/8 either way, so `REABSTRACT` stays on with a number behind it. **803 tests.**
- `4e833d5` `code_map` (from graphify's premise): `define`, `uses`, `imports`, `outline` from Python's own `ast`, no model call, cached per tree state, Python only and says so. **819 tests.**

**SWE-bench, and the harness bugs it hid.**

- `faa4610` SWE-bench Verified: 500 issues, graded by the maintainers' tests; the agent never sees the test names; arm64 images native. Four NP-hard math tasks whose optima were verified exhaustively and whose checkers reject named wrong methods. Golden set 20 items. **836 tests.**
- `882b7d3` README rewritten to describe the agent that exists (it still described the seven-node graph).
- `685d66e` Two harness bugs from a control run on an untouched repo: pytest colour escapes broke status parsing (PASS_TO_PASS 0/141 and 0/179 on healthy repos), and the promised x86_64 fallback did not exist (two thirds of instances "pull access denied"). **841 tests.**
- `a18f0e4` Two more: explicit node ids run nothing if one is stale (0/644 on a healthy repo; now by file, in batches), and grading shared the agent's deadline (a spent budget was scored as "patch would not apply"). Same control after: 0/2 on the bug, 644/644 on everything else. **845 tests.**
- `f896a0c` Windows: `code_map` and `list_files` returned OS-native separators; container-path tests skip on cmd.exe. CodeQL: a URL substring assertion replaced by an exact host comparison.
- `843db4d` The test workflow given a read-only token.
- `5fa3cf0` Empty commit to re-trigger a GitHub-managed scan.
- `20e4ac1` Seven review findings, all reproduced: SSRF in `browse` (`file:///etc/passwd`, the cloud metadata endpoint, obfuscated loopback forms), the mutation-gate `confirmed` set surviving an escalation wipe, a plan destroyed by its own "switch back" instruction, lessons truncated past their parseable tail, a `>` vs `>=` margin, `[-0:]`, and a 294-line docstring describing five nodes that no longer existed. **862 tests.**
- `38e2d51` **72 commits landed together; 845 tests.** Golden 20/20; LoCoMo 91% recalled / 95% answerable over 759 questions; tool use 2/4 mean 0.682; SWE-bench 2 of 3 soundly graded.

**A day of live use.**

- `ac98763`, `ce0484e`, `ec5e595`, `8987316` "It's acting weird and slow": `@work(exclusive=True)` does not stop a thread worker, so a second Enter ran a second pipeline; a modal's Enter escaped into a new turn; `RichLog` inside a collapsed `Collapsible` rendered nothing for a whole turn. Each fix asserted against the installed Textual. And interactive sessions got a real workspace (the file tools had only ever been reachable from a benchmark): the current directory by default, `--workspace`, `--no-workspace` winning over it, 120 s commands with a workspace. TUI tests wait on conditions, never counts. **898 tests.**
- `3fbc5fa`, `e883f30` An `ask_user` answer never reached the loop's transcript, so it asked the identical question again, deterministically, six times in a row. Tests assert on the messages handed to the model. **902 tests.**
- `cda6787` Three questions per turn; past the cap the ask is refused with a way forward; two refused asks end the run; "done" and "nothing else" (whole-answer match, never bare yes/no) spend the remaining asks.
- `f535fbf` Tokens were arriving and being dropped. A per-model ledger recorded at `_call` (retries are real spend), cache reads priced at a tenth (188k of 251k input in the checked case), a dated price table with a JSON override, "--" for unpriced rather than 0.
- `80c3a19` The turn opened with "hi", the answer to the follow-up question was the real request, and 32 calls of correct work were rejected against a checklist about a greeting. The request is now composed from the opening message and every asked/answered pair.
- `871b84d` Three evals stopped reporting numbers that meant nothing: math checkers passing on the right number anywhere (now `answer_is` with decoys), `eval-memory` printing "recalled 0%" as a pass when compaction never fired (now refuses with exit 1), `eval-claw` calling one run a mean (NOT EVIDENCE below three trials), `eval-swe` blending native and emulated timings (split reported, emulated budget scaled).
- `9ac40b0` A failure taxonomy (Recuris) computed from the action record with no model call: `zero_write`, `error_cascade`, `first_call_failed`, `no_actions`, and only those, because the other three need an oracle. Tool cost by tool and by seat.
- `50c59d4` "hi otto!" cost 13 model calls and about five minutes: told to run something that would fail if the task were not done, the loop invented a task and wrote a test file. The rubric call can answer NO TASK; a greeting is answered in one further call on the cheapest seat. Every way of being unsure keeps the old path.
- `2b740ba` Live progress (phase, tool, model, call count, a ticking clock, streaming answer, escape to stop) and the latency bug building it found: the evaluator handed provider failures back to the agent with no bound on the path. **68 requests and 190 s ending in a crash → 5 calls and 33 s.** Prompts audited against published harnesses; an injection hole closed; two test-isolation leaks fixed. **1,180 tests.**
- `8626160` A dropped line left two f-strings referring to `half`; every run with over 6,000 characters of evidence died with `NameError`. A regression test feeds the judge an oversized entry.

## Day 7 — 13 Sep: setup, documents, sessions

- `0350955` The TUI could not configure anything. A setup screen (keys written masked to `.env`, a live probe per vendor, every detected model with capabilities, per-seat pins in `~/.otto/routes.json` with an auto-map proposal), a directory browser, custom OpenAI-compatible endpoints as runtime provider classes, `Router()` no longer raising at construction, lessons and outcomes import/export, a redesign against the terminal agents that list well, and drag-to-copy through OSC 52 and a native command. `tui.py` split into modules. **1,336 tests.**
- `cdbacff` Asked for ten generations with a 500-word narrative each, the loop answered in 300 words in 23 s and was approved; nothing in that was laziness (the prompt asks for numbers, replies stop at `max_tokens`, the rubric may not mention length). The rubric call now says `KIND: research`, and a document goes to a sectioned workflow with a continuity ledger, code-checked lengths and headings, one revision, assembly, and a judge report from the files. **54 calls, 42 minutes, 42,000 words.** Three judge failures exposed and fixed. **1,379 tests.**
- `7a095bd` The judge failed on a model that refuses `temperature`. The Anthropic policy reads published capability flags; an unknown model's refusal is retried without the parameter and remembered in `~/.otto/temperature.json`; a refusal phrased "is deprecated" no longer reads as a retirement. Plus a directory-picker highlight overwriting a typed save path, caught by CI on Windows.
- `80fb070` A session was a uuid and a file only compaction wrote to, so nearly every conversation died with its process; 3,068 empty memory files had accumulated. The queue mirrors its live tiers to a `pending` table and restores from it; a session index; `otto sessions` with list, delete, rename, export, import and prune; `--resume` on both front ends; `Session.close()` for Windows, where an open SQLite handle blocks unlink. **1,441 tests.**
- `39f24ae` A run with a workspace and no container had no browser, so a chess page whose script threw at load was approved on `node --check` alone. A local browser through the existing Playwright driver (`browse open index.html` fails when the page throws; `look` screenshots it); `exercise`, one tool with nine kinds of walkthrough (a page, a served app, a command line, an API, a terminal program, and an app on Android, iOS, macOS, Linux or Windows), each reported step by step as the machine saw it and shown to the judge as code-computed evidence; two once-only holds so a run cannot finish on its own tests or on a launch alone; UTF-8 hardening of the tool layer. Measured: five runs in a row wrote a harness, passed it and finished without using what they built; with the hold, all five walked through. Pages, served pages, a curses app, a prompt, CLIs, an API and an AppKit app were driven for real; the other four device drivers are covered by faked commands and tracked in issues #5 to #8. **1,534 tests.**
- `9bc5e0e` Recording the demo found two things. The REPL was unreadable with `LANGFUSE_BASE_URL` pointing at a collector that was not running: the exporter retried on stderr every couple of seconds. The host is now probed once per process with a one-second TCP connect; unreachable means one warning and no tracing at all. And quitting the TUI mid-turn printed `ValueError: <Token ...> was created in a different Context`, because the stream generator was finalised on another thread. Every binder restores by value when its token cannot be reset, and the streaming entry points recognise the SDK's own version of that error and finish quietly. **1,552 tests.**
- `08c90a3` The README plays the one-minute recording inline and follows it with seven screenshots taken mid-run, with the thinking block open; the cli, pipeline and memory READMEs carry the ones that belong to them.

## Day 8 — 14 Sep: a persistent interpreter

- Issue #1. Every `execute_python` call was a standalone script in a fresh process, so an agent that loaded a dataset or built an index had to rebuild it on the next call, from the transcript or through disk; Prime Agent (arXiv:2608.23552) names the persistent interpreter as the layer that makes long-horizon work compositional. One interpreter per run now, behind the same contextvar seam as the workspace and the command runner, started on the first call and closed with the run. Written tests-first against the three forks the issue asked to settle: a delegated subagent gets its own session, never the parent's; inside a container the tool keeps its one-shot script and says so; a call the interrupt cannot stop is killed and the result says the session restarted. The first run of the suite found the plumbing's own dependency on the builtins: a snippet that rebound `len` broke the shim's `json.loads`, and every call after it timed out silently; builtins are now restored after each call. The acceptance measurement the issue names, Claw-Eval score and `model_calls` with the session on and off (`OTTO_PYTHON_SESSION=0`), needs keys and is not taken here. **1,586 tests.**

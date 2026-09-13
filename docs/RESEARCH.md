# Research the design draws on

Each paper is listed with the finding that shaped Otto and where it landed.
The numbers are the papers' own, as recorded in the module docstrings.

| paper | finding applied | where |
| --- | --- | --- |
| [RefineBench: Evaluating Refinement Capability of Language Models via Checklists](https://arxiv.org/abs/2511.22173) (Lee et al., ICLR 2026) | self-refinement over five turns is 31.3% for the best model and near 0% for most; the same models reach 90–98% given an external checklist | the rubric-first evaluator, `agent/pipeline/nodes.py` |
| [The Art of Building Verifiers for Computer Use Agents](https://arxiv.org/abs/2604.06240) (2026) | a verifier structured as criteria-then-judgment reaches human-level agreement (κ 0.64, inside the 0.53–0.57 human band) with a 0.01 false-positive rate, and the gain is architectural | two-phase judging, non-overlapping checkable criteria |
| [SABER: Small Actions, Big Errors](https://arxiv.org/abs/2512.07850) (2025) | mutating actions are 14–18% of steps and a single mutating deviation cuts success odds by up to 92–96% | the mutation gate, `TOOL_TIERS` |
| [The Verifier Tax: Horizon-Dependent Safety–Success Tradeoffs in Tool-Using LLM Agents](https://arxiv.org/abs/2603.19328) (2026) | enforcement that blocked 94% of non-compliant actions left safe task completion under 5%, because the actor fabricated a way around the block | every hold offers a way forward; the judge treats "blocked" as not "failed" |
| [The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents](https://arxiv.org/abs/2608.24358) (2026) | across 58,000 runs, handing a stronger model a weaker one's trajectory recovers under half the gain; discarding it moves recovery from 47% to 64%, and the reverse hurts | escalate by restarting, de-escalate by carrying |
| [The Compaction Cliff in Long-Running AI Agent Memory](https://arxiv.org/abs/2608.22752) (2026) | type-blind compaction keeps 53% of constraints at 50% compression and 24% at 10%, falling to 10% over five rounds; type-aware holds 96% | type-aware compaction; `otto eval-compaction` |
| [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978) (Zhang et al., 2026) | consolidating its own memories made a model fail 54% of problems it had solved; retaining raw episodes doubles accuracy | no bullet layer on the context tier; lessons distilled from raw trajectories, never from other lessons |
| [Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?](https://arxiv.org/abs/2511.13646) (2025) | asking the agent after each step whether it should build itself a tool took 62% → 76%, and the system to 77.4% on SWE-bench Verified | the periodic tool-building reminder |
| [The Bitter Lesson of Tool Calling](https://arxiv.org/abs/2608.06370) (2026) | programmatic tool calling matches or beats JSON in 11 of 14 models and holds under fan-out where JSON collapses | the `ACTION:` / `CODE:` text protocol |
| [Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline](https://arxiv.org/abs/2601.12307) (2026) | a single agent reusing its KV cache matches homogeneous multi-agent workflows at lower cost; only genuine model heterogeneity justifies sub-agents | one loop with modes; `delegate` refuses your own mode |
| [Can Small Agents Collaborate to Beat a Single Large Language Model?](https://arxiv.org/abs/2601.11327) (2026) | reasoning at the orchestrator was worth +18.2 (GAIA) and +36.7 (AIME) at 8% latency; sub-agent size was flat (23.0 / 23.0 / 23.6) | the delegate child is thin and bounded |
| [CoAct-1: Computer-using Agents with Coding as Actions](https://arxiv.org/abs/2508.03923) (2025) | routing subtasks to code or GUI and preferring code reaches 60.76% on OSWorld in 10.15 steps against ~15 for GUI-only | the desktop image keeps a shell; `look`/`look_act` are the fallback |
| [Beyond Browsing: API-Based Web Agents](https://arxiv.org/abs/2410.16464) (ACL Findings 2025) | API plus browser beats browsing alone by 24 absolute points | `browse` is the fallback to `execute_bash` |
| [AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents](https://arxiv.org/abs/2410.13825) (ICLR 2025) | refining only the observation and action space beat every scaffolding trick by +9.8 points (+29.4%) | pages come back as a digest, never raw DOM; a small action vocabulary |
| [Building Effective AI Coding Agents for the Terminal](https://arxiv.org/abs/2603.05344) (OpenDev, 2026) | long runs suffer instruction fade-out; event-driven reminders in the conversation work where rewriting the system prompt does not | periodic checklist and tool-building reminders on tool results already being sent |
| [Recursive Experiential–Working Memory Evolution for Long-Horizon Agent Harnesses](https://arxiv.org/abs/2608.24876) (Recuris, 2026) | a structured trace localises a fault 64.8% of the time against 13.0% from the outcome alone; a named failure-mode taxonomy | `agent/eval/failures.py`, the tags decidable from the action record |
| [Agent-as-a-Router: Agentic Model Routing for Coding Tasks](https://arxiv.org/abs/2606.22902) (2026) | routing on logged per-task outcome statistics is worth +15.3% relative | outcome-based reordering of fallback chains, `agent/router/outcomes.py` |
| [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/abs/2607.12227) (2026) | harness evolution does not consistently beat repeated sampling under matched budgets | `--trials`, matched baseline arms, learning off in the baseline |
| [DarwinX: Evolving Agent Harnesses Through Natural Selection](https://arxiv.org/abs/2608.07545) (2026) | a 31.7-point gap between the proxy the search maximised and held-out truth | `--split holdout`, read-only lessons on the held-out split |
| [SEA-Eval: Evaluating Self-Evolving Agents Beyond Episodic Assessment](https://arxiv.org/abs/2604.08988) (2026) | identical success rates hide up to 31x differences in token cost on a stream of related tasks | cost recorded beside every score |
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
(the over-build ladder in the agent prompt, measured at 54% fewer lines and
22% fewer tokens with safety held at 100%), **Terminal-Bench** and
**Claw-Eval** (the container harnesses).

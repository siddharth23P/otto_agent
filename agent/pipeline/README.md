# agent/pipeline/

The graph is two nodes, `agent -> evaluator`, with an `ask_user` pause. The
agent works the task in one conversation and switches mode when the kind of
work changes; the evaluator judges the result against criteria written
before the attempt existed.

| module | what it is |
| --- | --- |
| `nodes.py` | the agent loop, the evaluator, the rubric call, the prompts, the gates, compaction, delegation |
| `run.py` | the entry points: `run_pipeline`, `run_pipeline_stream`, `resume_pipeline_stream`; binds the budget, store, workspace and usage ledger for a run |
| `state.py` | `AgentState`: what one run carries between the two nodes |
| `modes.py` | the four modes as data: a routing seat plus that role's guidance |
| `tools.py` | the tool registry, tiered by reversibility, and the tools themselves |
| `toolkit.py` | tools bound for one run only (a benchmark's own tools) |
| `workspace.py` | the bound workspace directory and path confinement |
| `execution.py` | the command-runner seam: where a shell command actually runs (host or container) |
| `evidence.py` | a ledger of what the run proved, and the three once-only holds it drives |
| `budget.py` | the per-run model-call ceiling, counted at the one point every request passes through |
| `usage.py`, `pricing.py` | tokens per model, and what they cost |
| `progress.py` | what the run is doing right now, for whoever is watching |
| `research.py` | the document workflow |
| `browsing.py`, `screen.py`, `vision.py` | the browser, the desktop, and the one multi-block message |
| `walkthrough.py`, `native.py` | `exercise`: using what was built, on a page, a server, a shell, a terminal, or a device |
| `codemap.py`, `rag.py` | where a name is defined; semantic search over workspace files |

## What a run does

![a run mid-judgment: the trace, the evidence hold, the exercise walkthrough, the verdict](../../docs/media/run-trace-and-judge.png)

1. **The rubric.** One call writes what a correct answer must contain, from
   the task alone, before any attempt exists. It is the only information in
   the judgment the actor did not produce: a verifier that re-reads the
   actor's own output measures at approximately nothing, while the same
   models reach 90–98% given an external checklist. The same call says when
   there is no task (a greeting is answered in one further call on the
   cheapest seat, two calls for the turn) and when the task is a document
   (`KIND: research`, see below). The checklist lives in state, the loop works
   against it, and only the judgment may move a criterion's status.

2. **The loop.** A text protocol, `ACTION:` then `CODE:`, one tool call per
   reply. A reply is validated on the emit path: the tool name is resolved
   from prose, backticks, bold or a typo, a call is checked against its
   declared schema before dispatch, an empty body is refused by name, and
   whichever of `ACTION`/`FINAL` comes first wins. Old tool results are
   compacted in place to their one-line `actions` summary past a threshold,
   at no model call, with the full text stored for `recall_memory`.

3. **Modes.** `solve`, `plan`, `summarize`, `find`: each a routing seat and a
   short guidance block. Escalating to a deeper mode restarts from the task
   and the checklist and carries only the last thing produced; de-escalating
   carries the whole conversation. Handing a stronger model a weaker one's
   trajectory recovers under half the gain at four to six times the cost,
   while discarding it moves recovery from 47% to 64%.

4. **Delegate.** One bounded job on a different mode's model, one level,
   eight exchanges. A contract goes down, a report comes back, the child's
   working is discarded and its action record joins the parent's. Delegating
   to your own mode is refused: a single agent with a shared model matches
   the multi-agent version at lower cost.

5. **The holds.** Each fires once and offers a way forward, because a block
   with no way through is what makes an agent invent the thing it was
   denied.
   - The **mutation gate** holds the first call to an irreversible tool
     against a target and asks for the evidence that identifies it. Mutating
     actions are 14–18% of steps and one mutating mistake cuts success odds
     by 55–96%. Run-scoped tools default to gated.
   - The **evidence gate** holds an answer that changed code with nothing
     passing since; prose is exempt and "nothing to run" is accepted.
   - A turn that edited a page and never loaded it, or finished on its own
     tests without using what it built, is held once and asked to `exercise`
     the thing.

6. **The evaluator.** Scores the attempt against the checklist, as a share
   of criteria met, and separates "not met" from "blocked by the environment"
   so an environment blocker does not invite an identical retry. It may check
   one thing with a tool, within two exchanges (four for a document).
   Rejections are capped at two; past that the answer stands, said plainly
   as unverified. It sees the action record, the modes used, the tail of the
   loop's working, and the code-written walkthrough report when one exists.

7. **Lessons.** A finished run distils at most three transferable lessons on
   the plan seat; the next run reads at most one, only when relevant.

Periodic reminders ride on tool results already being sent: what is still
open on the checklist, and whether the agent should build itself a tool.
Periodic, not constant, because said every turn they become wallpaper.

## Tools

![todo.py written, exercised in three steps, judged 3/3](../../docs/media/exercise-todo-cli.png)

| tool | tier | does |
| --- | --- | --- |
| `execute_bash`, `execute_python` | read-only | a shell command or a script in the workspace or the container; nothing may detach, long output keeps both ends |
| `read_file`, `list_files` | read-only | line-numbered reads with ranges; listings that skip `.git` and friends |
| `write_file`, `edit_file` | workspace | writes into the bound root only; `edit_file` matches through a cascade (exact, trailing whitespace, indentation, first-and-last-line anchor) and still demands a unique hit |
| `code_map` | read-only | `define`, `uses`, `imports`, `outline` from Python's `ast`, cached per tree state, no model call |
| `rag` | read-only | semantic search over the workspace's files, on the memory engine |
| `recall_memory` | read-only | semantic search over what this session said and what was compacted away |
| `view_image` | read-only | a vision model answers a question about an image file; words travel, never pixels |
| `web_search` | read-only | Anthropic's server-side search; unconfigured is a clean failure, never an answer from memory |
| `browse`, `browse_act` | read-only, mutating | a page digest (url, title, headings, links, fields, clipped text), and clicks and typing behind the gate |
| `exercise` | read-only | a whole sequence against a page, a served app, a shell, an API, a terminal program or a device app, reported step by step as the machine saw it |
| `look`, `look_act` | read-only, mutating | the container desktop through the vision model; clicks and typing behind the gate |
| `complete_code`, `predict_edit` | read-only | Mercury's fill-in-the-middle and edit endpoints |

Every shell and file tool runs either on the host workspace or inside a task
container through `execution.py`'s command runner, so the tools never learn
what a container is. Content crosses into a container base64-encoded, never
in a heredoc. The tier invariant is a test: anything irreversible is held
before it runs, and a tier that disagrees with the gate fails the suite.

## The document workflow

When the rubric call says the deliverable is a long written document, the
run goes to `research.py` rather than the loop: an outline on the plan seat,
one plain prose call per section on the reason seat with no tool protocol,
a JSON continuity ledger of named things and settled facts carried section
to section (trimmed from the middle so founding facts survive), word counts
and required sub-headings checked in code with one revision, assembly into
`otto_research/<task>/document.md` (and docx, pdf or xlsx when asked), and a
report computed from the files for the evaluator. Gathering workers are
bounded find-mode loops.

## Budget and usage

`budget.py` counts model requests at `_call`, retries included, because the
empty-stream retry and the diffusion doubling are real requests. Default 120
per turn, `OTTO_MAX_MODEL_CALLS` overrides, and a harness's own deadline wins.
Exhaustion returns the best answer so far rather than raising. `usage.py`
records tokens per model at the same point; `pricing.py` prices them from a
dated table with cache reads separated, and reports "unknown" rather than 0
for anything it cannot price.

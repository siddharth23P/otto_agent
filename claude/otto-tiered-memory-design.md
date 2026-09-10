# Otto short-term memory — the X/Y tiered queue (2026-09-10)

Not to be confused with `otto-memory-design.md` (2026-09-09), which is Otto's long-term learning system — project profile, cross-project lessons, community sharing. This doc is short-term memory: what keeps a single session's conversation and a single task's in-progress context from growing without bound inside one model's context window. Verbatim spec from the person, preserved because every design choice below traces back to it:

> Otto short term memory is like a queue with X+Y space. Once X gets filled memory moves to Y. When Y gets filled all memory in Y is flushed and stored with a hash in a DB. Then we take all the memory in Y and make a summarized bullet list and place it in the last slot of Y with each bullet having hash of reference to DB. Once it gets filled again we take all and do the same but we dont map any hash from before in this list unless it's a bullet point. A bullet point can have multiple hash associated with it. [...] If agent decides it needs more details on something finder will find the relevant stuff by first pulling all the hash and compiling the list then finding relevant information (not all) and sending that as new X for agent to work on with. The goal is to make it seem that the agent has infinite memory while making sure we are in smart zone of model context at all time.

## Status

Phase 1 — the standalone engine — is built, tested, and committed (`agent/memory/`, commit `56a58ef`, 50 new tests).

Phase 2 — wiring the `kind="history"` instance into the graph and CLI — is built, tested, and committed (`agent/memory/session.py`, `agent/memory/wiring.py`, commit `8a791a2`). `otto chat`/`otto tui` now bound each session's conversation history to a real `TieredQueue`; a `recall_memory` tool exists and is wired into every role prompt. See "What's built (Phase 2)" below for exactly what changed and what's still open.

A LoCoMo-based evaluation script is also built, tested, and committed (`agent/eval/memory_bench.py`, `otto eval-memory`, commit `71d1f59`) — see "Evaluation" below for what it measures and what a real run found.

A citation-coverage bug found by that script is fixed and committed (`agent/memory/queue.py`), along with the retrieval rework that followed from it — see "Finding: uncited items were silently unreachable" below.

Still not started: the `kind="context"` instance (in-turn context/board growth across unbounded overseer rounds, `agent/pipeline/nodes.py`) — see item 1 under "What's still pending."

## Scope: one engine, two independent uses

Confirmed with the person rather than assumed: conversation history growing across turns (`agent/pipeline/run.py`'s `history` parameter) and in-turn context/board growth across unbounded overseer rounds (`agent/pipeline/nodes.py`) both get the same `TieredQueue` mechanism, as two separate instances rather than one merged pool — a `kind` string (`"history"` vs `"context"`) namespaces them in the store so mixing conversation turns with task-internal scratch work never happens.

## Two tiers, then permanent storage

* **X** — small, in-memory, holds the most recent items verbatim. What a prompt reads as "the detailed, recent part."
* **Y** — larger, in-memory. Once X's own token budget is exceeded, X's entire current content moves to Y in one step (not a trickle) and X starts over empty.
* **DB** (SQLite, one file per session, `~/.otto/memory/<session_id>.db`) — once Y's own token budget is exceeded (raw items plus whatever bullets are already sitting in Y from an earlier round), everything in Y is compacted in one step:
  1. Every raw item in Y is flushed to a `chunks` table, keyed by its sha256 content hash — permanent, content-addressed, never rewritten (`agent/memory/hashing.py`).
  2. The whole of Y — raw items and any prior bullets, each shown as one numbered item — goes to an injected `summarize` callback, asked to return one bullet per line, each ending in `[sources: N,N,...]`.
  3. Each new bullet's `hash_refs` is built from what it cites: a cited raw item contributes its own single hash; a cited prior bullet contributes its own `hash_refs` unchanged — never re-hashed as a new leaf. This is the spec's "we dont map any hash from before in this list unless it's a bullet point," and it's what lets one bullet transitively cover many hashes generations later while every `hash_refs` entry still resolves to real, permanently stored text.
  4. Every item the summarizer did *not* cite gets its own extra bullet anyway, so the new generation's `hash_refs` collectively cover everything that compaction flushed. Added after the finding below; the prompt asks for full coverage, this is what makes it true.
  5. Y is replaced by just that new bullet list. The store's older generation of bullets for that `kind` is marked `superseded`, not deleted — an audit trail, not live state.

`current_view()` is what a prompt actually reads: Y's bullets (oldest, compact), then Y's raw overflow not yet compacted (still verbatim), then X's raw items (newest) — never the DB directly. `earlier_view()` (added in Phase 2) is `current_view()` minus X's own "RECENT:" section — everything older than X, on its own — and `recent_items` is X's own items as a plain list rather than pre-formatted text. Both exist so a caller (Phase 2's `agent/memory/wiring.py`) can put "recent, verbatim" and "older, likely compacted" in two different places in its own prompt, rather than re-parsing `current_view()`'s single blob back apart.

## Budget: 40% of 260K, per the person's explicit sizing call

Every conversational graph node — router, planner, solver, summarizer, finder, evaluator — shares one model, `mercury-2.5` (`agent/router/mapping.py`'s `TASK_ROUTES`, checked directly: all pinned to `inception:mercury-2.5`), whose context window is 260,000 tokens. The person's instruction was explicit after an initial smaller proposal was rejected: "make sure it's 40% of 260K window."

```
MERCURY_2_5_CONTEXT_WINDOW = 260_000
TARGET_FRACTION            = 0.40
TOTAL_BUDGET               = 104_000   # 40% of 260K
X_BUDGET                   = 24_000    # first-pass split, tunable
Y_BUDGET                   = 80_000    # TOTAL_BUDGET - X_BUDGET
```

The X:Y split (roughly 1:3) is the implementer's own first pass, not a figure the person specified — flagged as a module constant meant to be tuned from real usage. Mercury/FIM/edit models were also named (fim/edit 32K, mercury-2 128K) but don't factor into this budget: FIM/edit calls receive only a `CODE:` body, never the conversation, so nothing here applies to them.

## Retrieval — `finder`'s recall path

Per the spec's "finder will find the relevant stuff by first pulling all the hash and compiling the list then finding relevant information (not all)": `agent/memory/retrieval.py`'s `recall(store, kind, query, top_k)` ranks the current generation's bullets by cosine similarity between the query and each bullet's stored embedding, pulls back only the underlying chunks those top-`k` bullets cite (not the whole store), and returns them as readable text — meant to become a fresh slice of `X` for the agent to keep working with, exactly as the spec describes. Falls back to the most recent bullets, unranked, whenever the embedding model can't be reached.

As of Phase 2, this is reachable mid-task: `agent/pipeline/tools.py`'s `recall_memory` tool calls it directly, using whichever session's `MemoryStore` is bound for the current run (`agent/memory/session.py`'s `bind_store()`/`current_store()`, a `contextvars`-based binding — see "What's built (Phase 2)").

## Embedding source: local and offline, confirmed with the person

Inception Labs has no embeddings endpoint (checked their docs directly — chat, FIM, edit, and model-listing are the only REST families they document). This repo also made a deliberate, documented call to go Inception-only for every LLM call. Reaching for a hosted embeddings API would have reopened that decision; a local model doesn't, since an embedding isn't an LLM call. Confirmed with the person via a targeted follow-up question rather than assumed. Chose `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim, ONNX-based, ~50MB of deps, no torch) — lazily loaded, so a process that never calls `recall()` never loads it. Its first-ever use needs network access once, to fetch the model from Hugging Face Hub (cached afterward). Both this and `tiktoken`'s `cl100k_base` (used for approximate token counting — Mercury's own tokenizer isn't public) degrade gracefully rather than crash when that first fetch can't reach the network: `EmbeddingUnavailable` triggers `retrieval.py`'s unranked-recent fallback, and `tokens.py` falls back to a `len(text) // 4` heuristic. Confirmed as a real, useful resilience feature (not just a workaround) — this was actually exercised during development, since the sandbox this was built in blocks Hugging Face Hub and OpenAI's tokenizer blob storage at the network layer; a real `otto chat`/`otto tui` run on the person's own machine, with ordinary internet access, downloads and caches both once — confirmed for real during the Phase 2/eval-script work on the person's own Mac (`otto eval-memory`'s stress-test run there actually loaded and used the real `fastembed` model, onnxruntime and all).

## Storage: session-owned SQLite, not LangGraph state

`MemoryStore` (`agent/memory/store.py`) is plain Python/SQLite that `summarizer`/`finder` code reads and writes directly — deliberately not a LangGraph state channel. Every external turn's own graph thread is disposable by design (`agent/pipeline/run.py`'s `_graph_thread_id`), so anything meant to survive past one turn has to live somewhere that isn't thrown away with it. Lives at `~/.otto/memory/<session_id>.db`, sibling to the `~/.otto` convention `agent/cli/output.py` already establishes for internal Otto state (ledger/profile), as distinct from `otto_output/`, which is for the person to open. As of Phase 2, `session_id` comes from `agent/cli/shell.py`'s `Session` (a `uuid.uuid4().hex` generated once per session, reset on `otto new`).

## What's built (Phase 1)

`agent/memory/` — six modules, fully unit-tested, no live LLM or network required to exercise:

* `tokens.py` — approximate token counting (tiktoken + graceful fallback).
* `hashing.py` — sha256 content-addressing.
* `store.py` — SQLite-backed `MemoryStore` (chunks + per-`kind` bullets).
* `embeddings.py` — local `fastembed` wrapper, `EmbeddingUnavailable`.
* `queue.py` — `TieredQueue`: the X/Y mechanics, citation-based compaction, `current_view()`/`earlier_view()`/`recent_items`.
* `retrieval.py` — `recall()`: embedding-ranked search over current bullets with graceful fallback.

`TieredQueue.__init__` takes `summarize` as a plain injected `str -> str` callable rather than importing `agent.pipeline`/`agent.router` directly — this package has zero dependency on the graph or the router, so it never needs a live model to test (a real caller wires it to `ROUTER.chat_model(Task.SUMMARIZE, ...)`).

## What's built (Phase 2)

Two new modules bridge the standalone engine to a live session — the one deliberate exception to Phase 1's "zero dependency on `agent.pipeline`/`agent.router`" rule, since bridging the two is their whole job:

* `agent/memory/session.py` — `bind_store()`/`current_store()`, a `contextvars`-based binding so `agent/pipeline/tools.py`'s `recall_memory` tool (a plain `str -> ToolResult` function with no visibility into session state) can find out which session's `MemoryStore` is active for the current run. Its own tiny module specifically to avoid a circular import: `agent/pipeline/tools.py` is imported by `agent/pipeline/nodes.py`, so it can't import `wiring.py` (which imports `nodes.py`) without closing a cycle; this module imports only `agent.memory.store`.
* `agent/memory/wiring.py` — `new_history_queue()`, `record_turn()`, `history_for_graph()`, and the `summarize_for_memory()` callback that routes a `TieredQueue`'s compaction through the same `ROUTER.chat_model(Task.SUMMARIZE, ...)` path the summarizer role already uses. `history_for_graph()` is the actual split that keeps `agent/pipeline/nodes.py`'s existing prompt-building code (`_conversation_so_far()`) completely unchanged: X's `recent_items` become real `HumanMessage`/`AIMessage` objects (round-tripped via `"you: "`/`"otto: "` prefixes chosen to match `_conversation_so_far()`'s own speaker labels) fed as `history`, exactly as before Phase 2; Y's `earlier_view()` becomes a new `memory_context: str` parameter that seeds `state["context"]` — reusing the existing "CONTEXT GATHERED SO FAR:" prompt section every role already has, rather than inventing a new one.

Also landed as part of Phase 2:

* `agent/cli/shell.py`'s `Session` now owns a `history_queue: TieredQueue` (built in `__post_init__`, reset by `otto new`) instead of a flat, unbounded `history: list[BaseMessage]`.
* `agent/pipeline/tools.py` gained a seventh tool, `recall_memory` — added to every role/evaluator prompt's `ACTION:` enumeration (an existing test, `tests/test_prompt_tool_sync.py`, asserts every dispatchable tool appears in every prompt), with the strongest phrasing in `FINDER_PROMPT` per the spec's own "finder will find the relevant stuff" framing.
* `agent/pipeline/run.py`'s `run_pipeline`/`run_pipeline_stream`/`resume_pipeline_stream` all open the session's `MemoryStore` and wrap the run in `bind_store()`, so `recall_memory` has something to find.

333 → 351 tests passing across Phase 2 + the eval script; 362 as of the citation-coverage fix below.

## Evaluation

`agent/eval/memory_bench.py` / `otto eval-memory` (commit `71d1f59`): replays a real long-conversation dataset — LoCoMo (`snap-research/locomo`, arXiv:2402.17753, "Evaluating Very Long-Term Conversational Memory of LLM Agents"; raw JSON at `raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`, ~2.8MB, no auth/license gate) — through a `TieredQueue`, and scores every QA item's cited evidence for:

* `store_coverage` — is it reachable at all (verbatim or as a permanent chunk)? An engine-correctness check, not a retrieval-quality one; should always be ~100% — a miss here means the engine actually lost information.
* `visible_verbatim_coverage` — is it still sitting un-compacted in the live view, so a prompt reading `current_view()` already has it with no `recall()` call needed?
* `recall_coverage` — does `recall()`, given only the question text, actually surface it via semantic search once it's been compacted away?
* `answerable_coverage` (`visible_verbatim` OR `recalled`) — the one number that answers "is this still findable at all, through either path."

Real finding, worth keeping in mind: real LoCoMo conversations are 11,000–24,000 tokens (369–663 turns) — comfortably inside Otto's real production budget (`X_BUDGET + Y_BUDGET = 104,000`). At that budget, compaction never fires for any LoCoMo conversation: `store_coverage`, `visible_verbatim_coverage`, and `answerable_coverage` all score 100%, `recall_coverage` stays 0% because there's nothing compacted yet to recall. This is correct engine behavior, not a bug — but it also means a production-budget run alone never exercises the compaction+recall code path at all. `run_one_conversation()`/`run_benchmark()` take optional `x_budget`/`y_budget` overrides (`otto eval-memory --x-budget/--y-budget`) specifically to force it: a real run on the person's own Mac, at a deliberately tiny budget with the real local `fastembed` model (not the cloud sandbox's fallback path), held `store_coverage` at 100% throughout while `recall_coverage` rose from 0% to ~39% (`visible_verbatim_coverage` dropped to ~19% as expected) — confirming the compaction+recall mechanism actually works once something has genuinely been compacted away, not just in the offline canned-summarizer/no-embeddings fallback path.

Two summarizer backends, `otto eval-memory --live`/`--no-live` (default off): offline uses a deterministic, no-network `_canned_summarize` (still exercises the engine's own citation-propagation and budget mechanics, just with placeholder bullet text); `--live` routes through the real `agent.memory.wiring.summarize_for_memory` (`Task.SUMMARIZE`), needing `INCEPTION_API_KEY`.

## Finding: uncited items were silently unreachable

A harder live sweep than the ~39% one above — 250 turns at `--x-budget 200 --y-budget 400`, forcing 19 compaction generations instead of a handful — showed `recall_coverage` collapsing to roughly 10%, with `recall()` returning the same few generic bullets for nearly every one of 101 unrelated questions. `unparsed_bullet_count` was 0, so the unparsed-reply fallback was not the cause.

**Cause.** Compaction built each new bullet's `hash_refs` from exactly the item numbers the summarizer's reply cited, and nothing checked that every item got cited by something. The prompt asked for full coverage; the model was free not to comply. An omitted item's raw text still reached the permanent chunk store, so `store_coverage` stayed at 100% and the miss was invisible to that metric, but no bullet in the live generation pointed at it any more. Since `recall()` only ever searches the live generation (older ones are `superseded`), that text became unreachable by any path short of a direct hash lookup, which nothing performs.

The damage was a cliff rather than a proportional loss because a prior bullet is itself one of the numbered items, carrying every hash accumulated behind it. One compaction that folded a prior bullet into prose without citing its number severed the whole chain at once.

**Measured**, replaying one LoCoMo conversation at 250 turns and `x=200`, `y=400` (19 generations, 249 stored chunks):

| Summarizer | Chunks reachable from live bullets | `recall_coverage` |
| --- | --- | --- |
| Cites every item | 249 / 249 | 100% |
| Cites ~70% of items | 9–11 / 249 | 2–3% |
| Cites ~70%, after the fix | 249 / 249 | 47% |

The offline benchmark could not see this on its own: `_canned_summarize` cites perfectly by construction, so only a real model in the loop ever omitted anything.

**Fix** (`_uncited_bullets()`, `agent/memory/queue.py`, commit `bcfdf97`). After parsing the summarizer's reply, every item index no bullet cited gets its own extra bullet. An uncited prior bullet is carried forward exactly as it was, text and `hash_refs` both. An uncited raw item becomes a bullet holding a truncated excerpt of its own text, which keeps the distinguishing words `recall()` ranks on rather than hiding it in a contentless catch-all. Cost: Y drains less completely when the summarizer omits a lot — the same replay went from 19 compactions to about 25, and from 3–4 live bullets to about 9. More, more specific bullets is the better trade for retrieval.

The live sweep's `recall_coverage` went from ~10% to 99% after the fix. That number is not as good as it looks — see below.

## Recall precision: what was wrong and what it is now

The 99% above was a dump, not retrieval. `recall()` expanded a matched bullet into the full text of every hash it cited, with no cap, and because compaction accumulates citations forward, a late-generation bullet cites nearly everything. Across all 101 questions exactly **two** distinct bullets ever surfaced and every call returned 38,033 characters — very nearly the whole conversation. It scored a hit for the same reason a real prompt would drown: the answer was in there because everything was.

Three changes, each measured on the same 250-turn replay.

**1. `recall()` is now two stages** (`agent/memory/retrieval.py`), which is what the spec described all along — "first pulling all the hash and compiling the list then finding relevant information (not all)". Stage 1 compiles the candidate list from every live bullet's `hash_refs`. Stage 2 ranks those raw chunks by their own stored embedding (written at flush time, `chunks.embedding`) and returns only the best handful.

Ranking the *bullets* first to narrow the candidate set was tried, and it is a trap worth naming so it does not get reintroduced as an efficiency win. Compaction does not produce comparable bullets: on a real live replay the store held seven, six citing one to three chunks each and one citing 237. Which bullet a question matched had almost nothing to do with where its answer was.

| Stage 1 | Coverage on the live store |
| --- | --- |
| Narrow to the best 3 bullets | 5.0% |
| Narrow to the best 5 | 67.3% |
| Search every live bullet's hashes | 90.1% |

Bullets summarize; they do not index. `top_k` now only controls how many bullet summaries print as context above the results.

**2. Each hit comes back with its neighbours.** A dialogue turn is a poor retrieval unit — the answer to "when did she join the group?" is routinely in the turn *after* the one that names it. `chunks.seq` records flush order so `chunks_near()` can find them. This was the single largest win available, and widening the window buys coverage more cheaply than returning more hits.

**3. We now send the query prefix.** `BAAI/bge-small-en-v1.5` is trained asymmetrically: a stored passage is embedded as-is, but a query is meant to arrive behind `"Represent this sentence for searching relevant passages: "`. We were embedding questions as plain passages. `embed_query()` is a separate function precisely so documents never get the prefix. Worth about four points for nothing.

Measured against the live store, scoring how often `recall()` surfaces a question's cited evidence and how much text it returns to do it:

| max_chunks | neighbours | Coverage | Chars returned |
| --- | --- | --- | --- |
| 5 | 0 | 65.3% | 922 |
| 5 | 1 | 79.2% | 2,112 |
| 5 | 2 | 84.2% | 3,189 |
| 10 | 2 | 90.1% | 6,293 |
| 20 | 2 | **96.0%** | 11,526 |
| 40 | 2 | 98.0% | 19,660 |
| uncapped (before) | — | 99.0% | 38,033 |

Shipped defaults are 20 chunks, window 2: **96% coverage at about 2,600 tokens**, under 3% of `TOTAL_BUDGET`, against 99% for the whole conversation. The last two points cost more than the first eighty.

**A token budget is the ceiling that actually holds.** Both knobs above are counts of items, which is only a proxy for size. LoCoMo's dialogue turns average ~150 characters; an Otto turn is a person's whole message or a full assistant reply and can be thousands, so the same 20 items could be ten times the text here. `DEFAULT_TOKEN_BUDGET` (3,000) is spent best-first, so a session of long turns returns fewer, longer items rather than blowing the budget, and the budget binds on the least relevant material. On LoCoMo it barely binds at all (11,459 characters against 11,526 unbounded), which is what a safety ceiling should look like.

Confirmed end to end on a fresh `--live` run (a different summarizer pass, so a different set of bullets — which is the point, now that bullet quality no longer gates what can be found):

| Category | n | store | recalled |
| --- | --- | --- | --- |
| single-hop | 28 | 100% | 96% |
| temporal | 24 | 100% | 92% |
| multi-hop | 8 | 100% | 88% |
| open-domain | 41 | 100% | 100% |
| **overall** | **101** | **100%** | **96%** |

Temporal and multi-hop are the weakest, which is what you would expect: both need more than one turn, and multi-hop needs turns that are far apart, so neighbour expansion does not help them the way it helps the rest.

All of it is tunable from the CLI: `otto eval-memory --max-chunks --neighbours --token-budget`. `run_one_conversation(keep_store=True)` leaves the SQLite file behind so a live replay can be re-scored under different settings without paying for another pass of the summarizer — the expensive part of a `--live` run is producing the chunks and bullets, not scoring them.

## What's still pending

1. In-turn `context`/`board` growth (`agent/pipeline/nodes.py`) — currently no cap across overseer retry rounds. Needs a `TieredQueue(kind="context", ...)` per run, which needs a place to persist a live `TieredQueue` across multiple node calls within one LangGraph run — `history`'s per-`Session` object didn't have to solve this (a `Session` already lives for the whole CLI process); still open.
2. The `ask_user` Q&A history gap (found while answering an earlier memory question, pre-Phase-2): a question asked mid-turn via `ask_user` and its answer aren't yet synced back into the history queue for future turns. Worth revisiting — may be naturally subsumed by how `record_turn()` is called rather than needing a separate fix.
3. Tuning `X_BUDGET`/`Y_BUDGET` themselves from real usage, per the person's own framing of the 104K figure as "use this as a first pass" — the LoCoMo finding above (real conversations never fill even the current budget) suggests there's real headroom to learn from once Otto sees heavier day-to-day use, though it doesn't by itself argue for changing the numbers.
4. Lexical (BM25) blending was measured and **rejected** on this evidence: it moved top-5 from 65.3% to 51.5%. LoCoMo questions paraphrase rather than quote, so exact-token overlap mostly adds noise. It may still help for dates and proper nouns, which is a narrower use than a global blend, and would need its own evidence.
5. Every number here comes from one LoCoMo conversation at a deliberately tiny budget. The defaults are tuned to that shape of data; a real Otto session has far longer, far fewer turns, and the count knobs in particular should be re-checked once there is real usage to replay.

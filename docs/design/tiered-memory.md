# Otto short-term memory: the X/Y tiered queue

This is the design of Otto's short-term memory: what keeps a session's
conversation, and one turn's tool output, from growing without bound inside a
model's context window, while keeping everything retrievable. Long-term
learning (the lesson bank, `agent/memory/lessons.py`) is a separate mechanism
and is not covered here.

The design started from this spec, preserved verbatim because every choice
below traces back to it:

> Otto short term memory is like a queue with X+Y space. Once X gets filled memory moves to Y. When Y gets filled all memory in Y is flushed and stored with a hash in a DB. Then we take all the memory in Y and make a summarized bullet list and place it in the last slot of Y with each bullet having hash of reference to DB. Once it gets filled again we take all and do the same but we dont map any hash from before in this list unless it's a bullet point. A bullet point can have multiple hash associated with it. [...] If agent decides it needs more details on something finder will find the relevant stuff by first pulling all the hash and compiling the list then finding relevant information (not all) and sending that as new X for agent to work on with. The goal is to make it seem that the agent has infinite memory while making sure we are in smart zone of model context at all time.

The doc was written on 10 September against a seven-node graph with a
`finder` role and a single vendor. The agent has since become one loop with
modes over four vendors, and the memory system grew with it. Everything below
describes the system as it is on `main`; where a measurement was taken under
the older shape, it says so. Commit hashes are on the current `main`.

## What exists, and where it landed

| piece | where | commit |
| --- | --- | --- |
| the standalone engine: `TieredQueue`, `MemoryStore`, hashing, tokens, embeddings, retrieval | `agent/memory/` | `d190b72` |
| the history tier wired into `otto chat` / `otto tui`, and the `recall_memory` tool | `agent/memory/wiring.py`, `session.py`, `agent/pipeline/tools.py` | `8ca337f` |
| `otto eval-memory` on LoCoMo | `agent/eval/memory_bench.py` | `71364e8` |
| every compacted item stays reachable (recall ~10% → 99%) | `_uncited_bullets`, `agent/memory/queue.py` | `3437a03` |
| two-stage recall with neighbours and a token budget (→ 96% at a tenth of the text) | `agent/memory/retrieval.py` | `d044ed8` |
| swappable, stamped embedding backends | `agent/memory/embeddings.py`, `store.py` | `7934cc1` |
| `gemini-embedding-001` as the measured default, local BGE as the floor | `agent/memory/embeddings.py` | `983747e` |
| a narrow read policy for mid-task recall | `agent/memory/retrieval.py` | `4c89f58` |
| the context tier: evicted tool output stored and searchable | `agent/pipeline/nodes.py`, `retrieval.recall_chunks` | `1f5249f` |
| type-aware compaction, and `otto eval-compaction` to prove it | `agent/memory/queue.py`, `agent/eval/compaction_bench.py` | `c607b3a` |
| the live tiers mirrored to disk, so a session can be resumed | `MemoryStore.pending`, `TieredQueue(restore=True)`, `agent/memory/sessions.py` | `da9ada9` |

## Two uses of one engine

The same `TieredQueue` mechanism serves two things, as two instances rather
than one pool, namespaced in the store by a `kind` string so conversation
turns and task-internal scratch never mix:

- `kind="history"`: the conversation across turns, owned by the CLI's
  `Session`.
- `kind="context"`: tool output evicted from the agent loop's own conversation
  within a turn. This tier has no bullet layer (see "The context tier").

## Two tiers, then permanent storage

- **X**: small, in memory, holds the most recent items verbatim. What a prompt
  reads as "the detailed, recent part".
- **Y**: larger, in memory. Once X's token budget is exceeded, X's entire
  content moves to Y in one step and X starts over empty.
- **DB** (SQLite, one file per session, `~/.otto/memory/<session_id>.db`):
  once Y's budget is exceeded, everything in Y is compacted in one step:
  1. Every raw item in Y is flushed to a `chunks` table keyed by its sha256
     content hash: permanent, content-addressed, never rewritten.
  2. The whole of Y, raw items and prior bullets each shown as one numbered
     item, goes to an injected `summarize` callback asked to return one bullet
     per line, each ending in `[sources: N,N,...]`.
  3. Each new bullet's `hash_refs` is built from what it cites: a cited raw
     item contributes its own hash; a cited prior bullet contributes its own
     `hash_refs` unchanged, never re-hashed as a new leaf. This is the spec's
     "we dont map any hash from before in this list unless it's a bullet
     point", and it lets one bullet cover many hashes generations later while
     every entry still resolves to real stored text.
  4. Every item the summariser did not cite gets its own extra bullet anyway,
     so the new generation covers everything compaction flushed. Added after
     the finding below; the prompt asks for full coverage and this is what
     makes it true.
  5. Y is replaced by the new bullet list. The store's previous generation of
     bullets is marked `superseded`, not deleted: an audit trail, not live
     state.

`current_view()` is what a prompt reads: Y's bullets (oldest, compact), then
Y's raw overflow not yet compacted, then X's raw items (newest), never the DB
directly. `earlier_view()` is everything older than X on its own, and
`recent_items` is X as a plain list, so the wiring can place "recent,
verbatim" and "older, compacted" in two different parts of its prompt.

## Type-aware compaction

What the person said is never handed to the summariser. In a conversation the
type is visible from the speaker: the person's turn is the requirement, Otto's
is a report of work, and a report can be summarised without losing anything
the next turn has to honour. Protected items ride through compaction as
themselves, in every generation, and are never fed back into the summariser
(a first version protected them for exactly one generation, which a test
caught at 2/8).

Measured with `otto eval-compaction`: eight constraints planted through a
synthetic conversation, each policy replayed over them with a deterministic
summariser so the policy is measured and not the model of the day.

| policy | 120 turns | 400 turns |
| --- | --- | --- |
| type-blind | 3/8 | 0/8 |
| type-blind, tight budget | 0/8 | 0/8 |
| protected | 8/8 | 8/8 |
| protected, tight budget | 8/8 | 8/8 |

`REABSTRACT` (feeding prior bullets back through the summariser) was expected
to be the other culprit, so an abstract-once mode with clean shedding of the
oldest bullets was built and measured: 8/8 either way. What destroyed
constraints was the first compaction, and no later generation recovers what
generation one dropped. `REABSTRACT` stays `True`, now with a number behind
it, and the alternative stays in the file as the arm that made the comparison.

## Budget

```
MERCURY_2_5_CONTEXT_WINDOW = 260_000
TARGET_FRACTION            = 0.40
TOTAL_BUDGET               = 104_000   # 40% of 260K
X_BUDGET                   = 24_000    # first-pass split, tunable
Y_BUDGET                   = 80_000    # TOTAL_BUDGET - X_BUDGET
```

The total was sized as 40% of Mercury's 260K window when every seat ran on
Mercury. Seats now span four vendors, and the constants stand: the budget sits
well inside every routed model's window, and a real LoCoMo conversation
(11K–24K tokens) never fills it, which is why the benchmark takes budget
overrides to force compaction. The X:Y split is a first pass meant to be tuned
from real usage, not a specified figure.

## Retrieval

`recall(store, kind, query)` runs the two stages the spec described: compile
the candidate list from every live bullet's `hash_refs`, then rank those raw
chunks by their own embedding (written at flush time, one batched `embed()`
per flush) and return the best handful, each with the turns either side of it.
It falls back to the most recent bullets, unranked, whenever the embedding
model cannot be reached.

Ranking the bullets first to narrow the candidate set was tried and is a trap
worth naming so it is not reintroduced as an efficiency win. Compaction does
not produce comparable bullets: on a live replay the store held seven, six
citing one to three chunks each and one citing 237, so which bullet a question
matched had almost nothing to do with where its answer was.

| stage 1 | coverage on the live store |
| --- | --- |
| narrow to the best 3 bullets | 5.0% |
| narrow to the best 5 | 67.3% |
| search every live bullet's hashes | 90.1% |

Bullets summarise; they do not index. `top_k` only controls how many bullet
summaries print as context above the results.

Two read policies share the store. Accuracy on history questions rises with
the number of items returned, while task-time recall falls when retrieved
context starves attention from the thing being acted on, so `recall_memory`
called mid-task asks for the narrow read (`PROCEDURAL_TOP_K = 1`) and
everything else uses the default.

Measured against the live store, scoring how often `recall()` surfaces a
question's cited evidence and how much text it returns to do it:

| max_chunks | neighbours | coverage | chars returned |
| --- | --- | --- | --- |
| 5 | 0 | 65.3% | 922 |
| 5 | 1 | 79.2% | 2,112 |
| 5 | 2 | 84.2% | 3,189 |
| 10 | 2 | 90.1% | 6,293 |
| 20 | 2 | **96.0%** | 11,526 |
| 40 | 2 | 98.0% | 19,660 |
| uncapped (the first version) | | 99.0% | 38,033 |

Shipped defaults are 20 chunks, window 2: 96% coverage at about 2,600 tokens,
under 3% of the total budget, against 99% for returning the whole
conversation. Both knobs are counts, which is only a proxy for size (a LoCoMo
turn is ~150 characters, an Otto turn can be thousands), so a 3,000-token
budget spent best-first is the ceiling that actually holds; on LoCoMo it
barely binds (11,459 characters against 11,526 unbounded), which is what a
safety ceiling should look like.

Lexical BM25 blending was measured and rejected: it moved top-5 from 65.3% to
51.5%, because LoCoMo questions paraphrase rather than quote. It may still
help for dates and proper nouns, which is a narrower use and would need its
own evidence.

On a fresh `--live` run (a different summariser pass, so a different set of
bullets, which is the point now that bullet quality no longer gates what can
be found):

| category | n | store | recalled |
| --- | --- | --- | --- |
| single-hop | 28 | 100% | 96% |
| temporal | 24 | 100% | 92% |
| multi-hop | 8 | 100% | 88% |
| open-domain | 41 | 100% | 100% |
| **overall** | **101** | **100%** | **96%** |

Temporal and multi-hop are the weakest, as expected: both need more than one
turn, and multi-hop needs turns far apart, so neighbour expansion helps them
least (issue #22).

## Embeddings

The backend is swappable. The local model, `fastembed`'s
`BAAI/bge-small-en-v1.5` (384-dim, ONNX, no torch), is lazily loaded and is
the floor: the offline test suite, the evaluation harnesses' no-network paths
and any machine without a key all depend on embeddings working with no
credentials. `OTTO_EMBEDDING_MODEL` names a hosted model as `provider:model`,
and when a Gemini key is configured the default is `gemini:gemini-embedding-001`,
because it was measured, not chosen: three LoCoMo conversations, 221
questions, scored on `eval-memory`'s recall-coverage metric.

| model | conv 1 | conv 2–3 (held out) | all |
| --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | 96.0% | 92.5% | 93.7% |
| openai:text-embedding-3-small | 96.0% | | |
| gemini:gemini-embedding-001 | 98.0% | 96.7% | 97.3% |

The gain is larger on the two conversations never used for tuning, which is
the direction a real result moves in. The cost is recorded alongside because
it is the number that would reverse this: a recall query pays ~520 ms hosted
against ~17 ms local, on the person's turn.

Every vector records the model that produced it, and retrieval refuses to
rank across two embedding spaces. Three failure modes made the stamp
necessary, and the third is why a dimension check would not have been enough:
different dimensions raise from numpy (and were being swallowed into one line
of stderr, leaving recall dead for the session); `INSERT OR IGNORE` keyed on
the text hash means a stale vector is never overwritten; and the same
dimension from a different model raises nothing at all while ranking
nonsense. Refusing degrades to the unranked most-recent path with a warning
naming `MemoryStore.reembed()` as the fix.

The BGE query prefix ("Represent this sentence for searching relevant
passages: ") is a property of the local backend, not the module: it is worth a
measured 3.9 points on BGE and is harmful elsewhere, since OpenAI's embeddings
are symmetric and Gemini signals query-versus-passage with a task type.

Both the embedding model and `tiktoken`'s `cl100k_base` (approximate token
counting; Mercury's tokenizer is not public) need one network fetch on first
use and degrade rather than crash without it: `EmbeddingUnavailable` triggers
the unranked fallback, and `tokens.py` falls back to `len(text) // 4`.

## The context tier

The agent loop's own conversation grows without bound on a tool-heavy run
(around 500K characters at the default 120-call ceiling). Past
`LOOP_COMPACT_AT` the loop rewrites its oldest tool results in place to the
one-line `actions` summary already written when the call ran, keeping the seed
and the most recent `KEEP_VERBATIM` results untouched. That costs no model
call and bounds the transcript on its own (a test asserts the size stops
growing with the number of tool calls). The threshold is high and the cut is
chunky on purpose: every vendor caches its own prefix and rewriting history
invalidates it from the rewrite point.

Before a message is overwritten its full text goes into the session store
under `kind="context"`, and `recall_memory` searches it alongside the
conversation, reporting the two kinds under separate headings so evicted tool
output never comes back labelled as something the person said. This tier has
no bullet layer: the result was already reduced to a line in `actions`, so
summarising it again would pay a model call to abstract an abstraction, the
operation that made one model fail 54% of problems it had previously solved.
`recall_chunks` ranks the chunks directly. The compacted stub tells the model
the output is searchable only when a store is actually bound; with none (every
run outside a chat session, the benchmark harnesses included) "run it again"
is still the honest advice.

## Storage

`MemoryStore` is plain Python and SQLite that the loop and the recall tool
read and write directly, deliberately not a LangGraph state channel. Every
turn's graph thread is disposable by design, so anything meant to survive past
one turn lives somewhere that is not thrown away with it. The file is
`~/.otto/memory/<session_id>.db`, alongside the `~/.otto` convention for
internal state, as distinct from `otto_output/`, which is for the person to
open. `session_id` comes from the CLI's `Session` and changes on `/new`.

The binding between a run and its store is a `contextvars` variable
(`agent/memory/session.py`'s `bind_store()` / `current_store()`), so the
`recall_memory` tool, a plain function of one string, can find the right
session's store without the tools module importing the graph. The `summarize`
callback is injected into `TieredQueue` rather than imported, so the memory
package has no dependency on the router or the pipeline and never needs a live
model to test; the wiring module is the one place that connects the queue's
compaction to the router's `SUMMARIZE` seat.

## Resuming a session

Until 13 September the file only ever received what compaction retired, so X
and Y's raw items, the whole conversation for any session shorter than X's
24K tokens, lived in the process and died with it. `MemoryStore` now carries a
`pending` table mirroring the live tiers (`add_pending` on append,
`demote_pending` when X overflows, `clear_pending` once a compaction has
written the raw text to `chunks`), and `TieredQueue(restore=True)` rebuilds X,
Y's raw overflow, Y's bullets and the generation counter from it. Opt-in,
because the benchmarks build queues over stores they control and must start
from what they constructed. `agent/memory/sessions.py` is the index of
sessions a person had (`~/.otto/sessions.db`: id, title, workspace,
timestamps, turns), written on the first finished turn so an abandoned prompt
leaves nothing behind. An export is one JSON file: recent turns verbatim,
current bullets, the chunks they cite; embeddings stay out because they belong
to the model that made them, so an imported session ranks recall by recency
until it is re-embedded.

## Evaluation

`otto eval-memory` replays LoCoMo (`snap-research/locomo`, arXiv:2402.17753,
"Evaluating Very Long-Term Conversational Memory of LLM Agents"; ~2.8MB of
raw JSON, fetched on demand) through a `TieredQueue` and scores every QA
item's cited evidence for:

- `store_coverage`: is it reachable at all, verbatim or as a permanent chunk?
  An engine-correctness check that should always be ~100%; a miss here means
  the engine lost information.
- `visible_verbatim_coverage`: is it still sitting uncompacted in the live
  view, so a prompt already has it with no `recall()` call?
- `recall_coverage`: does `recall()`, given only the question, surface it once
  it has been compacted away?
- `answerable_coverage` (visible or recalled): the one number that answers "is
  this still findable at all".

Real LoCoMo conversations never exceed the production budget, so at the
defaults compaction never fires and `recall_coverage` is 0% for the right
reason. The command refuses to print a coverage table in that case (exit 1,
naming the budgets used), because a benchmark that can quietly measure
nothing will do it again to somebody else; `--x-budget` / `--y-budget` force
compaction. Two summariser backends: offline uses a deterministic
`_canned_summarize` that exercises the citation and budget mechanics with
placeholder text; `--live` routes through the real `SUMMARIZE` seat.
`--show-items failures|all` prints the evidence beside what recall returned,
because `recalled` is an exact-substring match and reading the two side by
side is how a real semantic find is told from a coincidental one.
`run_one_conversation(keep_store=True)` leaves the SQLite file behind so a
live replay can be re-scored under different settings without paying for the
summariser again.

## Finding: uncited items were silently unreachable

A hard live sweep (250 turns at `--x-budget 200 --y-budget 400`, forcing 19
compaction generations) showed `recall_coverage` collapsing to roughly 10%,
with `recall()` returning the same few generic bullets for nearly every one of
101 unrelated questions. The unparsed-reply fallback was not the cause
(`unparsed_bullet_count` was 0).

**Cause.** Compaction built each bullet's `hash_refs` from exactly the item
numbers the summariser cited, and nothing checked that every item got cited.
The prompt asked for full coverage; the model was free not to comply. An
omitted item's raw text still reached the chunk store, so `store_coverage`
stayed at 100% and the miss was invisible to that metric, but no live bullet
pointed at it any more, and `recall()` only searches the live generation. The
damage was a cliff rather than proportional because a prior bullet is itself
a numbered item carrying every hash behind it: one compaction that folded a
prior bullet into prose without citing its number severed the whole chain.

| summariser | chunks reachable from live bullets | `recall_coverage` |
| --- | --- | --- |
| cites every item | 249 / 249 | 100% |
| cites ~70% of items | 9–11 / 249 | 2–3% |
| cites ~70%, after the fix | 249 / 249 | 47% |

The offline benchmark could not see this: its canned summariser cites
perfectly by construction. Only a real model in the loop ever omitted
anything.

**Fix** (`_uncited_bullets()`, commit `3437a03`). Every item index no bullet
cited gets its own bullet. An uncited prior bullet is carried forward exactly
as it was, text and `hash_refs` both; an uncited raw item becomes a bullet
holding a truncated excerpt of its own text, which keeps the words `recall()`
ranks on. Y drains less completely when the summariser omits a lot (the same
replay went from 19 compactions to about 25, and from 3–4 live bullets to
about 9), and more, more specific bullets is the better trade for retrieval.
The live sweep went from ~10% to 99%, and that 99% was a dump rather than
retrieval, which is what the two-stage recall above fixed.

## What is still open

1. `X_BUDGET` and `Y_BUDGET` are a first pass. Real LoCoMo conversations
   never fill them; a real Otto session has far longer, far fewer turns. The
   count-based knobs in particular should be re-checked against replayed real
   sessions, which `otto sessions --export` now makes possible.
2. Every retrieval number here comes from LoCoMo conversations at a
   deliberately tiny budget, so the defaults are tuned to that shape of data.
3. Multi-hop recall is the weakest category (issue #22). Neighbour expansion
   does not help questions whose evidence is far apart.
4. BM25 blending is rejected on the evidence above. A narrower use for dates
   and proper nouns would need its own measurement.
5. An imported session ranks recall by recency until `MemoryStore.reembed()`
   has run, and nothing runs it automatically yet.

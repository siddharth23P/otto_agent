# agent/memory/

Short-term memory that keeps a session's conversation and a turn's tool
output inside a model's context window while keeping everything retrievable,
plus the lesson bank and the session index. This package imports nothing
from `agent/pipeline` or `agent/router`; the summariser is injected, so
nothing here needs a live model to test.

| module | what it is |
| --- | --- |
| `queue.py` | `TieredQueue`: the X/Y tiers, citation-based compaction, type-aware protection, restore from disk |
| `store.py` | `MemoryStore`: SQLite, content-addressed chunks with embeddings, bullets per generation, the pending mirror of the live tiers |
| `retrieval.py` | `recall()` and `recall_chunks()`: two-stage ranked search with neighbours under a token budget |
| `embeddings.py` | the swappable, stamped embedding backends: local BGE, OpenAI, Gemini |
| `hashing.py`, `tokens.py` | sha256 content addressing; approximate token counting with an offline fallback |
| `wiring.py`, `session.py` | the bridge to a live session: the summarise callback on the `SUMMARIZE` seat, and the per-run store binding the `recall_memory` tool reads |
| `lessons.py` | the lesson bank |
| `sessions.py` | the index of saved sessions |

## The tiers

`X` holds recent items verbatim. When its token budget overflows, its whole
content moves to `Y`. When `Y` overflows, every raw item is flushed to a
`chunks` table keyed by its sha256 and the whole of `Y` is summarised into a
cited bullet list. Each bullet's hash references resolve to real stored text
no matter how many generations later, because a cited prior bullet
contributes its own references unchanged, and every item the summariser did
not cite gets its own bullet, so nothing flushed is ever unreachable from
the live generation. Budget: 104,000 tokens total, 24,000 for `X`, 80,000
for `Y`, well inside every routed model's window.

Compaction is type-aware. What the person said is never handed to the
summariser: the person's turn is the requirement, Otto's is a report of work,
and a report can be summarised without losing anything the next turn has to
honour. Measured with `otto eval-compaction` on eight planted constraints,
the protected policy keeps 8/8 at 120 and 400 turns through 23 compaction
rounds, at every budget tried.

The loop's own tool output is the second tier, `kind="context"`: past a
threshold the loop rewrites its oldest results to their one-line `actions`
summary and stores the full text here first. This tier has no bullet layer,
since the result was already reduced to a line when the call ran and
summarising it again would pay a model call to abstract an abstraction.
`recall_chunks` ranks the chunks directly, and `recall_memory` reports the
two kinds under separate headings so evicted tool output never comes back
labelled as something the person said.

## Recall

Two stages: compile the candidate chunks from every live bullet's
references, then rank the chunks by their own embedding (written at flush
time, one batched call per flush) and return the best twenty, each with the
turns either side of it, under a 3,000-token ceiling spent best-first.
Bullets summarise; they do not index, so they are never used to narrow the
search. Queries go through `embed_query()`, which adds the search
instruction the local model is trained to expect and nothing for backends
that are symmetric.

Measured on LoCoMo (101 questions, a live summariser): 96% of cited evidence
recalled at about 2,600 tokens per query, store coverage 100%; single-hop
96%, temporal 92%, multi-hop 88%, open-domain 100%.

Two read policies share the store: `recall_memory` called mid-task asks for
the narrow read (`PROCEDURAL_TOP_K = 1`), because retrieved context starves
attention from the thing being acted on; everything else uses the default.

## Embeddings

Every vector records the model that made it, and retrieval refuses to rank
across two embedding spaces, degrading to the unranked most-recent path with
a warning that names `MemoryStore.reembed()` as the fix. The local model,
`BAAI/bge-small-en-v1.5` through `fastembed`, is the floor and needs no
credentials. When a Gemini key is configured the default is
`gemini:gemini-embedding-001`, measured on 221 LoCoMo questions at 97.3%
against 93.7% for local, with the gain larger on the held-out conversations.
`OTTO_EMBEDDING_MODEL` (`provider:model`) wins over both. Every backend
returns float32 and translates every failure to `EmbeddingUnavailable`, so a
rate limit degrades a compaction flush instead of taking it down.

## Sessions

The live tiers are mirrored to a `pending` table on every append, demoted on
`X` overflow and cleared once a compaction has retired their text to
`chunks`, so `TieredQueue(restore=True)` rebuilds the exact in-memory state.
`sessions.py` keeps the index at `~/.otto/sessions.db` (id, title, workspace,
timestamps, turns), written on the first finished turn so an abandoned prompt
leaves nothing behind. An export is one JSON file: recent turns verbatim,
current bullets, the chunks they cite; embeddings stay out because they
belong to the model that made them.

## Lessons

A finished run distils at most three short lessons about method, adjudicated
against the bank at write time and never derived from another lesson; the
next run reads at most one, only above a relevance threshold. Both the bank
and the routing outcome log are off in the `--no-learning` arm and read-only
on the held-out split of any measurement. `otto lessons` prints, clears,
exports and imports the bank.

The full design with its measurement tables: [docs/design/tiered-memory.md](../../docs/design/tiered-memory.md).

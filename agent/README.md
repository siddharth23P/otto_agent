# agent/

The Python package behind the `otto` command. Five sub-packages, each with
its own README:

| package | role |
| --- | --- |
| [pipeline/](pipeline/README.md) | the agent loop and the evaluator, the tools, the gates, the workspace and container seams, the document workflow |
| [memory/](memory/README.md) | the tiered short-term memory, the lesson bank, the session index |
| [router/](router/README.md) | which model serves which kind of work, across four vendors |
| [cli/](cli/README.md) | the TUI, the REPL and every `otto` sub-command |
| [eval/](eval/README.md) | the benchmark harnesses |
| [config/](config/README.md) | the `.env` file, and where Otto's state lives |
| [phone/](phone/README.md) | the phone as a place Otto works: the backend Protocol, the screen digest, the money guard, the phone tools |
| [server/](server/README.md) | `otto serve`: the agent behind a WebSocket for a client that has the hands |

Beside them, one module: `embed.py`, the surface an embedding host (the
Android app) depends on -- `configure()`, keys, `Runtime`, `SessionHandle` --
versioned by `API_VERSION` and deliberately narrow.

Dependency direction, enforced by the modules' own docstrings and a few
tests: `memory/` imports nothing from `pipeline/` or `router/`, so it never
needs a live model to test; `pipeline/` reaches models only through
`router/`; `cli/` and `eval/` sit on top of both. Per-run state (the budget,
the workspace, the command runner, the memory store, the run-scoped tools)
is bound with `contextvars` at the pipeline entry point rather than threaded
through arguments, which is what lets a tool be a plain function of one
string.

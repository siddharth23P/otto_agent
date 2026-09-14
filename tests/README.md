# tests/

1,589 tests across 75 files, passing on macOS, Linux and Windows in under two
minutes with no API keys and no network. CI runs the full matrix on every
push and pull request with a read-only token.

```bash
uv run pytest -q
```

## How the suite stays honest

- **No keys, no network.** `conftest.py` sets placeholder keys with
  `setdefault` (a real key in the shell is left alone), points
  `OTTO_EMBEDDING_MODEL` at the local model, and gives every test its own
  empty lesson bank, outcome log, temperature store and session memory, so a
  full run leaves `~/.otto` at the same file count it started with. Tests
  that need a real vendor are marked `live_anthropic` and skip without a key.
- **Behaviour, not output.** Regression tests assert on the messages handed
  to the model (the only thing that distinguishes a resumed conversation
  from a restarted one), on the exact inputs that broke real runs (a prose
  sentence where a path goes, chat-template tokens in a command, an outline
  truncated at a seat's token limit, evidence over the judge's clip length),
  and on call counts against the real compiled graph with a scripted model
  (`test_call_budget.py`).
- **Library behaviour is asserted, not assumed.** The Textual and LangGraph
  behaviours the front end and the stream depend on are pinned against the
  installed versions in `test_tui.py` and `test_pipeline_run.py`, so an
  upgrade that changes one fails loudly here.
- **Conditions, not clocks.** Every TUI test waits on a predicate with a
  deadline through one helper and never on an iteration count; the file
  passes when run four times concurrently under CPU load.
- **Invariants are tests.** Every dispatchable tool appears in the agent
  prompt and the prompt stays under a measured character cap
  (`test_prompt_tool_sync.py`); every irreversible tool is behind the
  mutation gate (`test_tools_stubs.py`); no directory in the repository
  shadows a site package (`test_no_import_shadowing.py`); the memory
  package imports nothing from the pipeline or the router.
- **The container path is exercised without Docker.** Tests bind a command
  runner that executes the generated POSIX commands on the host shell, so
  the base64 shipping, the `find` pruning and the edit script all run in the
  suite; those tests skip on Windows with a reason that says why.
- **Platform bugs are tests.** Path separators from `code_map` and
  `list_files` are POSIX on every OS; a session's SQLite handle is closed
  before its file is deleted, which only Windows enforces.

## Where things are

| area | files |
| --- | --- |
| the agent loop, evaluator, gates, modes, budget | `test_agent_loop.py`, `test_evaluator_node.py`, `test_pipeline_nodes.py`, `test_evidence.py`, `test_modes.py`, `test_call_budget.py`, `test_budget.py`, `test_chat_fast_path.py`, `test_ask_user_node.py` |
| tools, workspace, container seam, python session, browser, screen, walkthroughs | `test_tools_stubs.py`, `test_tool_loop.py`, `test_workspace_tools.py`, `test_workspace_session.py`, `test_python_session.py`, `test_run_scoped_toolkit.py`, `test_browsing.py`, `test_screen.py`, `test_walkthrough.py`, `test_native.py`, `test_codemap.py`, `test_vision_tool.py` |
| the document workflow | `test_research_workflow.py`, `test_research_router.py` |
| memory | `test_memory_*.py`, `test_evicted_context.py`, `test_retrieval_method.py`, `test_embedding_backends.py`, `test_lessons.py`, `test_sessions.py` |
| routing and providers | `test_router.py`, `test_mapping.py`, `test_health.py`, `test_seat_outcomes.py`, `test_overrides.py`, `test_automap.py`, `test_reload.py`, `test_model_policies.py`, `test_temperature_learning.py`, `test_custom_endpoints.py`, `test_inception_*.py`, `test_diffusion_retry.py` |
| front ends | `test_tui.py`, `test_tui_progress.py`, `test_tui_setup.py`, `test_setup.py`, `test_clipboard.py`, `test_art.py`, `test_output.py`, `test_progress.py`, `test_usage.py`, `test_envfile.py`, `test_lessons_cli.py` |
| harnesses | `test_eval_runner.py`, `test_eval_honesty.py`, `test_swe_bench.py`, `test_claw_bench.py`, `test_memory_bench.py`, `test_compaction_bench.py`, `test_hle_bench.py`, `test_terminal_bench_adapter.py`, `test_failures.py`, `test_langfuse_sync.py` |

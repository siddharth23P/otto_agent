# Otto

A terminal AI agent that works on a codebase, a container, a browser or a
desktop, uses what it built, and judges its own work against criteria it
wrote before it started. One agent loop with modes over four model vendors,
a rubric-first evaluator, tiered memory, six benchmark harnesses, and 1,591
tests that need no API key.

![One minute of otto tui](https://raw.githubusercontent.com/siddharth23P/otto_agent/main/docs/media/otto-demo.gif)

## Install

```bash
pip install otto-cli-agent
otto doctor
otto tui
```

Put keys in a `.env` in the directory you run it from. `INCEPTION_API_KEY`
is required (it alone serves the fill-in-the-middle and edit endpoints);
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are optional and
each unlocks the seats routed to that vendor. `otto tui` opens its setup
screen on first start when nothing is configured. Python 3.12 or newer.

## What it does

- One agent loop with four modes that swap the model underneath without
  losing the conversation; routing that learns from outcomes, per-model
  cooldowns and a per-provider circuit breaker.
- A rubric written from the task before any attempt, shared by the loop and
  the judge, so the judge never grades against the actor's own output.
- Once-only holds before an irreversible action, before finishing on unrun
  code, and before finishing without using what was built.
- 18 tools: shell and Python, files, `code_map`, a browser, `exercise` (walk
  through a page, a served app, a CLI, an API, a terminal program or a device
  app and report each step as the machine saw it), a desktop, web search,
  workspace search and memory recall.
- Tiered memory with type-aware compaction and two-stage semantic recall
  (96% on LoCoMo), a lesson bank, and sessions that survive the process.
- A Textual TUI with a setup screen, live progress, a token and dollar
  ledger; a REPL with the same pipeline.
- Six benchmark harnesses: a golden set, SWE-bench Verified, Claw-Eval,
  LoCoMo, a compaction bench and Humanity's Last Exam.

## Links

- Source, folder-by-folder READMEs and screenshots:
  https://github.com/siddharth23P/otto_agent
- Website: https://siddharth23p.github.io/otto_agent/
- How it was built, commit by commit, with the measurement behind each
  change: https://github.com/siddharth23P/otto_agent/blob/main/docs/HISTORY.md
- The research each design decision draws on:
  https://github.com/siddharth23P/otto_agent/blob/main/docs/RESEARCH.md

MIT licensed.

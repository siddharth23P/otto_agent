# Contributing

Thanks for looking. The bar for a change here is the same one the project was
built to: say what you measured, or what you would need to measure.

## Running it

```bash
uv sync
uv run pytest -q          # 1,589 tests, no API key, no network, about two minutes
uv run otto doctor        # once you have keys in .env
```

The suite is designed to pass on a machine with no keys at all. If a test
you add needs a real vendor, mark it with the `live_*` markers in
`tests/conftest.py` so it skips without one.

## Where things are

Every folder has a README describing what is in it and why. Start at the
[root README](README.md), then the folder you are changing. The design
decisions carry their measurements in the module docstrings; if you change a
behaviour that has a number behind it, the pull request should carry the new
number.

## What a good pull request looks like

- One change, with the reason in the description: what was wrong, how you
  know, what changed, what you measured.
- A test that fails without the change. For a bug, the exact input that
  broke a real run is the best test there is.
- No new dependency without a sentence on why the standard library or an
  existing one would not do.
- Prompt changes are measured, not argued: this codebase records a case
  where adding one more instruction erased the effect of the four before it.
- Commits under your own name.

## Reporting a bug

Open an issue with the command you ran, what you expected, and what
happened, including the exact error text. If it involved a model, say which
seat and vendor (`otto route <task>` shows it).

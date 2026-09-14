# agent/config/

`envfile.py` reads and writes the one `.env` file Otto loads, at the
repository root. It is what the setup screen writes keys and named endpoints
into, and what `agent/cli/main.py` loads at start. Keys are only ever shown
masked; a named custom endpoint `local` becomes `LOCAL_API_KEY` and
`LOCAL_BASE_URL`. Nothing else in the package reads the file directly.

`home.py` says where that file is and where every per-installation file
goes: `OTTO_HOME` (default `~/.otto`) for the memory stores, the session
index, the outcome log, routes and lessons; `OTTO_ENV_FILE` for the `.env`
(default: the checkout's, or `<OTTO_HOME>/.env` for an installed wheel).
Both are read when the state modules are imported, so an embedding host sets
them first -- `agent/embed.py`'s `configure()` does. `OTTO_OUTPUT_DIR` moves
`otto_output/` the same way.

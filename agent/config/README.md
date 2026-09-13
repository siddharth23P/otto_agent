# agent/config/

`envfile.py` reads and writes the one `.env` file Otto loads, at the
repository root. It is what the setup screen writes keys and named endpoints
into, and what `agent/cli/main.py` loads at start. Keys are only ever shown
masked; a named custom endpoint `local` becomes `LOCAL_API_KEY` and
`LOCAL_BASE_URL`. Nothing else in the package reads the file directly.

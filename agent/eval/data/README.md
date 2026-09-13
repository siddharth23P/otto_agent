# agent/eval/data/

Downloaded benchmark data lives here and is not committed: LoCoMo's JSON,
HLE's parquet, and Claw-Eval's fixture archives (about 2.9GB) are fetched on
demand by their harnesses, and `.gitignore` excludes them by file type rather
than by directory so hand-authored files alongside them stay tracked.

| path | what it is |
| --- | --- |
| [claw/](claw/README.md) | the Claw-Eval run configuration, the Gemini judge patch, and the runbook for measuring a change honestly |

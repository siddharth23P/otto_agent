# agent/router/

Which model serves which kind of work. The pipeline never names a vendor; it
asks for a `Task` seat and the router resolves it through a fallback chain.

| module | what it is |
| --- | --- |
| `mapping.py` | the seats and the routing table: per seat, an ordered chain of candidates with per-route params, validated at import |
| `router.py` | `Router`: resolves a seat to a chat model, two passes over the chain (skip unwell candidates, then take the chain as it stands) |
| `outcomes.py` | what each seat's model achieved, recorded so the chain can be reordered on evidence |
| `health.py` | per-model cooldowns and a per-provider circuit breaker |
| `overrides.py` | what a person pinned on top of the table (`~/.otto/routes.json`) and where pins may not go |
| `automap.py` | a proposed model for every seat from the models this machine can reach, with the reason |
| `setup.py`, `reload.py` | the data layer under the setup screen; one call after keys, endpoints or pins changed in the running process |
| [llm_provider/](llm_provider/README.md) | the vendor adapters and the per-model policies |

## Seats

`chat_fast`, `reason`, `evaluate`, `plan`, `summarize`, `vision`, `web`,
`code_complete`, `code_edit`. The evaluator has its own seat because a judge
and an actor want different models. `vision` and `web` have no fallback
floor on purpose: a model that cannot see an image or reach the web would
answer confidently from memory while looking like a real look or a real
search, so `NoViableRoute` is the correct outcome. `code_complete` and
`code_edit` are served by Inception alone, which is why `INCEPTION_API_KEY`
is the one key Otto cannot start without; every other vendor is optional and
its routes skip with a legible reason when its key is absent.

## Learned ordering

Each approved single-mode run records an outcome for its (seat, model) pair.
A pair needs twelve runs before its number moves anything; only two measured
candidates ever trade places, so the declared order in the table is not
unseated by one side's absence of evidence; a challenger must lead by ten
points, or match within that and cost a fifth fewer calls. One run in ten
goes to the trusted candidate with the fewest runs behind it, so a demoted
model keeps accruing evidence and can win its place back. Exploration is off
whenever the log is read-only, which keeps a held-out measurement
reproducible. Multi-mode runs record nothing, and the evaluator's own seat is
never credited with its verdict.

## Health

A rate limit cools one model, since quota is metered per model. A
five-hundred or a dead connection counts toward the provider, and three in a
row trip its breaker. The vendor's own `Retry-After` wins over the guess;
windows double on re-entry and stop at five minutes. Cooldowns are a
preference, never a prohibition: if skipping everything unwell leaves
nothing, the chain is taken as it stands. A missing key or a model that does
not exist still raises, because those are not health.

## Pins and setup

`otto route <task>` prints a seat's chain with a pinned head starred and the
observed outcomes under it. Pins persist in `~/.otto/routes.json` and apply
to the running session at once; `OTTO_IGNORE_ROUTES=1` makes a run use the
shipped table untouched, which the evaluation harnesses do. The web seat
stays on Anthropic (the `web_search` tool binds Anthropic's server-side
search) and the fill-in-the-middle and edit seats stay on Inception;
`validate_pin` refuses a pin that would break either.

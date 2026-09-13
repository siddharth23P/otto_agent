# agent/router/llm_provider/

One adapter per vendor behind one `BaseProvider` contract, plus the two
per-model policies that only became visible with four vendors in one table.

| module | what it is |
| --- | --- |
| `base.py` | the contract: list models with capabilities, build a chat model, and let nothing but `ProviderError` cross the boundary; `translate_unknown()` duck-types the errors LangChain's own chat classes raise |
| `openai_provider.py` | the reference implementation |
| `anthropic_provider.py`, `gemini_provider.py` | the same contract on their SDKs |
| `inception_provider.py` | Inception (Mercury): chat with the diffusion view, plus the fill-in-the-middle and edit endpoints no other vendor serves |
| `custom.py` | named OpenAI-compatible endpoints as runtime provider classes: OpenRouter, a remote vLLM, Ollama, LM Studio |
| `retired.py` | models a vendor still lists but will not serve, filtered at the catalogue; a retirement discovered at call time is remembered for the rest of the process, never persisted |
| `temperature.py` | what temperature each model will actually honour |

## Temperature

One number fails three different ways: Inception silently resets anything
under 0.5 to the model default of 1.0, OpenAI's o-series rejects any value
but its default, and Anthropic and Gemini honour different ranges, with
Gemini publishing a ceiling per model (39 cap at 2, 7 at 1). So a route asks
for what it wants and the policy decides what the model can be given: an
explicit per-model entry first, then a model that admits no choice, then the
vendor's own published ceiling, then a per-provider default. Anthropic's
policy reads the published capability flags, so a new Claude release that
drops the parameter needs no table edit. A model no table knows is asked:
a refused temperature is retried without one and remembered in
`~/.otto/temperature.json`, because a refusal is deterministic and the worst
a wrong entry does is run at the default.

## Catalogue filtering

Capability detection is generous (a vendor tags every model in a family as
vision-capable), so `retired.py` also drops models that are alive and simply
not chat models: text-to-speech, transcription, a model that answers with
"only supports Interactions API". Patterns rather than ids, because these
families grow faster than any list of exact names stays true. Every pin in
the routing table is verified to actually serve a request.

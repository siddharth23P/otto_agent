"""Inception provider -- SDK only, no HTTP client, no OpenAI compatibility shim.

Every network call in this module goes through `inceptionai`. Chat is still
exposed as a LangChain `BaseChatModel` (`ChatInception` below) because that is
what `BaseProvider.chat_model` promises and what keeps Inception usable inside
a LangGraph node and visible to Langfuse -- but the transport underneath is
`client.chat.completions.create`, nothing else.

Inception is the only one of the four vendors that returns real capability
metadata (`supported_features`, `input_modalities`, `context_length`), so
nothing here is guessed from model-name prefixes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Iterator, Literal, Sequence

from inceptionai import Inception
from inceptionai._exceptions import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    InceptionError,
    NotFoundError,
    PermissionDeniedError,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from agent.router.llm_provider.base import (
    AuthError,
    Completion,
    BaseProvider,
    Capability,
    ModelInfo,
    ProviderError,
    ProviderUnavailable,
)

__all__ = ["ChatInception", "InceptionProvider", "build_edit_prompt"]


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------


def _field(obj: Any, key: str) -> Any:
    """Read `key` off a usage payload that may be a model *or* a plain dict.

    The SDK declares `usage` on ChatCompletion but not on ChatCompletionChunk,
    so on the streaming path pydantic keeps the API's usage payload in model
    extras as an untyped dict. Both shapes reach us, so never assume either.
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _usage(u: Any) -> dict[str, int] | None:
    """Vendor usage payload -> Langfuse's usage_details key names."""
    if u is None:
        return None
    details = {
        "input": _field(u, "prompt_tokens"),
        "output": _field(u, "completion_tokens"),
        "total": _field(u, "total_tokens"),
    }
    details = {k: v for k, v in details.items() if v is not None}
    if not details:
        return None
    for attr, key in (("cached_input_tokens", "cached_input"),
                      ("reasoning_tokens", "reasoning")):
        value = _field(u, attr)
        if value:
            details[key] = value
    return details


def _translate(exc: Exception) -> ProviderError:
    """Map an `inceptionai` exception onto the provider error hierarchy.

    Order matters: AuthenticationError, PermissionDeniedError and NotFoundError
    all subclass APIStatusError, so the narrow cases must be tested first or
    they get swallowed by the broad one.
    """
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return AuthError(f"inception: key rejected ({exc})")
    if isinstance(exc, APIConnectionError):  # APITimeoutError subclasses this
        return ProviderUnavailable(f"inception: {exc}")
    if isinstance(exc, APIStatusError):
        if exc.status_code in (401, 403):
            return AuthError(f"inception: HTTP {exc.status_code}")
        return ProviderError(f"inception: HTTP {exc.status_code}")
    return ProviderError(f"inception: {exc}")


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------

_ROLES: dict[type, str] = {
    SystemMessage: "system",
    HumanMessage: "user",
    AIMessage: "assistant",
    ToolMessage: "tool",
}


def _to_sdk_message(message: BaseMessage) -> dict[str, Any]:
    """LangChain message -> `ChatCompletionMessageParam` shape."""
    if isinstance(message, ChatMessage):
        role = message.role
    else:
        role = _ROLES.get(type(message), "")
        if not role:
            for cls, name in _ROLES.items():
                if isinstance(message, cls):
                    role = name
                    break
        if not role:
            raise ProviderError(f"inception: unsupported message {type(message).__name__}")

    payload: dict[str, Any] = {"role": role, "content": message.content}

    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
    elif isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["args"]),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _from_sdk_tool_calls(tool_calls: Sequence[Any] | None) -> list[dict[str, Any]]:
    """`ChatCompletionToolCall` -> LangChain tool-call dicts.

    `function.arguments` arrives as a JSON *string*. A model can emit malformed
    JSON, so a parse failure is preserved verbatim rather than raised -- losing
    the whole turn over one bad argument blob is worse than surfacing it.
    """
    parsed: list[dict[str, Any]] = []
    for call in tool_calls or []:
        raw_args = call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"__raw__": raw_args}
        parsed.append(
            {
                "name": call.function.name or "",
                "args": args,
                "id": call.id,
                "type": "tool_call",
            }
        )
    return parsed


# --------------------------------------------------------------------------
# Edit prompt construction
# --------------------------------------------------------------------------

CURSOR = "<|cursor|>"


def build_edit_prompt(
    code_to_edit: str,
    *,
    current_file: str = "",
    recently_viewed: Sequence[str] = (),
    edit_history: Sequence[str] = (),
) -> str:
    """Assemble the tagged prompt the edit endpoint requires.

    All four sections must be present even when empty -- the documented
    examples send empty `recently_viewed_code_snippets` and `edit_diff_history`
    blocks rather than omitting them.

    Kept at module level, and pure, so it can be tested without a client.
    """
    if CURSOR not in code_to_edit:
        code_to_edit = f"{code_to_edit}{CURSOR}"

    return (
        f"<|recently_viewed_code_snippets|>\n"
        f"{chr(10).join(recently_viewed)}"
        f"<|/recently_viewed_code_snippets|>\n"
        f"<|edit_diff_history|>\n"
        f"{chr(10).join(edit_history)}"
        f"<|/edit_diff_history|>\n"
        f"<|current_file_content|>\n"
        f"{current_file or code_to_edit.replace(CURSOR, '')}\n"
        f"<|/current_file_content|>\n"
        f"<|code_to_edit|>\n"
        f"{code_to_edit}\n"
        f"<|/code_to_edit|>"
    )


# --------------------------------------------------------------------------
# LangChain chat model over the Inception SDK
# --------------------------------------------------------------------------


class ChatInception(BaseChatModel):
    """LangChain chat model whose transport is the `inceptionai` SDK.

    Implementing `_generate` and `_stream` is the entire contract -- LangChain
    builds `invoke`, `stream`, `batch`, `bind_tools` and callback dispatch on
    top of them, which is why Langfuse works here without extra wiring.
    """

    # Typed as Any deliberately: the SDK client is not a pydantic-friendly type
    # and BaseChatModel is a pydantic model. Excluded from serialisation so the
    # client (and the key inside it) never lands in a trace payload.
    client: Any = Field(exclude=True, repr=False)
    model: str

    temperature: float | None = None
    max_tokens: int | None = None
    diffusing: bool | None = None
    realtime: bool | None = None
    reasoning_effort: Literal["instant", "low", "medium", "high"] | None = None
    reasoning_summary: bool | None = None
    #: Anything else `chat.completions.create` accepts (response_format, ...).
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    #: Called with each denoising snapshot when `diffusing` is on, so a caller
    #: can render the answer resolving in place. Excluded from serialisation:
    #: LangChain builds invocation_params from the model's fields, and a
    #: callable in there would be handed to every callback -- including
    #: Langfuse, which would store its repr in the trace.
    frame_sink: Callable[[str], None] | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "inception"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def _payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_sdk_message(m) for m in messages],
        }
        optional = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "diffusing": self.diffusing,
            "realtime": self.realtime,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
        }
        # Omit unset keys rather than sending None -- the SDK distinguishes
        # "not given" from an explicit null.
        body.update({k: v for k, v in optional.items() if v is not None})
        if stop:
            body["stop"] = stop
        body.update(self.model_kwargs)
        body.update(kwargs)
        return body

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            completion = self.client.chat.completions.create(
                **self._payload(messages, stop, **kwargs)
            )
        except InceptionError as exc:
            raise _translate(exc) from exc

        choice = completion.choices[0]
        usage = completion.usage

        message = AIMessage(
            content=choice.message.content or "",
            tool_calls=_from_sdk_tool_calls(choice.message.tool_calls),
            response_metadata={
                "model": completion.model,
                "finish_reason": choice.finish_reason,
                "warning": completion.warning,
            },
            usage_metadata={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"finish_reason": choice.finish_reason},
                )
            ],
            llm_output={"model": completion.model, "id": completion.id},
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._payload(messages, stop, **kwargs)
        payload["stream"] = True
        # Without this the API never sends the usage chunk at all, and every
        # streamed call reports zero tokens -- and therefore zero cost.
        payload.setdefault("stream_options", {"include_usage": True})

        try:
            stream = self.client.chat.completions.create(**payload)
        except InceptionError as exc:
            raise _translate(exc) from exc

        # Diffusion does not stream deltas. Every chunk is a full snapshot of
        # the entire answer, re-denoised -- so yielding snapshots as content
        # makes LangChain concatenate every intermediate draft into the final
        # message, and that message is what Langfuse records as the output.
        # Snapshots therefore go to `frame_sink` for display, and only the last
        # one is ever yielded.
        snapshot = ""
        stamped = False

        for chunk in stream:
            if not chunk.choices:
                # The usage-only chunk: `choices` is empty and `usage` is
                # populated. The SDK's ChatCompletionChunk does not declare a
                # `usage` field even though the API documents sending one, so
                # read it defensively rather than trusting the type.
                usage = getattr(chunk, "usage", None)
                if usage is None and isinstance(getattr(chunk, "model_extra", None), Mapping):
                    usage = chunk.model_extra.get("usage")
                input_tokens = _field(usage, "prompt_tokens")
                output_tokens = _field(usage, "completion_tokens")
                if input_tokens is not None or output_tokens is not None:
                    input_tokens = input_tokens or 0
                    output_tokens = output_tokens or 0
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            usage_metadata={
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": _field(usage, "total_tokens")
                                or input_tokens + output_tokens,
                            },
                        )
                    )
                continue
            choice = chunk.choices[0]
            text = choice.delta.content or ""
            if not text:
                continue

            if self.diffusing:
                snapshot = text
                if self.frame_sink is not None:
                    self.frame_sink(text)
                if run_manager and not stamped:
                    # Stamps Langfuse's completion_start_time on the first
                    # frame. Without it, time-to-first-token would equal total
                    # latency for every diffusion call, since the only content
                    # chunk is yielded after the stream is exhausted.
                    stamped = True
                    run_manager.on_llm_new_token("")
                continue

            generation = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                # Drives Langfuse / CLI token callbacks. Without this, streamed
                # tokens never reach any handler.
                run_manager.on_llm_new_token(text, chunk=generation)
            yield generation

        if snapshot:
            # The settled answer, emitted once, so the accumulated message is
            # the final text rather than every draft glued together.
            generation = ChatGenerationChunk(message=AIMessageChunk(content=snapshot))
            if run_manager:
                run_manager.on_llm_new_token(snapshot, chunk=generation)
            yield generation


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

#: `Model.supported_features` values that map onto our capability vocabulary.
#: These are the values the API is documented to return -- mercury-2 reports
#: ["tools", "json_mode", "structured_outputs"], mercury-edit-2 reports [].
#: Unrecognised features are ignored, not dropped: the untouched payload is
#: always available on `ModelInfo.raw`.
#:
#: "reasoning" is deliberately absent. `reasoning_effort` is a real chat request
#: parameter, so REASONING stays declared at provider level, but no documented
#: model payload advertises it as a feature -- so no model is tagged with it
#: until one actually does.
_FEATURE_MAP: dict[str, Capability] = {
    "tools": Capability.TOOLS,
    "json_mode": Capability.STRUCTURED_OUTPUT,
    "structured_outputs": Capability.STRUCTURED_OUTPUT,
}


def _capabilities_of(model: Any) -> set[Capability]:
    caps = {
        _FEATURE_MAP[feature]
        for feature in getattr(model, "supported_features", []) or []
        if feature in _FEATURE_MAP
    }
    if "image" in (getattr(model, "input_modalities", []) or []):
        caps.add(Capability.VISION)
    return caps


class InceptionProvider(BaseProvider):
    name = "inception"
    env_var = "INCEPTION_API_KEY"
    # Left as None on purpose: the SDK already defaults to
    # https://api.inceptionlabs.ai and honours INCEPTION_BASE_URL. Hardcoding
    # it here would silently override that.
    default_base_url = None
    capabilities = frozenset(
        {
            Capability.CHAT,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.FIM,
            Capability.EDIT,
        }
    )

    # -- 1. client --------------------------------------------------------

    def _build_client(self) -> Inception:
        return Inception(api_key=self._api_key, base_url=self._base_url)

    # -- 2. model discovery -----------------------------------------------

    def _fetch_models(self) -> list[ModelInfo]:
        """Union the three task-specific model lists into one catalogue.

        A model can appear in more than one list, so results are keyed by id
        and their capabilities merged -- this is where the old `ModelTask`
        enum dissolves into per-model capability flags.
        """
        found: dict[str, ModelInfo] = {}

        sources = (
            (Capability.CHAT, self._client.models.list_chat),
            (Capability.FIM, self._client.models.list_fim),
            (Capability.EDIT, self._client.models.list_edit),
        )

        for capability, fetch in sources:
            try:
                page = fetch()
            except NotFoundError:
                # Endpoint not enabled for this account. One unavailable task
                # must not blank out the other two.
                continue
            except InceptionError as exc:
                raise _translate(exc) from exc

            for model in page.data:
                caps = {capability} | _capabilities_of(model)
                existing = found.get(model.id)
                if existing is not None:
                    found[model.id] = replace(
                        existing, capabilities=existing.capabilities | frozenset(caps)
                    )
                else:
                    found[model.id] = ModelInfo(
                        id=model.id,
                        provider=self.name,
                        display_name=model.name,
                        capabilities=frozenset(caps),
                        context_window=model.context_length,
                        max_output_tokens=model.max_output_length,
                        raw=model.model_dump(),
                    )

        return list(found.values())

    # -- 3. chat ----------------------------------------------------------

    def chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        return ChatInception(client=self._client, model=model_id, **kwargs)

    # -- optional capabilities (satisfy SupportsFIM / SupportsEdit) --------

    def fim(
        self,
        model_id: str,
        prefix: str,
        suffix: str = "",
        **kwargs: Any,
    ) -> Completion:
        """Fill-in-the-middle completion. No LangChain equivalent exists."""
        try:
            completion = self._client.fim.completions.create(
                model=model_id, prompt=prefix, suffix=suffix, **kwargs
            )
        except InceptionError as exc:
            raise _translate(exc) from exc
        return Completion(
            text=completion.choices[0].text,
            usage=_usage(getattr(completion, "usage", None)),
        )

    def code_edit(
        self,
        model_id: str,
        code_to_edit: str,
        *,
        current_file: str = "",
        recently_viewed: Sequence[str] = (),
        edit_history: Sequence[str] = (),
        **kwargs: Any,
    ) -> Completion:
        """Predict the next edit to `code_to_edit`.

        There is no instruction parameter: this endpoint infers the edit from
        context. Streaming and tool calling are not supported on it, so unlike
        `chat_model` there is nothing here for LangChain to wrap.

        `code_to_edit` may contain `<|cursor|>` to mark the caret; if it does
        not, the cursor is placed at the end of the region.
        """
        prompt = build_edit_prompt(
            code_to_edit,
            current_file=current_file,
            recently_viewed=recently_viewed,
            edit_history=edit_history,
        )
        try:
            completion = self._client.edit.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except InceptionError as exc:
            raise _translate(exc) from exc
        return Completion(
            text=completion.choices[0].message.content or "",
            usage=_usage(getattr(completion, "usage", None)),
        )

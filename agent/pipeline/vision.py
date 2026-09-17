"""The one place in Otto that builds a multi-block message.

Everything else in this codebase is text-only, deliberately and load-bearingly
so. `agent/pipeline/nodes.py`'s `_content_text` silently drops any non-text
block; `task_text` is read straight off `state["messages"][-1].content` in
three prompt builders; `agent/memory/wiring.py` asserts message content is
always a plain string and does `str(message.content)` on it. That last one is
not a style note -- a real image block reaching the memory queue would write
megabytes of base64 into the compaction buffer, blow the X budget on a single
turn, and then be handed to the summarizer as a prompt.

So images do not enter the graph. A vision-capable model looks at the file and
returns *words*, and those words travel as ordinary tool output. The cost is
real and worth stating: the model doing the reasoning is not the model doing
the looking, it cannot re-examine the image while writing code against it, and
it has no way to tell that the describer was wrong. What makes that workable
rather than useless is that the caller asks a QUESTION rather than requesting a
caption, and can ask again -- narrowing across several calls turns a captioner
into something closer to an oracle.

Deliberately not routed through `nodes._call`: importing nodes from tools would
close an import cycle, and `_call`'s diffusion-retry logic is specific to
Mercury and wrong for a vision model.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_image_block

#: Magic bytes, not the file extension and not `imghdr` (removed in Python
#: 3.13). An agent that has just written a file may well not have named it
#: helpfully, and a wrong media type is rejected by the vendor rather than
#: quietly mishandled.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

DEFAULT_QUESTION = (
    "Describe this image in full detail. Transcribe any text in it verbatim, "
    "and describe layout, colours and structure precisely enough that someone "
    "who cannot see it could reproduce it."
)


def sniff_media_type(data: bytes) -> str | None:
    """The image's media type from its own first bytes, or None if it is not
    an image this code recognises."""
    for signature, media_type in _SIGNATURES:
        if data.startswith(signature):
            return media_type
    # WEBP is "RIFF" + 4 size bytes + "WEBP", so it needs an offset check.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _flatten(content) -> str:
    """A reply's text, whether the vendor returned a string or a block list.

    A local copy rather than an import of nodes._content_text, for the cycle
    reason in the module docstring. Gemini does return block lists.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def describe_image(llm, image_b64: str, media_type: str, question: str) -> str:
    """Ask `llm` about an image and return its answer as plain text."""
    message = HumanMessage(content=[
        {"type": "text", "text": question or DEFAULT_QUESTION},
        create_image_block(base64=image_b64, mime_type=media_type),
    ])
    return _flatten(llm.invoke([message]).content).strip()


def describe_with_fallback(router, image_b64: str, media_type: str, question: str,
                           **overrides) -> tuple[str, str]:
    """The answer from the first model in the VISION chain that gives one, and "provider:model".

    A failed call is reported to agent/router/health.py -- an account out of credit cools its
    whole vendor for an hour, a rate limit cools the model -- and the next candidate is tried.
    When none answers, the last failure is raised with every model's reason in its message.
    `overrides` go to each model (a retry count, a timeout)."""
    from agent.router import health
    from agent.router.llm_provider.base import ProviderError, translate_unknown
    from agent.router.mapping import Task

    reasons: list[str] = []
    last: Exception | None = None
    for decision in router.chain(Task.VISION):
        name = f"{decision.provider}:{decision.model.id}"
        if why := health.HEALTH.cooling(decision.provider, decision.model.id):
            reasons.append(f"{name} skipped ({why})")
            continue
        try:
            llm = router.model_for(decision, **overrides)
            answer = describe_image(llm, image_b64, media_type, question)
        except Exception as exc:
            health.note_failure(exc, provider=decision.provider, model_id=decision.model.id)
            last = exc if isinstance(exc, ProviderError) else translate_unknown(
                exc, provider=decision.provider, model_id=decision.model.id)
            reasons.append(f"{name}: {str(last)[:160]}")
            continue
        health.HEALTH.note_success(decision.provider, decision.model.id)
        return answer, name
    detail = "; ".join(reasons) or "no vision model is configured"
    raise ProviderError(f"no vision model could answer -- {detail}") from last

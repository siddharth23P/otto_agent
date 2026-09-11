"""Per-run binding of tools that only exist for one run.

agent/pipeline/tools.py's TOOL_DISPATCH is Otto's permanent toolbox: twelve
tools that mean the same thing in every run, named in every prompt, and
derived once at import into nodes.py's _TOOL_MENU/_ACTION_BLOCK. That is the
right shape for a toolbox whose contents are a property of the agent.

It is the wrong shape for a benchmark that hands the agent a DIFFERENT set of
tools per task. Claw-Eval (agent/eval/claw_bench.py) is the case that forced
this: each of its 300 tasks declares its own JSON-schema tools --
`gmail_search`, `calendar_create_event`, `crm_update_contact` -- backed by a
mock HTTP service started for that task alone. None of them belong in
TOOL_DISPATCH; all of them have to be callable by name, with a real schema,
for the run to be an attempt at the task rather than a refusal.

So this is the seam, and it is the same contextvar shape agent/memory/
session.py, agent/pipeline/workspace.py and agent/pipeline/execution.py
already use, for the same reason: a tool reached through _tool_loop is a plain
function of one string, with no way to see the run it belongs to.

NOTHING BOUND IS THE NORMAL CASE. An `otto chat` turn binds no extra tools,
every prompt keeps exactly the menu it was built with at import, and the merge
below is a dict copy of nothing. Only a harness that deliberately binds a
toolkit changes what the agent may call, and only for the duration of its own
`with` block.

What a caller has to supply, beyond the callable: a `description` and a
`schema`. Otto's protocol is text -- `ACTION: <name>` then a `CODE:` body --
not native tool-calling, so the model cannot be handed a tool definition by
the API. It has to be TOLD, in the prompt, that the tool exists and what its
body should look like. `render_note()` below is that telling, and it is the
reason description and schema are required rather than optional niceties.
"""
from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

from agent.pipeline.tools import TOOL_DISPATCH, ToolResult


@dataclass(frozen=True, slots=True)
class ExtraTool:
    """One run-scoped tool.

    `call` takes the raw CODE: body exactly as a TOOL_DISPATCH entry does, so
    _tool_loop needs no branch for which kind of tool it just invoked. A tool
    whose arguments are structured parses that body itself -- see
    `json_body()` below, which is what a JSON-schema tool should use so that
    a malformed body comes back as a failed ToolResult the model can read and
    correct, rather than as an exception that kills the node.
    """

    name: str
    description: str
    call: Callable[[str], ToolResult]
    #: JSON Schema for the body, when the body is a JSON object. Empty means
    #: the body is free text and the description says what it should contain.
    schema: dict[str, Any] = field(default_factory=dict)


_current: contextvars.ContextVar[Mapping[str, ExtraTool]] = contextvars.ContextVar(
    "otto_current_extra_tools", default={},
)


@contextmanager
def bind_extra_tools(tools: list[ExtraTool] | Mapping[str, ExtraTool] | None) -> Iterator[None]:
    """Make `tools` callable by name for this block and anything it calls.

    Replaces rather than merges with an outer binding: a harness running task
    B must not inherit task A's tools, and nesting two toolkits is not a case
    that exists. Passing None or an empty collection unbinds, so a harness can
    bind per task without leaking one into the next.
    """
    if tools is None:
        mapping: Mapping[str, ExtraTool] = {}
    elif isinstance(tools, Mapping):
        mapping = dict(tools)
    else:
        mapping = {t.name: t for t in tools}
    token = _current.set(mapping)
    try:
        yield
    finally:
        _current.reset(token)


def current_extra_tools() -> Mapping[str, ExtraTool]:
    """The toolkit bound by the innermost `bind_extra_tools()`, or an empty
    mapping -- the normal state for a chat turn, and not an error."""
    return _current.get()


def dispatch_table() -> dict[str, Callable[[str], ToolResult]]:
    """TOOL_DISPATCH plus whatever is bound, as one lookup for _tool_loop.

    Extra tools win on a name collision, deliberately: a benchmark that
    declares its own `read_file` against its own service means that one, and
    silently serving Otto's would make the agent act on the wrong filesystem.
    """
    extra = current_extra_tools()
    if not extra:
        return dict(TOOL_DISPATCH)
    return {**TOOL_DISPATCH, **{name: t.call for name, t in extra.items()}}


def _one_line_schema(schema: dict[str, Any]) -> str:
    """The parts of a JSON Schema a caller needs to write a valid body:
    each property's name and type, and which ones are required.

    Not the whole schema. A task with five tools would spend several thousand
    characters on `"additionalProperties": false` and nested `"title"` keys,
    and nodes.py's own prompt-length finding (a fifth debugging habit erased
    the effect of the four before it) says that is not free.
    """
    props = schema.get("properties") or {}
    if not props:
        return "{}"
    required = set(schema.get("required") or ())
    parts = []
    for key, spec in props.items():
        kind = (spec or {}).get("type", "any")
        if (spec or {}).get("enum"):
            kind = "|".join(json.dumps(v) for v in spec["enum"])
        parts.append(f"{key}: {kind}" + ("" if key in required else " (optional)"))
    return "{" + ", ".join(parts) + "}"


def render_note(tools: Mapping[str, ExtraTool] | None = None) -> str:
    """The block that tells a prompt these tools exist. Empty string when
    nothing is bound, so a caller can append it unconditionally.

    Written as an ADDITION to the menu each prompt already carries rather than
    a replacement for it: the standing tools still work, and saying so stops
    the model treating the note as a narrowing.
    """
    tools = current_extra_tools() if tools is None else tools
    if not tools:
        return ""
    lines = [
        "TOOLS FOR THIS TASK, in addition to the ones already listed. "
        "Same protocol: ACTION: <name> then CODE: with the body below. "
        "The body is a single JSON object on one line -- no prose around it, "
        "no code fence.",
    ]
    for tool in tools.values():
        shape = _one_line_schema(tool.schema) if tool.schema else "(free text)"
        lines.append(f"- {tool.name} {shape} -- {tool.description.strip()}")
    return "\n".join(lines)


def json_body(tool_name: str, body: str) -> dict[str, Any] | ToolResult:
    """Parse a CODE: body as the JSON object a schema-shaped tool expects.

    Returns the object, or a FAILED ToolResult saying what was wrong -- which
    the caller returns as-is. Never raises: a model that emitted prose instead
    of JSON should see a tool result it can correct on the next iteration,
    which is exactly what a returncode of 1 gets it (nodes.py's _tool_loop
    feeds stdout/stderr/returncode straight back).

    Tolerant of the two things models reliably do anyway -- wrapping the object
    in a ```json fence, and adding a sentence before it -- because refusing
    those costs a whole iteration to teach something the parse can just handle.
    """
    text = body.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return ToolResult(
            stdout="",
            stderr=f"{tool_name}: the CODE: body must be a JSON object, e.g. {{\"query\": \"...\"}}",
            returncode=1,
        )
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return ToolResult(
            stdout="", stderr=f"{tool_name}: the CODE: body is not valid JSON ({exc})", returncode=1,
        )
    if not isinstance(parsed, dict):
        return ToolResult(
            stdout="", stderr=f"{tool_name}: the CODE: body must be a JSON OBJECT, not a {type(parsed).__name__}",
            returncode=1,
        )
    return parsed

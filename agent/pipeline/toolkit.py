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
import logging
import re
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
    #: Whether calling this changes something outside Otto -- sends a message,
    #: writes a record, moves money. agent/pipeline/tools.py's TOOL_TIERS says
    #: this for the standing tools; a run-scoped tool has to say it itself,
    #: and a benchmark's `gmail_send_message` is exactly the case that matters.
    #:
    #: Defaults to True, which is the safe direction: an unmarked tool is
    #: treated as irreversible and gated. A caller that knows a tool only
    #: reads says so explicitly.
    mutates: bool = True
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

logger = logging.getLogger(__name__)


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


#: Longest a task-supplied name or description may be once it reaches the
#: prompt. A description is a hint about a tool's shape, and one longer than
#: this is either a mistake or an attempt to spend the prompt.
MAX_TOOL_NAME_CHARS = 64
MAX_TOOL_DESCRIPTION_CHARS = 300

#: A tool name the ACTION parser can actually resolve. `_resolve_tool` in
#: agent/pipeline/nodes.py pulls identifiers out of an ACTION line with
#: exactly this shape, so a name with a space in it is UNREACHABLE through the
#: protocol -- advertising one promises something that cannot be called.
_CALLABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Anything that could start a new line of framework text, plus the control
#: characters that render as nothing and hide what follows them.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


def _inert(text: str, limit: int) -> str:
    """One line of at most `limit` characters, safe to put in a prompt.

    THIS IS A TRUST BOUNDARY, and it was open. `render_note` builds a
    SystemMessage, and `ExtraTool.name` and `.description` arrive verbatim
    from a benchmark's own task file -- agent/eval/claw_bench.py passes
    `spec.name` and `spec.description` straight through, deliberately and
    documented as such. `.strip()` trims the ends and leaves the middle alone,
    so a description containing a newline followed by `FINAL:` or
    `TOOL RESULT:` forges framework-level text inside the system role.

    deer-flow escapes tool names because it renders them into XML-ish tags a
    crafted name could close. Otto has no tags, so escaping is the wrong
    mechanism; what matters here is that untrusted text cannot start a line.
    Collapsing control characters does that, and the length bound stops a
    description being used to spend the prompt.
    """
    collapsed = _CONTROL.sub(" ", text or "").strip()
    collapsed = " ".join(collapsed.split())
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1].rstrip() + "\u2026"
    return collapsed


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
        name = _inert(tool.name, MAX_TOOL_NAME_CHARS)
        if not _CALLABLE_NAME.match(name):
            # Advertising a name the parser cannot resolve promises a tool
            # that cannot be called, and the model spends turns finding out.
            logger.warning("run-scoped tool %r is not a callable name", tool.name)
            continue
        shape = _inert(_one_line_schema(tool.schema) if tool.schema else "(free text)", 400)
        lines.append(f"- {name} {shape} -- {_inert(tool.description, MAX_TOOL_DESCRIPTION_CHARS)}")
    return "\n".join(lines)


#: JSON Schema types Otto checks. Anything else in a schema is accepted
#: without comment -- the point is catching a call that CANNOT work, not
#: reimplementing a validator.
_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_against(schema: dict[str, Any], parsed: dict[str, Any]) -> str:
    """What is wrong with this call, or "" if nothing is.

    Checked in code rather than described in a prompt, because the failure this
    addresses is the one named as dominant in production: plausible reasoning
    decoupled from the output contract -- an action that reads perfectly and
    cannot possibly work. A text protocol like Otto's widens that gap, since
    nothing structurally enforces the shape of a call. The counter-evidence is
    that closing it in code works: one system eliminated every illegal move
    across 145 environments by validating rather than instructing.

    Catching it HERE rather than at the far end of an HTTP round trip means the
    model is told what it got wrong in the terms of its own call -- "you left
    out `query`" -- instead of reading a 400 back from someone else's service
    and inferring.

    Deliberately shallow: missing required fields, wrong primitive types,
    unknown fields. Not a JSON Schema implementation, and it never rejects
    something it merely does not understand.
    """
    if not schema:
        return ""
    properties = schema.get("properties") or {}
    required = [k for k in (schema.get("required") or ()) if isinstance(k, str)]

    missing = [k for k in required if k not in parsed]
    if missing:
        return (
            f"missing required field(s): {', '.join(missing)}. "
            f"This call takes {_one_line_schema(schema)}"
        )

    for key, value in parsed.items():
        spec = properties.get(key)
        if spec is None:
            if properties:
                return (
                    f"unknown field {key!r}. This call takes "
                    f"{_one_line_schema(schema)}"
                )
            continue
        expected = _TYPE_CHECKS.get((spec or {}).get("type", ""))
        # bool is an int in Python, and a schema asking for a number does not
        # mean True.
        if expected and (not isinstance(value, expected)
                         or (expected != (bool,) and isinstance(value, bool))):
            return (
                f"field {key!r} should be {spec['type']}, got "
                f"{type(value).__name__}"
            )
        allowed = (spec or {}).get("enum")
        if allowed and value not in allowed:
            return f"field {key!r} must be one of {allowed}, got {value!r}"
    return ""


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

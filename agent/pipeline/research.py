"""The research workflow: a long document, built section by section.

A predefined code path, not a mode. The agent loop (nodes.py) is the right
shape for a task with a checkable result and the wrong shape for a document:
its prompt asks for "the numbers, the names, the decision", its replies stop
at the seat's max_tokens, `read_file` clips at 4000 characters, `_compact`
never shrinks the model's own prose, and the budget tells it to stop
gathering at a fifth of its calls. Asked for ten generations of a fictional
empire with a 500-word narrative each, it wrote a simulation, printed the
numbers, and answered in 300 words -- and the judge, whose criteria may not
mention length, approved. Every one of those pushes toward compression, and
none of them is the model being lazy.

So here the model never decides control flow. The workflow does, in this
order, and the model fills in content:

  1. OUTLINE (one call, the plan seat): a JSON plan -- sections, each with a
     brief, a word minimum, the sub-headings it must contain, one-line checks,
     and questions to look up first -- plus the seed of the continuity ledger.
  2. per section, in order:
     GATHER (optional, a bounded agent loop in `find` mode) writes notes to a
     file; WRITE (one call, the reason seat) returns the section as plain
     Markdown ending in a fenced ledger block -- no ACTION protocol, so the
     8192 tokens go to prose and nothing is truncated at a line that happens
     to start with `FINAL:`; CHECK in code (words, required parts, protocol
     leak, truncation) and optionally one call against the outline's checks;
     REVISE once if anything failed, keeping the better draft.
  3. ASSEMBLE in code: title, abstract, contents, the sections in order.
  4. CONVERT when the request named docx, pdf or xlsx.
  5. Hand the evaluator a REPORT computed from the files -- words against
     minimums, parts found, checks passed -- and the weakest section in
     full. It cannot read thirty thousand words through a 4000-character
     clip, so it judges the report and may spend its one check on the file.

Continuity is the ledger: a small JSON of named entities, settled facts and
open threads that each writer receives, updates, and hands to the next, so
the tenth section can refer back to a law the first one passed. The writer
also sees the tail of the previous section. When a writer returns no usable
ledger the previous one is carried, so continuity is never worse than "the
last 1500 characters plus whatever ledger survived".

Everything the workflow spends goes through nodes.py's `_call`, so the
budget, the usage ledger and `model_calls` are right without any plumbing
of their own. What it does with a budget: skip the reconnaissance note
(workers must not be told to stop looking on their first iteration), drop
gathering and checks in the wrap-up stretch, and when the budget is spent,
assemble what exists, say which sections are missing, and end without a
judgment -- the same exit the loop takes, for the same reason.

Files, under `otto_research/<task-slug>/` in the workspace: outline.json,
ledger.json, sections/NN-slug.md, notes/NN-slug.md, document.md and the
converted file if one was asked for. A re-entry after an evaluator rejection
reads those back rather than carrying them in state, rewrites the weakest
sections with the judge's feedback, and reassembles.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from agent.pipeline import nodes as pn
from agent.pipeline.budget import Budget, current_budget
from agent.pipeline.profile import disabled_tools
from agent.pipeline.progress import report as report_progress
from agent.pipeline.state import AgentState
from agent.pipeline.tools import execute_python, reachable_tools, write_file
from agent.pipeline.toolkit import render_note
from agent.pipeline.python_session import python_session_note
from agent.pipeline.workspace import OutsideWorkspace, resolve_in_workspace, workspace_note
from agent.router.llm_provider.base import ProviderError
from agent.router.mapping import Task

logger = logging.getLogger(__name__)

#: Where a document and its working files live, relative to the workspace.
RESEARCH_DIR = "otto_research"
#: An outline longer than this is a book, and a run that long is not what a
#: single turn's budget can pay for.
MAX_SECTIONS = 40
#: The floor and ceiling on a section's word minimum. The ceiling is the
#: writer's reply: the reason seat allows 8192 tokens, reasoning included,
#: and a section that needs more than 2000 words of prose plus a ledger is
#: one the outline must split.
MIN_SECTION_WORDS = 150
MAX_SECTION_WORDS = 2000
#: How much of the previous section the writer sees, so the join reads as
#: one document even when the ledger is thin.
PREVIOUS_TAIL_CHARS = 1500
#: How much gathered material the writer sees for one section.
NOTES_CHARS = 6000
#: How much request text the writer sees. The outline saw all of it.
REQUEST_CHARS = 4000
#: The ledger's size cap. Past it, facts are dropped from the MIDDLE: the
#: founding facts and the latest ones are the two ends a late section most
#: needs, and the first live run lost the Merchant Charter of Generation 1
#: by Generation 10 under an oldest-first rule at 3000.
LEDGER_MAX_CHARS = 6000
#: A gathering worker's iterations -- fewer than a delegate's, because it has
#: one question list and one file to write.
GATHER_ITERATIONS = 6
#: How many times the outline call may be asked for one JSON object.
MAX_OUTLINE_ATTEMPTS = 2
#: How many sections an evaluator rejection rewrites, weakest first.
MAX_REJECTION_REWRITES = 3
#: What a section costs at the full tier: gather, write, check, revise. The
#: tier drops when the budget cannot afford that for every section left.
CALLS_PER_SECTION_FULL = 4
#: Calls held back for the judgment at the end.
JUDGMENT_RESERVE = 2
#: Reply room for the outline and the writers, in tokens. The seats' own
#: settings are sized for the loop's replies -- a tool call, an answer. A
#: ten-section outline is 17,000 characters of JSON and stopped at the plan
#: seat's 4096 on the first live run (parse failed twice, the run fell back
#: to the loop); a 2000-word section plus a ledger on a reasoning model, whose
#: reasoning counts against the same cap, needs more than 8192.
OUTLINE_MAX_TOKENS = 16384
SECTION_MAX_TOKENS = 16384

FORMATS = ("md", "docx", "pdf", "xlsx")


@dataclass(frozen=True, slots=True)
class Section:
    index: int
    slug: str
    title: str
    brief: str
    min_words: int
    required_parts: tuple[str, ...]
    checks: tuple[str, ...]
    research_questions: tuple[str, ...]

    @property
    def path(self) -> str:
        return f"sections/{self.index:02d}-{self.slug}.md"

    @property
    def notes_path(self) -> str:
        return f"notes/{self.index:02d}-{self.slug}.md"


@dataclass(frozen=True, slots=True)
class Outline:
    title: str
    abstract: str
    format: str
    ledger: dict
    sections: tuple[Section, ...]


@dataclass(slots=True)
class SectionCheck:
    """What the code can say about one section without a model."""

    words: int
    min_words: int
    missing_parts: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    leaked_protocol: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures()

    def failures(self) -> list[str]:
        out = []
        if self.leaked_protocol:
            out.append("the reply was a tool call, not the section -- write the prose itself")
        if self.words < self.min_words:
            out.append(f"{self.words} words of prose; the minimum is {self.min_words}")
        for part in self.missing_parts:
            out.append(f"no `### {part}` heading -- that part is required, named exactly")
        if self.truncated:
            out.append("the section appears cut off: it does not end in a sentence "
                       "and has no ledger block")
        out.extend(f"failed: {check}" for check in self.failed_checks)
        return out


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

OUTLINE_PROMPT = (
    "Plan a long document that satisfies the request below. Reply with ONE "
    "JSON object and nothing else -- no fence, no prose -- shaped like:\n"
    '{"title": str, "abstract": str, "format": "md" | "docx" | "pdf" | "xlsx", '
    '"ledger": {"entities": {"<name>": "<one-line current state>"}, '
    '"facts": [str], "open_threads": [str]}, '
    '"sections": [{"title": str, "brief": str, "min_words": int, '
    '"required_parts": [str], "checks": [str], "research_questions": [str]}]}\n\n'
    "The CRITERIA are binding: every count, length and required part in them "
    "must be reachable from this outline. If the request names N parts, there "
    "are N sections (or N groups of sections) -- never fewer. A section is "
    "what one writer produces in one sitting: 150 to 2000 words; split "
    "anything longer into more sections. `brief` says what the section covers "
    "and how it connects to what came before. `required_parts` are the "
    "sub-headings it must contain, in order, named exactly; when the request "
    "names parts every section must have, list them for every section. "
    "`min_words` is the least the whole section may run to, and it must "
    "cover every minimum the request states for its parts. `checks` are "
    "one-line statements someone could verify from the section alone; leave "
    "it empty when the counts say it all. `research_questions` are things to "
    "look up on the web or in the workspace before writing -- empty for "
    "anything written from the request itself, such as fiction or analysis "
    "of material already given. `ledger` seeds the continuity state: the "
    "named things and settled facts the whole document must keep straight. "
    "`format` is what the request asked the file to be; md when it did not say. "
    "KEEP IT COMPACT: the abstract under 80 words, each brief under 60, "
    "required part names short (the heading text only, such as \"Ruler "
    "Profile\", never a sentence). The plan is read by machines; the writing "
    "happens later."
)

WRITER_PROMPT = (
    "You are writing ONE section of a longer document that is being assembled "
    "section by section. You will be shown the request, the outline, this "
    "section's brief, and a CONTINUITY LEDGER: the named things and settled "
    "facts from the sections already written. Everything in the ledger is "
    "true in this document. Build on it, refer back to it by name where the "
    "brief calls for it, and never contradict or re-introduce what earlier "
    "sections established.\n\n"
    "Write the section in full, in Markdown, starting with the heading you "
    "are given and using `### ` headings for each required part, named "
    "exactly as listed. The minimum length is counted in words of prose -- "
    "write the thing itself, not a plan or a summary of what it would "
    "contain. No preamble and no closing remarks about the document.\n\n"
    "After the prose, add a fenced block that starts with ```ledger and "
    "holds the ledger as JSON, updated with what this section established: "
    "entities whose state changed, facts added, threads opened or closed. "
    "Keep it under 6000 characters -- drop the least important facts rather "
    "than exceed it, but never the founding ones. Nothing after the closing "
    "fence."
)

SECTION_CHECK_PROMPT = (
    "Below is one section of a longer document and a list of statements it "
    "must satisfy. Read the section, then reply with exactly:\nFAILED:\n"
    "followed by one line per statement that is NOT satisfied, quoting the "
    "statement, or the single word NONE. Nothing else."
)

RESEARCH_WORKER_CONTRACT = (
    "You have one bounded job for a longer document another process is "
    "assembling. Find out the following and nothing beyond it:\n\n{questions}\n\n"
    "Use web_search for the open web and read_file / execute_bash for this "
    "workspace. Write what you found -- specifics, numbers, names, and the "
    "source of each -- to the file {notes_path} with write_file. Then reply "
    "with exactly\nFINAL:\n<five lines: the findings someone who cannot see "
    "your working can write from>"
)

RESEARCH_JUDGE_NOTE = (
    "The table above is EXACT: the word counts, the parts found and the "
    "check results were computed by code from the section files, not "
    "reported by the model that wrote them, so there is nothing to recount. "
    "Judge whether those numbers and the weakest section (below, in full) "
    "satisfy the criteria. The document is at {path} in the workspace, one "
    "file per section under {sections_dir}; if one criterion truly cannot "
    "be settled from what is here, read a section file with read_file -- "
    "one tool call per reply, and your LAST reply must be the verdict in "
    "the FINAL format, whatever you have or have not managed to check."
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

_WORD = re.compile(r"\b\w+\b")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_LEDGER_BLOCK = re.compile(r"```ledger[^\n]*\n(.*?)\n[ \t]*```", re.S)
_SENTENCE_END = re.compile(r"[.!?:;\"'”’)\]*`_]\s*$")
_FORMAT_WORDS = re.compile(
    r"\b(docx|word doc\w*|ms word|pdf|xlsx|excel|spreadsheet)\b", re.I,
)


def _slugify(text: str, *, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "section"


def _words(body: str) -> int:
    """Words of prose: headings and fenced blocks do not count."""
    total = 0
    in_fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or _HEADING.match(line):
            continue
        total += len(_WORD.findall(line))
    return total


def _headings(body: str) -> list[str]:
    found = []
    in_fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and (m := _HEADING.match(line)):
            found.append(m.group(1))
    return found


def _anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).strip().replace(" ", "-")


def _write(rel: str, content: str) -> bool:
    result = write_file(f"{rel}\n{content}")
    if result.returncode != 0:
        logger.warning("research: could not write %s: %s", rel, result.stderr)
        return False
    return True


def _read(rel: str) -> str | None:
    try:
        path = resolve_in_workspace(rel)
    except OutsideWorkspace:
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _exists(rel: str) -> bool:
    try:
        return resolve_in_workspace(rel).exists()
    except OutsideWorkspace:
        return False


def _task_slug(task_text: str) -> str:
    base = _slugify(" ".join(task_text.split()[:6]), limit=48) or "document"
    slug, n = base, 2
    while _exists(f"{RESEARCH_DIR}/{slug}"):
        slug = f"{base}-{n}"
        n += 1
    return slug


def _with_room(llm, max_tokens: int):
    """The seat's model with at least `max_tokens` of reply allowed.

    The routed model is a pydantic object whose cap is `max_tokens` on
    three vendors and `max_output_tokens` on Gemini; a model_copy with a
    bigger one is the same move nodes.py's _call makes for a diffusing
    retry. A model with no cap set, or one already roomier, is returned as
    it is."""
    for attr in ("max_tokens", "max_output_tokens"):
        current = getattr(llm, attr, None)
        if isinstance(current, int) and current < max_tokens:
            try:
                return llm.model_copy(update={attr: max_tokens})
            except Exception:  # not a pydantic model; use it as it is
                return llm
    return llm


def _emit_board(*lines: str) -> None:
    pn._emit({"research": {"board": list(lines)}})


def _spent(budget: Budget | None) -> bool:
    return budget is not None and budget.spent()


def _tier(budget: Budget | None, remaining_sections: int) -> str:
    """How much each remaining section may cost: "full", "lean" or "bare".

    Full is gather + write + check + revise. Lean drops gathering and the
    model check, keeping one revision for a section the code checks fail.
    Bare is the write alone. The wrap-up stretch is always bare: the budget
    has said so, and a workflow that argued would be the loop it replaced.
    """
    if budget is None:
        return "full"
    if budget.phase() == "wrap_up":
        return "bare"
    if budget.max_model_calls is None:
        return "full"
    remaining = budget.max_model_calls - budget.calls - JUDGMENT_RESERVE
    per_section = remaining / max(remaining_sections, 1)
    if per_section >= CALLS_PER_SECTION_FULL:
        return "full"
    if per_section >= 2:
        return "lean"
    return "bare"


# --------------------------------------------------------------------------
# outline
# --------------------------------------------------------------------------

def _parse_outline(text: str) -> Outline:
    """One JSON object out of the outline reply, validated and clamped.
    Raises ValueError with a reason the model can act on."""
    stripped = pn._strip_code_fence(text).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    try:
        data = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} at character {exc.pos}") from None
    if not isinstance(data, dict):
        raise ValueError("the JSON must be an object")
    raw_sections = data.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("`sections` must be a non-empty list")
    title = str(data.get("title") or "").strip() or "Document"
    fmt = str(data.get("format") or "md").strip().lower()
    if fmt not in FORMATS:
        fmt = "md"

    sections: list[Section] = []
    used: set[str] = set()
    for i, raw in enumerate(raw_sections[:MAX_SECTIONS], start=1):
        if not isinstance(raw, dict) or not str(raw.get("title") or "").strip():
            raise ValueError(f"section {i} has no title")
        stitle = str(raw["title"]).strip()
        slug, n = _slugify(stitle), 2
        while slug in used:
            slug = f"{_slugify(stitle)}-{n}"
            n += 1
        used.add(slug)
        try:
            min_words = int(raw.get("min_words") or MIN_SECTION_WORDS)
        except (TypeError, ValueError):
            min_words = MIN_SECTION_WORDS
        min_words = max(MIN_SECTION_WORDS, min(MAX_SECTION_WORDS, min_words))
        sections.append(Section(
            index=i, slug=slug, title=stitle,
            brief=str(raw.get("brief") or "").strip(),
            min_words=min_words,
            required_parts=_strings(raw.get("required_parts")),
            checks=_strings(raw.get("checks")),
            research_questions=_strings(raw.get("research_questions")),
        ))
    return Outline(
        title=title,
        abstract=str(data.get("abstract") or "").strip(),
        format=fmt,
        ledger=_normalise_ledger(data.get("ledger")) or _empty_ledger(),
        sections=tuple(sections),
    )


def _strings(value) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v).strip() for v in value if str(v).strip())


def _outline(task_text: str, criteria: list[str]) -> Outline:
    """Phase one: the plan, on the plan seat. Raises ValueError when two
    attempts produced no usable JSON, ProviderError when the seat is down."""
    llm = _with_room(pn.ROUTER.chat_model(Task.PLAN), OUTLINE_MAX_TOKENS)
    body = f"TASK:\n{task_text}"
    if criteria:
        body += "\n\nCRITERIA (binding):\n" + "\n".join(f"- {c}" for c in criteria)
    if note := workspace_note():
        body += f"\n\nWORKSPACE:\n{note}"
    messages: list = [SystemMessage(OUTLINE_PROMPT), HumanMessage(body)]
    error: ValueError | None = None
    for _ in range(MAX_OUTLINE_ATTEMPTS):
        reply = pn._call(llm, messages)
        try:
            return _parse_outline(reply)
        except ValueError as exc:
            error = exc
            messages.append(AIMessage(reply or "(empty)"))
            messages.append(HumanMessage(
                f"That was not one usable JSON object: {exc}. Reply again with "
                "only the object, shaped exactly as described."
            ))
    raise error or ValueError("no outline")


def _outline_to_json(outline: Outline) -> str:
    return json.dumps(dataclasses.asdict(outline), indent=1)


def _outline_from_json(text: str) -> Outline:
    data = json.loads(text)
    return Outline(
        title=data["title"], abstract=data.get("abstract", ""),
        format=data.get("format", "md"),
        ledger=_normalise_ledger(data.get("ledger")) or _empty_ledger(),
        sections=tuple(Section(
            index=s["index"], slug=s["slug"], title=s["title"], brief=s.get("brief", ""),
            min_words=s.get("min_words", MIN_SECTION_WORDS),
            required_parts=tuple(s.get("required_parts", ())),
            checks=tuple(s.get("checks", ())),
            research_questions=tuple(s.get("research_questions", ())),
        ) for s in data["sections"]),
    )


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

def _empty_ledger() -> dict:
    return {"entities": {}, "facts": [], "open_threads": []}


def _normalise_ledger(value) -> dict | None:
    """The three keys, each the right shape, or None for anything that is
    not a ledger at all. Missing keys are filled; wrong-typed ones dropped."""
    if not isinstance(value, dict):
        return None
    entities = value.get("entities")
    facts = value.get("facts")
    threads = value.get("open_threads")
    return {
        "entities": ({str(k): str(v) for k, v in entities.items()}
                     if isinstance(entities, dict) else {}),
        "facts": [str(f) for f in facts] if isinstance(facts, list) else [],
        "open_threads": [str(t) for t in threads] if isinstance(threads, list) else [],
    }


def _split_ledger(reply: str) -> tuple[str, dict | None]:
    """The section's prose and its ledger, apart. The LAST ```ledger block
    is the ledger; the prose is everything else, fence-stripped if the model
    wrapped the whole reply. None when there is no block or it is not JSON."""
    blocks = list(_LEDGER_BLOCK.finditer(reply))
    if not blocks:
        return pn._strip_code_fence(reply).strip(), None
    last = blocks[-1]
    body = (reply[:last.start()] + reply[last.end():]).strip()
    body = pn._strip_code_fence(body).strip()
    try:
        ledger = _normalise_ledger(json.loads(last.group(1)))
    except json.JSONDecodeError:
        ledger = None
    return body, ledger


def _merge_ledger(old: dict, new: dict | None) -> dict:
    """The next section's ledger: the old one when the writer returned none.

    Otherwise a UNION, not a replacement. A writer rewriting the ledger
    drops what its own section did not touch -- on the first live run the
    founding charter was gone from the ledger by the tenth section, and the
    tenth section did not mention it. So an entity the new ledger omits is
    kept with its old state, a fact it omits is kept, and the writer's
    versions win where both exist. Open threads are the writer's: closing
    one is the point of the field.

    Trimmed to LEDGER_MAX_CHARS from the middle of the facts, so the
    founding facts and the latest survive; then open threads; entities go
    last, oldest first, because a name is the cheapest thing to keep and the
    most expensive to lose.
    """
    if new is None:
        return old
    entities = {**old.get("entities", {}), **new.get("entities", {})}
    facts = list(old.get("facts", []))
    facts += [f for f in new.get("facts", []) if f not in facts]
    merged = {"entities": entities, "facts": facts,
              "open_threads": list(new.get("open_threads", []))}
    while len(json.dumps(merged)) > LEDGER_MAX_CHARS:
        if len(merged["facts"]) > 2:
            middle = len(merged["facts"]) // 2
            merged["facts"] = merged["facts"][:middle] + merged["facts"][middle + 1:]
        elif merged["open_threads"]:
            merged["open_threads"] = merged["open_threads"][1:]
        elif merged["entities"]:
            first = next(iter(merged["entities"]))
            merged["entities"] = {k: v for k, v in merged["entities"].items() if k != first}
        else:
            break
    return merged


# --------------------------------------------------------------------------
# gathering
# --------------------------------------------------------------------------

def _spawn_worker(state: AgentState, instruction: str, *, mode: str = "find",
                  max_iterations: int = GATHER_ITERATIONS,
                  actions: list[str]) -> str:
    """One bounded agent loop with a contract, no parent conversation, and
    a report back -- the delegate shape (nodes.py's _delegate), with the
    caller rather than a model choosing the mode. The child's actions join
    the record; its conversation is discarded."""
    child: list = [SystemMessage(pn.compose_agent_prompt(reachable_tools(),
                                                         may_delegate=False))]
    for extra in (workspace_note(), python_session_note(), render_note()):
        if extra:
            child.append(SystemMessage(extra))
    child.append(HumanMessage(instruction))
    child.append(pn._mode_message(mode))
    taken: list[str] = []
    try:
        output, why, _ = pn._agent_loop(
            state, child, mode=mode, actions=taken, mode_log=[],
            max_iterations=max_iterations, may_delegate=False,
        )
    except pn.NeedsUserInput as exc:
        # A worker cannot pause the run; the workflow has no way to relay a
        # question and the writer can work without the answer.
        actions.append(f"research(worker): asked a question and was refused: "
                       f"{exc.question[:80]}")
        return ""
    except ProviderError as exc:
        actions.append(f"research(worker): failed: {exc}")
        return ""
    actions.extend(f"{mode}(worker): {line}" for line in taken)
    return output.strip()


def _gather(state: AgentState, section: Section, dir_rel: str,
            actions: list[str]) -> str:
    notes_path = f"{dir_rel}/{section.notes_path}"
    instruction = RESEARCH_WORKER_CONTRACT.format(
        questions="\n".join(f"- {q}" for q in section.research_questions),
        notes_path=notes_path,
    )
    _emit_board(f"research: gathering for section {section.index}: "
                f"{section.research_questions[0][:60]}")
    report = _spawn_worker(state, instruction, actions=actions)
    notes = _read(notes_path) or ""
    combined = "\n\n".join(part for part in (notes.strip(), report) if part)
    return combined[:NOTES_CHARS]


# --------------------------------------------------------------------------
# writing and checking
# --------------------------------------------------------------------------

def _writer_body(section: Section, *, task_text: str, outline: Outline, ledger: dict,
                 previous_tail: str, notes: str,
                 revise: tuple[str, list[str]] | None) -> str:
    listing = "\n".join(
        f"{s.index}. {s.title}" + ("   <- this one" if s.index == section.index else "")
        for s in outline.sections
    )
    parts = [
        f"THE REQUEST:\n{task_text[:REQUEST_CHARS]}",
        f"THE DOCUMENT: {outline.title}\n{listing}",
        f"THIS SECTION -- start with this heading:\n## {section.title}\n\n{section.brief}",
    ]
    if section.required_parts:
        parts.append("REQUIRED PARTS (as ### headings, in this order, named exactly):\n"
                     + "\n".join(f"- {p}" for p in section.required_parts))
    parts.append(f"MINIMUM: {section.min_words} words of prose, not counting headings.")
    if section.checks:
        parts.append("CHECKS THIS SECTION MUST PASS:\n"
                     + "\n".join(f"- {c}" for c in section.checks))
    parts.append(f"CONTINUITY LEDGER:\n{json.dumps(ledger, indent=1)}")
    if previous_tail:
        parts.append(f"THE PREVIOUS SECTION ENDS:\n...{previous_tail}")
    if notes:
        parts.append(f"NOTES GATHERED FOR THIS SECTION:\n{notes}")
    if revise:
        draft, failures = revise
        parts.append(
            f"YOUR DRAFT:\n{draft}\n\nIT FAILS:\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\n\nReturn the whole section again, complete, fixed, with the ledger block."
        )
    return "\n\n".join(parts)


def _write_section(section: Section, *, task_text: str, outline: Outline, ledger: dict,
                   previous_tail: str, notes: str,
                   revise: tuple[str, list[str]] | None = None,
                   ) -> tuple[str | None, dict | None]:
    """One call on the reason seat; the section's prose and its ledger.
    (None, None) when the seat failed twice."""
    llm = _with_room(pn.ROUTER.chat_model(Task.REASON), SECTION_MAX_TOKENS)
    messages = [
        SystemMessage(WRITER_PROMPT),
        HumanMessage(_writer_body(
            section, task_text=task_text, outline=outline, ledger=ledger,
            previous_tail=previous_tail, notes=notes, revise=revise,
        )),
    ]
    reply = ""
    for attempt in range(2):
        try:
            reply = pn._call(llm, messages)
            break
        except ProviderError as exc:
            logger.warning("research: writing section %d failed (%d/2): %s",
                           section.index, attempt + 1, exc)
            if attempt == 1:
                return None, None
    body, new_ledger = _split_ledger(reply)
    if body and not body.lstrip().startswith("## "):
        body = f"## {section.title}\n\n{body}"
    return body, new_ledger


def _check_section(section: Section, body: str, *, had_ledger: bool) -> SectionCheck:
    check = SectionCheck(words=_words(body), min_words=section.min_words)
    first = next((line for line in body.splitlines() if line.strip()), "")
    check.leaked_protocol = bool(pn._NEXT_DIRECTIVE.match(first))
    headings = [h.lower() for h in _headings(body)]
    check.missing_parts = [
        part for part in section.required_parts
        if not any(part.lower() in h for h in headings)
    ]
    check.truncated = not had_ledger and not _SENTENCE_END.search(body.rstrip())
    return check


def _judge_section(section: Section, body: str) -> list[str]:
    """One call on the evaluate seat against the outline's own checks. The
    statements it names as failed, or [] -- including on a provider error,
    which is not the section's fault."""
    llm = pn.ROUTER.chat_model(Task.EVALUATE)
    try:
        reply = pn._call(llm, [
            SystemMessage(SECTION_CHECK_PROMPT),
            HumanMessage("STATEMENTS:\n" + "\n".join(f"- {c}" for c in section.checks)
                         + f"\n\nSECTION:\n{body}"),
        ])
    except ProviderError as exc:
        logger.warning("research: checking section %d failed: %s", section.index, exc)
        return []
    failed = []
    after = reply.split("FAILED:", 1)[1] if "FAILED:" in reply else reply
    for line in after.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line and line.upper() != "NONE":
            failed.append(line)
    return failed


# --------------------------------------------------------------------------
# assembly, conversion, report
# --------------------------------------------------------------------------

def _assemble(outline: Outline, written: list[tuple[Section, str]],
              incomplete: list[Section]) -> str:
    total = sum(_words(body) for _, body in written)
    lines = [f"# {outline.title}", ""]
    if outline.abstract:
        lines += [f"_{outline.abstract}_", ""]
    status = f"{len(written)} sections, {total:,} words."
    if incomplete:
        status += (" INCOMPLETE -- not written: "
                   + ", ".join(f"{s.index} ({s.title})" for s in incomplete)
                   + " (budget ran out).")
    lines += [status, "", "## Contents", ""]
    lines += [f"{s.index}. [{s.title}](#{_anchor(s.title)})" for s, _ in written]
    lines.append("")
    for _, body in written:
        lines += [body.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def _wanted_format(task_text: str, outline: Outline) -> str | None:
    if outline.format != "md":
        return outline.format
    found = _FORMAT_WORDS.search(task_text)
    if not found:
        return None
    word = found.group(1).lower()
    if word.startswith("word") or word == "ms word" or word == "docx":
        return "docx"
    if word in ("xlsx", "excel", "spreadsheet"):
        return "xlsx"
    return "pdf"


def _convert(dir_rel: str, fmt: str) -> tuple[bool, str]:
    """document.md -> document.<fmt> in the same directory, by a script on
    the workspace's own interpreter. (ok, detail).

    Where the host has switched code execution off (agent/pipeline/
    profile.py) -- Otto embedded in the phone app has no interpreter to hand
    the script to -- the document is written in this process instead
    (agent/pipeline/documents.py). The Markdown stands either way."""
    if "execute_python" in disabled_tools():
        return _convert_here(dir_rel, fmt)
    try:
        source = resolve_in_workspace(f"{dir_rel}/document.md")
        target = resolve_in_workspace(f"{dir_rel}/document.{fmt}")
    except OutsideWorkspace as exc:
        return False, str(exc)
    script = (CONVERT_SCRIPTS[fmt]
              .replace("__SOURCE__", str(source))
              .replace("__TARGET__", str(target)))
    result = execute_python(script)
    if result.returncode == 0 and target.exists():
        return True, f"document.{fmt}"
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    return False, (detail[-1] if detail else f"exit {result.returncode}")


def _convert_here(dir_rel: str, fmt: str) -> tuple[bool, str]:
    from agent.pipeline import documents

    try:
        source = resolve_in_workspace(f"{dir_rel}/document.md")
        target = resolve_in_workspace(f"{dir_rel}/document.{fmt}")
        documents.write(source.read_text(encoding="utf-8"), target, fmt)
    except OutsideWorkspace as exc:
        return False, str(exc)
    except Exception as exc:  # a missing library, a writer's complaint: the Markdown stands
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"document.{fmt}"


def _report(outline: Outline, document_path: str, rows: list[dict],
            incomplete: list[Section], converted: tuple[str, bool, str] | None) -> str:
    total = sum(r["words"] for r in rows)
    head = (f"Document written to `{document_path}` -- {len(rows)} sections, "
            f"{total:,} words.")
    if converted:
        fmt, ok, detail = converted
        head += (f" Also `{detail}`." if ok
                 else f" Conversion to {fmt} failed ({detail}); the Markdown stands.")
    lines = [head]
    if incomplete:
        lines.append("INCOMPLETE -- the budget ran out before: "
                     + ", ".join(f"{s.index}. {s.title}" for s in incomplete))
    if outline.abstract:
        lines += ["", outline.abstract]
    lines += ["", "| # | section | words | min | parts | checks |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['index']} | {r['title']} | {r['words']:,} | {r['min']:,} | "
                     f"{r['parts']} | {r['checks']} |")
    return "\n".join(lines)


def _row(section: Section, check: SectionCheck, revised: bool) -> dict:
    parts = (f"{len(section.required_parts) - len(check.missing_parts)}/"
             f"{len(section.required_parts)}" if section.required_parts else "-")
    if check.ok:
        status = "ok"
    else:
        status = "; ".join(check.failures())[:120]
    if revised:
        status = f"revised once -- {status}"
    return {"index": section.index, "title": section.title, "words": check.words,
            "min": section.min_words, "parts": parts, "checks": status,
            "ratio": check.words / max(section.min_words, 1), "ok": check.ok}


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

def run_research(state: AgentState) -> Command:
    task_text = pn._requested(state)
    criteria = [c["text"] for c in (state.get("checklist") or [])]
    budget = current_budget()
    if budget is not None:
        budget.skip_recon()
    actions: list[str] = []

    document_path = state.get("document_path")
    if state.get("feedback") and document_path and _exists(
            f"{document_path.rsplit('/', 1)[0]}/outline.json"):
        return _revise(state, task_text, document_path, budget, actions)

    report_progress("phase", "outlining the document")
    try:
        outline = _outline(task_text, criteria)
    except (ValueError, ProviderError) as exc:
        _emit_board(f"research: could not outline the document ({exc}) -- "
                    "running it as an agent task")
        return Command(
            update={
                "node": "research", "route": "agent",
                "board": [f"research: could not outline the document ({exc}) -- "
                          "running it as an agent task"],
                **_calls(budget),
            },
            goto="agent",
        )
    dir_rel = f"{RESEARCH_DIR}/{_task_slug(task_text)}"
    _write(f"{dir_rel}/outline.json", _outline_to_json(outline))
    _emit_board(f"research: outlined {len(outline.sections)} sections -- {outline.title}")
    actions.append(f"research: outlined {len(outline.sections)} sections into {dir_rel}/")

    ledger = outline.ledger
    _write(f"{dir_rel}/ledger.json", json.dumps(ledger, indent=1))
    written: list[tuple[Section, str]] = []
    rows: list[dict] = []
    incomplete: list[Section] = []
    previous_tail = ""
    n = len(outline.sections)
    for section in outline.sections:
        if _spent(budget):
            incomplete.append(section)
            continue
        tier = _tier(budget, n - section.index + 1)
        report_progress("phase", f"writing section {section.index} of {n}: {section.title}")

        notes = ""
        if tier == "full" and section.research_questions:
            notes = _gather(state, section, dir_rel, actions)
            if _spent(budget):
                incomplete.append(section)
                continue

        body, new_ledger = _write_section(
            section, task_text=task_text, outline=outline, ledger=ledger,
            previous_tail=previous_tail, notes=notes,
        )
        if body is None:
            incomplete.append(section)
            actions.append(f"research: section {section.index} could not be written "
                           "(provider failure)")
            continue
        check = _check_section(section, body, had_ledger=new_ledger is not None)
        if tier == "full" and section.checks and not _spent(budget):
            check.failed_checks = _judge_section(section, body)

        revised = False
        if not check.ok and tier != "bare" and not _spent(budget):
            _emit_board(f"research: section {section.index} needs a revision -- "
                        + "; ".join(check.failures())[:100])
            body2, ledger2 = _write_section(
                section, task_text=task_text, outline=outline, ledger=ledger,
                previous_tail=previous_tail, notes=notes,
                revise=(body, check.failures()),
            )
            if body2 is not None:
                check2 = _check_section(section, body2, had_ledger=ledger2 is not None)
                if check.failed_checks and tier == "full" and not _spent(budget):
                    check2.failed_checks = _judge_section(section, body2)
                if len(check2.failures()) <= len(check.failures()):
                    body, new_ledger, check = body2, ledger2, check2
                revised = True

        _write(f"{dir_rel}/{section.path}", body)
        ledger = _merge_ledger(ledger, new_ledger)
        if new_ledger is None:
            _emit_board(f"research: section {section.index} returned no ledger -- "
                        "carrying the previous one")
        _write(f"{dir_rel}/ledger.json", json.dumps(ledger, indent=1))
        previous_tail = body[-PREVIOUS_TAIL_CHARS:]
        written.append((section, body))
        rows.append(_row(section, check, revised))
        actions.append(f"research: wrote {section.path} ({check.words} words"
                       + (", revised once" if revised else "") + ")")
        _emit_board(f"research: section {section.index}/{n} -- {check.words} words"
                    + ("" if check.ok else " (" + "; ".join(check.failures())[:80] + ")"))

    return _finish(state, task_text, outline, dir_rel, written, rows, incomplete,
                   budget, actions)


def _revise(state: AgentState, task_text: str, document_path: str,
            budget: Budget | None, actions: list[str]) -> Command:
    """Re-entry after an evaluator rejection: rewrite the weakest sections
    with the judge's feedback, reassemble, and go back for judgment."""
    dir_rel = document_path.rsplit("/", 1)[0]
    feedback = state.get("feedback") or ""
    try:
        outline = _outline_from_json(_read(f"{dir_rel}/outline.json") or "")
    except (ValueError, KeyError, TypeError) as exc:
        return Command(
            update={"node": "research", "route": "agent",
                    "board": [f"research: could not reload the outline ({exc}) -- "
                              "handing the rejection to the agent"],
                    **_calls(budget)},
            goto="agent",
        )
    # The ledger as it stands at the END -- the one before each rewritten
    # section is not kept. A rewrite of section 3 may therefore see facts
    # from 4 onward; one round of that is a smaller wrong than reconstructing
    # ten ledgers would cost.
    ledger = _normalise_ledger(json.loads(_read(f"{dir_rel}/ledger.json") or "{}")) \
        or outline.ledger
    report_progress("phase", "revising the document")
    resubmit = _no_verdict(feedback)
    _emit_board("research: the judge reached no verdict -- resubmitting the document unchanged"
                if resubmit else f"research: revising after rejection -- {feedback[:80]}")

    bodies: dict[int, str] = {}
    checks: dict[int, SectionCheck] = {}
    incomplete: list[Section] = []
    for section in outline.sections:
        body = _read(f"{dir_rel}/{section.path}")
        if body is None:
            incomplete.append(section)
            continue
        bodies[section.index] = body
        checks[section.index] = _check_section(section, body, had_ledger=True)

    candidates = sorted(
        (s for s in outline.sections if s.index in bodies),
        key=lambda s: (checks[s.index].ok, checks[s.index].words / max(s.min_words, 1)),
    )
    rewritten: set[int] = set()
    for section in ([] if resubmit else candidates[:MAX_REJECTION_REWRITES]):
        if _spent(budget):
            break
        previous = bodies.get(section.index - 1, "")[-PREVIOUS_TAIL_CHARS:]
        old = bodies[section.index]
        body, new_ledger = _write_section(
            section, task_text=task_text, outline=outline, ledger=ledger,
            previous_tail=previous, notes="",
            revise=(old, [f"the evaluator rejected the document: {feedback}",
                          *checks[section.index].failures()]),
        )
        if body is None:
            continue
        check = _check_section(section, body, had_ledger=new_ledger is not None)
        if len(check.failures()) <= len(checks[section.index].failures()):
            bodies[section.index] = body
            checks[section.index] = check
            ledger = _merge_ledger(ledger, new_ledger)
            _write(f"{dir_rel}/{section.path}", body)
        rewritten.add(section.index)
        actions.append(f"research: rewrote {section.path} after rejection "
                       f"({check.words} words)")
    _write(f"{dir_rel}/ledger.json", json.dumps(ledger, indent=1))

    written = [(s, bodies[s.index]) for s in outline.sections if s.index in bodies]
    rows = [_row(s, checks[s.index], s.index in rewritten) for s, _ in written]
    return _finish(state, task_text, outline, dir_rel, written, rows, incomplete,
                   budget, actions,
                   board=["research: the judge reached no verdict -- resubmitted unchanged"
                          if resubmit else
                          f"research: revising after rejection -- {feedback[:80]}"])


def _finish(state: AgentState, task_text: str, outline: Outline, dir_rel: str,
            written: list[tuple[Section, str]], rows: list[dict],
            incomplete: list[Section], budget: Budget | None,
            actions: list[str], board: list[str] | None = None) -> Command:
    board = list(board or [])
    report_progress("phase", "assembling the document")
    document_path = f"{dir_rel}/document.md"
    _write(document_path, _assemble(outline, written, incomplete))
    actions.append(f"research: assembled {document_path} ({len(written)} sections)")

    converted = None
    if (fmt := _wanted_format(task_text, outline)) and written:
        report_progress("phase", f"converting to {fmt}")
        ok, detail = _convert(dir_rel, fmt)
        converted = (fmt, ok, detail)
        actions.append(f"research: converted to {fmt}" if ok
                       else f"research: conversion to {fmt} failed: {detail}")

    report = _report(outline, document_path, rows, incomplete, converted)
    context = _judge_context(document_path, dir_rel, written, rows)

    update = {
        "node": "research", "route": "research",
        "output": report, "context": context,
        "document_path": document_path,
        "actions": actions, "feedback": "",
        **_calls(budget),
    }
    if _spent(budget) or not written:
        why = ("ran out of budget" if _spent(budget)
               else "wrote no sections")
        _emit_board(f"research: {why} -- answering with what exists, unverified")
        return Command(
            update={**update, "final_output": report,
                    "board": board + [f"research: {why} after {len(written)} of "
                                      f"{len(outline.sections)} sections -- answering "
                                      "with what exists, unverified"]},
            goto=END,
        )
    return Command(
        update={**update, "board": board + [f"research: document assembled -- "
                                            f"{len(written)} sections, ready for judgment"]},
        goto="evaluator",
    )


def _judge_context(document_path: str, dir_rel: str,
                   written: list[tuple[Section, str]], rows: list[dict]) -> str:
    """What the evaluator sees under CONTEXT GATHERED: that the table is
    exact, where the files are, and the weakest section in full (the
    evaluator clips it to its own limit)."""
    context = RESEARCH_JUDGE_NOTE.format(path=document_path,
                                         sections_dir=f"{dir_rel}/sections/")
    weakest = min(rows, key=lambda r: r["ratio"], default=None)
    if weakest is not None:
        body = next(b for s, b in written if s.index == weakest["index"])
        context += (f"\n\nWEAKEST SECTION IN FULL ({weakest['title']}, "
                    f"{weakest['words']} words, minimum {weakest['min']}):\n{body}")
    return context


def _no_verdict(feedback: str) -> bool:
    """Whether a rejection is the judge failing to judge rather than a
    finding: it ran out of its own replies, or answered in a tool-call
    format the loop does not read. Seen live on the first document run --
    three rejections, none about the document, each one paying for a
    revision pass. Not something to rewrite sections over."""
    head = pn.NO_VERDICT_NOTE.split(",", 1)[0]
    return feedback.startswith(head) or "<function_calls>" in feedback or "<invoke " in feedback


def _calls(budget: Budget | None) -> dict:
    return {"model_calls": budget.calls} if budget is not None else {}


# --------------------------------------------------------------------------
# conversion scripts. Run by execute_python on the workspace's interpreter;
# __SOURCE__ / __TARGET__ are replaced with absolute paths (str.replace, not
# .format, so the braces below are safe). Each prints "wrote <target>".
# --------------------------------------------------------------------------

_DOCX_SCRIPT = r'''
import re
from docx import Document

SOURCE = r"""__SOURCE__"""
TARGET = r"""__TARGET__"""
text = open(SOURCE, encoding="utf-8").read()
doc = Document()


def clean(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return s


para = []


def flush():
    global para
    if para:
        doc.add_paragraph(clean(" ".join(para)))
        para = []


in_fence = False
for line in text.splitlines():
    if line.strip().startswith("```"):
        flush()
        in_fence = not in_fence
        continue
    if in_fence:
        doc.add_paragraph(line, style="No Spacing")
        continue
    s = line.rstrip()
    if not s.strip():
        flush()
        continue
    m = re.match(r"^(#{1,6})\s+(.*)$", s)
    if m:
        flush()
        doc.add_heading(clean(m.group(2)), min(len(m.group(1)) - 1, 4))
        continue
    m = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)$", s)
    if m:
        flush()
        style = "List Number" if s.lstrip()[0].isdigit() else "List Bullet"
        doc.add_paragraph(clean(m.group(1)), style=style)
        continue
    if s.lstrip().startswith("|"):
        flush()
        continue
    para.append(s.strip())
flush()
doc.save(TARGET)
print("wrote", TARGET)
'''

_PDF_SCRIPT = r'''
import re
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

SOURCE = r"""__SOURCE__"""
TARGET = r"""__TARGET__"""
text = open(SOURCE, encoding="utf-8").read()
styles = getSampleStyleSheet()
heading = {1: styles["Title"], 2: styles["Heading1"], 3: styles["Heading2"]}


def clean(s):
    s = escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return s


story = []
para = []
fence = []
in_fence = False


def flush():
    global para
    if para:
        story.append(Paragraph(clean(" ".join(para)), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * cm))
        para = []


for line in text.splitlines():
    if line.strip().startswith("```"):
        if in_fence:
            story.append(Preformatted("\n".join(fence), styles["Code"]))
            fence = []
        else:
            flush()
        in_fence = not in_fence
        continue
    if in_fence:
        fence.append(line)
        continue
    s = line.rstrip()
    if not s.strip():
        flush()
        continue
    m = re.match(r"^(#{1,6})\s+(.*)$", s)
    if m:
        flush()
        story.append(Paragraph(clean(m.group(2)),
                               heading.get(len(m.group(1)), styles["Heading3"])))
        continue
    m = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.*)$", s)
    if m:
        flush()
        story.append(Paragraph("• " + clean(m.group(1)), styles["BodyText"]))
        continue
    if s.lstrip().startswith("|"):
        flush()
        continue
    para.append(s.strip())
flush()
SimpleDocTemplate(TARGET, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm).build(story)
print("wrote", TARGET)
'''

_XLSX_SCRIPT = r'''
import re
from openpyxl import Workbook
from openpyxl.styles import Font

SOURCE = r"""__SOURCE__"""
TARGET = r"""__TARGET__"""
text = open(SOURCE, encoding="utf-8").read()
wb = Workbook()
contents = wb.active
contents.title = "Contents"
contents.append(["#", "Section", "Words"])
for cell in ("A1", "B1", "C1"):
    contents[cell].font = Font(bold=True)
contents.column_dimensions["B"].width = 60


def sheet_name(title, used):
    base = re.sub(r"[\[\]:*?/\\]", " ", title).strip()[:28] or "Section"
    name, n = base, 2
    while name in used or name == "Contents":
        name = f"{base[:25]}-{n}"
        n += 1
    used.add(name)
    return name


used = set()
sheet = None
para = []
index = 0
words = 0


def flush():
    global para
    if para and sheet is not None:
        sheet.append([" ".join(para)])
        para = []


in_fence = False
for line in text.splitlines():
    if line.strip().startswith("```"):
        in_fence = not in_fence
        continue
    s = line.rstrip()
    m = re.match(r"^## (.*)$", s)
    if m and not in_fence:
        flush()
        if sheet is not None:
            contents.append([index, sheet.title, words])
        if m.group(1).strip() == "Contents":
            sheet = None
            continue
        index += 1
        words = 0
        sheet = wb.create_sheet(sheet_name(m.group(1), used))
        sheet.append([m.group(1)])
        sheet["A1"].font = Font(bold=True)
        sheet.column_dimensions["A"].width = 120
        continue
    if sheet is None:
        continue
    if not s.strip():
        flush()
        continue
    m = re.match(r"^(#{3,6})\s+(.*)$", s)
    if m:
        flush()
        sheet.append([m.group(2)])
        sheet.cell(row=sheet.max_row, column=1).font = Font(bold=True)
        continue
    words += len(re.findall(r"\b\w+\b", s))
    para.append(s.strip())
flush()
if sheet is not None:
    contents.append([index, sheet.title, words])
wb.save(TARGET)
print("wrote", TARGET)
'''

CONVERT_SCRIPTS: dict[str, str] = {
    "docx": _DOCX_SCRIPT,
    "pdf": _PDF_SCRIPT,
    "xlsx": _XLSX_SCRIPT,
}

"""One agent, one evaluator, and the seams that keep them honest.

The graph is `agent -> evaluator -> END`, with an `ask_user` pause. It used to
be seven nodes -- an overseer re-invoked after every step, four specialists and
a judge -- and collapsing it is the largest measured change in this file:
node boundaries were 45-69% of a run's wall time against 0.1-0.4 seconds of
actual tool execution per task, and three of every five model calls were
overhead. Mean score went 0.54 to 0.62 on the measured tasks.

WHAT A RUN DOES, in order.

1. `_criteria` writes down what a correct answer must contain, FROM THE TASK
   ALONE, before any attempt exists. Its own call, its own prompt. This is the
   only information in the whole judgment the actor did not produce, and it is
   why the evaluator is worth its calls: self-refinement without external
   information measures at -2.5% to 0% over five turns, where the same models
   reach 90-98% given an external checklist.

2. `_agent_loop` runs one conversation until it answers, pauses, or runs out
   of budget. The protocol is text -- `ACTION:` then `CODE:` -- not native
   tool calling, which is deliberate: programmatic tool calling matches or
   beats JSON in 11 of 14 models, and under fan-out JSON collapses from 100%
   to 0% between 70 and 72 tools where the code path holds.

   Inside it, four things earn their place:

   - `_parse_worker_reply` validates on the EMIT path. A tool name wrapped in
     backticks, bold, prose or a typo resolves instead of costing a round trip
     to be told it does not exist, and whichever of ACTION/FINAL comes first
     wins so an answer is never silently discarded in favour of a trailing
     suggestion.
   - The mutation gate holds a tool that cannot be undone, once per target,
     BEFORE it runs. Mutating actions are 14-18% of steps and one mutating
     mistake cuts success odds 55-96%, so the gate is cheap and precisely
     aimed. `TOOL_TIERS` is what it reads.
   - `_switch_mode` changes model and guidance without changing node. Modes
     are (task, guidance, depth); escalating restarts from the seed and
     carries the last thing produced, de-escalating keeps everything. A
     stronger model handed a weaker one's trajectory recovers 47% of the gain
     at 4-6x the cost, which is why the directions differ.
   - `_compact` shrinks old tool results for free and keeps the full text
     retrievable through `recall_memory`.

3. The evidence gate. An answer that changed code with nothing run since is
   held once and asked for the check -- no model call unless it fires, prose
   edits exempt, and "there is nothing to run here" accepted. See
   agent/pipeline/evidence.py for why this is structural rather than a prompt.

4. `evaluator` scores the answer against the criteria from step 1, separates
   "not met" from "blocked by the environment", and may check one thing with a
   tool. It is capped at MAX_EVALUATOR_ITERATIONS because, given a tool budget
   and no cap, it spent it: 24 calls on a task the loop did in six.

5. `_distil` leaves at most three transferable lessons behind, on a cheaper
   seat than the executor. `_record_seat` credits the model that produced the
   answer, but only when the run used ONE mode -- crediting any of several is
   guessing, and a log that mis-attributes cannot be trusted to change routing.

WHAT IS BOUND, NOT IMPORTED. Every seam this file depends on is a contextvar
so a harness can redirect it per run without the graph knowing: the workspace,
the command runner (a container, for the benchmarks), extra tools, the spend
budget, the memory store, and the lesson bank. That is what lets Claw-Eval and
SWE-bench run Otto's real graph rather than a copy of it.

BUDGET IS IN MODEL REQUESTS, counted in `_call`, retries included -- a budget
that counted logical calls would undercount by up to 3x exactly when a run is
going badly.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.memory.embeddings import current_model_name as embedding_model_name
from agent.memory.embeddings import embed
from agent.memory.hashing import content_hash
from agent.memory.retrieval import EVICTED_KIND
from agent.memory.session import current_store
from agent.memory.lessons import (
    Lesson, learning_enabled, parse_distilled, recall_lessons, record_lessons,
)
from agent.pipeline.evidence import Ledger, render_note as render_unproven
from agent.pipeline.state import AgentState
from agent.pipeline.budget import Budget, current_budget, default_budget
from agent.pipeline.modes import DEFAULT_MODE, MODES, mode_names, mode_reason, parse_mode_body
from agent.pipeline.tools import (
    MUTATING, READ_ONLY, TOOL_DISPATCH, TOOL_TIERS, ToolResult,
)
from agent.pipeline.toolkit import current_extra_tools, dispatch_table, render_note
from agent.router.llm_provider.base import ProviderError, translate_unknown
from agent.router import outcomes as seat_outcomes
from agent.router import health as provider_health
from agent.router.mapping import Task
from agent.router.router import Router

ROUTER = Router()  # the LLM-provider router every node calls into -- NOT
                    # the router() graph node below (same naming collision
                    # noted in the first revision: the two are unrelated,
                    # case matters, Python doesn't confuse them, only a
                    # reader skimming might).

#: The four specialists that actually produce work. Always exactly these
#: four; adding a fifth means adding its prompt, its node function, a line
#: in the graph wiring at the bottom of this file, and a mention in
#: ROUTER_PROMPT.
#: Cut from 150 to 30 with the loop rewrite. 150 was sized for a graph that
#: bounced router -> specialist -> router for every few tool calls, so a real
#: run legitimately used dozens of super-steps. One loop uses three or four
#: (agent -> evaluator -> agent -> evaluator), so 30 is still a wide margin and
#: catches a genuine runaway in seconds rather than minutes.
_RECURSION_SAFETY_NET = 30

#: Same rationale as the first revision's identical constant: a diffusing
#: (Mercury) call cut off at max_tokens is not a clean prefix, it's an
#: unconverged snapshot, and must be retried rather than accepted.
MAX_DIFFUSION_RETRIES = 3

#: Appended to a tool result the model has already produced, verbatim, earlier
#: in this same attempt.
#:
#: A plain success gives it nothing to react to. Measured on a hard task: the
#: solver wrote the same file over and over -- every write succeeding, every
#: result saying so -- because "wrote main.c.rs (30 lines)" is not a reason to
#: do anything different, and continuing an established pattern is the easiest
#: thing for a model to do. Naming the repetition is: it turns an
#: indistinguishable success into a fact about the attempt's own history.
#:
#: Deliberately not an error and not a refusal -- the call really did succeed,
#: and re-running something is sometimes right (re-reading a file after
#: changing it). This only says it happened and that repeating cannot change
#: the outcome.
REPEATED_CALL_NOTE = (
    "NOTE: that is {n} {tool} calls in a row against `{target}`, with nothing "
    "run in between. Rewriting it again is guessing. Run it -- compile it, "
    "execute it, test it -- and use what that actually reports."
)

#: Appended when a call fails against something that has already failed in
#: this attempt.
#:
#: Separate from REPEATED_CALL_NOTE, and fires on the SECOND occurrence rather
#: than the third, because the two are not the same mistake. Re-running
#: something that succeeded is wasteful; re-running something that failed,
#: unchanged, means the error was never read. It is also the cheapest thing
#: for a model to produce, which is why it needs saying out loud.
REPEATED_FAILURE_NOTE = (
    "NOTE: `{target}` has already failed once in this attempt and has now "
    "failed again. Do not run it a third time. Read the error above and work "
    "out what is actually wrong -- check the state of the thing you are acting "
    "on, and whether something else is undoing or blocking your change. Fix "
    "what you find, then try."
)

#: How many calls in a row against the same target before the note appears.
#: 3, not 2: a second attempt straight after the first is ordinary (a typo
#: spotted on re-reading), while a third with still nothing run against it is
#: the pattern this exists to break.
REPEATS_BEFORE_NOTE = 3


def _action_target(tool_name: str, body: str) -> str:
    """What a tool call is acting ON -- the file path for the file tools, the
    command itself for the shell ones.

    The target rather than the whole body, because the loop worth catching
    does not repeat verbatim: rewriting the same file with a slightly
    different draft every time looks different byte for byte and is the same
    non-progress. Measured on a hard task: 166 writes to one path, of 8, 721,
    36, 37 and 35 lines, without the compiler ever being run once.
    """
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    if tool_name == "view_image":
        # The path alone is the wrong target here. Asking a second, narrower
        # question about one image is exactly how this tool is meant to be
        # used -- a vision model returns words, so narrowing across calls is
        # the only way to get at detail -- and keying on the path would flag
        # that as repetition and tell the agent to stop.
        return f"view_image:{body.strip()[:200]}"
    if tool_name in {"write_file", "edit_file", "read_file", "list_files"}:
        return first_line.strip()[:120]
    return f"{tool_name}:{first_line.strip()[:120]}"

#: How many of the run's most recent recorded actions get shown (state.py's
#: `actions`). Bounded because this goes into every role and overseer prompt
#: and a long run accumulates hundreds; the most recent describe the world as
#: it is now, and an older attempt since superseded is the one worth
#: forgetting.
_ACTIONS_SHOWN = 40

#: How many unparseable replies IN A ROW end a node's tool loop early.
#:
#: The corrective retry (UNPARSEABLE_FEEDBACK) is worth having: a model that
#: forgot the format once usually gets it right when told. What it cannot fix
#: is a model that is not answering at all. Measured on a hard task: a broken
#: reasoning_effort setting (agent/router/mapping.py's Task.REASON, now
#: corrected) made the provider return empty streams, and the solver spent 39
#: of its 40 turns feeding "you didn't follow the format" to a model that had
#: sent back nothing, issuing 2 real commands in six minutes. The route is
#: fixed; nothing about the loop knew to stop, and the next provider hiccup
#: would spend a whole budget the same way.
#:
#: 3 rather than 2 because the retry genuinely does recover a one-off, and
#: rather than 5 because by the third identical non-answer the budget is
#: better spent letting the overseer pick a different step.
MAX_CONSECUTIVE_DEAD_REPLIES = 3

#: Bound on one node's OWN tool-calling loop (ACTION/execute_TOOL
#: round-trips) before its last reply is used as-is. Unrelated to the
#: (now-removed) overseer retry cap -- this bounds a single node's single
#: turn, not how many turns the overseer may hand out. Applies identically
#: to every role node and the evaluator (_tool_loop is shared by all of
#: them).
#: How many ACTION/tool exchanges one node gets before it must answer with
#: what it has. 5 is right for what this graph was built for -- a node
#: checking its own work, where a sixth exchange usually means it is stuck
#: rather than making progress -- and it stays the default.
#:
#: It is overridable because agentic benchmarks are a genuinely different
#: shape of task: published SWE-bench and terminal-agent traces routinely run
#: dozens of tool calls in one attempt (read a file, edit it, run the tests,
#: read the failure, edit again), and a cap of 5 would measure the cap rather
#: than the agent. An env var rather than a parameter because _tool_loop is
#: reached through five node functions and a LangGraph call, none of which
#: take agent-level configuration today, and threading one through all of
#: them to serve the harness would be a worse trade than this.
#: Temperature now lives in the routing table, per candidate, because only a
#: candidate knows which vendor it is talking to: Anthropic, OpenAI and Gemini
#: honour 0.0, while Inception resets anything below 0.5 to the model default
#: of 1.0 (agent/router/llm_provider/inception_provider.py clamps as a
#: backstop). A single constant here could not be right for both ends of a
#: chain, and passing one at the call site actively overrode the table --
#: Router.model_for merges {**route_params, **overrides}.
MAX_TOOL_ITERATIONS = int(os.environ.get("OTTO_MAX_TOOL_ITERATIONS", "5"))

#: The ACTION: enumeration every role/evaluator prompt shows, built FROM the
#: registry rather than typed out in each prompt. It was hand-kept in sync
#: through three rounds of tool additions, with tests/test_prompt_tool_sync.py
#: standing guard over the copies; deriving it means the next tool added to
#: TOOL_DISPATCH appears in all five prompts by construction, and that test
#: now checks the derivation reaches them rather than checking five hand-
#: written lists against one registry. `ask_user` is appended by hand because
#: it is deliberately NOT in TOOL_DISPATCH -- it unwinds the whole tool loop
#: by raising (see NeedsUserInput below) instead of returning a ToolResult.
#: `ask_user` and `switch_mode` are appended by hand because neither is in
#: TOOL_DISPATCH: they change the LOOP's own state rather than returning a
#: ToolResult -- one pauses the run, the other changes which model answers
#: next and under what guidance.
_TOOL_MENU = "|".join((*TOOL_DISPATCH, "ask_user", "switch_mode", "delegate"))

#: KEEP THIS SHORT. A fifth habit was added and measured -- "search for the
#: exact text of an error rather than reasoning about it", which for the task
#: in question was one command that would have named the culprit outright. It
#: did not merely fail to take (zero such searches); it wiped out the
#: behaviour the first four had produced, taking system-inspection commands
#: from 17 to 0 and total commands from 80 to 53 on the same task. One run
#: each, so treat the size of that with suspicion but not the sign: past some
#: length the model acts on none of this rather than more of it. Adding a
#: habit here means measuring that the others survive it.
#: General debugging habits, shared by the prompts of the roles that actually
#: change things. Not advice about any particular kind of task -- these are the
#: two things a competent engineer does that a model, left alone, reliably
#: does not.
#:
#: The first is looking at the system rather than at the thing that failed.
#: Observed on a benchmark task: 99 commands, 28 of them re-running the one
#: command that was broken, and not a single `ps`, `crontab` or look at what
#: was running. The fix it applied was correct and kept being undone by
#: processes it never went looking for.
#:
#: The second is not repeating an action that already failed. A model will
#: re-issue a failing command verbatim several times over, because re-trying
#: is cheaper to produce than diagnosing. Saying plainly that a repeat needs a
#: reason is what turns the second attempt into a question about the first.
#: The CODE: body format for each tool, written once instead of five times.
#:
#: It used to be spelled out inline in every prompt at 1054 characters -- 35%
#: of the solver's whole prompt, repeated five ways, and hand-edited five ways
#: every time a tool was added. Together with the habits block below that left
#: the solver's own instructions as about a sixth of what it was reading,
#: which matters more here than it would elsewhere: a fifth habit measurably
#: erased the effect of the four before it, so length on this model is not
#: free.
#:
#: Trimmed to the formats a caller cannot guess. execute_bash takes a command,
#: execute_python takes code, web_search and rag take a query -- saying so
#: costs tokens and tells the model nothing it did not already know from the
#: tool's name.
_TOOL_BODY_HINT = (
    "read_file: a path, optionally `path:START-END`. "
    "list_files: a directory. "
    "write_file: the path on the FIRST line, the file's whole content after "
    "it -- no separator, no JSON. "
    "edit_file: the path, then a line `---OLD---`, the exact text to replace, "
    "a line `---NEW---`, the replacement. "
    "complete_code: code, optionally `---SUFFIX---` then trailing code. "
    "predict_edit: code only, no instruction. "
    "recall_memory: a search query. "
    "code_map: `define <name>`, `uses <name>`, `imports <module>` or "
    "`outline <path>` -- exact names, Python only. "
    "switch_mode: one word from " + "|".join(mode_names()) + ", optionally why "
    "after it -- the conversation carries on. "
    "delegate: that same word, then one bounded job. It runs on that mode's "
    "model with none of this conversation and reports back. "
    # The WHEN of asking used to live here too, and it is said again by
    # MUTATION_GATE_NOTE at the moment it applies -- which is where a model
    # can act on it. This block's job is what goes in the body.
    "ask_user: a question, optionally then `CHOICES: a | b`."
)

#: Which standing tools change things, named from the registry rather than
#: typed out, so the next tool added to a mutating tier says so by itself.
#:
#: This is agent/pipeline/tools.py's TOOL_TIERS finally having a reader in
#: production. It has always been declared and always been enforced only by a
#: unit test, which is a strange place for the one distinction that decides
#: whether a mistake can be taken back. Claw-Eval task T026 is what it costs:
#: three contacts matched "Manager Zhang", the grader required asking which,
#: and the agent sent to the first -- then sent twice more. Safety is a
#: multiplier in that benchmark's formula, so the whole task scored zero.
#:
#: The rule in the ask_user hint above is about the ACTION rather than the
#: tool, because a benchmark's own tools (agent/pipeline/toolkit.py) carry no
#: tier and sending mail is exactly the case that matters. This line is the
#: half that CAN be derived, and it keeps the registry honest.
_MUTATING_TOOLS = tuple(
    name for name, tier in TOOL_TIERS.items() if tier != READ_ONLY
)

#: The ACTION/CODE protocol, assembled once for every prompt that offers tools.
_ACTION_BLOCK = (
    "reply with exactly\nACTION: <" + _TOOL_MENU + ">\nCODE:\n<"
    + _TOOL_BODY_HINT
    + ">\nand you will be shown the result, then you can continue. "
    + ", ".join(_MUTATING_TOOLS) + " change things and cannot be undone. "
)

_DIAGNOSTIC_HABITS = (
    "Four habits, whatever the task:\n"
    "1. Look at the system, not just at the thing that broke. Before you "
    "conclude why something fails, find out what state it is actually in -- "
    "what is running, what is scheduled, what a file really contains, what "
    "changed recently. If a fix does not hold, or something reverts, then "
    "something else is acting on it and finding THAT is the task.\n"
    "2. Never run the same thing twice hoping for a different answer. If a "
    "command failed, or did not achieve what you expected, work out why "
    "before you touch it again -- read the actual error, check your "
    "assumption about what it was going to do. Repeating an action is only "
    "worth doing when you have changed something that would change its "
    "result, and you should be able to say what.\n"
    "3. Look wide before you look narrow. A filter hides everything you did "
    "not think to ask for, so a `grep` that comes back empty or unsurprising "
    "is not evidence -- it is a guess about what the answer looks like. When "
    "a filtered search tells you nothing, run it again unfiltered and read "
    "what is really there before you conclude anything.\n"
    "4. If something goes back to how it was, that is not your change "
    "failing -- it is something else changing it. Ask what could: a process "
    "still running, a scheduled or repeating job, a service, something "
    "watching the file. Enumerate them and look, rather than applying the "
    "same fix harder.\n\n"
)

#: What to check before writing code, and the one place this prompt argues for
#: doing LESS.
#:
#: Deliberately NOT a fifth diagnostic habit. The comment above records a
#: measurement where adding one erased the effect of the four before it,
#: taking system-inspection commands from 17 to 0; this is a different kind of
#: instruction -- a check before a write rather than a habit while
#: investigating -- so it is its own block, and its cost is visible on its own
#: line if it ever needs removing.
#:
#: The ladder is measured. Against the same agent with no such instruction, on
#: twelve real tickets in a real repository: 54% fewer lines, 22% fewer
#: tokens, 20% lower cost, 27% faster, and safety held at 100%. It was the
#: only variant tested that cut every metric at once -- a bare "write
#: one-liners" prompt was cheaper too and dropped a safety guard doing it,
#: which is why the last sentence here is not optional decoration.
#:
#: The cut is largest where there is a real over-build trap and near zero
#: where the code is already minimal, so this costs almost nothing on tasks it
#: does not apply to.
_MINIMALITY_LADDER = (
    "Before writing code, stop at the first that holds: it need not exist; "
    "this codebase already has it; the standard library does it; the platform "
    "does it; an installed dependency does it; it is one line. Then write the "
    "minimum that works. Be lazy about the SOLUTION, never about the reading. "
    "Never cut validation at a trust boundary, data-loss handling, security "
    "or accessibility: those are the job.\n\n"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

#: Phase one of judging: work out what a good answer would have to contain,
#: from the TASK ALONE, before the attempt is visible.
#:
#: This is the whole reason the evaluator is worth its calls. RefineBench
#: measured self-refinement over five turns at 31.3% for the best model and
#: -2.5% to 0% for most; the SAME models reach 90-98% when given an external
#: checklist. An evaluator that only re-reads the actor's own output is
#: measured at approximately nothing, and that is what this one was: it saw the
#: answer, the action record, the transcript tail and the gathered context --
#: all of it produced by the actor it was judging.
#:
#: A rubric derived from the task is the cheapest thing the actor does not have.
#: Generating it in a SEPARATE phase is load-bearing rather than tidy: writing
#: criteria while looking at an answer produces criteria that the answer
#: happens to meet. The verifier paper reports this structure taking agreement
#: with humans from 0.26-0.31 to 0.64 -- inside the human inter-annotator band
#: of 0.53-0.57 -- and false positives from 0.40-0.45 to 0.01, and shows the
#: gain survives giving the baselines the same model, so it is architectural
#: rather than a better judge.
RUBRIC_PROMPT = (
    "You are about to judge someone else's attempt at the task below. First, "
    "before you see any attempt, write down what a correct answer would have "
    "to contain.\n\n"
    "Write criteria about the ANSWER and what it claims, not about the route "
    "taken to it. \"The reported value is what the code actually prints\" is a "
    "criterion; \"the agent called the read tool first\" is a route.\n\n"
    "COVERAGE COUNTS AS THE ANSWER. If the task is about a set of things -- "
    "every meeting, each file, all the records -- then \"every one of them is "
    "accounted for, not just the first\" is a criterion, and it is usually the "
    "one that decides whether the work was actually done. A report that covers "
    "one item and says nothing about the rest is a wrong answer, not a short "
    "one.\n\n"
    "Reply with 2 to 4 criteria, one per line, each starting with `- `. Each "
    "must be something you could CHECK rather than an opinion: a value that "
    "must be right, a file that must exist, a command that must succeed, a "
    "question that must be answered. Make them independent -- overlapping "
    "criteria double-count one mistake. Fewer is better.\n\n"
    "Do not write criteria about style, effort or presentation. Nothing else "
    "in your reply, no preamble.\n\n"
    "If the message asks for nothing that could be checked -- a greeting, a "
    "thank-you, an acknowledgement, small talk -- reply with exactly NONE and "
    "no criteria. That is a normal answer, not a failure to understand."
)

EVALUATOR_PROMPT = (
    "Judge whether the {target} below actually satisfies the original "
    "request -- {target_note}.\n\n"
    "CRITERIA, written before this attempt was visible. Judge against these "
    "and nothing else:\n{rubric}\n\n"
    "Take each in turn. A criterion is met, not met, or blocked -- blocked "
    "meaning something outside the agent's control stopped it, or you could "
    "not check it from what you were shown and had no budget left to look. "
    "BLOCKED IS NOT FAILED. A criterion you could not verify is not evidence "
    "the work is wrong, and rejecting on one sends the agent back to redo "
    "something that may already be right.\n\n"
    "Judge what the answer CLAIMS against what you can see. If the answer "
    "states a result and nothing you were shown contradicts it, that is met.\n\n"
    "CHECK IT, DO NOT TAKE ITS WORD. If the request asked for something to "
    "be changed, fixed, built or made to work, run a command that would fail "
    "if it had not been -- and judge what that command actually reports, not "
    "what the output above claims. An output that says the work is done is "
    "not evidence the work is done. If your own check contradicts it, reject "
    "and quote exactly what you saw.\n\n"
    "Check the RESULT, not the steps. A step that succeeded does not mean "
    "the thing the request asked for is true now -- something else may have "
    "undone it, or the step may have been aimed at the wrong target. If a "
    "check passes, consider whether it would still pass a minute from now, "
    "and if you have reason to doubt it, look for what would change it "
    "back.\n\n"
    "You may check ONE thing with a tool if a criterion genuinely cannot be "
    "settled from what you were shown: "
    + _ACTION_BLOCK +
    "When you are done, reply with exactly\nFINAL:\n"
    "MET: the number of criteria met, then / then the number of criteria\n"
    "BLOCKED: yes or no -- whether anything unmet was outside the agent's "
    "control\n"
    "APPROVE: yes or no\n"
    "WHY: one sentence naming the first criterion that failed, or why it "
    "passed\n"
    "You have at most {max_iter} exchanges before your last reply is used "
    "as-is."
)
#: What a finished run is asked to leave behind for the next one.
#:
#: Deliberately asked of BOTH outcomes. A bank built only from successes
#: throws away the half of the signal that says what not to do, and in a
#: failed run the sharpest lesson is usually the one nobody would have
#: written down after a win.
#:
#: Asked for at most three, and each one short, because injecting a pile of
#: skill-like items every turn and letting the model decide when they apply
#: measured 16.4 points BELOW a variant that injected none. Retrieved text is
#: not free: it competes with the task for attention.
DISTIL_PROMPT = (
    "A run just finished. Write down what a DIFFERENT task could reuse from "
    "it.\n\n"
    "Look for FRICTION. What was retried, what was surprising, what took two "
    "attempts, what nearly went wrong, what turned out not to be where it "
    "looked. That is where a reusable lesson lives. A run that went smoothly "
    "usually still contains one -- the thing that made it go smoothly and "
    "would not have been obvious beforehand.\n\n"
    "A lesson is about METHOD, never about subject matter. It has to make "
    "sense to someone working on something completely unrelated, so do not "
    "name what this task was about -- no emails, no invoices, no repository, "
    "whatever it happened to be. If your cue only makes sense to someone "
    "doing this same kind of task, you have written a note, not a lesson.\n"
    "  no  -- \"the config lives in /etc/app.conf\" (a fact about one "
    "machine)\n"
    "  no  -- \"rank urgent client requests above internal deadlines\" (a "
    "judgment about one domain's content)\n"
    "  no  -- \"read files before editing them\" (true of every task, so it "
    "tells the next run nothing)\n"
    "  yes -- \"when a tool reports a path that does not exist, check the "
    "working directory before assuming the file is missing\"\n"
    "  yes -- \"when the request names a category of thing, count them first "
    "and check the count against the answer before finishing\"\n\n"
    "Write ONE to THREE. Keep each field under 25 words -- a lesson nobody "
    "can read at a glance is a lesson nobody uses. Each has a `cue` -- the "
    "SITUATION it applies in, never this task's subject -- an `action`, and "
    "an `outcome` of "
    "\"worked\" or \"failed\". A lesson from something that went wrong is "
    "worth more than one from something that went right; say \"failed\" and "
    "describe what to do instead. Reply with an empty array only if the run "
    "genuinely contained no friction and no non-obvious choice.\n\n"
    "Before you answer, re-read each cue and action. If either names anything "
    "from this particular task, rewrite it in general terms or drop it.\n\n"
    "Reply with a JSON array and nothing else:\n"
    '[{{"cue": "...", "action": "...", "outcome": "worked"}}]'
)

#: Asked only of a run that was REJECTED AND RETRIED.
#:
#: Automatically extracted principles scale and come out too generic to act
#: on; hand-authored ones are actionable and do not scale. The ingredient that
#: closes the gap is contrastive analysis -- naming one aspect and comparing a
#: better attempt against a worse one on it, rather than describing the better
#: one alone.
#:
#: A retried run contains both attempts by construction, so this costs nothing
#: extra to ask: no second call, no second trajectory, just the comparison made
#: explicit instead of left implied by an outcome label. On a run that was
#: accepted first time there is nothing to compare and this is not sent.
CONTRAST_NOTE = (
    "Because there were two attempts, say what CHANGED between them. Pick the "
    "one aspect that actually differed -- what was checked, what order things "
    "were done in, what assumption was dropped -- and write the lesson as that "
    "contrast: what the weaker attempt did, and what the better one did "
    "instead. A lesson that describes only the better attempt is the generic "
    "kind nobody can act on."
)

#: Added when the criteria call found nothing to check -- a greeting, a
#: thank-you, an acknowledgement.
#:
#: The prompt above is written for tasks: find out what is true, do the work,
#: confirm it holds. Handed "hello, how are you?", an agent following it looks
#: for work to do. Measured on a clean session, that was four model calls and
#: 48 seconds to produce "I'm doing well, thank you!" -- the judge and the
#: lesson were already being skipped by then, and this is the rest of the bill.
#:
#: Deliberately not a separate prompt or a separate path. One line, added only
#: when the run has already established there is nothing to verify, so a task
#: that merely looks chatty never sees it.
CONVERSATION_NOTE = (
    "There is nothing to check here -- this is conversation, not a task. "
    "Answer it directly in your next reply with FINAL:, briefly and like a "
    "person. Do not use a tool, do not look anything up, and do not go "
    "looking for work to do."
)

#: The whole agent, in one prompt.
#:
#: This replaces PLANNER_PROMPT, SOLVER_PROMPT, SUMMARIZER_PROMPT and
#: FINDER_PROMPT, and it is SHORTER than the four of them put together by a
#: wide margin -- each of those carried its own copy of _ACTION_BLOCK (831
#: characters), so the protocol alone was stated four times. Said once here,
#: with what is specific to each role moved into agent/pipeline/modes.py as a
#: guidance block under 400 characters, the text a call actually carries goes
#: DOWN even though the agent can now do everything all four roles could.
#:
#: That matters more here than it would elsewhere. The comment above
#: _DIAGNOSTIC_HABITS records a measurement where a fifth habit did not merely
#: fail to take -- it erased the effect of the four before it, taking
#: system-inspection commands from 17 to 0. Length is not free on this model,
#: so the budget freed by de-duplicating the protocol is the budget that pays
#: for the mode machinery.
AGENT_PROMPT = (
    "You are an engineer with a shell, working a task end to end: find out "
    "what is true, do the work, and confirm it actually holds.\n\n"
    + _DIAGNOSTIC_HABITS + _MINIMALITY_LADDER +
    "You work in a MODE, which sets both how you are thinking and which model "
    "you are running on. You start in " + DEFAULT_MODE + ". Switch when the "
    "KIND of work changes -- `ACTION: switch_mode` with one of "
    + "|".join(mode_names()) + " -- not to restate what you are already doing. "
    "The conversation carries over: everything above stays, and you keep every "
    "tool.\n\n"
    + _ACTION_BLOCK +
    "One tool call per reply.\n\n"
    "Before you finish, run something that would FAIL if the task were not "
    "done, and read what it prints. An explanation is not evidence. When you "
    "have seen it work, reply with exactly\nFINAL:\n<the answer itself -- the "
    "numbers, the names, the decision -- not a description of what you did>"
)


#: Fed back inside _tool_loop when a reply has neither ACTION: nor FINAL:
#: (or a FINAL: with nothing after it). A marker-less or empty reply is
#: never silently accepted as an answer -- the node is told what went wrong
#: and made to retry, spending one of MAX_TOOL_ITERATIONS exchanges.
UNPARSEABLE_FEEDBACK = (
    "Your last reply didn't match either required format -- it had no "
    "ACTION: with a tool call, and no FINAL: followed by an answer (or "
    "FINAL: was there but empty). Reply again using exactly one of those "
    "two formats, with no other text."
)


# --------------------------------------------------------------------------
# Shared LLM-call plumbing -- unchanged from the first revision.
# --------------------------------------------------------------------------

def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _call(llm, messages: list) -> str:
    """Run one streamed chat call and return its settled text.

    A diffusing (Mercury) call whose stream ends with finish_reason "length"
    is retried with doubled max_tokens rather than accepted -- see
    MAX_DIFFUSION_RETRIES. A non-diffusing truncation is returned as-is: it
    is an incomplete but internally clean prefix, the ordinary "ran out of
    budget" case every LLM caller already has to tolerate.
    """
    # Every model REQUEST is counted here, not every logical call, and the
    # retries below are the reason. An empty-stream retry and a diffusion
    # max_tokens doubling are each a fresh HTTP request that costs real money,
    # so a budget that counted _call once would undercount by up to 3x exactly
    # when a run is going badly. agent/pipeline/budget.py has the rest of the
    # argument; nothing bound is the normal case and makes this a no-op.
    budget = current_budget()
    current = llm
    for attempt in range(MAX_DIFFUSION_RETRIES):
        if budget is not None:
            budget.spend()
        reply = None
        try:
            for chunk in current.stream(messages):
                reply = chunk if reply is None else reply + chunk
        except ValueError as exc:
            # langchain_core raises ValueError("No generation chunks were
            # returned") from inside stream() when the provider yields nothing
            # at all -- so the `reply is None` check below never gets the
            # chance to see it. This module always meant to treat an empty
            # stream as an empty answer (the caller's unparseable-reply retry
            # then does its job); an uncaught ValueError instead unwinds the
            # whole graph, which is how one benchmark task died outright
            # rather than scoring badly.
            if "No generation chunks" not in str(exc):
                raise
            logger.warning(
                "%s returned an empty stream (attempt %d/%d)",
                _model_label(current), attempt + 1, MAX_DIFFUSION_RETRIES,
            )
            if attempt < MAX_DIFFUSION_RETRIES - 1:
                continue  # transient often enough to be worth one more try
            return ""
        except ProviderError:
            # Inception's own provider already translated this one; re-wrapping
            # would bury the specific subclass the callers branch on.
            raise
        except Exception as exc:
            # The chat model here may be a LangChain class Otto does not own,
            # which raises its vendor's SDK errors straight out of .stream().
            # _run_role and evaluator catch only ProviderError, so an untranslated
            # one unwinds the whole graph instead of becoming a clean edge back
            # to the overseer.
            #
            # Clause order is load-bearing twice over. The ValueError clause
            # must stay FIRST, or a bare `except Exception` swallows the
            # empty-stream retry above. And a `raise` from an earlier clause is
            # not caught by a later clause of the same try, which is what keeps
            # an unrelated ValueError propagating. Collapsing these into one
            # handler with isinstance checks breaks both.
            # Before translating: the raw exception still carries the status
            # and any Retry-After header, and agent/router/health.py needs
            # both. Translating first would throw away the one thing that
            # tells a rate limit from an outage.
            provider_health.note_failure(
                exc, provider=getattr(current, "_otto_provider", ""),
                model_id=_model_label(current))
            raise translate_unknown(
                exc,
                provider=getattr(current, "_otto_provider", ""),
                model_id=_model_label(current),
            ) from exc
        if reply is None:
            return ""

        # A call that came back settles "is this vendor reachable", whichever
        # model answered it -- so this clears the provider breaker as well as
        # the model's own cooldown.
        provider_health.HEALTH.note_success(
            getattr(current, "_otto_provider", ""), _model_label(current))

        finish_reason = (reply.response_metadata or {}).get("finish_reason")
        if not (getattr(current, "diffusing", False) and finish_reason == "length"):
            return _content_text(reply.content)

        # Everything below is Inception-only by construction: `diffusing` is a
        # ChatInception field, so no other vendor's model reaches it.
        if attempt == MAX_DIFFUSION_RETRIES - 1:
            raise ProviderError(
                f"{_model_label(current)} truncated a diffusing response at "
                f"max_tokens={getattr(current, 'max_tokens', None)!r} on every one of "
                f"{MAX_DIFFUSION_RETRIES} attempts -- refusing to hand an "
                f"unconverged diffusion snapshot to the rest of the graph"
            )
        bumped = max(getattr(current, "max_tokens", None) or 1024, 1024) * 2
        logger.warning(
            "%s truncated a diffusing response at max_tokens=%r "
            "(attempt %d/%d) -- retrying at max_tokens=%d",
            _model_label(current), getattr(current, "max_tokens", None),
            attempt + 1, MAX_DIFFUSION_RETRIES, bumped,
        )
        current = current.model_copy(update={"max_tokens": bumped})
    return ""  # unreachable -- the loop always returns or raises


def _parse_rubric(text: str) -> list[str]:
    """The criteria out of a rubric reply. Bullet lines only.

    Bounded at RUBRIC_MAX because criteria are scored one by one: an
    unbounded list turns one judgment into an unbounded number of them, and
    the verifier paper's own guidance is a small set of NON-OVERLAPPING
    criteria, since overlapping ones double-count a single mistake.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        for marker in ("- ", "* ", "\u2022 "):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break
        else:
            continue
        if line:
            lines.append(line)
    return lines[:RUBRIC_MAX]


def _parse_verdict(text: str) -> "Verdict":
    """The four things a judgment has to say, out of one FINAL body.

    Separated because one number cannot carry them. An attempt can take every
    right step and be stopped by a missing credential, or reach the right
    answer by accident -- and a single approve/reject collapses those into the
    same signal, which is how a judge ends up training the agent on noise.
    """
    approve, reason = _parse_approval(text)
    met = total = 0
    if "MET:" in text:
        fragment = text.split("MET:")[1].split("\n")[0]
        numbers = [int(n) for n in re.findall(r"\d+", fragment)[:2]]
        if len(numbers) == 2:
            met, total = numbers
    blocked_line = text.split("BLOCKED:")[1].split("\n")[0].lower() if "BLOCKED:" in text else ""
    return Verdict(
        approved=approve,
        reason=reason,
        # The continuous score. Kept apart from `approved` on purpose: "three
        # of four criteria met" and "nothing worked" are both rejections and
        # should not look alike to anything reading this back.
        process=round(met / total, 3) if total else (1.0 if approve else 0.0),
        criteria=(met, total),
        blocked="yes" in blocked_line,
    )


def _parse_approval(text: str) -> tuple[bool, str]:
    after = text.split("APPROVE:")[-1]
    line, _, rest = after.partition("\n")
    approve = "yes" in line.strip().lower()
    reason = rest.split("WHY:")[-1].strip() if "WHY:" in rest else rest.strip()
    return approve, reason


def _strip_code_fence(text: str) -> str:
    """Strip a whole-reply ```lang\\n...\\n``` wrapper, if present.

    Applied to every FINAL body regardless of which node produced it: the
    solver's code needs to be raw, fence-free text to be usable directly
    (e.g. by a golden-set checker that execs it), and stripping a fence
    from a plain-text answer that happens to be entirely fenced is
    harmless. Only strips when the ENTIRE reply is one fenced block (starts
    with ``` and ends with a lone ``` line) -- a reply that merely mentions
    backticks mid-text is left alone.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


class NeedsUserInput(Exception):
    """Raised by _tool_loop (below) when a reply's ACTION: is ask_user --
    a signal, not a tool result. Every other ACTION gets dispatched via
    TOOL_DISPATCH and its result fed straight back into this same loop's
    local `messages`, but an ask_user answer has to come from an actual
    human, which means the WHOLE GRAPH has to pause -- not just this one
    node's own tool-calling loop. Raising unwinds out of _tool_loop and out
    of whichever role node (or evaluator()) called it, all the way to a
    dedicated `ask_user` graph node built to do nothing else (module
    docstring, seventh refinement, explains why the interrupt() call
    itself can't just live here: LangGraph re-executes a node's ENTIRE
    body from the top on resume, so anything with side effects before it
    -- including the LLM calls this loop already made -- would replay a
    second time).
    """

    def __init__(self, question: str, choices: list[str]):
        self.question = question
        self.choices = choices
        super().__init__(question)


def _parse_ask_user_body(body: str) -> tuple[str, list[str]]:
    """Split an ask_user ACTION's CODE: body into (question, choices).

    A line starting with "CHOICES:" (case-insensitive, anywhere in the
    body) holds pipe-separated options; everything else is the question.
    No such line -- the common case, a genuinely open-ended question --
    means choices=[] and the whole body is the question.
    """
    lines = body.splitlines()
    choices: list[str] = []
    question_lines = []
    for line in lines:
        if line.strip().upper().startswith("CHOICES:"):
            raw = line.split(":", 1)[1]
            choices = [c.strip() for c in raw.split("|") if c.strip()]
        else:
            question_lines.append(line)
    return "\n".join(question_lines).strip(), choices


#: A line that STARTS a new directive, and therefore ENDS the CODE: body
#: above it. Anchored to the start of a line so a mention of the word inside
#: a command ("echo ACTION: done") doesn't truncate anything.
_NEXT_DIRECTIVE = re.compile(r"^[ \t]*(?:ACTION|FINAL):", re.MULTILINE)


#: Chat-template control tokens a model sometimes emits into its own visible
#: output -- "<|tool_call_start|>" and friends. They are markup for the
#: serialiser, never content, and they arrive mid-body where nothing else
#: strips them: observed once in a Terminal-Bench run, where the first command
#: of the task became `curl -v example.com\n\n<|tool_call_start|> 2>&1 | head`
#: and bash answered "syntax error near unexpected token `|'". Rare, and free
#: to remove.
_SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")


def _code_body(text: str) -> str:
    """The CODE: body of the FIRST action in `text`, ending where the next
    ACTION:/FINAL: begins.

    It used to be "everything after the first CODE:, to the end of the reply".
    That is correct only while the model emits exactly one tool call per reply,
    which it does not: asked for one step it will sometimes lay out the whole
    plan at once, and the entire rest of the reply -- the literal lines
    "ACTION: execute_bash", "CODE:", and a trailing "FINAL:" block -- was then
    handed to the shell as part of the command. Terminal-Bench transcripts
    caught it: bash reporting `ACTION:: command not found` mid-task, an exit
    code of 127 for a command whose real work had actually succeeded, and an
    agent reading that as failure. Only the first call is executed either way;
    the loop feeds its result back and the model reissues the rest.
    """
    if "CODE:" not in text:
        return ""
    after = text.split("CODE:", 1)[1]
    match = _NEXT_DIRECTIVE.search(after)
    body = after[: match.start()] if match else after
    return _SPECIAL_TOKEN.sub("", body).strip()


#: A bare identifier, for pulling a tool name out of whatever the model wrapped
#: it in. Tool names are ASCII snake_case by construction (TOOL_DISPATCH).
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: The loop's own pseudo-tools. They never appear in TOOL_DISPATCH -- they
#: change the loop's state rather than returning a result -- but a reply naming
#: one is naming a real thing, so name RESOLUTION has to know about them. Whether
#: the loop in question will honour it is a separate question, answered by that
#: loop: `_tool_loop` (the evaluator's) rejects `delegate` on purpose.
_LOOP_TOOLS = ("ask_user", "switch_mode", "delegate")

#: How close a misspelling has to be before it is treated as the tool it
#: resembles. 0.8 accepts `execute_pyton` and `read_files`; it does not accept
#: `read_file` for `edit_file`, which differ by more than a typo and where
#: guessing wrong would run the wrong tool rather than waste a round trip.
_NEAR_MISS_CUTOFF = 0.8


def _resolve_tool(line: str, allowed) -> str:
    """The tool named on an ACTION: line, out of whatever decorated it.

    Every shape here was produced by a real model against this protocol:

        ACTION: execute_bash to list the files      prose after the name
        ACTION: `execute_bash`                      backticks
        ACTION: **execute_bash**                    bold
        ACTION: execute_bash.                       a full stop

    All four used to reach the dispatch table verbatim, miss, and come back as
    "tool 'execute_bash.' is not available" -- a wasted model call each time,
    for a reply that named the right tool. A text protocol has nothing
    structurally enforcing its action schema, which is the dominant production
    failure class (plausible reasoning decoupled from the action contract), and
    the counter-evidence is that it closes in code: one harness eliminated all
    illegal moves across 145 environments by validating on the emit path.

    Resolution order is deliberate. An exact token match anywhere on the line
    wins first, so the prose case resolves rather than being guessed at. Only
    then does a near miss get considered, and only for the FIRST token -- a
    fuzzy match against a word buried in a sentence is how a parser starts
    inventing tool calls. When nothing resolves, the first token is returned
    unchanged so the caller's "not available" message still names what the
    model actually said.
    """
    import difflib

    vocabulary = set(allowed) | set(_LOOP_TOOLS)
    tokens = _IDENTIFIER.findall(line)
    if not tokens:
        return line.strip()
    for token in tokens:
        if token in vocabulary:
            return token
    near = difflib.get_close_matches(tokens[0], sorted(vocabulary), n=1,
                                     cutoff=_NEAR_MISS_CUTOFF)
    return near[0] if near else tokens[0]


def _parse_worker_reply(
    text: str, *, allowed=(),
) -> tuple[Literal["action", "final", "unparseable"], str, str]:
    """Split a reply into (kind, tool_name, body).

    For an ACTION: (`"action"`, the tool named on that line resolved against
    `allowed`, the CODE: body). For a FINAL: with a non-empty body:
    (`"final"`, `""`, the answer text). Anything else -- no ACTION:/FINAL:
    marker anywhere, or a FINAL: with nothing after it -- is `"unparseable"`,
    which both loops treat as a signal to retry with corrective feedback
    rather than accepting it.

    WHICHEVER MARKER COMES FIRST WINS. This used to check ACTION: first
    regardless of position, so a reply that answered and then suggested a
    follow-up step --

        FINAL:
        the report is written to /workspace/report.md
        ACTION: execute_bash
        CODE:
        cat /workspace/report.md

    -- silently discarded the answer and ran the command instead. The loop
    then had no output to return, and the run paid for another exchange to
    get back an answer it had already been given. Reading in document order is
    also simply what the reply means.
    """
    action_at = text.find("ACTION:")
    final_at = text.find("FINAL:")
    if action_at != -1 and (final_at == -1 or action_at < final_at):
        line = text[action_at + len("ACTION:"):].split("\n", 1)[0]
        return "action", _resolve_tool(line, allowed), _code_body(text)
    if final_at != -1:
        body = text[final_at + len("FINAL:"):].strip()
        if body:
            return "final", "", body
        return "unparseable", "", text.strip()
    return "unparseable", "", text.strip()


def _action_problem(tool_name: str, body: str, allowed) -> str:
    """Why this action cannot be run, or "" if it can.

    The emit-path contract, checked before anything is dispatched. Both
    conditions used to be discovered by the tool itself, several seconds and
    one confusing error later: an unavailable tool, and an ACTION: with no
    CODE: body at all -- which reached the shell as an empty command and came
    back as a returncode the model then had to interpret.
    """
    if tool_name not in allowed:
        return (f"tool {tool_name!r} is not available "
                f"(allowed: {sorted(allowed)})")
    if not body.strip():
        return (f"{tool_name} was called with no CODE: body. Put what it "
                "should act on under a CODE: line.")
    return ""


def _model_label(llm) -> str:
    """A model's id for a log line, whichever vendor's chat class this is.

    `ChatInception`, `ChatAnthropic` and `ChatGoogleGenerativeAI` expose
    `model`; `ChatOpenAI` exposes `model_name` and keeps `model` only as a
    populate-by-name alias, so the plain attribute access this replaced raised
    AttributeError there. The messages that used it were also hardcoded to say
    "inception:", which is simply untrue for any other vendor.
    """
    for attr in ("model", "model_name", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str):
            return value
    return type(llm).__name__


def _summarise_action(tool_name: str, body: str, result) -> str:
    """One line describing a tool call and how it went, for the record the
    overseer and a re-invoked role read (agent/pipeline/state.py's `actions`).

    Deliberately lossy: the point is "you already tried this, here is what came
    back", not a replayable transcript. The first line of the body identifies
    the call (a path, a command), and only a FAILING result's text is carried
    forward -- a compiler error is exactly what the next attempt needs, the
    full stdout of a successful build is noise.
    """
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    call = f"{tool_name} {first_line.strip()[:90]}"
    if result.returncode == 0:
        detail = (result.stdout or "").strip().splitlines()
        return f"{call} -> ok{': ' + detail[0][:120] if detail else ''}"
    problem = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    return f"{call} -> FAILED (exit {result.returncode}): {problem[:200]}"


def _tool_loop(llm, messages: list, actions: list[str] | None = None,
               *, max_iterations: int | None = None) -> str:
    """Run the shared ACTION/FINAL tool-calling loop -- every role node AND
    the evaluator drive their conversation through this one function. A
    role node's FINAL body IS its candidate answer; the evaluator's FINAL
    body is instead parsed by _parse_approval() at the call site, since
    "APPROVE: yes/no\\nWHY: ..." is answer-shaped text as far as this loop
    is concerned, just interpreted differently by its caller.

    `output` (returned if the loop exhausts without ever hitting the
    `return` below) is ONLY ever set from an actual attempted answer -- a
    FINAL body, or, failing that, the last unparseable reply. An ACTION
    reply never touches it: a tool call is not a candidate answer.
    """
    # TOOL_DISPATCH plus whatever agent/pipeline/toolkit.py has bound for this
    # run -- empty in every ordinary turn, so this is a dict copy of nothing.
    # A benchmark that hands the agent task-specific tools (agent/eval/
    # claw_bench.py) binds them there rather than mutating the registry, and
    # the note is how the prompt finds out they exist: the menu in
    # _ACTION_BLOCK is derived at import and cannot know about them.
    dispatch = dispatch_table()
    note = render_note()
    if note:
        messages.insert(1, SystemMessage(note))

    budget = current_budget()
    output = ""
    dead_replies = 0
    last_target: str | None = None
    repeats = 0
    failed_targets: set[str] = set()
    for _ in range(max_iterations or MAX_TOOL_ITERATIONS):
        # The evaluator drives this loop, and a judgment that kept checking
        # past the ceiling would spend the budget the agent was stopped to
        # protect. Its own MAX_TOOL_ITERATIONS cap stays as the inner bound.
        if budget is not None and budget.spent():
            return output
        text = _call(llm, messages)
        kind_of_reply, tool_name, body = _parse_worker_reply(text, allowed=dispatch)

        if kind_of_reply == "final":
            return _strip_code_fence(body)

        if kind_of_reply == "unparseable":
            dead_replies += 1
            output = text  # fallback if the loop ends here, exhausted or not
            if dead_replies >= MAX_CONSECUTIVE_DEAD_REPLIES:
                logger.warning(
                    "tool loop: %d unparseable replies in a row -- giving up "
                    "on this node rather than spending the rest of its budget",
                    dead_replies,
                )
                return output
            # Never an EMPTY AIMessage. `_call` returns "" on an empty stream,
            # Anthropic rejects empty text blocks, and this loop is the
            # evaluator's -- which runs on Anthropic. One empty reply here
            # breaks every later call in the same judgment. The same guard
            # exists in `_agent_loop`; this copy did not get it, which is the
            # argument for there being one loop rather than two.
            if text:
                messages.append(AIMessage(text))
            messages.append(HumanMessage(UNPARSEABLE_FEEDBACK))
            continue

        dead_replies = 0

        # ACTION -- deliberately does NOT touch `output` (see docstring).
        if tool_name == "ask_user":
            # Not a normal tool -- see NeedsUserInput's own docstring for
            # why this has to unwind all the way out rather than being
            # just another TOOL_DISPATCH entry.
            question, choices = _parse_ask_user_body(body)
            raise NeedsUserInput(question or "(no question given)", choices)
        if problem := _action_problem(tool_name, body, dispatch):
            evidence = problem
        else:
            result = dispatch[tool_name](body)
            if actions is not None:
                actions.append(_summarise_action(tool_name, body, result))
            evidence = (
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                f"returncode: {result.returncode}"
            )
            target = _action_target(tool_name, body)
            repeats = repeats + 1 if target == last_target else 1
            last_target = target
            failed_before = target in failed_targets
            if result.returncode != 0:
                failed_targets.add(target)
            if result.returncode != 0 and failed_before:
                # A failing call repeated is the strongest possible signal
                # that the last one was not understood -- act on it at once
                # rather than waiting for a run of three.
                evidence += "\n\n" + REPEATED_FAILURE_NOTE.format(target=target)
            elif repeats >= REPEATS_BEFORE_NOTE:
                evidence += "\n\n" + REPEATED_CALL_NOTE.format(
                    n=repeats, tool=tool_name, target=target,
                )
        messages.append(AIMessage(text))
        messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))
    return output


# --------------------------------------------------------------------------
# The agent loop -- one conversation, many modes.
#
# What this replaces: router -> role -> router -> evaluator -> router, where
# every node rebuilt [SystemMessage, HumanMessage] from scratch and the tool
# conversation died with the node. Measured on Claw-Eval traces, the boundaries
# between those nodes cost 20 to 126 seconds each and were 45 to 69% of a run's
# wall time, while every tool call a task made totalled 0.1 to 0.4 seconds. One
# round of work was five model calls, three of them the router and evaluator.
#
# The agent forgot what it had just done AND paid to be reminded. Those were
# never two problems: they were the same boundary. So there is one loop now, and
# changing role is an appended message rather than a node transition.
# --------------------------------------------------------------------------

#: How many times a run may switch mode before it is told to finish in the one
#: it is in. A backstop, not the main guard -- a swap already costs a model call
#: and yields no tool result, so it competes for budget on its own, and
#: `_refuse_idle_swap` below catches ping-ponging much earlier.
MAX_MODE_SWAPS = 8

#: Said when a swap follows a swap with nothing done in between. Deliberately
#: the same shape as REPEATED_CALL_NOTE, which measurement already showed this
#: model acts on.
IDLE_SWAP_NOTE = (
    "NOTE: you switched to {last} and now to {want} with nothing done in "
    "between. Switching is not progress. Do the work in the mode you are in."
)


#: Said once, immediately before a tool that changes something runs for the
#: first time against a given target.
#:
#: Measured justification: a single MUTATING deviation cuts a task's success
#: odds by 55-96%, where a non-mutating one costs 7-21% -- and mutating actions
#: are only 14-18% of steps, so gating them is cheap and precisely aimed
#: (SABER, 2512.07850). Separately, 82.5% of analysed agent failures are the
#: agent failing to compare against evidence it already holds: "evidence
#: contradicting the failure already exists in the agent's execution directory,
#: yet comparison never occurs."
#:
#: Claw-Eval T026 is that failure exactly. Three contacts matched "Manager
#: Zhang", all three were in the transcript, and the agent sent to the first.
#: Rewriting the ask_user guidance took it from three sends to two. A prompt
#: cannot fix this because the problem is not what the agent knows, it is that
#: nothing makes it look.
#:
#: It must offer a way FORWARD, not a refusal. One enforcement study blocked
#: 94% of non-compliant actions and still had safe task completion below 5%,
#: because the agent fabricated credentials to route around the block.
MUTATION_GATE_NOTE = (
    "HOLD. `{tool}` changes something outside this conversation and cannot be "
    "undone.\n\n"
    "Before it runs, check it against what you already know -- not against "
    "what you intended. Name the exact target you are about to act on, and the "
    "specific thing you read that identifies it as the right one.\n\n"
    "If more than one candidate fits what you were asked for -- more than one "
    "recipient, record, file or account -- you do not know which is meant. "
    "Use ask_user. Picking one and hoping is the worst option available.\n\n"
    "If it is the right target and you have the evidence, say so in one line "
    "and issue the same call again; it will run."
)


#: How often the loop is reminded of things it was told once.
#:
#: Long runs suffer instruction fade-out: what was said in the system prompt
#: stops steering behaviour as the conversation grows past it. The answer that
#: works is event-driven reminders delivered in the CONVERSATION rather than by
#: rewriting the system prompt -- rewriting it would invalidate every vendor's
#: prefix cache, and the model has stopped attending to that region anyway.
#:
#: These ride on a tool result that was being sent regardless, so they cost no
#: model call at all.
REMINDER_EVERY = 6

#: The reflection from the strongest cheap result in the survey. Its ablation
#: is the point: an agent that COULD write itself tools scored 62% -> 64%;
#: adding this question after each step took it to 76%, and the system to 77.4%
#: on SWE-bench Verified -- beating offline self-improvers that cost 1231 GPU
#: hours. Deciding WHEN to build a tool is the mechanism; being able to is not.
TOOL_BUILDING_NOTE = (
    "You have been at this a while. Is there a small script you could write "
    "once and run repeatedly that would make the rest of this faster or more "
    "reliable than doing it by hand each time? Write it if so; if not, carry "
    "on."
)


def _reminders(iteration: int, checklist: list[dict] | None) -> str:
    """What to re-say at this point in the loop, if anything.

    Deliberately periodic rather than every turn. Said constantly these become
    part of the wallpaper, which is the failure mode they exist to fix.
    """
    if iteration == 0 or iteration % REMINDER_EVERY:
        return ""
    parts = [TOOL_BUILDING_NOTE]
    # `seen` is not open. Something has already been written for it, and
    # re-listing it is how a reminder turns into wallpaper.
    open_items = [i for i in (checklist or []) if i.get("status") == "pending"]
    if open_items:
        parts.append(
            "Still open:\n" + "\n".join(f"- {i['text']}" for i in open_items)
        )
    return "\n\n".join(parts)


def _mutates(tool_name: str) -> bool:
    """Whether this tool changes something that cannot be taken back.

    TOOL_TIERS has recorded this since it was written and had no production
    reader until the prompt started naming the mutating tools; this is the
    second. Run-scoped tools carry their own flag and default to True, so an
    unmarked benchmark tool is gated rather than waved through.
    """
    extra = current_extra_tools().get(tool_name)
    if extra is not None:
        return extra.mutates
    # MUTATING only, not WORKSPACE. Writing a file into a throwaway workspace
    # or a task container is recoverable -- write it again. The measured case
    # for gating is about actions that change something OUTSIDE: a refund, a
    # cancellation, a sent message. Gating workspace writes cost a model call
    # per new file and bought nothing; measured live, a two-file task went from
    # 6 calls to 9.
    #
    # No standing tool is MUTATING today, which tools.py's own invariant
    # asserts, so in practice this gate applies to run-scoped tools -- which is
    # exactly where T026's `gmail_send_message` lived.
    return TOOL_TIERS.get(tool_name, MUTATING) == MUTATING


def _emit(payload: dict) -> None:
    """Send one live event to whoever is streaming this run, or do nothing.

    LangGraph's "updates" stream only emits when a node RETURNS. That was fine
    when a node returned every few seconds; with one loop that runs for minutes
    it would mean `otto chat` sits silent and then prints one panel. This is the
    "custom" stream (agent/pipeline/run.py's _STREAM_MODES), and the payload is
    deliberately shaped exactly like a node update so agent/cli/chat.py and
    agent/cli/tui.py need no changes at all.

    Best-effort by design: outside a graph run, or when nothing subscribes to
    the custom stream, the writer is absent or a no-op -- so run_pipeline() and
    every offline test are unaffected.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(payload)
    except Exception:  # not in a graph, or no custom subscriber
        pass


def _mode_message(name: str) -> HumanMessage:
    """The message that puts the loop into a mode.

    A HumanMessage, NEVER a SystemMessage, and this is load-bearing rather than
    stylistic: langchain_anthropic raises ValueError("Received multiple
    non-consecutive system messages") and langchain_google_genai hoists a
    mid-list system message into `system_instruction` out of position or drops
    it at a bare `else: pass`. Three of the routed Tasks are pinned to
    Anthropic, so a system message here would be a hard 400 in some modes and a
    silent no-op in others.
    """
    return HumanMessage(
        f"MODE: {name}\n{MODES[name].guidance}\n"
        "Same task, same tools, same conversation -- everything above still applies."
    )


def _seed_transcript(state: AgentState, task_text: str, checklist=None) -> list:
    """The conversation a run starts from. Built once per run, never rebuilt.

    The two system messages are adjacent at indices 0 and 1 on purpose: a run of
    system messages at the very start is legal on every vendor here, and the
    moment one appears later it is not (see _mode_message). Everything after
    them is Human/AI for the life of the run.
    """
    history = _conversation_so_far(state)
    context = state.get("context") or ""
    body = "\n\n".join(part for part in (
        f"CONVERSATION SO FAR:\n{history}" if history else "",
        f"CONTEXT GATHERED SO FAR:\n{context}" if context else "",
        f"TASK:\n{task_text}",
        _lessons_block(task_text),
        _render_checklist(checklist),
        CONVERSATION_NOTE if checklist == [] else "",
    ) if part)

    note = render_note()
    messages: list = [SystemMessage(AGENT_PROMPT)]
    if note:
        messages.append(SystemMessage(note))
    messages.append(HumanMessage(body))
    messages.append(_mode_message(state.get("mode") or DEFAULT_MODE))
    return messages


def _lessons_block(task_text: str) -> str:
    """At most ONE lesson from an earlier run, and only if it is about this.

    Three deliberate restraints, each measured. One, not several: task-time
    procedural recall peaks at k=1 and loses about 7 points by k=5. Here at
    the seed, not on every turn: always-on injection of skill-like items
    scored 16.4 points below injecting none. And nothing at all when nothing
    is relevant -- agent/memory/lessons.py returns an empty list rather than
    the closest match, because an off-topic lesson is worse than silence.
    """
    lessons = recall_lessons(str(task_text or ""))
    if not lessons:
        return ""
    return (
        "FROM AN EARLIER RUN (might not apply -- ignore it if it does not):\n"
        + "\n".join(f"- {lesson.rendered()}" for lesson in lessons)
    )


def _plain(messages: list) -> list[dict]:
    """A transcript as JSON-able dicts, for the checkpointer and the trace.

    LangGraph serialises state on every super-step and Langfuse serialises it
    into every span, so storing message OBJECTS would put the whole
    conversation through both on each pass.
    """
    kinds = {HumanMessage: "human", AIMessage: "ai"}
    return [
        {"kind": kinds[type(m)], "content": _content_text(m.content)}
        for m in messages
        # System messages are deliberately NOT stored. The only ones in the list
        # are the opening pair, which `agent` rebuilds on resume; storing them
        # would round-trip the same two constants through the checkpointer and
        # every Langfuse span on each super-step, and would risk one being
        # revived into the middle of a list -- the exact shape that raises on
        # Anthropic and is silently dropped on Gemini.
        if type(m) in kinds and _content_text(m.content).strip()
    ]


def _revive(stored) -> list:
    """The inverse of _plain. A stored SystemMessage is dropped rather than
    revived: the only ones ever stored are the opening pair, which
    _seed_transcript rebuilds, and reviving one into the middle of a list is
    the exact failure _mode_message exists to avoid."""
    if not stored:
        return []
    builders = {"human": HumanMessage, "ai": AIMessage}
    out = []
    for entry in stored:
        builder = builders.get(entry.get("kind", "human"))
        content = entry.get("content") or ""
        if builder is not None and content:
            out.append(builder(content))
    return out


def _conversation_so_far(state: AgentState) -> str:
    """Prior turns of this session's conversation, as plain dialogue lines
    -- everything in state["messages"] except the very last one (today's
    task, which every caller below already shows separately as TASK:).

    2026-09-10 design call, live-tested: "improve above solution" as a
    fresh turn had nothing to improve -- state["messages"] held only that
    one sentence, because agent/pipeline/run.py's _initial() never seeded
    it with anything earlier. That half of the fix is run.py's new
    `history` parameter; THIS half is making every prompt below actually
    show what it carries once it's there. Empty on a session's first turn,
    and for every eval/debug caller that never passes `history` at all
    (run.py's default `()`) -- in both cases `state["messages"]` has
    exactly one entry, so this returns "" and nothing changes for them.
    """
    prior = state["messages"][:-1]
    if not prior:
        return ""
    lines = []
    for m in prior:
        speaker = "you" if isinstance(m, HumanMessage) else "otto"
        lines.append(f"{speaker}: {_content_text(m.content)}")
    return "\n".join(lines)


def _actions_block(state: AgentState) -> str:
    """What has already been done in this run, as prompt text -- or "" if
    nothing has.

    Shown to BOTH the overseer and the role it dispatches, because the loop
    this fixes needed both to see it. A role's tool conversation lives in a
    local list inside _tool_loop and dies when that node returns, so a
    re-invoked role started blind; and the overseer, deciding what to do next,
    could see only that "solver produced an output", never what solver had
    tried. Measured on a hard task: the solver wrote the same file 166 times
    across five rounds, a different draft each time, never compiling any of
    them. Neither half of the loop knew the work had already been done.
    """
    already = state.get("actions") or []
    if not already:
        return ""
    return (
        "WHAT HAS ALREADY BEEN DONE (tool calls from earlier attempts in this "
        "run, oldest first). These already happened and their effects are "
        "real. Do not repeat one expecting a different result -- build on it, "
        "or find out why it did not work:\n"
        + "\n".join(f"- {line}" for line in already[-_ACTIONS_SHOWN:])
    )


#: How many exchanges the evaluator gets to reach a verdict.
#:
#: Measured, after giving it the evidence it had been missing: it went from
#: judging blind to spending its whole five-iteration budget checking, on every
#: judgment. On a task as small as "write fib.py and run it", one run cost 24
#: model calls and 106 seconds -- and FIFTEEN of those 24 were the evaluator,
#: three judgments of five calls each. The loop doing the actual work used six.
#:
#: That is the opposite of what the evidence was for. The point was one
#: better-informed judgment, not four extra checks per judgment: it can now see
#: the commands that were run and what they printed, so the common case needs no
#: tool call at all. Two leaves room for one real check when something genuinely
#: cannot be taken on trust.
MAX_EVALUATOR_ITERATIONS = 2

#: Criteria per judgment. Small and non-overlapping is the point -- each is
#: scored in turn, and overlapping criteria double-count one mistake.
RUBRIC_MAX = 5


@dataclass(frozen=True, slots=True)
class Verdict:
    """What one judgment concluded, kept as four separate facts."""

    approved: bool
    reason: str
    #: 0.0-1.0, the share of criteria met. The continuous half.
    process: float
    #: (met, total), so a caller can say "3 of 4" rather than "0.75".
    criteria: tuple[int, int]
    #: Whether what went unmet was outside the agent's control. An environment
    #: blocker is not the agent being wrong, and counting it as one teaches
    #: the wrong lesson to anything downstream.
    blocked: bool


#: How many rejections a run may collect before the answer stands anyway.
#: Judgment is worth paying for; judgment without a bound is a way to spend a
#: whole budget re-reading the same answer.
MAX_REJECTIONS = 2

#: Ceiling on each piece of evidence handed to the evaluator. Two-ended, like
#: agent/pipeline/tools.py's own clip, for the same reason: the start says what
#: was attempted and the end says how it came out, and keeping only the head
#: loses the half that decides the verdict.
_EVIDENCE_CHARS = 6000

#: How much of the loop's own conversation the evaluator sees. The tail, because
#: what it needs to check is how the run FINISHED -- the commands that were meant
#: to prove the work, and what they actually printed.
_EVIDENCE_MESSAGES = 6


def _clip_evidence(text: str) -> str:
    if len(text) <= _EVIDENCE_CHARS:
        return text
    half = _EVIDENCE_CHARS // 2
    return (
        f"{text[:half]}\n... [{len(text) - _EVIDENCE_CHARS} characters omitted] "
        f"...\n{text[-half:]}"
    )


def _evidence_tail(state: AgentState) -> str:
    """The end of the agent's own working, for the evaluator to check against.

    This is the evidence that used not to exist. The evaluator saw an answer and
    no way to verify it, so "I cannot confirm this" came back as a rejection --
    and each rejection cost a full extra round of the graph.
    """
    stored = state.get("transcript") or []
    tail = stored[-_EVIDENCE_MESSAGES:]
    if not tail:
        return ""
    speaker = {"ai": "otto", "human": "result"}
    lines = [
        f"{speaker.get(entry.get('kind'), 'result')}: {entry.get('content', '')}"
        for entry in tail
    ]
    return _clip_evidence("\n".join(lines))


#: When the loop starts compacting itself, in characters of conversation.
#: Roughly 40k tokens at four characters a token -- comfortably inside every
#: routed model's window, and chosen so compaction is rare and chunky rather
#: than incremental. That matters for cost as well as noise: every vendor
#: caches its own prefix, and rewriting history invalidates the cache from the
#: rewrite point, so many small compactions would cost more than they save.
LOOP_COMPACT_AT = 160_000

#: How many recent messages stay verbatim. The tail is what the model is
#: actually reasoning over; the head is what it has already acted on.
KEEP_VERBATIM = 8

#: What a compacted tool result is cut down to. Enough to remember the shape
#: of what came back -- an error class, a count, the first line of output --
#: without carrying the whole thing for the rest of the run.
COMPACTED_RESULT_CHARS = 240


#: Message prefixes that are never compacted, whatever their age.
#:
#: Type-blind compaction is measured as destructive: constraint recall falls to
#: 53% at 50% compression and 24% at 10%, and 53% -> 10% over five successive
#: rounds. Type-AWARE compaction holds 100/95/80 and stabilises at 96%, and the
#: behavioural difference is real -- 37.7% against 29.2% on one benchmark,
#: p=0.005 (The Compaction Cliff, 2608.22752).
#:
#: What must survive is anything that says what the run is FOR or what it may
#: not do. Losing a tool result costs a re-run; losing a constraint means the
#: agent does the wrong thing confidently for the rest of the session.
_NEVER_COMPACT = (
    "TASK:",
    "THIS IS WHAT HAS TO BE TRUE",
    "MODE:",
    "CONVERSATION SO FAR:",
    "CONTEXT GATHERED SO FAR:",
    "EVALUATOR REJECTED",
    "HOLD.",
)


def _compact(messages: list, actions: list[str] | None = None) -> int:
    """Shrink the oldest tool results in place. Returns how many it rewrote.

    Costs NOTHING -- no model call -- which is why it is the first tier and why
    it runs before anything cleverer. On a tool-heavy run the transcript is
    mostly tool output by volume, so cutting the old ones down reclaims most of
    it, and `actions` still carries the one-line record of everything.

    Deliberately lossy in the same way `_summarise_action` is: "you already
    tried this, here is roughly what came back". What it must never touch is
    the seed (the task, the opening prompts) or the recent tail, so the run
    keeps both what it was asked and what it is in the middle of.

    This BOUNDS the transcript on its own -- after compaction a run at the
    default 120-call ceiling holds roughly 8 verbatim results plus a hundred
    240-character stubs, some 15k tokens, well inside every routed window.

    It used to stop there, and the dropped bytes were simply gone. `_keep_evicted`
    below is the second tier: the full text goes into the session store under
    EVICTED_KIND before the message is overwritten, so `recall_memory` can
    search it. Still no model call, and still nothing summarised.
    """
    rewritten = 0
    protected = len(messages) - KEEP_VERBATIM
    summaries = list(actions or [])
    seen_results = 0
    for i, message in enumerate(messages):
        if not isinstance(message, HumanMessage):
            continue
        text = _content_text(message.content)
        if not text.startswith("TOOL RESULT:"):
            continue
        # Count every result, compacted or not, so the summary line still lines
        # up with the call it describes once some have been rewritten.
        index, seen_results = seen_results, seen_results + 1
        if i >= protected or len(text) <= COMPACTED_RESULT_CHARS:
            continue
        if any(marker in text for marker in _NEVER_COMPACT):
            continue
        # A real summary rather than a truncation, at no cost: `actions` already
        # holds one line per call from `_summarise_action`, written when the
        # call ran. Truncating to the first 240 characters keeps whatever
        # happened to come first, which for a failing command is usually the
        # banner and not the error.
        summary = summaries[index] if index < len(summaries) else ""
        # The full text, kept where `recall_memory` can find it again, BEFORE
        # the message is overwritten. This is the second tier.
        kept = _keep_evicted(text)
        # The stub says where the rest went -- but ONLY when it actually went
        # somewhere. A model that can see the bytes are missing and is not told
        # they are searchable will re-run the command, which is what it was
        # correctly told to do before the second tier existed; and a model told
        # to search for something nothing stored will spend a call being told
        # no memory is bound. Both are wrong in the other's situation, so the
        # hint follows the storing.
        messages[i] = HumanMessage(
            (f"TOOL RESULT (compacted): {summary}" if summary else
             text[:COMPACTED_RESULT_CHARS] + "\n... [older result, compacted]")
            + (EVICTED_HINT if kept else "")
        )
        rewritten += 1
    return rewritten


#: Appended to every compacted stub. One short line, because it is appended
#: to as many stubs as a long run has old tool calls.
EVICTED_HINT = "\n(the full output is searchable: ACTION: recall_memory)"


def _keep_evicted(text: str) -> bool:
    """Put an evicted tool result somewhere `recall_memory` can still find it.

    Until this existed, `_compact` was one-way: a result older than the recent
    tail was replaced by its one-line summary and the bytes were gone. The
    honest thing to tell the model then was "you can run that again", which is
    true, and not "you can search for it", which was not. Now it is.

    No model call, and no bullets. Chunks with their own embeddings, ranked
    directly by agent/memory/retrieval.py's `recall_chunks`.

    Returns whether the text was actually kept, which is what decides whether
    the stub left behind may claim it is searchable.

    False and silent when no store is bound -- every run outside a chat
    session, the whole benchmark harness included. An embedder that is down
    still returns True: an unembedded chunk is unrankable, not lost, and
    becomes rankable again when the store is re-embedded.
    """
    store = current_store()
    if store is None or not text.strip():
        return False
    try:
        vector = embed([text])[0]
        model = embedding_model_name()
    except Exception as exc:  # noqa: BLE001 -- never fail a run over this
        logger.debug("evicted result stored without an embedding: %s", exc)
        vector, model = None, None
    try:
        store.add_chunk(EVICTED_KIND, content_hash(text), text, vector, model)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not store an evicted result: %s", exc)
        return False
    return True


#: Asked when a run has spent its budget without producing an answer.
#:
#: The run-level budget already tells the loop to wrap up at 80% -- see
#: agent/pipeline/budget.py's WRAP_UP_NOTE -- and this is what happens when
#: that did not take. Deliberately blunt, and deliberately not asking for more
#: work: the one thing left worth doing is writing down what is already known.
OUT_OF_BUDGET_NOTE = (
    "You are out of budget and this is your last reply. Do not call another "
    "tool. Answer now with FINAL: and the best answer you can give from what "
    "you already have -- partial is fine, and say what is uncertain. An "
    "incomplete answer somebody can read beats no answer at all."
)


def _answer_from_what_is_here(llm, messages: list) -> str:
    """One last call, to turn a spent run into a readable answer.

    Costs one model call on exactly the runs that were going to report
    nothing, which is the cheapest possible place to spend one. Failures here
    return "" -- the same nothing the caller already had -- so this can never
    make the outcome worse than it was.
    """
    try:
        reply = _call(llm, [*messages, HumanMessage(OUT_OF_BUDGET_NOTE)])
    except Exception as exc:  # noqa: BLE001 -- a spent run must not also raise
        logger.info("could not salvage an answer from a spent run: %s", exc)
        return ""
    kind, _, body = _parse_worker_reply(reply)
    if kind == "final":
        return _strip_code_fence(body)
    if kind == "action":
        # A TOOL CALL IS NOT AN ANSWER. Returning the raw reply here would put
        # "ACTION: execute_bash\nCODE:\n..." in front of the person as the
        # result of their run -- the ACTION-protocol leak this loop has a rule
        # against, reintroduced by the one path that exists to salvage
        # something. It asked for an answer and got another tool call, so
        # there is no answer, and "" is the honest report.
        return ""
    # Prose that simply forgot the marker is still an answer somebody can read.
    return reply.strip()


def _transcript_size(messages: list) -> int:
    return sum(len(_content_text(m.content)) for m in messages)


def _agent_loop(state: AgentState, messages: list, *, mode: str,
                actions: list[str], mode_log: list[str],
                seed: list | None = None,
                checklist: list[dict] | None = None,
                max_iterations: int | None = None,
                may_delegate: bool = True) -> tuple[str, str, str]:
    """Run one conversation until it answers, pauses, or runs out of budget.

    Returns `(output, why_it_stopped, mode)`. `output` is only ever set from an
    attempted ANSWER -- a FINAL body, or failing that the last unparseable
    reply. A tool call is not a candidate answer, which is the same rule the
    loop this replaces had and the same reason: the ACTION protocol's raw text
    leaking out as an answer is a bug that already happened once.

    `messages` is mutated in place. The caller persists it, so a run that pauses
    for a question or dies on a budget still hands back everything it learned.
    """
    budget = current_budget()
    output = ""
    dead_replies = 0
    last_target: str | None = None
    repeats = 0
    failed_targets: set[str] = set()
    swaps = 0
    did_work_since_swap = True
    #: (tool, target) pairs already held once. A gate that fired every time
    #: would either loop forever or teach the model to ignore it.
    confirmed: set[str] = set()
    #: What this run has actually proved. See agent/pipeline/evidence.py; it
    #: costs nothing until the moment a run tries to finish with code changed
    #: and nothing run, and it is allowed to say that once.
    ledger = Ledger()
    asked_for_proof = False
    iteration = 0
    llm = ROUTER.chat_model(MODES[mode].task)

    while max_iterations is None or iteration < max_iterations:
        if _transcript_size(messages) > LOOP_COMPACT_AT:
            dropped = _compact(messages, actions)
            if dropped:
                logger.info("agent loop: compacted %d old tool result(s)", dropped)
                _emit({"agent": {"board": [f"compacted {dropped} older tool result(s)"]}})

        if budget is not None:
            if budget.spent():
                # Out of budget with nothing to show is the worst outcome
                # available, and until now it was a common one: the loop
                # returned "" and the run reported nothing at all. One more
                # call buys an answer from what it already has.
                #
                # This is the half that was missing under per-turn rationing.
                # Dividing a budget across turns turned one mediocre answer
                # into nine empty ones precisely because a stretch that ran
                # out ended with nothing; nothing can be rationed into a loop
                # that fails this way.
                if not output:
                    output = _answer_from_what_is_here(llm, messages)
                return output, "budget", mode
            note = budget.wrap_up_once()
            if note:
                messages.append(HumanMessage(note))

        text = _call(llm, messages)
        # Resolved against what this run can actually call, which includes
        # whatever agent/pipeline/toolkit.py bound for it -- a benchmark's
        # task-specific tools are exactly the names a model is most likely to
        # decorate or misspell, having seen them once in a prompt.
        dispatch = dispatch_table()
        kind_of_reply, tool_name, body = _parse_worker_reply(text, allowed=dispatch)

        if kind_of_reply == "final":
            # The one place this is worth an exchange: code changed, nothing
            # ran. Asked ONCE -- a guard that keeps asking is a guard the model
            # learns to answer rather than act on. It is also allowed to be
            # told there is nothing to run, because enforcement with no way to
            # say no gets routed around rather than obeyed.
            if ledger.needs_check and not asked_for_proof:
                asked_for_proof = True
                _emit({"agent": {"board": [
                    "holding the answer: " + ", ".join(ledger.unproven[:3])
                    + " changed with nothing run"
                ]}})
                messages.append(AIMessage(text))
                messages.append(HumanMessage(render_unproven(ledger)))
                continue
            return _strip_code_fence(body), "final", mode

        if kind_of_reply == "unparseable":
            dead_replies += 1
            output = text or output
            if dead_replies >= MAX_CONSECUTIVE_DEAD_REPLIES:
                logger.warning(
                    "agent loop: %d unparseable replies in a row -- giving up on "
                    "this attempt rather than spending the rest of the budget",
                    dead_replies,
                )
                return output, "dead", mode
            # An empty reply must not become an empty AIMessage. `_call` returns
            # "" on an empty stream, Anthropic rejects empty text blocks, and in
            # a transcript that never resets one of those would break every
            # later Anthropic call for the rest of the run.
            if text:
                messages.append(AIMessage(text))
            messages.append(HumanMessage(UNPARSEABLE_FEEDBACK))
            continue

        dead_replies = 0

        if tool_name == "ask_user":
            question, choices = _parse_ask_user_body(body)
            raise NeedsUserInput(question or "(no question given)", choices)

        if tool_name == "delegate":
            if not may_delegate:
                # One level only. A sub-agent that delegates is a subtask that
                # was never bounded, and the depth would compound silently.
                evidence = ("delegate is not available inside a delegated "
                            "subtask -- do this one yourself")
            else:
                result = _delegate(state, body, actions=actions, parent_mode=mode)
                evidence = (
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                    f"returncode: {result.returncode}"
                )
            iteration += 1
            messages.append(AIMessage(text))
            messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))
            did_work_since_swap = True
            continue

        if tool_name == "switch_mode":
            messages.append(AIMessage(text))
            mode, swaps, did_work_since_swap = _switch_mode(
                messages, body, mode=mode, swaps=swaps,
                did_work=did_work_since_swap, mode_log=mode_log,
                calls=budget.calls if budget else 0, seed=seed,
                confirmed=confirmed,
            )
            llm = ROUTER.chat_model(MODES[mode].task)
            continue

        did_work_since_swap = True

        # The gate. Before a tool that cannot be undone runs for the first time
        # against a given target, make the model check it against what it
        # already read -- see MUTATION_GATE_NOTE for why a prompt alone does
        # not do this. Costs one call on the ~15% of steps that mutate.
        gate_key = f"{tool_name}:{_action_target(tool_name, body)}"
        if tool_name in dispatch and _mutates(tool_name) and gate_key not in confirmed:
            confirmed.add(gate_key)
            messages.append(AIMessage(text))
            messages.append(HumanMessage(MUTATION_GATE_NOTE.format(tool=tool_name)))
            _emit({"agent": {"board": [f"holding {tool_name} for a check"]}})
            continue

        if problem := _action_problem(tool_name, body, dispatch):
            evidence = problem
        else:
            result = dispatch[tool_name](body)
            ledger.record(tool_name, body, result.returncode)
            line = _summarise_action(tool_name, body, result)
            actions.append(f"{mode}: {line}")
            _emit({"agent": {"board": [f"{mode}: {line}"]}})
            evidence = (
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                f"returncode: {result.returncode}"
            )
            target = _action_target(tool_name, body)
            repeats = repeats + 1 if target == last_target else 1
            last_target = target
            failed_before = target in failed_targets
            if result.returncode != 0:
                failed_targets.add(target)
            if result.returncode != 0 and failed_before:
                evidence += "\n\n" + REPEATED_FAILURE_NOTE.format(target=target)
            elif repeats >= REPEATS_BEFORE_NOTE:
                evidence += "\n\n" + REPEATED_CALL_NOTE.format(
                    n=repeats, tool=tool_name, target=target,
                )
        iteration += 1
        # The LIVE checklist, not state's. On a first run the loop builds it
        # and state still holds None, so reading state here silently reminded
        # the model of nothing -- caught by a test, not by reading.
        # Settle what the action record already grounds, before deciding what
        # is still worth re-stating. A criterion whose artefact was written
        # twenty actions ago is not open, and nagging about it is how the
        # reminder becomes wallpaper.
        if checklist is not None:
            checklist[:] = _note_evidence(checklist, actions)
        reminder = _reminders(iteration, checklist)
        if reminder:
            evidence += "\n\n" + reminder
        messages.append(AIMessage(text))
        messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))

    # Only a bounded sub-loop reaches here. The parent loop has no iteration
    # cap -- it runs until it answers, pauses, or runs out of budget -- so its
    # `while` is always true and every exit above is a `return`.
    return output, "exhausted", mode


#: How many exchanges a delegated subtask gets. Small on purpose: a sub-agent
#: that needs a long conversation is a subtask that was not bounded properly,
#: and the parent is better placed to notice that than the child is.
MAX_DELEGATE_ITERATIONS = 8

#: Ceiling on what one mode hands forward across an escalating switch. Long
#: enough for a real plan, short enough that it cannot reintroduce the
#: transcript the escalation exists to drop.
MAX_CARRIED_CHARS = 3000


#: Told to the sub-agent instead of the parent's conversation.
#:
#: The measured shape. Reasoning belongs at the ORCHESTRATOR: putting it there
#: was worth +18.2 and +36.7 points on two benchmarks at 8% added latency,
#: where putting it in the sub-agents was marginal-to-negative at +77%. And
#: sub-agent SIZE barely mattered once the orchestrator thought -- 23.0 / 23.0
#: / 23.6 across models twenty times apart. So the child is thin by design, not
#: by economy.
DELEGATE_CONTRACT = (
    "You have been given one bounded job by another agent, which is handling "
    "the wider task. Do exactly this and nothing beyond it:\n\n{instruction}\n\n"
    "You do not have the conversation it came from and you do not need it. "
    "When you are done reply with exactly\nFINAL:\n<what you found or did, "
    "stated so someone who cannot see your working can act on it>"
)


def _delegate(state: AgentState, body: str, *, actions: list[str],
              parent_mode: str) -> ToolResult:
    """Run one bounded subtask in a fresh context, on another mode's model.

    This is the one shape of multi-agent the evidence actually supports. Where
    every agent shares a model, a single agent role-playing the workflow
    matches or beats the multi-agent version at lower cost -- so a sub-agent
    earns its keep only when the model genuinely differs, or the context must
    be isolated, or both. Otto's modes differ by model, which is what makes
    this worth having rather than a second way to spend calls.

    What crosses in each direction is the point. Down: a contract, never the
    parent's transcript. Up: a report, never a trajectory. The child's working
    is discarded when it returns, which is what keeps a long delegation from
    costing the parent its context window.
    """
    head, _, instruction = body.partition("\n")
    want = parse_mode_body(head)
    if want is None:
        return ToolResult(
            stdout="",
            stderr=(f"delegate: first line must be a mode -- one of "
                    f"{', '.join(mode_names())}. Then the job on the lines after it."),
            returncode=1,
        )
    if not instruction.strip():
        return ToolResult(
            stdout="", stderr="delegate: say what the job is on the lines after the mode.",
            returncode=1,
        )
    if want == parent_mode:
        # Same model, no isolation gained that a fresh prompt would not give.
        # This is the degenerate case the literature warns about: sub-agents
        # used purely as context-isolation threads, paying coordination for
        # something the parent could do itself.
        return ToolResult(
            stdout="",
            stderr=(f"delegate: you are already in {want} mode, so this would run on "
                    "the same model with no context you do not have. Just do it."),
            returncode=1,
        )

    child: list = [SystemMessage(AGENT_PROMPT)]
    if note := render_note():
        child.append(SystemMessage(note))
    child.append(HumanMessage(DELEGATE_CONTRACT.format(instruction=instruction.strip())))
    child.append(_mode_message(want))

    taken: list[str] = []
    _emit({"agent": {"board": [f"delegated to {want}: {instruction.strip()[:60]}"]}})
    try:
        output, why, _ = _agent_loop(
            state, child, mode=want, actions=taken, mode_log=[],
            max_iterations=MAX_DELEGATE_ITERATIONS, may_delegate=False,
        )
    except NeedsUserInput:
        # The child cannot pause the run -- it does not own the conversation
        # with the person. Hand the question up as its result and let the
        # parent decide whether to ask it.
        raise
    except ProviderError as exc:
        return ToolResult(stdout="", stderr=f"delegate: the subtask failed: {exc}", returncode=1)

    # The child's actions join the parent's record; its conversation does not.
    actions.extend(f"{want}(delegated): {line}" for line in taken)
    if not output.strip():
        return ToolResult(
            stdout="", stderr=f"delegate: {want} finished without an answer ({why})", returncode=1,
        )
    return ToolResult(stdout=output, stderr="", returncode=0)


def _last_ai_text(messages: list) -> str:
    """The most recent thing the model actually said, clipped.

    Used to carry one mode's output across an escalation that otherwise wipes
    the conversation. Deliberately the last AI turn rather than a search for
    something plan-shaped: the loop does not know what a plan looks like, and
    a heuristic that did would quietly fail on the day a mode produced
    something else worth keeping.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _content_text(message.content).strip()
            if text:
                return text[:MAX_CARRIED_CHARS]
    return ""


def _switch_mode(messages: list, body: str, *, mode: str, swaps: int,
                 did_work: bool, mode_log: list[str], calls: int,
                 seed: list | None = None,
                 confirmed: set[str] | None = None) -> tuple[str, int, bool]:
    """Handle one `switch_mode` request. Returns `(mode, swaps, did_work)`.

    Three refusals, cheapest first, and none of them raises -- a refusal is a
    message the model reads and acts on, exactly like a failed tool result.

    And an asymmetry, which is the measured part. Handing a stronger model the
    weaker one's trajectory recovers under half the quality it should, at four
    to six times the cost; DISCARDING that trajectory takes recovery from 47%
    to 64%. The reverse is not true -- removing a strong model's trajectory
    before handing down HURTS. In the paper's words: strong trajectories guide
    weak receivers, weak trajectories burden strong ones.

    So escalating restarts from the seed, and de-escalating carries everything.
    The restart is only affordable because the checklist survives it: what the
    run has established is state, not conversation, so a clean restart loses
    the weaker model's phrasing and none of its findings.
    """
    want = parse_mode_body(body)
    if want is None:
        messages.append(HumanMessage(
            f"switch_mode: that is not a mode. Choose one of "
            f"{', '.join(mode_names())}. You are still in {mode}."
        ))
        mode_log.append(f"call {calls}: refused (unknown mode)")
        return mode, swaps, did_work

    if want == mode:
        # Deliberately does NOT re-append the guidance: it is already above,
        # and repeating it would grow the prompt every time the model asked
        # for what it already has.
        messages.append(HumanMessage(
            f"You are already in {mode} mode; its guidance is above. Continue."
        ))
        mode_log.append(f"call {calls}: refused (already in {mode})")
        return mode, swaps, did_work

    if not did_work:
        messages.append(HumanMessage(IDLE_SWAP_NOTE.format(last=mode, want=want)))
        mode_log.append(f"call {calls}: refused ({mode} -> {want}, no work between)")
        return mode, swaps, did_work

    if swaps >= MAX_MODE_SWAPS:
        messages.append(HumanMessage(
            f"You have switched modes {swaps} times. Finish in the mode you "
            "are in."
        ))
        mode_log.append(f"call {calls}: refused (swap limit)")
        return mode, swaps, did_work

    reason = mode_reason(body)
    escalating = MODES[want].depth > MODES[mode].depth
    messages_before = list(messages)
    if escalating and seed is not None:
        # Clean restart. Everything the run established is in the checklist and
        # in the workspace; what is dropped is the weaker model's account of
        # getting there, which is the part measured as a burden.
        messages[:] = list(seed)
        # The gate's memory goes with the conversation it was recorded in.
        #
        # `confirmed` holds "this tool has already been held once against this
        # target, so let the reissue through". That is only true while the
        # model can REMEMBER being asked. An escalating switch wipes the
        # transcript back to the seed, so the model that arrives next has no
        # record of the hold -- and without this line it would find the gate
        # already satisfied and run the irreversible call unchecked. Holding
        # the same target twice costs one exchange; not holding it costs the
        # thing the gate exists to prevent.
        if confirmed is not None:
            confirmed.clear()
        # The LAST thing this mode produced comes with it.
        #
        # Without this, `plan` is self-defeating: its own guidance says to
        # "switch back and carry the steps out yourself", solve is the deepest
        # mode so switching back always escalates, and escalating wipes the
        # transcript -- so following the instruction destroys the plan at the
        # moment it is needed. Nothing else held it: `context` is only written
        # by ask_user and the checklist is fixed at run start.
        #
        # One message, not the whole conversation. What escalation is for is
        # dropping the weaker model's account of getting somewhere; the thing
        # it arrived at is the part worth carrying.
        carried = _last_ai_text(messages_before)
        if carried:
            messages.append(HumanMessage(
                f"What you produced in {mode} mode, to work from:\n{carried}"
            ))
        messages.append(HumanMessage(
            f"Starting fresh in {want} mode. The work so far stands -- what is "
            "already true is listed above, and anything written is still "
            "written. What you do not have is the earlier back-and-forth, "
            "which you do not need."
        ))
    messages.append(_mode_message(want))
    direction = "escalated" if escalating else "switched"
    mode_log.append(
        f"call {calls}: {mode} -> {want}"
        + (" (restarted)" if escalating and seed is not None else "")
        + (f" ({reason})" if reason else "")
    )
    _emit({"agent": {"board": [f"{direction} to {want} mode" + (f" -- {reason}" if reason else "")]}})
    return want, swaps + 1, False


def agent(state: AgentState) -> Command[Literal["evaluator", "ask_user", "__end__"]]:
    """The one working node. Everything the four specialists did, in one
    conversation that is never thrown away.

    Every exit writes `transcript` back. That is a correctness rule rather than
    an optimisation: a run that pauses on `ask_user` and never persisted its
    conversation would resume having forgotten the work it paused in the middle
    of, which is the failure this whole rewrite exists to remove.
    """
    task_text = state["messages"][-1].content
    mode = state.get("mode") or DEFAULT_MODE
    stored = _revive(state.get("transcript"))
    resuming = bool(stored)

    # The run's working state, written once from the task alone before any
    # attempt exists. Nothing here can be shaped by an attempt trying to
    # satisfy it -- which is stronger than generating it after the fact, and
    # the same call now serves both the loop and the judgment instead of one
    # each. See AgentState.checklist.
    checklist = state.get("checklist")
    if checklist is None:
        checklist = _new_checklist(_criteria(ROUTER.chat_model(Task.EVALUATE), task_text))

    if resuming:
        messages = [SystemMessage(AGENT_PROMPT), *(
            [SystemMessage(render_note())] if render_note() else []
        ), *stored]
        feedback = state.get("feedback") or ""
        if feedback:
            messages.append(HumanMessage(
                f"EVALUATOR REJECTED THAT:\n{feedback}\n"
                + (_render_checklist(checklist) + "\n" if checklist else "")
                + "Fix what is still open. You have everything above."
            ))
    else:
        messages = _seed_transcript(state, task_text, checklist)

    actions: list[str] = []
    mode_log: list[str] = []
    budget = current_budget()

    def carry(**extra) -> dict:
        """Every return path persists the same four things."""
        update = {
            "transcript": _plain(messages),
            "checklist": checklist,
            "mode": mode,
            "model_calls": budget.calls if budget else state.get("model_calls") or 0,
        }
        if actions:
            update["actions"] = actions
        if mode_log:
            update["mode_log"] = mode_log
        update.update(extra)
        return update

    try:
        output, why, mode = _agent_loop(
            state, messages, mode=mode, actions=actions, mode_log=mode_log,
            # What an escalation restarts from: the prompts, the task and the
            # checklist, with none of the working conversation.
            seed=list(messages[:3]),
            checklist=checklist,
        )
    except NeedsUserInput as exc:
        return Command(
            update=carry(
                pending_question=exc.question,
                pending_choices=exc.choices,
                asking_role="agent",
                board=["otto is asking you a question"],
            ),
            goto="ask_user",
        )
    except ProviderError as exc:
        # A provider failure is not the agent being wrong. Hand over whatever
        # was already produced rather than losing the run -- the evaluator can
        # judge a partial answer, and an empty one is what the old graph
        # produced here.
        return Command(
            update=carry(
                node="agent",
                node_error=f"agent: {exc}",
                feedback=f"a provider/network failure interrupted the run: {exc}",
                board=[f"otto hit a provider error: {exc}"],
            ),
            goto="evaluator",
        )

    if why == "budget":
        # Straight to the end, not to the evaluator. There is nothing left to
        # pay a judgment with, and handing on would put the run in a loop: the
        # evaluator rejects for lack of evidence, the loop returns immediately
        # because it is still out of budget, and the two bounce until the
        # recursion limit. Answering with what it has is the whole reason
        # exhaustion returns instead of raising.
        return Command(
            update=carry(
                node="agent",
                output=output,
                final_output=output,
                board=["otto ran out of budget -- answering with what it has, unverified"],
            ),
            goto=END,
        )

    board = {
        "final": "otto has an answer",
        "dead": "otto could not produce a usable reply and is handing over what it has",
    }[why]
    if not checklist and why == "final":
        # Nothing to verify, so nothing downstream runs.
        #
        # The criteria call is the first thing a turn does, and an empty
        # checklist is it saying there was no task in the message. Running the
        # rest anyway is how "thanks, that's helpful" cost 22 model calls and
        # 263 seconds: the judge measured a chatty reply against criteria that
        # did not exist, rejected it, and the loop retried twice. The answer it
        # finally produced was "1|Problem to solve|No problem to solve".
        #
        # Only on a clean `final`. A run that died or ran out still goes to the
        # evaluator, because "no criteria" and "no answer" are different
        # problems and only one of them is a conversation.
        return Command(
            update=carry(node="agent", output=output, final_output=output,
                         feedback="", board=["otto answered"]),
            goto=END,
        )
    return Command(
        update=carry(node="agent", output=output, feedback="", board=[board]),
        goto="evaluator",
    )


def _new_checklist(criteria: list[str]) -> list[dict]:
    return [{"text": c, "status": "pending", "evidence": ""} for c in criteria]


#: A path-shaped token inside a criterion: something with a slash or a dot in
#: it and no whitespace. Deliberately narrow -- this is used to decide that a
#: criterion has been ACTED ON, and a looser pattern would match ordinary
#: prose and mark work done that nobody did.
_PATH_IN_TEXT = re.compile(r"[A-Za-z0-9_./~-]*[/.][A-Za-z0-9_./~-]*[A-Za-z0-9_]")

#: Tools whose SUCCESS is evidence that a named artefact now exists. Reading a
#: file is not: a criterion about a report is not satisfied by having looked
#: at one.
_PRODUCING_TOOLS = ("write_file", "edit_file", "predict_edit")


def _note_evidence(checklist: list[dict] | None, actions: list[str] | None) -> list[dict]:
    """Attach grounded evidence to criteria that have already been acted on.

    NOT a judgment, and deliberately not a `met`. A successful write to the
    exact path a criterion names is environment-grounded -- it is a returncode,
    not the agent's account of itself -- but it says the artefact exists, not
    that its contents satisfy anything. So this records `seen` and leaves the
    verdict to the evaluator, which is the only thing allowed to say `met`.

    What it buys is an honest reminder block. Criteria are re-stated every
    REMINDER_EVERY iterations while they are open, and without this a run
    keeps being nagged about something it did twenty actions ago -- which is
    how a reminder becomes wallpaper, the exact failure the reminder exists to
    prevent. Grounding the next step in what is still outstanding rather than
    in the whole history is what makes a long-horizon advantage grow instead
    of decay.

    A wrong `seen` costs a missing nag; a wrong `met` would cost the next
    attempt skipping the thing that is actually absent. Only one of those is
    worth the risk, which is why this stops short of the stronger claim.
    """
    if not checklist or not actions:
        return checklist or []
    done = {
        target
        for line in actions
        for target in (_produced_path(line),)
        if target
    }
    if not done:
        return checklist
    updated = []
    for item in checklist:
        if item.get("status") != "pending":
            updated.append(item)
            continue
        hit = next((p for p in _PATH_IN_TEXT.findall(item.get("text", "")) if p in done), "")
        updated.append({**item, "status": "seen", "evidence": f"wrote {hit}"} if hit else item)
    return updated


def _produced_path(action_line: str) -> str:
    """The path a successful producing action wrote, or "".

    Reads the one-line action record `_summarise_action` writes, because that
    is what survives compaction -- the transcript it came from may be gone.
    """
    if " ok " not in f" {action_line} " and "returncode: 0" not in action_line:
        # `_summarise_action` marks a failure explicitly; anything that says
        # so is not evidence of anything existing.
        if "failed" in action_line or "error" in action_line.lower():
            return ""
    if not any(tool in action_line for tool in _PRODUCING_TOOLS):
        return ""
    found = _PATH_IN_TEXT.findall(action_line)
    return found[0] if found else ""


def _render_checklist(checklist: list[dict] | None) -> str:
    """The working state as the loop sees it: what is settled and what is not.

    Status first on every line, because the question the loop is answering is
    "what is left", and a list that reads as prose makes that the reader's job.
    """
    if not checklist:
        return ""
    mark = {"met": "[done]", "blocked": "[blocked]", "seen": "[acted on]",
            "pending": "[  ]"}
    lines = []
    for item in checklist:
        line = f"{mark.get(item.get('status'), '[  ]')} {item.get('text', '')}"
        if item.get("evidence"):
            line += f"  <- {item['evidence']}"
        lines.append(line)
    return "THIS IS WHAT HAS TO BE TRUE WHEN YOU ARE DONE:\n" + "\n".join(lines)


def _settle(checklist: list[dict], verdict: "Verdict") -> list[dict]:
    """Apply one judgment to the run's records.

    Coarse on purpose. The verdict says how many criteria were met, not which,
    so this marks all of them or none rather than guessing an assignment --
    a wrong `met` is worse than an honest `pending`, because the next attempt
    would skip the thing that is actually missing.

    What it does carry is the distinction that matters: `blocked` is not the
    same as `pending`. An obstacle outside the agent's control should not be
    retried identically, and a plain rejection invites exactly that.
    """
    if not checklist:
        return checklist or []
    if verdict.approved:
        status, evidence = "met", verdict.reason
    elif verdict.blocked:
        status, evidence = "blocked", verdict.reason
    else:
        status, evidence = "pending", ""
    return [{**item, "status": status, "evidence": evidence} for item in checklist]


def _criteria(llm, task_text: str) -> list[str]:
    """Phase one: the rubric, from the task alone.

    A separate call on purpose. Criteria written while looking at an answer
    are criteria the answer happens to meet, which is the failure mode that
    makes a self-judging loop measure zero.

    Fails soft: a provider hiccup here must not cost the judgment. An empty
    rubric degrades the evaluator to what it was before this change, which is
    worse but not broken.
    """
    try:
        reply = _call(llm, [
            SystemMessage(RUBRIC_PROMPT),
            HumanMessage(f"TASK:\n{task_text}"),
        ])
    except ProviderError as exc:
        logger.warning("rubric generation failed, judging without one: %s", exc)
        return []
    return _parse_rubric(reply)


def _record_seat(state: AgentState, *, approved: bool) -> None:
    """Credit the seat that produced this answer, for agent/router/outcomes.py.

    ONLY SINGLE-MODE RUNS. If the agent changed modes, several models touched
    the answer and one verdict came back; crediting any of them is guessing.
    The data thrown away is real and the alternative is worse -- a log that
    mis-attributes cannot be trusted enough to change routing, which is the
    only thing it is for.

    Not the evaluator's own seat either. Nothing in a run says whether the
    JUDGE was right, so recording the judge's verdict against the judge would
    be a system marking its own homework.

    Costs no model call. This is the whole appeal of it: the evidence is a
    by-product of runs that were happening anyway.
    """
    if state.get("mode_log"):
        return
    mode = state.get("mode") or DEFAULT_MODE
    task = MODES[mode].task
    try:
        model_id = ROUTER.resolve(task).model.id
    except ProviderError:
        return
    seat_outcomes.record(
        task.value, model_id,
        approved=approved, calls=(current_budget().calls if current_budget() else 0),
    )


def _distil(state: AgentState, *, succeeded: bool) -> list[Lesson]:
    """One cheap call at the end of a run, turning the trajectory into at most
    three lessons for the next one.

    RUN ON A CHEAP MODEL, BUT NOT THE CHEAPEST. The evidence says
    harness-UPDATING is flat in base capability -- a 9B model's updates
    measure as good as a frontier model's, and a 7B meta-agent trained in one
    GPU hour gave +2.9 to +24.6% designing for stronger executors -- while
    harness-BENEFIT is not flat. So this deliberately does not get the seat
    the executor gets.

    It was Task.SUMMARIZE for exactly one measurement. On a real 29-message
    trajectory that model returned `[]` every time, including from a run the
    judge had rejected; the same body on Task.PLAN produced three usable
    lessons. "Flat in capability" is about whether the updates are GOOD,
    which says nothing about whether a model will emit any at all -- and a
    distiller that always answers "nothing to learn" is a bank that never
    fills. Task.PLAN it is, one call per run.

    FROM THE RAW TRAJECTORY, NEVER FROM THE BANK. The bank is not shown to
    this call. Consolidating a model's own distillations and feeding them back
    made one model fail 54% of problems it had previously solved; raw-episode
    retention doubled accuracy against forced consolidation. Abstract once.

    Failure here is silent by design. A run that produced an answer has done
    its job, and losing the lesson is not worth losing the answer.
    """
    if not learning_enabled():
        return []
    if not (evidence := _evidence_tail(state)) and not state.get("actions"):
        return []
    # A retried run holds BOTH a worse attempt and a better one, which is the
    # one situation where the comparison can be asked for rather than implied.
    rejections = state.get("rejections") or 0
    body = "\n\n".join(part for part in (
        f"TASK:\n{state['messages'][-1].content}",
        f"HOW IT WENT: {'the answer was accepted' if succeeded else 'it was NOT accepted'}",
        (f"IT WAS REJECTED AND RETRIED {rejections} time(s). The earlier attempt "
         f"and the later one are both above.\n{CONTRAST_NOTE}" if rejections else ""),
        _actions_block(state),
        f"THE END OF THE WORKING:\n{evidence}" if evidence else "",
    ) if part)
    try:
        # Through _call, not .invoke, so this shows up in `model_calls` like
        # every other request. A learning step whose cost the cost axis cannot
        # see is exactly the kind of thing milestone 7 exists to prevent.
        # BudgetExhausted is caught below along with everything else: a run
        # with nothing left to spend learns nothing, which is correct.
        reply = _call(ROUTER.chat_model(Task.PLAN),
                      [SystemMessage(DISTIL_PROMPT), HumanMessage(body)])
    except Exception as exc:  # noqa: BLE001 -- never fail a finished run
        logger.info("distilling lessons failed, learning nothing: %s", exc)
        return []
    return record_lessons(parse_distilled(
        reply, outcome_default="worked" if succeeded else "failed",
    ))


def evaluator(state: AgentState) -> Command[Literal["agent", "__end__", "ask_user"]]:
    task_text = state["messages"][-1].content
    node = state.get("node") or "agent"
    output = state.get("output") or ""
    judging_plan = False

    llm = ROUTER.chat_model(Task.EVALUATE)
    target, target_note = "ANSWER", "as a finished answer"
    human_label = "ANSWER"
    # The SAME number the loop below is actually run with. It used to be
    # MAX_TOOL_ITERATIONS, so the prompt promised five exchanges and the loop
    # allowed two -- a model told it has budget it does not have will plan to
    # use it.
    # The criteria the run has been working against, written from the task
    # alone before any attempt existed (AgentState.checklist). They are the
    # only information in the whole judgment that the actor did not produce,
    # and the reason it is worth its calls -- see RUBRIC_PROMPT.
    #
    # Read from state rather than regenerated: it is the same list the loop
    # has been carrying, so judge and actor cannot be working to different
    # bars, and a re-judgment after a rejection costs nothing to set up.
    # Generated here only for a caller that drove the evaluator directly.
    checklist = state.get("checklist")
    if checklist is None:
        checklist = _new_checklist(_criteria(llm, task_text))
    rubric = [item["text"] for item in checklist]

    system_prompt = EVALUATOR_PROMPT.format(
        target=target, target_note=target_note, max_iter=MAX_EVALUATOR_ITERATIONS,
        rubric="\n".join(f"- {c}" for c in rubric) or "- the request is satisfied",
    )
    # It used to judge blind: conversation, request, output, nothing else. So
    # "I cannot verify this" came back as a rejection, and a rejection cost a
    # whole extra round. Everything added below already exists and is already
    # bounded -- _actions_block caps at _ACTIONS_SHOWN, the transcript tail is
    # clipped -- so one better-informed judgment costs a fraction of the round
    # it avoids.
    human_body = "\n\n".join(part for part in (
        (f"CONVERSATION SO FAR:\n{_conversation_so_far(state)}"
         if _conversation_so_far(state) else ""),
        f"ORIGINAL REQUEST:\n{task_text}",
        f"{human_label}:\n{output}",
        _actions_block(state),
        (f"MODES USED:\n" + "; ".join(state.get("mode_log") or [])
         if state.get("mode_log") else ""),
        (f"HOW IT FINISHED (the end of otto's own working):\n{_evidence_tail(state)}"
         if _evidence_tail(state) else ""),
        (f"CONTEXT GATHERED:\n{_clip_evidence(state.get('context') or '')}"
         if state.get("context") else ""),
    ) if part)
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    try:
        reply = _tool_loop(llm, messages, max_iterations=MAX_EVALUATOR_ITERATIONS)
    except NeedsUserInput as exc:
        # Seventh refinement (module docstring): the evaluator itself got
        # stuck judging something and needs the person's input to settle
        # it. `output` (still pending judgment) is left untouched.
        return Command(
            update={
                "pending_question": exc.question,
                "pending_choices": exc.choices,
                "asking_role": "evaluator",
                "board": ["evaluator is asking you a question"],
            },
            goto="ask_user",
        )
    except ProviderError as exc:
        # A provider/network failure interrupted the evaluator itself --
        # fifth refinement (module docstring). `output` (whatever is
        # pending judgment) is left untouched; the failure text goes into
        # `feedback` so planner sees it the same way it would see any
        # other specialist's rejected attempt, via _role_body's existing
        # background display.
        what = "plan" if judging_plan else "output"
        return Command(
            update={
                "node_error": f"evaluator: {exc}",
                "feedback": (
                    f"the evaluator was interrupted by a provider/network "
                    f"failure before it could judge {node}'s {what} -- not "
                    f"a real rejection: {exc}"
                ),
                "board": [f"evaluator failed with a provider error: {exc}"],
            },
            goto="agent",
        )
    # _parse_approval defaults to approve=False whenever "APPROVE:" isn't
    # found in `reply` at all (e.g. _tool_loop exhausted on unparseable
    # replies) -- fails CLOSED by construction: an evaluator that never
    # rendered a real verdict is not evidence the answer is fine.
    verdict = _parse_verdict(reply)
    approve, reason = verdict.approved, verdict.reason
    # The audit is the ONLY thing that may move a record's status. An
    # executor's claim about its own work is not evidence, which is the whole
    # separation the state layer exists for.
    judged = _settle(checklist, verdict)
    if verdict.criteria[1]:
        reason = f"{verdict.criteria[0]}/{verdict.criteria[1]} criteria met -- {reason}"
    if verdict.blocked and not approve:
        # An environment blocker is not the agent being wrong. Saying so keeps
        # a real obstacle from being counted as a failure -- and from being
        # retried identically, which is what a plain rejection invites.
        reason = f"blocked by the environment, not by the attempt: {reason}"
    # The judgment's own calls count too. Left out, `model_calls` reports what
    # the loop spent rather than what the run spent -- measured live at 4
    # against an actual 9, because the evaluator checks the answer with tools
    # and those are model calls like any other.
    def calls_so_far() -> dict:
        """Called at each return rather than computed once above, because the
        learning step below is itself a model call. Reading this first left
        the distilling call out of the very number it is meant to appear in --
        which is the blindness milestone 7 exists to remove."""
        return {"model_calls": budget.calls} if (budget := current_budget()) else {}

    def learned_from(succeeded: bool) -> list[str]:
        """Board lines for whatever the run leaves behind. The distilling call
        happens HERE, at a terminal edge, so a run that is going back to the
        agent for another attempt does not pay for a lesson about work that is
        not finished."""
        return [f"learned: {lesson.rendered()}"
                for lesson in _distil(state, succeeded=succeeded)]

    if approve:
        _record_seat(state, approved=True)
        board = ["evaluator approved the answer"] + learned_from(True)
        return Command(
            update={
                "checklist": judged,
                "final_output": output,
                "rejections": 0,
                "board": board,
                **calls_so_far(),
            },
            goto=END,
        )
    rejections = (state.get("rejections") or 0) + 1
    if rejections > MAX_REJECTIONS:
        # Judgment must not eat the whole budget. Past the cap the answer
        # stands, said plainly rather than silently -- an unverified answer the
        # reader is told about beats a run that spent everything re-judging.
        #
        # It still learns. A run the judge would not accept is the one most
        # worth learning from -- distilling only from accepted runs throws away
        # the half of the signal that says what NOT to do.
        _record_seat(state, approved=False)
        board = [
            f"evaluator rejected {node} {rejections} times; answering "
            "anyway, unverified: " + reason
        ] + learned_from(False)
        return Command(
            update={
                "checklist": judged,
                "final_output": output,
                "rejections": rejections,
                "board": board,
                **calls_so_far(),
            },
            goto=END,
        )
    return Command(
        update={
            "checklist": judged,
            "feedback": reason,
            "rejections": rejections,
            "board": [f"evaluator rejected {node}: {reason}"],
            **calls_so_far(),
        },
        goto="agent",
    )


# --------------------------------------------------------------------------
# ask_user -- seventh refinement (module docstring). The ONLY node in this
# graph that calls interrupt(), and deliberately does nothing else: read
# the pending question off state, pause, write the answer down, hand back
# to whoever asked. See NeedsUserInput's docstring above for why an LLM
# call or a tool dispatch has no business happening in this node -- resume
# re-executes this whole function from the top, so anything with a side
# effect before interrupt() would redo it every time the person answers.
# --------------------------------------------------------------------------

def ask_user(state: AgentState) -> Command[Literal["agent", "evaluator"]]:
    question = state.get("pending_question") or ""
    choices = state.get("pending_choices") or []
    answer = interrupt({"question": question, "choices": choices})

    # Into `context`, NOT `messages` -- every node here (router included)
    # reads state["messages"][-1] as THE TASK for the whole turn;
    # appending onto it would silently replace the actual task the next
    # time anything looked (module docstring). `context` already means
    # "material gathered so far for planner/solver to use" and is already
    # shown to every prompt below via "CONTEXT GATHERED SO FAR:" -- the
    # role that asked sees this Q&A as ordinary background on its next
    # (fresh) attempt, the same way it would see anything finder dug up.
    prior = state.get("context") or ""
    qa = f'you asked: "{question}"\nthe user answered: "{answer}"'
    role = state.get("asking_role") or "agent"
    return Command(
        update={
            "context": f"{prior}\n\n{qa}" if prior else qa,
            "pending_question": None,
            "pending_choices": None,
            "asking_role": None,
            "board": [f"you answered -- {role} is carrying on"],
        },
        goto=role,
    )


g = StateGraph(AgentState)
for _name, _fn in (
    ("agent", agent),
    ("evaluator", evaluator),
    ("ask_user", ask_user),
):
    g.add_node(_name, _fn)
g.add_edge(START, "agent")
# Every other edge is a Command(goto=...) from the node function itself --
# router -> whichever of the five it picks (or a plan step's route_to);
# every specialist -> router OR ask_user (got stuck, NeedsUserInput --
# seventh refinement); evaluator -> router (reject, or plan approved),
# END (final answer approved), or ask_user (same as a specialist);
# ask_user -> whichever specialist/evaluator asked, once answered --
# nothing else to wire here.

app = g.compile(checkpointer=InMemorySaver())

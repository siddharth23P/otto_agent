"""Coverage for the seventh refinement (agent/pipeline/nodes.py's module
docstring): a role node or the evaluator can pause the whole run and ask
the person something it's genuinely stuck without, via LangGraph's own
interrupt()/Command(resume=...) mechanism, instead of guessing or looping
("Solve N Queens with brute force" -> "improve above solution" getting
stuck trying to guess what "above solution" meant, live-tested the same
day -- see nodes.py's sixth refinement for the OTHER half of that fix).

Three pieces, tested separately here:

  * _parse_ask_user_body -- splitting an ask_user ACTION's CODE: body into
    (question, choices). Covered by test_role_nodes.py/test_evaluator_
    node.py too (via the full _run_role/evaluator path), but the parsing
    rules themselves (a CHOICES: line anywhere, case-insensitively, pipe-
    separated, empty entries dropped) are tested directly here.

  * _tool_loop raising NeedsUserInput on ACTION: ask_user, instead of
    dispatching it like any other tool through TOOL_DISPATCH -- the
    signal that unwinds all the way out to the dedicated ask_user node.

  * ask_user() itself -- the ONLY node that calls interrupt(), monkeypatched
    here (it needs a real LangGraph task context otherwise, which a bare
    unit test calling the node function directly doesn't have) so its own
    logic -- reading pending_question/pending_choices/asking_role off
    state, writing the Q&A into `context` (NOT `messages` -- see the
    node's own docstring for why), clearing all three pending fields, and
    handing back to whichever role asked -- can be checked without a real
    graph run.
"""
import pytest
from langchain_core.messages import HumanMessage

from agent.pipeline import nodes as pn


# --------------------------------------------------------------------------
# _parse_ask_user_body
# --------------------------------------------------------------------------

def test_parse_ask_user_body_with_no_choices_is_just_the_question():
    question, choices = pn._parse_ask_user_body("what output format do you want?")
    assert question == "what output format do you want?"
    assert choices == []


def test_parse_ask_user_body_splits_out_a_choices_line():
    question, choices = pn._parse_ask_user_body(
        "which language should this be in?\nCHOICES: python | rust | go"
    )
    assert question == "which language should this be in?"
    assert choices == ["python", "rust", "go"]


def test_parse_ask_user_body_choices_line_is_case_insensitive():
    question, choices = pn._parse_ask_user_body("pick one\nchoices: a | b")
    assert question == "pick one"
    assert choices == ["a", "b"]


def test_parse_ask_user_body_drops_empty_choice_entries():
    question, choices = pn._parse_ask_user_body("pick one\nCHOICES: a | | b |  ")
    assert choices == ["a", "b"]


def test_parse_ask_user_body_multiline_question_before_choices():
    question, choices = pn._parse_ask_user_body(
        "here is some context.\nwhat should I do next?\nCHOICES: x | y"
    )
    assert question == "here is some context.\nwhat should I do next?"
    assert choices == ["x", "y"]


# --------------------------------------------------------------------------
# _tool_loop raising NeedsUserInput on ACTION: ask_user
# --------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[list[str]] = []

    def stream(self, messages):
        from langchain_core.messages import AIMessageChunk
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._reply)


def test_tool_loop_raises_needs_user_input_on_ask_user_action():
    fake = _FakeModel("ACTION: ask_user\nCODE:\nwhat should I call the function?")

    with pytest.raises(pn.NeedsUserInput) as excinfo:
        pn._tool_loop(fake, [HumanMessage("do the thing")])

    assert excinfo.value.question == "what should I call the function?"
    assert excinfo.value.choices == []


def test_tool_loop_ask_user_only_makes_one_call_not_the_full_iteration_budget():
    # Confirms it's treated as an immediate unwind, not fed back into the
    # loop as a "tool result" and retried -- ask_user is not in
    # TOOL_DISPATCH, so falling through to the generic branch would have
    # produced a "tool not available" TOOL RESULT and kept looping instead.
    fake = _FakeModel("ACTION: ask_user\nCODE:\nwhich one?")

    with pytest.raises(pn.NeedsUserInput):
        pn._tool_loop(fake, [HumanMessage("do the thing")])

    assert len(fake.calls) == 1


def test_tool_loop_ask_user_with_no_question_falls_back_to_a_placeholder():
    fake = _FakeModel("ACTION: ask_user\nCODE:\n")

    with pytest.raises(pn.NeedsUserInput) as excinfo:
        pn._tool_loop(fake, [HumanMessage("do the thing")])

    assert excinfo.value.question  # never empty -- something is always shown


# --------------------------------------------------------------------------
# ask_user() -- the dedicated node. interrupt() itself is monkeypatched:
# calling it for real needs a live LangGraph task context (get_config()
# raises outside one), which a bare unit test calling the node function
# directly doesn't have -- exactly like every other node test in this repo
# calls _run_role/router()/evaluator() directly rather than through a
# compiled graph.
# --------------------------------------------------------------------------

def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("solve N queens with brute force")],
        "board": [],
        "round": 3,
        "node": None,
        "feedback": "",
        "output": None,
        "context": "",
        "plan": None,
        "active_step": None,
        "node_error": None,
        "pending_question": "which N?",
        "pending_choices": [],
        "asking_role": "solver",
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_ask_user_calls_interrupt_with_the_pending_question_and_choices(monkeypatch):
    seen = {}

    def fake_interrupt(payload):
        seen.update(payload)
        return "8"

    monkeypatch.setattr(pn, "interrupt", fake_interrupt)

    pn.ask_user(_state(pending_question="which N?", pending_choices=["4", "8"]))

    assert seen == {"question": "which N?", "choices": ["4", "8"]}


def test_ask_user_routes_back_to_whichever_role_asked(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8")

    result = pn.ask_user(_state(asking_role="finder"))

    assert result.goto == "finder"


def test_ask_user_defaults_to_the_loop_if_asking_role_is_somehow_unset(monkeypatch):
    # Defensive fallback -- should never happen in practice, since only
    # ask_user() clears asking_role and only after reading it. The default is
    # the working loop rather than the judge: an answer handed to the evaluator
    # with nothing to judge would end the run.
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8")

    result = pn.ask_user(_state(asking_role=None))

    assert result.goto == "agent"


def test_ask_user_writes_the_qa_into_context_not_messages(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8x8")

    result = pn.ask_user(_state(pending_question="which N?", context=""))

    assert "which N?" in result.update["context"]
    assert "8x8" in result.update["context"]
    assert "messages" not in result.update  # task text must stay untouched


def test_ask_user_appends_onto_existing_context_rather_than_replacing_it(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "recursive")

    result = pn.ask_user(_state(context="earlier: the repo uses pytest", pending_question="recursive or iterative?"))

    assert "earlier: the repo uses pytest" in result.update["context"]
    assert "recursive or iterative?" in result.update["context"]
    assert "recursive" in result.update["context"]


def test_ask_user_hands_the_answer_back_to_the_loop(monkeypatch):
    # The bug this exists for: agent() resumes from `transcript`, which is
    # the one place ask_user does NOT write, so the loop came back from the
    # pause with the prompt that had just produced the question and asked it
    # again. `user_answer` is what carries the reply into that transcript.
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8x8")

    result = pn.ask_user(_state(asking_role="agent"))

    assert result.update["user_answer"] == "8x8"


def test_ask_user_does_not_hand_the_answer_to_the_loop_when_the_evaluator_asked(monkeypatch):
    # The evaluator rebuilds its prompt from state (`context`) on every run,
    # so it needs nothing here -- and setting it would leave an answer to the
    # EVALUATOR's question sitting in state for the loop to splice into its
    # own conversation as a reply to something it never asked.
    monkeypatch.setattr(pn, "interrupt", lambda payload: "yes")

    result = pn.ask_user(_state(asking_role="evaluator"))

    assert "user_answer" not in result.update


def test_ask_user_clears_all_three_pending_fields(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "yes")

    result = pn.ask_user(_state())

    assert result.update["pending_question"] is None
    assert result.update["pending_choices"] is None
    assert result.update["asking_role"] is None


def test_ask_user_leaves_plan_and_active_step_untouched(monkeypatch):
    # A specialist that got stuck mid-plan-step must resume that same
    # step once answered -- ask_user() itself has no reason to touch
    # plan/active_step, only the role it hands back to does.
    monkeypatch.setattr(pn, "interrupt", lambda payload: "yes")
    plan = [{"task": "write it", "route_to": "solver", "output": None}]

    result = pn.ask_user(_state(plan=plan, active_step=0, asking_role="solver"))

    assert "plan" not in result.update
    assert "active_step" not in result.update


# --------------------------------------------------------------------------
# End to end through the REAL compiled graph (pn.app) -- interrupt()/
# Command(resume=...) is genuine LangGraph checkpointed-pause runtime
# machinery, not something a bare node-function call (everything above)
# can exercise on its own. No live LLM needed: every ROUTER.chat_model()
# call is monkeypatched to pop the next reply off a scripted queue, in the
# exact order this particular run is expected to make them -- the same
# "no live LLM" spirit as the rest of the offline suite, just also
# confirming the graph's own wiring (StateGraph edges, the checkpointer,
# app.compile()) actually pauses and resumes correctly, which nothing else
# in this suite touches.
# --------------------------------------------------------------------------

def _initial_state(text: str) -> dict:
    return {
        "messages": [HumanMessage(text)],
        "board": [],
        "round": 0,
        "node": None,
        "feedback": "",
        "output": None,
        "context": "",
        "plan": None,
        "active_step": None,
        "node_error": None,
        "pending_question": None,
        "pending_choices": None,
        "asking_role": None,
        "final_output": None,
    }


def test_a_real_graph_run_pauses_on_ask_user_and_resumes_via_command(monkeypatch):
    import uuid

    # Three replies now, not five. The two router decisions are gone with the
    # router: the loop asks, and after the answer it carries on in the SAME
    # conversation rather than being re-dispatched into a fresh one.
    queue = [
        "- an 8-queens solution is produced",           # the run's checklist
        "ACTION: ask_user\nCODE:\nhow many queens?",   # the loop pauses
        # The answer is part of the request, so the checklist is rewritten
        # around it before the loop carries on -- nodes.py's _requested().
        "- an 8-queens solution is produced",
        "FINAL:\n8-queens solution here",              # it carries on after the answer
        "FINAL:\nAPPROVE: yes\nWHY: looks right",      # evaluator
    ]

    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _FakeModel(queue.pop(0)))

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    initial = _initial_state("solve N queens with brute force")

    updates = list(pn.app.stream(initial, config, stream_mode="updates"))

    assert "__interrupt__" in updates[-1]
    triggered = updates[-1]["__interrupt__"][0]
    assert triggered.value == {"question": "how many queens?", "choices": []}
    # Nothing finished yet -- the checkpointed state still shows the pause.
    assert pn.app.get_state(config).values["final_output"] is None

    list(pn.app.stream(pn.Command(resume="8"), config, stream_mode="updates"))

    final_state = pn.app.get_state(config).values
    assert final_state["final_output"] == "8-queens solution here"
    # The pause itself leaves no trace once resumed.
    assert final_state["pending_question"] is None
    assert final_state["asking_role"] is None
    assert not queue  # every scripted reply was actually consumed, in order


def test_the_resumed_loop_actually_sees_the_answer_it_paused_for(monkeypatch):
    """The regression this whole pair of fields exists for.

    Live-tested failure: otto asked "what can I help you with?", the person
    answered, and otto asked the identical question again -- six times over,
    every answer ignored. The cause was not the pause machinery (which works)
    but what came back from it: agent() rebuilds its conversation from
    `transcript` on resume, the Q&A was written only into `context`, and so
    the second run got a prompt byte-identical to the one that had just
    produced the question. A scripted queue of replies cannot catch that --
    the fake answers differently on call two whatever it was shown -- so this
    asserts on the MESSAGES the model was handed, not on the run's output.
    """
    import uuid

    seen: list[list[str]] = []

    class _Recording(_FakeModel):
        def stream(self, messages):
            seen.append([str(m.content) for m in messages])
            yield from super().stream(messages)

    queue = [
        "- an 8-queens solution is produced",
        "ACTION: ask_user\nCODE:\nhow many queens?",
        "- an 8-queens solution is produced",   # the checklist, rewritten
        "FINAL:\n8-queens solution here",
        "FINAL:\nAPPROVE: yes\nWHY: looks right",
    ]
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _Recording(queue.pop(0)))

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    list(pn.app.stream(_initial_state("solve N queens with brute force"), config,
                       stream_mode="updates"))
    before = len(seen)
    list(pn.app.stream(pn.Command(resume="exactly 8 queens please"), config,
                       stream_mode="updates"))

    resumed = seen[before]
    assert any("exactly 8 queens please" in m for m in resumed), (
        "the resumed loop was never shown the answer it paused for"
    )
    assert any("how many queens" in m for m in resumed), (
        "the resumed loop was never shown the question it had asked"
    )


def test_the_answer_is_not_replayed_into_the_next_run_of_the_loop(monkeypatch):
    # `user_answer` is consumed, not accumulated: once spliced into the
    # transcript it must be cleared, or an evaluator rejection would send the
    # loop back in with the same "THE USER ANSWERED" block appended a second
    # time, below its own later work.
    import uuid

    seen: list[list[str]] = []

    class _Recording(_FakeModel):
        def stream(self, messages):
            seen.append([str(m.content) for m in messages])
            yield from super().stream(messages)

    queue = [
        "- an 8-queens solution is produced",
        "ACTION: ask_user\nCODE:\nhow many queens?",
        "- an 8-queens solution is produced",   # the checklist, rewritten
        "FINAL:\nfirst attempt",
        "FINAL:\nAPPROVE: no\nWHY: no board was printed",
        "FINAL:\nsecond attempt with a board",
        "FINAL:\nAPPROVE: yes\nWHY: looks right",
    ]
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _Recording(queue.pop(0)))

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    list(pn.app.stream(_initial_state("solve N queens with brute force"), config,
                       stream_mode="updates"))
    list(pn.app.stream(pn.Command(resume="exactly 8 queens please"), config,
                       stream_mode="updates"))

    assert not queue  # the run really did go all the way through the rejection
    retry = seen[-2]  # the loop's second post-answer run, after the rejection
    assert sum("exactly 8 queens please" in m for m in retry) == 1


# --------------------------------------------------------------------------
# MAX_USER_QUESTIONS -- the bound on how much of somebody else's attention one
# turn may spend. Nothing capped this: a rejection loop is capped by
# MAX_REJECTIONS and a tool loop by MAX_TOOL_ITERATIONS, but a run could pause,
# be answered, and pause again without limit. Live-tested twice, the second
# time with the work finished and the person having already said "done" and
# then "nothing else".
# --------------------------------------------------------------------------

def test_ask_user_counts_each_answered_pause(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8")

    result = pn.ask_user(_state(asks=2))

    assert result.update["asks"] == 3


def test_ask_user_counts_from_zero_when_nothing_has_asked_yet(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8")

    result = pn.ask_user(_state())

    assert result.update["asks"] == 1


def test_the_loop_still_asks_while_it_has_budget():
    fake = _FakeModel("ACTION: ask_user\nCODE:\nwhich one?")

    with pytest.raises(pn.NeedsUserInput):
        pn._tool_loop(fake, [HumanMessage("do the thing")],
                      asked=pn.MAX_USER_QUESTIONS - 1)


def test_the_loop_is_refused_rather_than_paused_once_the_asks_are_spent():
    # Refused, NOT ended: a wall with no way through it is what makes an
    # agent invent the thing it was denied (ASK_BUDGET_SPENT's own comment),
    # so the ask comes back as a failed tool call and the loop carries on.
    fake = _FakeModel("ACTION: ask_user\nCODE:\nanything else?")

    pn._tool_loop(fake, [HumanMessage("do the thing")],
                  asked=pn.MAX_USER_QUESTIONS)

    assert len(fake.calls) > 1  # it kept working instead of unwinding
    last = fake.calls[-1]
    assert any("asking again is not available" in str(m) for m in last)


def test_the_agent_loop_hands_over_rather_than_spin_on_a_refused_ask(monkeypatch):
    # _agent_loop is the one loop with no iteration ceiling -- the agent node
    # passes max_iterations=None, and what used to end a run of asks was the
    # pause itself unwinding out of it. Refusing the ask without
    # MAX_REFUSED_ASKS turns a model that asks on every reply into a loop that
    # spends the whole budget being told no. A regression here HANGS rather
    # than fails, which is why it is worth its own test.
    fake = _FakeModel("ACTION: ask_user\nCODE:\nanything else?")
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    state = _state(asks=pn.MAX_USER_QUESTIONS, transcript=None, mode=None)

    output, why, _mode = pn._agent_loop(
        state, [HumanMessage("say hi")], mode=pn.DEFAULT_MODE,
        actions=[], mode_log=[],
    )

    assert why == "dead"
    assert len(fake.calls) <= pn.MAX_REFUSED_ASKS + 1


def test_a_turn_cannot_pause_more_than_the_cap_however_many_times_it_tries(monkeypatch):
    """The runaway itself, through the real graph.

    The agent asks on every single reply. Without the cap this never returns
    an answer -- it pauses, is answered, and pauses again for as long as
    somebody keeps typing. With it, the turn pauses MAX_USER_QUESTIONS times
    and then finishes.
    """
    import uuid

    queue = ["- the request is answered"]
    monkeypatch.setattr(
        pn.ROUTER, "chat_model",
        lambda *a, **kw: _FakeModel(
            queue.pop(0) if queue else "ACTION: ask_user\nCODE:\nanything else?"
        ),
    )

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    initial = _initial_state("say hi")
    initial["asks"] = 0

    pauses = 0
    updates = list(pn.app.stream(initial, config, stream_mode="updates"))
    while "__interrupt__" in updates[-1]:
        pauses += 1
        assert pauses <= pn.MAX_USER_QUESTIONS, "the turn paused past its cap"
        # Deliberately NOT a closing answer -- this test is about the cap
        # holding when the person keeps engaging, not about _CLOSING_ANSWERS
        # (which has its own tests, and would end this after one pause).
        updates = list(pn.app.stream(pn.Command(resume="keep going"), config,
                                     stream_mode="updates"))

    assert pauses == pn.MAX_USER_QUESTIONS
    assert pn.app.get_state(config).values["asks"] == pn.MAX_USER_QUESTIONS


# --------------------------------------------------------------------------
# _CLOSING_ANSWERS -- "done", "nothing else". The live failure was that both
# of those were said, one after the other, and neither changed what happened
# next.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "done", "nothing else", "that's all", "Nothing else.", "  ALL DONE  ",
    "everything's done", "no thanks", "stop", "nothing more",
])
def test_these_answers_mean_stop_asking(answer):
    assert pn._is_closing_answer(answer)


@pytest.mark.parametrize("answer", [
    # Bare yes/no are deliberately absent: `CHOICES: yes | no` is the
    # commonest question shape there is.
    "no", "yes",
    # Anything carrying other content is an answer, not a request to stop.
    "no, use the second file", "done with the first one, now do the second",
    "nothing else matters for the schema", "8", "",
])
def test_these_answers_do_not_mean_stop_asking(answer):
    assert not pn._is_closing_answer(answer)


def test_a_closing_answer_spends_the_rest_of_the_turns_questions(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "nothing else")

    result = pn.ask_user(_state(asks=0))

    assert result.update["asks"] == pn.MAX_USER_QUESTIONS


def test_an_ordinary_answer_only_spends_the_one_question(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "use rust")

    result = pn.ask_user(_state(asks=0))

    assert result.update["asks"] == 1


def test_a_closing_answer_never_gives_asks_back(monkeypatch):
    # max(), not assignment -- a turn already over the cap must not be handed
    # a question back by saying "done".
    monkeypatch.setattr(pn, "interrupt", lambda payload: "done")

    result = pn.ask_user(_state(asks=pn.MAX_USER_QUESTIONS + 2))

    assert result.update["asks"] == pn.MAX_USER_QUESTIONS + 3


def test_saying_done_ends_the_turn_instead_of_being_asked_again(monkeypatch):
    """The reported failure, end to end.

    The agent asks on every reply. The person answers "done" to the first
    question, and that has to be the last question -- previously it was asked
    again, and again.
    """
    import uuid

    queue = ["- the request is answered"]
    monkeypatch.setattr(
        pn.ROUTER, "chat_model",
        lambda *a, **kw: _FakeModel(
            queue.pop(0) if queue else "ACTION: ask_user\nCODE:\nanything else?"
        ),
    )

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    initial = _initial_state("say hi")
    initial["asks"] = 0

    pauses = 0
    updates = list(pn.app.stream(initial, config, stream_mode="updates"))
    while "__interrupt__" in updates[-1]:
        pauses += 1
        assert pauses == 1, "it asked again after being told the work was done"
        updates = list(pn.app.stream(pn.Command(resume="done"), config,
                                     stream_mode="updates"))

    assert pauses == 1


def test_the_loop_is_told_the_answer_was_a_closing_one(monkeypatch):
    # The asking half is enforced; the FINISHING half is advice the loop sees
    # in its own conversation (_CLOSING_ANSWERS on why only one is a gate).
    import uuid

    seen: list[list[str]] = []

    class _Recording(_FakeModel):
        def stream(self, messages):
            seen.append([str(m.content) for m in messages])
            yield from super().stream(messages)

    queue = [
        "- the request is answered",
        "ACTION: ask_user\nCODE:\nanything else?",
        "FINAL:\nall finished",
        "FINAL:\nAPPROVE: yes\nWHY: fine",
    ]
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _Recording(queue.pop(0)))

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    list(pn.app.stream(_initial_state("say hi"), config, stream_mode="updates"))
    before = len(seen)
    list(pn.app.stream(pn.Command(resume="nothing else"), config, stream_mode="updates"))

    assert any("stop asking" in m for m in seen[before])


# --------------------------------------------------------------------------
# _requested / the checklist following the answer. The reported failure: the
# turn opened "hi", otto asked what was wanted, the person answered "analyse
# my repository and plan improvements to the TUI", otto did exactly that over
# 32 model calls -- and every bit of it was rejected, correctly by the
# checklist's own lights, because the checklist said the task was to answer a
# greeting. The run ended by saying hello, and the files it had written stayed
# on disk.
# --------------------------------------------------------------------------

def test_the_request_is_just_the_message_when_nothing_was_asked():
    assert pn._requested(_state(asked_qa=[])) == "solve N queens with brute force"


def test_what_the_person_said_when_asked_becomes_part_of_the_request():
    state = _state(
        messages=[HumanMessage("hi")],
        asked_qa=['you asked: "what can I help with?"\n'
                  'the user answered: "plan improvements to the TUI"'],
    )

    requested = pn._requested(state)

    assert "hi" in requested
    assert "plan improvements to the TUI" in requested


def test_the_request_keeps_every_answer_in_order():
    state = _state(messages=[HumanMessage("hi")], asked_qa=["first pair", "second pair"])

    requested = pn._requested(state)

    assert requested.index("first pair") < requested.index("second pair")


def test_ask_user_records_the_pair_for_the_request(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "plan the TUI work")

    result = pn.ask_user(_state(pending_question="what can I help with?"))

    assert len(result.update["asked_qa"]) == 1
    recorded = result.update["asked_qa"][0]
    assert "what can I help with?" in recorded
    assert "plan the TUI work" in recorded


def test_the_checklist_is_rewritten_around_what_the_person_answered(monkeypatch):
    """The fix for the reported failure, at the node.

    A greeting's checklist must not be what real work is judged against. The
    criteria call is asked for the WHOLE request, and the resulting checklist
    replaces the one written from the opening word.
    """
    seen: list[str] = []

    def fake_rubric(llm, task_text):
        seen.append(task_text)
        return pn.Rubric(["a TUI improvement plan exists"])

    monkeypatch.setattr(pn, "_rubric", fake_rubric)
    # agent() resolves the judge's model before handing it to _rubric, and
    # resolving it for real reaches the provider.
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _FakeModel(""))
    monkeypatch.setattr(pn, "_agent_loop",
                        lambda *a, **kw: ("a plan", "final", pn.DEFAULT_MODE))

    state = _state(
        messages=[HumanMessage("hi")],
        transcript=[{"type": "human", "content": "hi"}],
        checklist=[{"text": "a greeting is returned", "status": "pending", "evidence": ""}],
        user_answer="plan improvements to the TUI",
        asked_qa=['you asked: "what can I help with?"\n'
                  'the user answered: "plan improvements to the TUI"'],
        mode=None,
    )
    result = pn.agent(state)

    assert seen, "the criteria were never regenerated"
    assert "plan improvements to the TUI" in seen[0], (
        "the checklist was still derived from the opening word alone"
    )
    assert [item["text"] for item in result.update["checklist"]] == [
        "a TUI improvement plan exists"
    ]


def test_the_checklist_is_left_alone_when_nobody_has_answered_anything(monkeypatch):
    # The invariant this must not break: a checklist is written ONCE from the
    # request, not rewritten while an attempt is being made at it.
    monkeypatch.setattr(pn, "_rubric",
                        lambda llm, task: pytest.fail("regenerated for no reason"))
    monkeypatch.setattr(pn, "_agent_loop",
                        lambda *a, **kw: ("an answer", "final", pn.DEFAULT_MODE))
    existing = [{"text": "an 8-queens solution", "status": "pending", "evidence": ""}]

    result = pn.agent(_state(
        transcript=[{"type": "human", "content": "go"}],
        checklist=existing, user_answer=None, asked_qa=[], mode=None,
    ))

    assert result.update["checklist"] == existing


def test_a_closing_answer_does_not_rewrite_the_checklist(monkeypatch):
    # "done" is the person winding the turn up, not adding to the request --
    # criteria derived from it would be a checklist about saying goodbye.
    monkeypatch.setattr(pn, "_rubric",
                        lambda llm, task: pytest.fail("regenerated from a goodbye"))
    monkeypatch.setattr(pn, "_agent_loop",
                        lambda *a, **kw: ("an answer", "final", pn.DEFAULT_MODE))
    existing = [{"text": "an 8-queens solution", "status": "pending", "evidence": ""}]

    result = pn.agent(_state(
        transcript=[{"type": "human", "content": "go"}],
        checklist=existing, user_answer="nothing else", mode=None,
        asked_qa=['you asked: "anything else?"\nthe user answered: "nothing else"'],
    ))

    assert result.update["checklist"] == existing


def test_the_evaluator_judges_against_what_the_person_answered(monkeypatch):
    # The other half: the judge's own prompt has to carry the answer as part
    # of the request, or it rejects real work for not being a greeting.
    replies = ["FINAL:\nAPPROVE: yes\nWHY: fine"]
    seen: list[str] = []

    class _Capturing(_FakeModel):
        def stream(self, messages):
            seen.append("\n".join(str(m.content) for m in messages))
            yield from super().stream(messages)

    monkeypatch.setattr(pn.ROUTER, "chat_model",
                        lambda *a, **kw: _Capturing(replies[0]))

    pn.evaluator(_state(
        messages=[HumanMessage("hi")],
        output="a 162-line TUI improvement plan",
        checklist=[{"text": "a TUI plan exists", "status": "pending", "evidence": ""}],
        asked_qa=['you asked: "what can I help with?"\n'
                  'the user answered: "plan improvements to the TUI"'],
    ))

    assert any("plan improvements to the TUI" in text for text in seen), (
        "the judge never saw what the person actually asked for"
    )

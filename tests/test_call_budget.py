"""How many model calls one run costs, asserted exactly.

This is the number the loop rewrite exists to move, so it gets a test of its own
rather than being inferred from a benchmark afterwards.

The old graph was router -> role -> router -> evaluator -> router. One round of
real work -- the overseer decides, a specialist makes a tool call and answers,
the overseer decides again, the evaluator judges -- cost five model calls, three
of them overhead. Measured on Claw-Eval traces, the boundaries between those
nodes were 45 to 69% of a run's wall time, against 0.1 to 0.4 seconds of actual
tool execution per task.

These tests drive the REAL compiled graph with a scripted model and count the
requests, so a future change that quietly reintroduces a round-trip fails here
with a number rather than showing up as a slow benchmark weeks later.
"""
import tempfile
import uuid
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.memory.lessons import bind_bank
from agent.pipeline import nodes as pn
from agent.pipeline.budget import Budget, bind_budget
from agent.pipeline.run import _initial


class _Counting:
    """One scripted reply per request, and a count of the requests."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        reply = self._replies.pop(0) if self._replies else "FINAL:\nAPPROVE: yes\nWHY: ok"
        yield AIMessageChunk(content=reply)


def _run(monkeypatch, replies, budget=None, learning=False):
    model = _Counting(replies)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: model)
    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    # The lesson bank is OFF unless a test is about it. Counting calls is the
    # job here, and a learning step that writes into the real bank from a unit
    # test would be both a side effect and an extra call in every number below.
    with bind_budget(budget), bind_bank(None if not learning else _Bank()):
        final = pn.app.invoke(_initial("do the thing"), config)
    return model, final


def _Bank():
    """A real lesson bank in a throwaway file. Faking the store here would
    only test the fake -- and the learning step's whole job is to write."""
    from agent.memory.store import MemoryStore

    return MemoryStore(Path(tempfile.mkdtemp()) / "lessons.db")


def test_one_tool_call_and_an_answer_costs_four_model_calls(monkeypatch):
    """Was five under the old graph: two router decisions plus a judgment, for
    two calls of real work. The router is gone. The judgment now costs two --
    one to write the rubric from the task before the answer is visible, one to
    score against it -- because a judge that only re-reads the actor's own
    output measures at approximately zero (RefineBench: -2.5% to 0% over five
    turns, against 90-98% with an external checklist).

    So: same total as the old graph, spent on checking instead of routing."""
    model, final = _run(monkeypatch, [
        "- the sum is correct",                            # 1, the rubric
        "ACTION: execute_python\nCODE:\nprint(2 + 2)",   # 2, work
        "FINAL:\nthe answer is 4",                        # 3, work
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: checked it",
    ])

    assert model.calls == 4
    assert final["final_output"] == "the answer is 4"


def test_overhead_stays_flat_as_the_work_grows(monkeypatch):
    """The property that matters. Ten tool calls used to mean roughly three
    node boundaries and their router decisions; now the judgment is still one
    call, however long the work runs."""
    model, final = _run(monkeypatch, [
        "- the work is done",                              # the rubric
        *["ACTION: execute_python\nCODE:\nprint(1)"] * 10,
        "FINAL:\ndone",
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
    ])

    assert model.calls == 13
    assert final["final_output"] == "done"


def test_a_rejection_costs_one_round_trip_not_a_router_decision(monkeypatch):
    """A rejection used to return to the overseer, which spent a call deciding
    who should retry. The evaluator hands straight back to the loop."""
    model, final = _run(monkeypatch, [
        "- it is verified",                                  # 1, the checklist
        "FINAL:\nfirst attempt",                            # 2, work
        "FINAL:\nMET: 0/1\nBLOCKED: no\nAPPROVE: no\nWHY: not verified",
        "FINAL:\nsecond attempt",                           # 4, work again
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: now it checks out",
    ])

    # Five, not six. The criteria are written ONCE for the run rather than once
    # per judgment, so the re-judgment after a rejection costs nothing to set
    # up -- the judge and the actor are working from the same list.
    assert model.calls == 5
    assert final["final_output"] == "second attempt"


def test_a_mode_swap_costs_one_call_rather_than_a_node_boundary(monkeypatch):
    """Changing role used to mean a boundary: a router decision, a rebuilt
    prompt, and the tool conversation thrown away. It is one appended message
    now, and the swap request itself is the only call it costs."""
    model, final = _run(monkeypatch, [
        "- the work is done",                               # 1, rubric
        "ACTION: execute_python\nCODE:\nprint(1)",        # 2
        "ACTION: switch_mode\nCODE:\nplan",                # 3, the swap
        "FINAL:\nplanned and done",                        # 4
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
    ])

    assert model.calls == 5
    assert final["mode"] == "plan"


def test_the_budget_stops_a_run_that_will_not_finish(monkeypatch):
    """Before this there was no ceiling at all except LangGraph's super-step
    limit, which counts node transitions rather than money."""
    model, final = _run(
        monkeypatch,
        ["ACTION: execute_python\nCODE:\nprint(1)"] * 50,
        budget=Budget(max_model_calls=6),
    )

    assert model.calls <= 8, "the budget did not stop the loop"
    assert final["model_calls"] >= 6


def test_a_stopped_run_still_produces_an_answer(monkeypatch):
    """A run killed mid-command reports nothing; one that stops and hands over
    its best candidate is still gradable. That difference is why exhaustion
    returns rather than raising."""
    _, final = _run(
        monkeypatch,
        ["ACTION: execute_python\nCODE:\nprint(1)"] * 3
        + ["FINAL:\npartial but real"]
        + ["FINAL:\nAPPROVE: yes\nWHY: ok"],
        budget=Budget(max_model_calls=20),
    )

    assert final["final_output"] == "partial but real"


def test_model_calls_counts_the_judgment_too(monkeypatch):
    """It reported only what the loop spent until this. Measured live at 4
    against an actual 9, because the evaluator checks the answer with tools and
    those are model calls like any other."""
    model, final = _run(
        monkeypatch,
        [
            "- the work is done",                             # 1, the rubric
            "ACTION: execute_python\nCODE:\nprint(1)",      # 2
            "FINAL:\ndone",                                  # 3
            "FINAL:\nAPPROVE: yes\nWHY: verified",           # 4, the judge
        ],
        budget=Budget(max_model_calls=40),
    )

    assert model.calls == 4
    assert final["model_calls"] == 4


def test_the_judgment_does_not_go_on_a_checking_expedition(monkeypatch):
    """Measured regression, caught by a live probe rather than review.

    Giving the evaluator the evidence it had been missing made it MORE active,
    not less: it spent its whole five-iteration tool budget on every judgment.
    On "write fib.py and run it" one run cost 24 model calls and 106 seconds,
    fifteen of them the evaluator across three judgments, against six for the
    loop doing the actual work. Capping it took the same task to 7 calls and 31
    seconds -- and it approved first time, because judging from the evidence
    beats going looking for something to complain about.
    """
    judge = _Counting(["ACTION: execute_python\nCODE:\nprint(1)"] * 10)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: judge)

    pn.evaluator({
        "messages": [HumanMessage("do it")], "node": "agent", "output": "the answer",
        "board": [], "actions": [], "mode_log": [], "transcript": [],
        "context": "", "rejections": 0,
    })

    # One rubric call plus the capped judging exchanges.
    assert judge.calls <= 1 + pn.MAX_EVALUATOR_ITERATIONS


def test_learning_costs_exactly_one_call_at_the_end_of_a_run(monkeypatch):
    """Self-evolution is not free, and the point of this number is that it
    stays one. A distilling step that grew into its own tool loop would be the
    same mistake the evaluator made -- 24 calls on a task the loop did in six.

    Compare against the four in the first test above: same work, plus one.
    """
    model, _ = _run(
        monkeypatch,
        [
            "- the sum is correct",
            "ACTION: execute_python\nCODE:\nprint(2 + 2)",
            "FINAL:\nthe answer is 4",
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: checked it",
            '[{"cue": "arithmetic is asked for", "action": "run it", "outcome": "worked"}]',
        ],
        learning=True,
    )

    assert model.calls == 5


def test_a_run_with_nowhere_to_learn_does_not_pay_for_learning(monkeypatch):
    """Checked before the call, not after. Distilling lessons and then
    discarding them is the worst of both."""
    model, _ = _run(monkeypatch, [
        "- the sum is correct",
        "ACTION: execute_python\nCODE:\nprint(2 + 2)",
        "FINAL:\nthe answer is 4",
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: checked it",
    ])

    assert model.calls == 4


def test_the_learning_call_lands_in_the_reported_cost(monkeypatch):
    """It did not, at first: `model_calls` was read before the distilling
    step, so the one call self-evolution costs was the one call the cost axis
    could not see. That is precisely the blindness the measurement discipline
    exists to remove."""
    _, final = _run(
        monkeypatch,
        [
            "- the sum is correct",
            "ACTION: execute_python\nCODE:\nprint(2 + 2)",
            "FINAL:\nthe answer is 4",
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: checked it",
            '[{"cue": "arithmetic is asked for", "action": "run it"}]',
        ],
        budget=Budget(max_model_calls=40),
        learning=True,
    )

    assert final["model_calls"] == 5


# --------------------------------------------------------------------------
# A turn with no task in it
# --------------------------------------------------------------------------
#
# Reported from real use: "hi" and "hello" taking minutes and 11-20 model
# calls. Reproduced on "thanks, that's helpful" -- 22 calls and 263 seconds,
# answering `1|Problem to solve|No problem to solve`. The criteria call found
# nothing to check, the judge measured a chatty reply against criteria that did
# not exist, rejected it, and the loop restarted. Twice.
#
# Measured after, three clean trials on the same prompt: 3 calls, 26 seconds,
# every time.

def test_a_turn_with_nothing_to_check_skips_the_judge_and_the_lesson(monkeypatch):
    """An empty checklist is the criteria call saying there was no task. The
    two calls are the rubric and the answer."""
    model, final = _run(monkeypatch, [
        "NONE",                       # 1, the rubric: nothing to check
        "FINAL:\nI'm well, thanks!",  # 2, the answer
    ], learning=True)

    assert model.calls == 2
    assert final["final_output"] == "I'm well, thanks!"


def test_a_real_task_still_pays_for_verification(monkeypatch):
    """The fix must not buy speed by skipping verification on real work. This
    is the control: same path, one criterion, and the judge runs."""
    model, final = _run(monkeypatch, [
        "- the sum is correct",
        "FINAL:\n4",
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: checked",
    ])

    assert model.calls == 3
    assert final["final_output"] == "4"


def test_a_run_that_died_is_still_judged(monkeypatch):
    """"No criteria" and "no answer" are different problems, and only one of
    them is a conversation. A run that could not produce a usable reply goes
    to the evaluator even with an empty checklist."""
    model, _ = _run(monkeypatch, [
        "NONE",
        "not a parseable reply at all",
        "still not parseable",
        "nor this",
    ])

    assert model.calls > 2, "a dead run skipped the judge"


def test_a_turn_with_no_task_never_sees_the_engineer_prompt():
    """It used to. A 227-character note argued, from inside a HumanMessage,
    against 3,500 characters of system prompt telling the model to go find out
    what state the system is in and offering it seventeen tools -- so roughly
    3,200 characters of the turn went on setting up an argument with
    themselves. The turn gets a prompt its own size now."""
    seeded = pn._seed_transcript({"messages": [HumanMessage("hi")]}, "hi", [])
    body = "\n".join(pn._content_text(m.content) for m in seeded)

    assert pn.CONVERSATION_PROMPT in body
    assert "You are an engineer with a shell" not in body
    assert "execute_bash" not in body, "a greeting was offered a tool menu"
    assert sum(len(pn._content_text(m.content)) for m in seeded) < 900


def test_a_turn_with_a_task_still_gets_the_whole_prompt():
    """The control. Criteria exist, so this is work, and it pays for the
    habits, the ladder and the protocol."""
    seeded = pn._seed_transcript(
        {"messages": [HumanMessage("fix it")]}, "fix it",
        [{"text": "it works", "status": "pending", "evidence": ""}],
    )
    body = "\n".join(pn._content_text(m.content) for m in seeded)

    assert "You are an engineer with a shell" in body
    assert pn.CONVERSATION_PROMPT not in body


def test_the_conversation_path_can_still_answer():
    """One-way on purpose: a task misclassified as chatter costs a short
    answer, not a refusal to work."""
    assert "FINAL:" in pn.CONVERSATION_PROMPT


def test_a_failed_rubric_call_does_not_demote_a_task_to_chatter(monkeypatch):
    """An empty checklist means two different things and only one of them is
    evidence. A rubric that RAN and found nothing to check says the turn holds
    no task; a rubric CALL that died says nothing at all. Collapsing them let
    one exhausted API key seed a real task with the conversation prompt --
    no habits, no ladder, no tools -- which is how it was found."""
    def dead(llm, messages, **kw):
        raise pn.ProviderError("credit balance is too low")

    monkeypatch.setattr(pn, "_call", dead)
    assert pn._criteria(object(), "fix the failing test") is None

    seeded = pn._seed_transcript(
        {"messages": [HumanMessage("fix it")]}, "fix it", None,
    )
    body = "\n".join(pn._content_text(m.content) for m in seeded)
    assert "You are an engineer with a shell" in body
    assert pn.CONVERSATION_PROMPT not in body


def test_a_rubric_that_ran_and_found_nothing_still_means_no_task():
    """The other half of the same distinction."""
    assert pn._parse_rubric("this is just a greeting") == []


def test_a_failed_rubric_call_does_not_skip_the_judge():
    """The second site with the same conflation, and the more expensive one:
    `not checklist` read None as "nothing to verify" and returned straight to
    END. A dead rubric call skipped verification entirely, on every run, and
    each one still reported itself finished."""
    import inspect

    source = inspect.getsource(pn.agent)
    assert 'if checklist == [] and why == "final"' in source
    assert 'if not checklist and why' not in source

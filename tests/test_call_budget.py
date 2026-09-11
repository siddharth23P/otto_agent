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
import uuid

from langchain_core.messages import AIMessageChunk, HumanMessage

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


def _run(monkeypatch, replies, budget=None):
    model = _Counting(replies)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: model)
    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    with bind_budget(budget):
        final = pn.app.invoke(_initial("do the thing"), config)
    return model, final


def test_one_tool_call_and_an_answer_costs_four_model_calls(monkeypatch):
    """Was five under the old graph: two router decisions plus a judgment, for
    two calls of real work. The router is gone. The judgment now costs two --
    one to write the rubric from the task before the answer is visible, one to
    score against it -- because a judge that only re-reads the actor's own
    output measures at approximately zero (RefineBench: -2.5% to 0% over five
    turns, against 90-98% with an external checklist).

    So: same total as the old graph, spent on checking instead of routing."""
    model, final = _run(monkeypatch, [
        "ACTION: execute_python\nCODE:\nprint(2 + 2)",   # 1, work
        "FINAL:\nthe answer is 4",                        # 2, work
        "- the sum is correct",                            # 3, the rubric
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: checked it",
    ])

    assert model.calls == 4
    assert final["final_output"] == "the answer is 4"


def test_overhead_stays_flat_as_the_work_grows(monkeypatch):
    """The property that matters. Ten tool calls used to mean roughly three
    node boundaries and their router decisions; now the judgment is still one
    call, however long the work runs."""
    model, final = _run(monkeypatch, [
        *["ACTION: execute_python\nCODE:\nprint(1)"] * 10,
        "FINAL:\ndone",
        "- the work is done",
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
        "ACTION: execute_python\nCODE:\nprint(1)",        # 1
        "ACTION: switch_mode\nCODE:\nplan",                # 2, the swap
        "FINAL:\nplanned and done",                        # 3
        "- the work is done",                               # 4, rubric
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
            "ACTION: execute_python\nCODE:\nprint(1)",      # 1
            "FINAL:\ndone",                                  # 2
            "ACTION: execute_python\nCODE:\nprint(1)",      # 3, the judge checks
            "FINAL:\nAPPROVE: yes\nWHY: verified",           # 4
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

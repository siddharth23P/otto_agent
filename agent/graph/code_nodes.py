from __future__ import annotations
from collections import Counter
from typing import Any, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command, Send
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph.code_state import CodeTask, Proposal, ProposalVote, Submission, Review
from agent.router.mapping import Task
from agent.router.router import Router


ROUTER = Router()
MAX_ROUNDS = 3

DECOMP_PROMPT = (
    "Split the task below into exactly {agents} numbered parts. Each part must be "
    "self-contained enough for someone who cannot see the other parts to implement "
    "it alone. Number them 1 through {agents}, one paragraph each, no other text."
)
DECOMP_REVIEW_PROMPT = (
    "A task was split into parts, shown below. Judge whether the split is "
    "complete, non-overlapping, and buildable in isolation. Reply with exactly "
    "two lines:\nAPPROVE: yes or no\nWHY: one sentence."
)
AUTHOR_PROMPT = (
    "Write the code for the part described below. Return only the code, no "
    "explanation, no markdown fences."
)
REVIEW_PROMPT = (
    "Judge whether the code satisfies its spec, shown below. Reply with exactly "
    "two lines:\nAPPROVE: yes or no\nWHY: one sentence."
)
STITCH_PROMPT = (
    "Below is the original task, then independently authored parts of one "
    "program, each already reviewed and accepted, with their part specs. "
    "Reconcile naming, imports and interfaces into one coherent program. Do "
    "not rewrite the logic. Return only the final code."
)


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
    reply = None
    for chunk in llm.stream(messages):
        reply = chunk if reply is None else reply + chunk
    return _content_text(reply.content) if reply is not None else ""


def _parse_approval(text: str) -> tuple[bool, str]:
    after = text.split("APPROVE:")[-1]
    line, _, rest = after.partition("\n")
    approve = "yes" in line.strip().lower()
    reason = rest.split("WHY:")[-1].strip() if "WHY:" in rest else rest.strip()
    return approve, reason


def _parse_parts(text: str, agents: int) -> list[str] | None:
    specs: list[str] = []
    for i in range(1, agents + 1):
        marker = f"{i}."
        if marker not in text:
            return None
        after = text.split(marker, 1)[1]
        next_marker = f"{i + 1}."
        chunk = after.split(next_marker, 1)[0] if next_marker in after else after
        chunk = chunk.strip()
        if not chunk:
            return None
        specs.append(chunk)
    return specs if len(specs) == agents else None


def _resolve_specs(state: CodeTask, round_no: int) -> list[str] | None:
    proposal = next(p for p in state["proposals"] if p["round"] == round_no)
    return _parse_parts(proposal["text"], state["agents"])


def _best_submission(state: CodeTask, part: int) -> str:
    subs = {s["round"]: s["code"] for s in state["submissions"] if s["part"] == part}
    counts: dict[int, int] = {}
    for rv in state["reviews"]:
        if rv["part"] == part and rv["approve"]:
            counts[rv["round"]] = counts.get(rv["round"], 0) + 1
    best_round = max(subs, key=lambda rnd: counts.get(rnd, 0))
    return subs[best_round]


def propose(state: CodeTask) -> Command[Literal["propose_review", "decomp_consensus"]]:
    agents = state["agents"]
    r = state["decomp_round"] + 1
    author = (r - 1) % agents
    task_text = state["messages"][-1].content

    llm = ROUTER.chat_model(Task.PLAN, temperature=0.4)
    messages = [
        SystemMessage(DECOMP_PROMPT.format(agents=agents)),
        HumanMessage(task_text),
    ]
    text = _call(llm, messages)

    proposal = Proposal(round=r, author=author, text=text)
    auto_yes = ProposalVote(round=r, voter=author, approve=True, reason="proposed it")
    sends = [
        Send("propose_review", {"round": r, "text": text, "voter": seed})
        for seed in range(agents)
        if seed != author
    ]
    # agents == 1: no other seed exists to review the split. propose_review's
    # add_edge to decomp_consensus never fires without a Send reaching it, so
    # route there directly instead of stalling on an empty fan-out.
    goto = sends if sends else "decomp_consensus"
    return Command(
        update={
            "decomp_round": r,
            "proposals": [proposal],
            "proposal_votes": [auto_yes],
            "board": [f"decomp round {r}: seed {author} proposed a split"],
        },
        goto=goto,
    )


def propose_review(ballot: dict) -> dict:
    llm = ROUTER.chat_model(Task.CHAT_FAST, temperature=0.3)
    messages = [SystemMessage(DECOMP_REVIEW_PROMPT), HumanMessage(ballot["text"])]
    text = _call(llm, messages)
    approve, reason = _parse_approval(text)
    return {
        "proposal_votes": [
            ProposalVote(round=ballot["round"], voter=ballot["voter"], approve=approve, reason=reason)
        ]
    }


def decomp_consensus(state: CodeTask) -> Command[Literal["propose", "spawn_parts"]]:
    agents = state["agents"]
    r = state["decomp_round"]
    this_round = [v for v in state["proposal_votes"] if v["round"] == r]
    assert len(this_round) == agents
    approvals = sum(v["approve"] for v in this_round)

    specs = _resolve_specs(state, r) if approvals * 2 > agents else None
    if specs is not None:
        return Command(
            goto="spawn_parts",
            update={
                "part_specs": specs,
                "board": [f"decomp round {r}: accepted {approvals}/{agents}"],
            },
        )

    if r >= MAX_ROUNDS:
        tally = Counter()
        for v in state["proposal_votes"]:
            if v["approve"]:
                tally[v["round"]] += 1
        for round_no, _ in tally.most_common():
            fallback = _resolve_specs(state, round_no)
            if fallback is not None:
                return Command(
                    goto="spawn_parts",
                    update={
                        "part_specs": fallback,
                        "board": [f"decomp exhausted after {MAX_ROUNDS} rounds, using round {round_no}"],
                    },
                )
        # nothing ever parsed into a clean split: fall back to the first
        # proposal's raw text as a single part, repeated, rather than crash.
        first = state["proposals"][0]["text"]
        return Command(
            goto="spawn_parts",
            update={
                "part_specs": [first] * agents,
                "board": ["decomp exhausted, no clean split parsed"],
            },
        )

    return Command(
        goto="propose",
        update={"board": [f"decomp round {r}: split {approvals}/{agents}, retrying"]},
    )


def spawn_parts(state: CodeTask) -> Command[Literal["author"]]:
    agents = state["agents"]
    specs = state["part_specs"]
    part_round = {i: 1 for i in range(agents)}
    part_status: dict[int, str] = {i: "pending" for i in range(agents)}
    sends = [
        Send("author", {"part": i, "spec": specs[i], "round": 1, "feedback": "", "agents": agents})
        for i in range(agents)
    ]
    return Command(
        update={
            "part_round": part_round,
            "part_status": part_status,
            "board": [f"spawning {agents} parts"],
        },
        goto=sends,
    )


def author(ballot: dict) -> Command[Literal["review", "part_consensus"]]:
    part = ballot["part"]
    agents = ballot["agents"]
    spec = ballot["spec"]
    feedback = ballot["feedback"]
    round_ = ballot["round"]

    llm = ROUTER.chat_model(Task.REASON, temperature=0.5)
    prompt_text = spec if not feedback else f"{spec}\n\nThe last draft was rejected because: {feedback}"
    messages = [SystemMessage(AUTHOR_PROMPT), HumanMessage(prompt_text)]
    code = _call(llm, messages)

    submission = Submission(part=part, round=round_, code=code)
    auto_yes = Review(part=part, round=round_, voter=part, approve=True, reason="authored it")
    sends = [
        Send("review", {"part": part, "code": code, "spec": spec, "round": round_, "voter": seed})
        for seed in range(agents)
        if seed != part
    ]
    # agents == 1: zero reviewers exist. review's add_edge to part_consensus
    # never fires without a Send reaching it, so route there directly.
    goto = sends if sends else "part_consensus"
    return Command(
        update={"submissions": [submission], "reviews": [auto_yes]},
        goto=goto,
    )


def review(ballot: dict) -> dict:
    llm = ROUTER.chat_model(Task.CHAT_FAST, temperature=0.3)
    body = f"SPEC:\n{ballot['spec']}\n\nCODE:\n{ballot['code']}"
    messages = [SystemMessage(REVIEW_PROMPT), HumanMessage(body)]
    text = _call(llm, messages)
    approve, reason = _parse_approval(text)
    return {
        "reviews": [
            Review(part=ballot["part"], round=ballot["round"], voter=ballot["voter"], approve=approve, reason=reason)
        ]
    }


def part_consensus(state: CodeTask) -> Command[Literal["author", "stitch"]]:
    agents = state["agents"]
    pr = dict(state["part_round"])
    ps = dict(state["part_status"])
    sends: list[Send] = []
    board: list[str] = []

    for part in range(agents):
        if ps[part] != "pending":
            continue
        round_ = pr[part]  # read before any mutation below -- 11b.8's Trap
        this_round = [rv for rv in state["reviews"] if rv["part"] == part and rv["round"] == round_]
        assert len(this_round) == agents
        approvals = sum(rv["approve"] for rv in this_round)

        if approvals * 2 > agents:
            ps[part] = "accepted"
            board.append(f"part {part} accepted round {round_} ({approvals}/{agents})")
        elif round_ >= MAX_ROUNDS:
            ps[part] = "exhausted"
            board.append(f"part {part} exhausted after {MAX_ROUNDS} rounds")
        else:
            reasons = "; ".join(rv["reason"] for rv in this_round if not rv["approve"])
            next_round = round_ + 1
            pr[part] = next_round
            sends.append(Send("author", {
                "part": part,
                "spec": state["part_specs"][part],
                "round": next_round,
                "feedback": reasons,
                "agents": agents,
            }))

    update = {"part_round": pr, "part_status": ps, "board": board}
    return Command(update=update, goto=sends if sends else "stitch")


def stitch(state: CodeTask) -> dict:
    agents = state["agents"]
    stitcher = state["decomp_round"] % agents
    pieces = [_best_submission(state, part) for part in range(agents)]
    body = "\n\n".join(
        f"--- part {i} ---\nspec: {state['part_specs'][i]}\ncode:\n{pieces[i]}"
        for i in range(agents)
    )
    task_text = state["messages"][-1].content

    llm = ROUTER.chat_model(Task.REASON, temperature=0.3)
    messages = [
        SystemMessage(STITCH_PROMPT),
        HumanMessage(f"ORIGINAL TASK:\n{task_text}\n\n{body}"),
    ]
    final_code = _call(llm, messages)
    return {"final_code": final_code, "board": [f"stitched by seed {stitcher}"]}


g = StateGraph(CodeTask)
for _name, _fn in (
    ("propose", propose),
    ("propose_review", propose_review),
    ("decomp_consensus", decomp_consensus),
    ("spawn_parts", spawn_parts),
    ("author", author),
    ("review", review),
    ("part_consensus", part_consensus),
    ("stitch", stitch),
):
    g.add_node(_name, _fn)
g.add_edge(START, "propose")
g.add_edge("propose_review", "decomp_consensus")
g.add_edge("review", "part_consensus")

app = g.compile(checkpointer=InMemorySaver())

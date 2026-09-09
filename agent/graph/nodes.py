from __future__ import annotations
import re, string, emoji
from typing import Literal
from collections import Counter

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command, Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph.state import Ballot, Vote, Hive
from agent.router.mapping import Task
from agent.router.router import Router


ROUTER = Router()
HIVE_PROMPT = "Return two labelled lines -> ANSWER: and WHY:"
MAX_ROUNDS = 3

def _split(content: Any) -> tuple[str,str]:
    if isinstance(content, str):
        text = content
    else:
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "".join(parts)
                
    after_answer = text.split("ANSWER:")[-1]
    answer, _, rationale = after_answer.partition("WHY:")
    return answer.strip(),rationale.strip()
    
def _normalise(answer: str) -> str:
    if not answer:
        return ""
    ans = emoji.replace_emoji(answer, replace="")
    ans = ans.lower().strip()
    ans = ans.strip("'\"`")
    ans = ans.rstrip(string.punctuation)
    ans = re.sub(r'\s+', ' ', ans)
    return ans.strip()
    
def clone(ballot: Ballot, *, router=ROUTER) -> dict:
    llm = router.chat_model(Task.REASON, temperature=0.8, only=ballot["provider"])
    messages = [SystemMessage(HIVE_PROMPT),HumanMessage(ballot["question"] + "\n" + ballot["context"])]
    reply = None
    for chunk in llm.stream(messages):
        reply = chunk if reply is None else reply + chunk
    answer, rationale = _split(reply.content)
    answer = _normalise(answer)
    return {
        "votes":[Vote(
            round=ballot["round"],
            seed=ballot["seed"],
            answer=answer,
            rationale=rationale
        )]
    }

def spawn(state: Hive) -> Command[Literal["clone"]]:
    r = state["round"] + 1
    question = state["messages"][-1].content
    sends: list[Send] = []
    for seed in range(state["agents"]):
        provider = ROUTER.secondary if seed < state["secondary_seats"] else ROUTER.REQUIRED
        ballot = Ballot(question=question,context="",seed=seed,round=r,provider=provider)
        sends.append(Send("clone",ballot))
    return Command(
        update={"round":r, "board": [f"round {r}: {state['agents']} clones"]},
        goto=sends,
    )
    
def consensus(state: Hive) -> Command[Literal["spawn", "__end__"]]:
    this_round = [v for v in state["votes"] if v["round"] == state["round"]]
    assert len(this_round) == state["agents"]
    tally = Counter(v["answer"] for v in this_round)
    answer, n = tally.most_common(1)[0]
    if n * 2 > state["agents"]:
        return Command(update={"decision": answer, "board":[f"agreed {n}/{state['agents']}:{answer}"]}, goto=END)
    elif state["round"] >= MAX_ROUNDS:
        return Command(update={"decision": None, "board":[f"no consensus after {MAX_ROUNDS} rounds"]}, goto=END)
    return Command(update={"board":[f"split{dict(tally)}"]}, goto="spawn")

g = StateGraph(Hive)
g.add_node("spawn",spawn)
g.add_node("clone",clone)
g.add_node("consensus",consensus)
g.add_edge(START,"spawn")
g.add_edge("clone","consensus")

app = g.compile(checkpointer=InMemorySaver())
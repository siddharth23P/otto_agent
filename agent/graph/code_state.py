import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class Proposal(TypedDict):
    round: int
    author: int
    text: str


class ProposalVote(TypedDict):
    round: int
    voter: int
    approve: bool
    reason: str


class Submission(TypedDict):
    part: int
    round: int
    code: str


class Review(TypedDict):
    part: int
    round: int
    voter: int
    approve: bool
    reason: str


class CodeTask(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    board: Annotated[list[str], operator.add]
    agents: int
    decomp_round: int
    proposals: Annotated[list[Proposal], operator.add]
    proposal_votes: Annotated[list[ProposalVote], operator.add]
    part_specs: list[str] | None
    part_round: dict[int, int]
    part_status: dict[int, Literal["pending", "accepted", "exhausted"]]
    submissions: Annotated[list[Submission], operator.add]
    reviews: Annotated[list[Review], operator.add]
    final_code: str | None

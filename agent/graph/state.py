import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

ALLOWED_AGENTS = (1, 3, 5, 7, 9)

class Vote(TypedDict):
    round: int
    seed: int
    answer: str
    rationale: str

class Hive(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    board: Annotated[list[str], operator.add]
    votes: Annotated[list[Vote], operator.add]
    agents: int
    secondary_seats: int
    round: int
    decision: str | None

class Ballot(TypedDict):
    question: str
    context: str
    seed: int
    round: int
    provider: str
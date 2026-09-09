from langchain_core.messages import SystemMessage, HumanMessage

from agent.graph.state import ALLOWED_AGENTS
from agent.router.mapping import Task
from agent.router.router import Router


ROUTER = Router()
DEFAULT = 3

SIZE_PROMPT = (
    "How many independent agents does the task below actually need, to answer "
    "well? Reply with just a number.\n\n"
    "Worked examples:\n"
    '"hi" -> 1\n'
    '"what\'s 2+2" -> 1\n'
    '"is 7 prime" -> 1\n'
    "a short factual question with room for real disagreement -> 3\n"
    "a genuinely ambiguous, multi-part, or code-shaped task -> 5 or more\n\n"
    "Answer with just the number."
)


def _snap(n: int) -> int:
    if n in ALLOWED_AGENTS:
        return n
    return min(ALLOWED_AGENTS, key=lambda a: abs(a - n))


def size(text: str, *, router: Router = ROUTER) -> int:
    llm = router.chat_model(Task.CHAT_FAST, temperature=0.0)
    reply = None
    for chunk in llm.stream([SystemMessage(SIZE_PROMPT), HumanMessage(text)]):
        reply = chunk if reply is None else reply + chunk
    content = reply.content if reply is not None else ""
    if not isinstance(content, str):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
        content = "".join(parts)

    digits = "".join(ch for ch in content if ch.isdigit())
    if not digits:
        return DEFAULT
    try:
        n = int(digits)
    except ValueError:
        return DEFAULT
    return _snap(n)

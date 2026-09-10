"""Every role/evaluator prompt's ACTION: enumeration is meant to list
exactly what TOOL_DISPATCH actually dispatches (agent/pipeline/nodes.py's
prompts, agent/pipeline/tools.py's registry) -- a prompt advertising a tool
that isn't registered would send the model down a dead end, and a
registered tool the prompt never mentions would just never get used. This
was hand-kept in sync when complete_code/predict_edit were added
(2026-09-10); this test is what keeps it that way the next time a tool is
added or removed.
"""
from agent.pipeline import nodes as pn
from agent.pipeline.tools import TOOL_DISPATCH


def _formatted(prompt: str) -> str:
    # Every prompt has {max_iter}; EVALUATOR_PROMPT also has {role}.
    return prompt.format(max_iter=pn.MAX_TOOL_ITERATIONS, role="solver")


def test_every_tool_enabled_prompt_mentions_every_dispatchable_tool():
    prompts = {
        "PLANNER_PROMPT": pn.PLANNER_PROMPT,
        "SOLVER_PROMPT": pn.SOLVER_PROMPT,
        "SUMMARIZER_PROMPT": pn.SUMMARIZER_PROMPT,
        "FINDER_PROMPT": pn.FINDER_PROMPT,
        "EVALUATOR_PROMPT": pn.EVALUATOR_PROMPT,
    }
    for name, prompt in prompts.items():
        text = _formatted(prompt)
        for tool_name in TOOL_DISPATCH:
            assert tool_name in text, f"{name} never mentions tool {tool_name!r}"

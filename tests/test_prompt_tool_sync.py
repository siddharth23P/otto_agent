"""Every role/evaluator prompt's ACTION: enumeration is meant to list
exactly what TOOL_DISPATCH actually dispatches (agent/pipeline/nodes.py's
prompts, agent/pipeline/tools.py's registry) -- a prompt advertising a tool
that isn't registered would send the model down a dead end, and a
registered tool the prompt never mentions would just never get used. This
was hand-kept in sync when complete_code/predict_edit were added
(2026-09-10); this test is what keeps it that way the next time a tool is
added or removed.

EVALUATOR_PROMPT is dual-mode as of the second revision (2026-09-10) --
one shared template filled in with {target}/{target_note} depending on
whether it's judging a PLAN or a role's OUTPUT (see nodes.py's evaluator())
-- so it's checked once per mode here rather than once with a generic
{role} kwarg.
"""
from agent.pipeline import nodes as pn
from agent.pipeline.tools import TOOL_DISPATCH


def _formatted(prompt: str) -> str:
    return prompt.format(max_iter=pn.MAX_TOOL_ITERATIONS)


def _formatted_evaluator(*, target: str, target_note: str) -> str:
    return pn.EVALUATOR_PROMPT.format(
        max_iter=pn.MAX_TOOL_ITERATIONS, target=target, target_note=target_note,
    )


def test_every_tool_enabled_role_prompt_mentions_every_dispatchable_tool():
    prompts = {
        "PLANNER_PROMPT": pn.PLANNER_PROMPT,
        "SOLVER_PROMPT": pn.SOLVER_PROMPT,
        "SUMMARIZER_PROMPT": pn.SUMMARIZER_PROMPT,
        "FINDER_PROMPT": pn.FINDER_PROMPT,
    }
    for name, prompt in prompts.items():
        text = _formatted(prompt)
        for tool_name in TOOL_DISPATCH:
            assert tool_name in text, f"{name} never mentions tool {tool_name!r}"


def test_evaluator_prompt_mentions_every_dispatchable_tool_in_both_modes():
    plan_text = _formatted_evaluator(target="PLAN", target_note="would work if followed")
    final_text = _formatted_evaluator(target="SOLVER OUTPUT", target_note="as a finished answer")
    for tool_name in TOOL_DISPATCH:
        assert tool_name in plan_text, f"EVALUATOR_PROMPT (plan mode) never mentions tool {tool_name!r}"
        assert tool_name in final_text, f"EVALUATOR_PROMPT (final mode) never mentions tool {tool_name!r}"

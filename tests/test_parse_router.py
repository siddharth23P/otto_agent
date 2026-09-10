"""Coverage for _parse_router (agent/pipeline/nodes.py) -- router()'s
NODE:/WHY: reply parser. An unparseable or unrecognised NODE: value must
fail toward "solver" rather than crash or leave `node` unset, with the WHY
string saying so explicitly (visible on the board and to the dispatched
node's own prompt) -- this is the one place a malformed LLM reply could
otherwise take down the whole graph (an invalid `goto` on the returned
Command), so the fallback is tested directly rather than assumed.
"""
from agent.pipeline import nodes as pn


def test_valid_node_and_why_are_parsed():
    node, why = pn._parse_router("NODE: solver\nWHY: needs to write and run code")
    assert node == "solver"
    assert why == "needs to write and run code"


def test_parsing_is_case_insensitive_on_both_the_marker_and_the_value():
    node, why = pn._parse_router("node: Planner\nwhy: break it into steps")
    assert node == "planner"
    assert why == "break it into steps"


def test_every_role_node_name_round_trips():
    for role in pn.ROLE_NODES:
        node, _ = pn._parse_router(f"NODE: {role}\nWHY: x")
        assert node == role


def test_an_unrecognised_node_name_falls_back_to_solver_with_an_explanatory_why():
    node, why = pn._parse_router("NODE: astrologer\nWHY: x")
    assert node == "solver"
    assert "astrologer" in why or "NODE:" in why or "could not parse" in why


def test_no_node_line_at_all_falls_back_to_solver_with_an_explanatory_why():
    node, why = pn._parse_router("I don't know, maybe just answer it directly?")
    assert node == "solver"
    assert "could not parse" in why


def test_extra_surrounding_text_does_not_break_parsing():
    node, why = pn._parse_router("Sure, here goes:\nNODE: finder\nWHY: needs a lookup\nthanks!")
    assert node == "finder"
    assert why == "needs a lookup"

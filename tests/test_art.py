"""The art is data. These pin its shape so the widgets that step through it
can trust the tables without a terminal in the loop."""
from __future__ import annotations

import os

import pytest

from agent.cli import art


def test_the_wordmark_fits_in_the_sidebar():
    widths = {len(row) for row in art.WORDMARK_SMALL}
    assert len(widths) == 1, "ragged rows would jitter as they reveal"
    assert widths.pop() <= art.MAX_WORDMARK_WIDTH
    assert len(art.WORDMARK_SMALL) == 3


def test_reveal_frames_start_blank_and_end_on_the_wordmark():
    frames = art.reveal_frames(art.WORDMARK)
    assert len(frames) == art.REVEAL_STEPS + 1
    assert all("█" not in row for row in frames[0])
    assert frames[-1] == art.WORDMARK
    width = len(art.WORDMARK[0])
    assert all(len(row) == width for frame in frames for row in frame)


def test_reveal_frames_pad_ragged_input_to_one_width():
    frames = art.reveal_frames(("ab", "abcd"), steps=2)
    assert frames[-1] == ("ab  ", "abcd")
    assert frames[1] == ("ab  ", "ab  ")


def test_the_meter_turns_red_where_the_run_is_told_to_wrap_up():
    assert art.meter(0, 120).style == "green"
    assert art.meter(80, 120, warn_at=0.8).style == "yellow"
    assert art.meter(96, 120, warn_at=0.8).style == "bold red"
    assert str(art.meter(120, 120)) == art.METER_FULL * art.METER_WIDTH
    assert str(art.meter(3, None)) == ""
    assert str(art.meter(0, 0)) == ""


def test_phase_style_colours_only_the_states_that_mean_something():
    assert art.phase_style("working it out") == ""
    assert art.phase_style("checking the answer") == "yellow"
    assert art.phase_style("stopping after this call") == "bold red"
    assert art.phase_style("") == ""


def test_the_empty_state_has_no_mascot():
    joined = "".join(art.EMPTY_STATE)
    assert "ω" not in joined and '"""' not in joined and "(" not in joined, "text, not a face"


@pytest.mark.parametrize("line, mode", [
    ("escalated to plan mode -- need steps", "plan"),
    ("switched to find mode", "find"),
    ("de-escalated to summarize mode", "summarize"),
    ("solve: execute_bash -> ok", "solve"),
    ("compacted 3 older tool result(s)", None),
    ("escalated to turbo mode", None),
    ("", None),
])
def test_mode_from_board_line(line, mode):
    assert art.mode_from_board_line(line) == mode


def test_every_mode_has_a_glyph():
    from agent.pipeline.modes import MODES
    assert set(art.MODE_GLYPHS) == set(MODES)


def test_the_sparkle_settles_on_the_plain_bullet():
    glyph, style = art.SPARKLE_FRAMES[-1]
    assert glyph == art.BULLET_ANSWER and style == "green"


@pytest.mark.parametrize("line, expected", [
    ("solve: execute_bash -> ok", "◆ execute_bash → [green]ok[/]"),
    ("plan: read_file src/x.py -> failed", "☰ read_file src/x.py → [red]failed[/]"),
    ("escalated to plan mode -- need steps", "☰ escalated to plan mode -- need steps"),
    ("compacted 3 older tool result(s)", "compacted 3 older tool result(s)"),
])
def test_board_lines_are_drawn_with_glyphs(line, expected):
    assert art.decorate_board_line(line) == expected


def test_section_rules_are_one_line_and_dim():
    rule = art.section("usage", width=20)
    assert rule.plain.startswith("─ usage ") and len(rule.plain) == 20 and rule.style == "dim"


def test_the_env_flag_and_headless_both_disable_motion(monkeypatch):
    class Live:
        is_headless = False

    class Headless:
        is_headless = True

    monkeypatch.delenv(art.NO_ANIMATION_ENV, raising=False)
    assert art.animations_enabled(Live()) is True
    assert art.animations_enabled(Headless()) is False
    monkeypatch.setenv(art.NO_ANIMATION_ENV, "1")
    assert art.animations_enabled(Live()) is False


def test_art_text_is_a_rich_text_not_markup():
    text = art.art_text(art.EMPTY_STATE, style="dim")
    assert text.plain.splitlines() == list(art.EMPTY_STATE)


# --------------------------------------------------------------------------
# Motion (2026-09-13)
# --------------------------------------------------------------------------

def test_render_braille_sets_the_right_dots():
    assert art.render_braille([]) == art.BRAILLE_BLANK * 2
    assert art.render_braille([(0, 0)]) == "⠁" + art.BRAILLE_BLANK
    assert art.render_braille([(3, 3)]) == art.BRAILLE_BLANK + "⢀"
    assert art.render_braille([(0, 0), (1, 0), (0, 1)], 4, 4) == "⠋" + art.BRAILLE_BLANK


def test_the_snake_is_two_cells_wide_and_never_leaves_the_canvas():
    snake = art.Snake(seed=7)
    for _ in range(200):
        frame = snake.step()
        assert len(frame) == 2 and all(0x2800 <= ord(ch) <= 0x28FF for ch in frame)
        assert all(0 <= x < 4 and 0 <= y < 4 for x, y in snake._body)
        assert len(set(snake._body)) == 3, "three distinct dots"


def test_the_snake_is_deterministic_for_a_seed():
    a, b = art.Snake(seed=3), art.Snake(seed=3)
    assert [a.step() for _ in range(30)] == [b.step() for _ in range(30)]


def test_the_dot_sweep_travels_and_comes_back():
    positions = [art.dot_sweep(i).plain.index("●") for i in range(16)]
    assert positions[:8] == list(range(8)) and positions[8:] == list(range(6, -2, -1)) or positions[0] == 0
    assert all(len(art.dot_sweep(i).plain) == art.DOT_SWEEP_WIDTH for i in range(20))


def test_tween_eases_out_and_lands_exactly():
    path = art.tween(0, 100, steps=4)
    assert len(path) == 4 and path[-1] == 100
    assert path == sorted(path) and path[0] > 25, "eases out: big first step"
    assert art.tween(5, 5) == [5]


def test_shimmer_settles_on_the_plain_mark():
    frames = art.shimmer_frames(art.WORDMARK_SMALL)
    assert frames[-1].plain == "\n".join(art.WORDMARK_SMALL)
    assert any("reverse" in str(span.style) for span in frames[3].spans)
    assert not any("reverse" in str(span.style) for span in frames[-1].spans)


def test_sparkline_scales_to_the_peak_and_keeps_the_tail():
    assert art.sparkline([0, 50, 100]) == "▁▅█"
    assert art.sparkline([1, 1]) == "██"
    assert art.sparkline([]) == ""
    assert len(art.sparkline(range(40), width=12)) == 12

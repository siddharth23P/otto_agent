"""Coverage for looking at a container's desktop and clicking on it.

Otto had no OS control, and the first question was what it should point at. Not
a real machine: a screen-control tool aimed at somebody's actual desktop can
click anything they happen to have open, where one aimed at a container can
click the things somebody deliberately put in it. So it drives
`containers/otto-desktop` through the same command-runner seam the file and shell
tools use, and Otto ships no screen-control dependency.

That image is built on the one Otto already drives, deliberately: the measured
advice is that the CODE path beats the GUI path -- agents preferring code take
about a third fewer steps at a higher score -- and a desktop with no shell on
it would force the worse path.
"""
import agent.pipeline.tools as pt
from agent.pipeline import screen as screening
from agent.pipeline.execution import bind_command_runner


def _fake_desktop(out: str = "", err: str = "", code: int = 0):
    seen = {}

    def run(command: str, timeout: float):
        seen.setdefault("commands", []).append(command)
        return out, err, code

    run.seen = seen
    return run


# --------------------------------------------------------------------------
# Where it points
# --------------------------------------------------------------------------

def test_without_a_container_both_tools_refuse():
    """An ordinary chat turn binds no container, so there is no screen. Same
    refusal shape as the file tools with no workspace."""
    assert pt.look("what is on screen?").returncode == 1
    assert pt.look_act("click 10 10").returncode == 1
    assert "no container" in pt.look("what is on screen?").stderr


def test_the_capture_runs_in_the_container():
    runner = _fake_desktop(out="")
    with bind_command_runner(runner):
        pt.look("anything?")
    assert "import" in runner.seen["commands"][0]
    assert "$DISPLAY" in runner.seen["commands"][0]


def test_otto_ships_no_screen_control_dependency():
    """The point of driving it over there. agent/pipeline/ holds no vendor
    SDKs and this must not be the exception."""
    import inspect

    source = inspect.getsource(screening)
    for forbidden in ("pyautogui", "pynput", "Quartz", "screencapture"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Acting
# --------------------------------------------------------------------------

def test_a_click_becomes_a_pointer_move_and_a_click():
    runner = _fake_desktop()
    with bind_command_runner(runner):
        result = pt.look_act("click 840 512")
    assert result.returncode == 0
    assert "mousemove 840 512" in runner.seen["commands"][0]


def test_coordinates_may_be_comma_separated():
    runner = _fake_desktop()
    with bind_command_runner(runner):
        assert pt.look_act("click 840,512").returncode == 0


def test_a_click_without_two_numbers_is_refused():
    runner = _fake_desktop()
    with bind_command_runner(runner):
        assert pt.look_act("click the OK button").returncode == 1
        assert pt.look_act("click 840").returncode == 1


def test_typing_clears_modifiers_first():
    """A modifier stuck from an earlier action turns ordinary text into a
    stream of keyboard shortcuts."""
    runner = _fake_desktop()
    with bind_command_runner(runner):
        pt.look_act("type hello world")
    assert "--clearmodifiers" in runner.seen["commands"][0]


def test_typed_text_survives_a_quote():
    runner = _fake_desktop()
    with bind_command_runner(runner):
        assert pt.look_act("type it's fine").returncode == 0


def test_an_unknown_operation_names_the_ones_that_exist():
    runner = _fake_desktop()
    with bind_command_runner(runner):
        result = pt.look_act("scroll down")
    assert "click" in result.stderr and "type" in result.stderr


def test_looking_and_acting_are_separate_tools():
    """Clicking changes something and looking does not, which is what puts the
    acting half behind the same hold that covers sending a message."""
    from agent.pipeline import nodes as pn

    assert not pn._mutates("look")
    assert pn._mutates("look_act")


# --------------------------------------------------------------------------
# The capture coming back
# --------------------------------------------------------------------------

def test_a_wrapped_capture_still_decodes():
    """`base64` wraps at 76 columns by default and strict decoding rejects the
    newlines. That is exactly how this failed the first time it ran against a
    real desktop."""
    import base64 as b64

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    encoded = b64.b64encode(png).decode()
    wrapped = "\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76))
    runner = _fake_desktop(out=wrapped)
    with bind_command_runner(runner):
        result = pt.look("anything?")
    # It gets past decoding; the eight zero-padded bytes are not a real PNG,
    # so the vision model refuses them -- but as a failed tool call, which is
    # the point of the assertion below.
    assert "did not transfer cleanly" not in result.stderr


def test_a_capture_the_vision_model_refuses_is_one_failed_call(monkeypatch):
    """`look` caught ProviderError and nothing else, so a vendor 400 -- which
    is what a blank or half-drawn screen gets, verbatim "unable to process
    input image" -- unwound the whole tool loop rather than arriving as an
    ordinary failed call. Every other tool in this module already had the
    clause; this one did not, and the gap only showed once there was a
    working vision key to hit it with."""
    import base64 as b64

    def refuses(*args, **kwargs):
        raise RuntimeError("400 INVALID_ARGUMENT: Unable to process input image")

    monkeypatch.setattr(pt, "describe_image", refuses)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    runner = _fake_desktop(out=b64.b64encode(png).decode())

    with bind_command_runner(runner):
        result = pt.look("what is on screen?")

    assert not result.ok
    assert "could not look at the screen" in result.stderr


def test_an_empty_capture_says_the_desktop_may_not_be_running():
    runner = _fake_desktop(out="")
    with bind_command_runner(runner):
        assert "desktop" in pt.look("anything?").stderr


def test_a_capture_that_is_not_an_image_is_reported():
    import base64 as b64

    runner = _fake_desktop(out=b64.b64encode(b"not an image at all").decode())
    with bind_command_runner(runner):
        assert "readable image" in pt.look("anything?").stderr


def test_looking_needs_a_question():
    """A narrow question answered twice is close to being able to look; a
    generic caption is worth little."""
    runner = _fake_desktop()
    with bind_command_runner(runner):
        assert pt.look("   ").returncode == 1


def test_the_vocabulary_stays_small():
    assert len(screening.ACT_OPS) == 2

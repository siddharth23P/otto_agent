"""Coverage for `view_image` (agent/pipeline/tools.py) and the message it
builds (agent/pipeline/vision.py).

The contract worth pinning is narrow: an image file on disk -- or inside a
container -- reaches a vision model as a real image block with the right media
type and the agent's own question attached, and every way that can fail comes
back as a legible failing ToolResult rather than an exception. A benchmark run
with no GEMINI_API_KEY must degrade, not crash.

Entirely offline: the router is faked, and the container is a real shell bound
as a command runner (the pattern from tests/test_workspace_tools.py), so the
base64 transfer is exercised for real without Docker.
"""
import base64

import pytest

import agent.pipeline.tools as pt
from agent.pipeline.execution import bind_command_runner
from agent.pipeline.workspace import bind_workspace
from agent.router.llm_provider.base import ProviderError

#: A real 1x1 PNG, as bytes rather than a binary fixture file.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
GIF = b"GIF89a" + b"\x00" * 20


class _FakeLLM:
    def __init__(self, answer="a single dark pixel"):
        self.answer = answer
        self.seen = None

    def invoke(self, messages):
        self.seen = messages

        class _Reply:
            content = self.answer

        return _Reply()


class _FakeRouter:
    def __init__(self, llm=None, raises=None):
        self.llm, self.raises = llm or _FakeLLM(), raises
        self.task = None

    def chat_model(self, task, **overrides):
        self.task = task
        if self.raises is not None:
            raise self.raises
        return self.llm


@pytest.fixture
def fake_router(monkeypatch):
    router = _FakeRouter()
    monkeypatch.setattr(pt, "_get_router", lambda: router)
    return router


@pytest.fixture
def workspace(tmp_path):
    with bind_workspace(tmp_path) as ws:
        yield ws


# ---- the message that reaches the model ----------------------------------


def test_the_model_receives_a_real_image_block_and_the_agents_question(workspace, fake_router):
    """The contract everything else is plumbing for."""
    (workspace / "pixel.png").write_bytes(PNG)

    result = pt.view_image("pixel.png\ntranscribe the top staff bar by bar")

    assert result.ok
    [message] = fake_router.llm.seen
    text_block, image_block = message.content
    assert text_block == {"type": "text", "text": "transcribe the top staff bar by bar"}
    assert image_block["type"] == "image"
    assert image_block["mime_type"] == "image/png"
    assert base64.b64decode(image_block["base64"]) == PNG


def test_an_empty_question_falls_back_to_a_full_description(workspace, fake_router):
    (workspace / "pixel.png").write_bytes(PNG)

    pt.view_image("pixel.png")

    [message] = fake_router.llm.seen
    assert "Transcribe any text" in message.content[0]["text"]


def test_the_result_says_a_model_looked_rather_than_that_otto_saw(workspace, fake_router):
    """The reasoning model is reading another model's words, and the header is
    where that honesty lives."""
    (workspace / "pixel.png").write_bytes(PNG)

    out = pt.view_image("pixel.png\nwhat is it").stdout

    assert "a vision model looked at pixel.png" in out
    assert "image/png" in out
    assert "a single dark pixel" in out


def test_it_routes_to_the_vision_task(workspace, fake_router):
    from agent.router.mapping import Task

    (workspace / "pixel.png").write_bytes(PNG)
    pt.view_image("pixel.png\nwhat")

    assert fake_router.task is Task.VISION


# ---- every failure is a ToolResult, never an exception -------------------


def test_no_workspace_and_no_container_refuses_cleanly():
    result = pt.view_image("pixel.png\nwhat is this")

    assert not result.ok
    assert "no workspace is bound" in result.stderr


def test_a_path_escaping_the_workspace_is_refused(workspace, fake_router):
    result = pt.view_image("../../etc/passwd\nwhat is this")

    assert not result.ok
    assert "outside the workspace" in result.stderr


def test_a_text_file_is_refused_on_its_bytes_not_its_extension(workspace, fake_router):
    """An agent that just wrote a file may not have named it helpfully."""
    (workspace / "notes.png").write_text("#!/bin/sh\necho definitely not a png")

    result = pt.view_image("notes.png\nwhat is this")

    assert not result.ok
    assert "not an image" in result.stderr


def test_a_missing_file_is_refused(workspace, fake_router):
    assert not pt.view_image("absent.png\nwhat").ok


def test_an_oversized_image_is_refused_with_advice(workspace, fake_router, monkeypatch):
    monkeypatch.setattr(pt, "MAX_IMAGE_BYTES", 10)
    (workspace / "pixel.png").write_bytes(PNG)

    result = pt.view_image("pixel.png\nwhat")

    assert not result.ok
    assert "downscale" in result.stderr


def test_no_vision_key_degrades_instead_of_crashing_the_run(workspace, monkeypatch):
    """With no GEMINI_API_KEY, Task.VISION has no viable route. That arrives as
    a ProviderError and must look like any other failing tool."""
    monkeypatch.setattr(pt, "_get_router", lambda: _FakeRouter(raises=ProviderError("no viable route")))
    (workspace / "pixel.png").write_bytes(PNG)

    result = pt.view_image("pixel.png\nwhat")

    assert not result.ok
    assert "could not look at" in result.stderr


def test_a_vendor_sdk_error_is_also_contained(workspace, monkeypatch):
    monkeypatch.setattr(pt, "_get_router", lambda: _FakeRouter(raises=RuntimeError("vendor exploded")))
    (workspace / "pixel.png").write_bytes(PNG)

    assert not pt.view_image("pixel.png\nwhat").ok


# ---- the container path --------------------------------------------------


@pytest.fixture
def container(tmp_path):
    """A real shell in tmp_path bound as the command runner -- the same
    stand-in tests/test_workspace_tools.py uses, so the base64 transfer is
    exercised without Docker."""
    import subprocess

    def runner(command, timeout):
        proc = subprocess.run(
            command, shell=True, cwd=tmp_path, capture_output=True,
            text=True, timeout=timeout,
        )
        return proc.stdout, proc.stderr, proc.returncode

    with bind_command_runner(runner):
        yield tmp_path


def test_an_image_inside_a_container_arrives_byte_identical(container, fake_router):
    """This is the test that catches a newline or `base64 -w0` portability bug:
    a corrupted transfer would surface as a baffling model answer, not an
    error."""
    (container / "pixel.png").write_bytes(PNG)

    result = pt.view_image("pixel.png\nwhat colour")

    assert result.ok
    [message] = fake_router.llm.seen
    assert base64.b64decode(message.content[1]["base64"]) == PNG


def test_a_container_image_of_another_type_is_detected(container, fake_router):
    (container / "anim.gif").write_bytes(GIF)

    pt.view_image("anim.gif\nwhat")

    [message] = fake_router.llm.seen
    assert message.content[1]["mime_type"] == "image/gif"


def test_a_trailing_pwd_marker_does_not_corrupt_the_transfer(container, fake_router, tmp_path):
    """agent/eval/terminal_bench.py's runner appends a working-directory marker
    line to stdout. Nothing else tests that coupling."""
    import subprocess

    def marking_runner(command, timeout):
        proc = subprocess.run(command, shell=True, cwd=tmp_path, capture_output=True,
                              text=True, timeout=timeout)
        return proc.stdout + "\n__OTTO_PWD__/app\n", proc.stderr, proc.returncode

    (tmp_path / "pixel.png").write_bytes(PNG)
    with bind_command_runner(marking_runner):
        result = pt.view_image("pixel.png\nwhat")

    # Either it decodes cleanly or it refuses -- what it must never do is hand
    # a corrupt image to the model and call that success.
    if result.ok:
        [message] = fake_router.llm.seen
        assert base64.b64decode(message.content[1]["base64"]) == PNG
    else:
        assert "did not transfer cleanly" in result.stderr


def test_an_oversized_container_image_is_refused_before_transfer(container, fake_router, monkeypatch):
    monkeypatch.setattr(pt, "MAX_IMAGE_BYTES", 10)
    (container / "pixel.png").write_bytes(PNG)

    result = pt.view_image("pixel.png\nwhat")

    assert not result.ok
    assert "too big" in result.stderr


# ---- discoverability and the repeat detector -----------------------------


def test_read_file_points_at_view_image_instead_of_returning_mojibake(workspace):
    """Reading an image as text returns pages of line-numbered garbage. Naming
    the right tool at the moment it is needed is what makes it discoverable."""
    (workspace / "pixel.png").write_bytes(PNG)

    result = pt.read_file("pixel.png")

    assert not result.ok
    assert "use view_image" in result.stderr


def test_narrowing_questions_about_one_image_are_not_flagged_as_repetition():
    """A vision model returns words, so asking again with a narrower question
    is how the tool is meant to be used -- keying the repeat detector on the
    path alone would tell the agent to stop doing exactly that."""
    from agent.pipeline.nodes import _action_target

    first = _action_target("view_image", "score.png\ntranscribe bar 1")
    second = _action_target("view_image", "score.png\ntranscribe bar 2")
    same = _action_target("view_image", "score.png\ntranscribe bar 1")

    assert first != second
    assert first == same

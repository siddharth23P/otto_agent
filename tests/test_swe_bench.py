"""The SWE-bench harness: what it tells the agent, and how it decides.

The only benchmark here whose verdict nobody involved can argue with -- the
maintainers' own tests decide. That makes the harness itself the thing worth
testing: a grader that scores a deleted assertion as a fix, or a prompt that
leaks the test names, produces a number that looks like the published one and
measures something else entirely.

Nothing here needs Docker or the network.
"""
import pytest

from agent.eval import swe_bench as sb


@pytest.fixture
def instance():
    return sb.Instance(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        base_commit="d16bfe05a7",
        problem_statement="separability_matrix does not compute separability correctly",
        test_patch="",
        fail_to_pass=["t/test_a.py::test_one", "t/test_a.py::test_two"],
        pass_to_pass=["t/test_b.py::test_three"],
        difficulty="15 min - 1 hour",
    )


# --------------------------------------------------------------------------
# The image name
# --------------------------------------------------------------------------

def test_the_image_name_follows_the_upstream_convention(instance):
    """Docker tags cannot carry the double underscore SWE-bench ids use, so
    upstream substitutes `_1776_`. Getting this wrong is a pull that 404s on
    every instance."""
    assert "astropy_1776_astropy-12907" in instance.image
    assert "__" not in instance.image.split("/")[-1]


def test_it_asks_for_this_machine_s_architecture(monkeypatch):
    monkeypatch.setattr(sb.platform, "machine", lambda: "arm64")
    assert ".arm64." in sb.image_for("django__django-11099")

    monkeypatch.setattr(sb.platform, "machine", lambda: "x86_64")
    assert ".x86_64." in sb.image_for("django__django-11099")


# --------------------------------------------------------------------------
# What the agent is told
# --------------------------------------------------------------------------

def test_the_prompt_carries_the_issue_and_the_repository(instance):
    prompt = sb.build_prompt(instance)

    assert instance.problem_statement in prompt
    assert instance.repo in prompt
    assert sb.WORKDIR in prompt


def test_the_prompt_never_names_the_tests(instance):
    """An agent told which test to make pass writes to the test rather than to
    the bug, and the resulting number measures nothing."""
    prompt = sb.build_prompt(instance)

    for node in instance.fail_to_pass + instance.pass_to_pass:
        assert node not in prompt
    assert "test_one" not in prompt


def test_the_prompt_says_not_to_edit_tests(instance):
    assert "not edit or add tests" in sb.build_prompt(instance).lower()


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def _runner(output: str, *, patch_rc: int = 0):
    calls: list[str] = []

    def run(command: str, timeout: float):
        calls.append(command)
        if "git apply" in command:
            return "", "", patch_rc
        if "git diff" in command:
            return "", "", 0
        return output, "", 0

    run.calls = calls
    return run


def _report(passed, failed=()):
    return "\n".join([*(f"PASSED {t}" for t in passed),
                      *(f"FAILED {t}" for t in failed)])


def test_everything_passing_is_resolved(instance):
    run = _runner(_report(instance.fail_to_pass + instance.pass_to_pass))

    assert sb.grade("c", instance, run=run).resolved


def test_the_bug_still_failing_is_not_resolved(instance):
    run = _runner(_report(instance.pass_to_pass, failed=instance.fail_to_pass))
    verdict = sb.grade("c", instance, run=run)

    assert not verdict.resolved
    assert verdict.fail_to_pass_passed == 0
    assert verdict.pass_to_pass_passed == 1


def test_breaking_something_else_is_not_resolved(instance):
    """The half that stops an agent scoring by deleting the failing assertion:
    making the bug's tests pass while breaking another is not a fix."""
    run = _runner(_report(instance.fail_to_pass, failed=instance.pass_to_pass))
    verdict = sb.grade("c", instance, run=run)

    assert not verdict.resolved
    assert verdict.fail_to_pass_passed == 2


def test_a_test_that_never_ran_is_not_a_pass(instance):
    """Absence is not a pass. A node id missing from the report means the
    collection failed or the name moved, and either way it did not pass."""
    run = _runner(_report(["t/test_a.py::test_one"]))

    assert not sb.grade("c", instance, run=run).resolved


def test_a_test_patch_that_will_not_apply_is_reported_not_scored(instance):
    patched = sb.Instance(**{**instance.__dict__, "test_patch": "diff --git a/x b/x"})
    verdict = sb.grade("c", patched, run=_runner("", patch_rc=1))

    assert not verdict.resolved
    assert "would not apply" in verdict.error


def test_the_tests_are_only_applied_after_the_agent_stops(instance):
    """Same rule as Claw-Eval's grader files. If the patch went in first the
    agent could read the tests, and every score would be about that."""
    patched = sb.Instance(**{**instance.__dict__, "test_patch": "diff --git a/x b/x"})
    run = _runner(_report(patched.fail_to_pass + patched.pass_to_pass))
    sb.grade("c", patched, run=run)

    applied = next(i for i, c in enumerate(run.calls) if "git apply" in c)
    ran = next(i for i, c in enumerate(run.calls) if "pytest" in c)
    assert applied < ran


# --------------------------------------------------------------------------
# Reading pytest
# --------------------------------------------------------------------------

def test_the_report_is_parsed_rather_than_the_exit_code_trusted():
    """A run where one unrelated test errors has a non-zero exit and can still
    have every graded test passing, and the opposite is just as possible."""
    report = sb.parse_report(
        "PASSED tests/test_a.py::test_one\n"
        "FAILED tests/test_b.py::test_two\n"
        "ERROR tests/test_c.py::test_three\n"
        "SKIPPED tests/test_d.py::test_four\n"
    )

    assert report["tests/test_a.py::test_one"] == "PASSED"
    assert report["tests/test_b.py::test_two"] == "FAILED"
    assert report["tests/test_c.py::test_three"] == "ERROR"


def test_noise_around_the_summary_does_not_confuse_it():
    report = sb.parse_report(
        "collecting ...\n= short test summary info =\n"
        "PASSED tests/test_a.py::test_one\n= 1 passed in 0.4s =\n"
    )

    assert report == {"tests/test_a.py::test_one": "PASSED"}


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------

def test_json_encoded_test_lists_are_accepted():
    """The parquet stores them as JSON strings; some mirrors hand back lists.
    Depending on which is how a harness breaks on someone else's machine."""
    assert sb._as_list('["a::b", "c::d"]') == ["a::b", "c::d"]
    assert sb._as_list(["a::b"]) == ["a::b"]
    assert sb._as_list(None) == []


def test_selection_filters_without_reordering():
    items = [
        sb.Instance(f"r__r-{i}", "django/django" if i % 2 else "astropy/astropy",
                    "c", "p", "", [], [], difficulty="<15 min")
        for i in range(6)
    ]

    assert len(sb.select(items, repo="django")) == 3
    assert [i.instance_id for i in sb.select(items, limit=2)] == ["r__r-0", "r__r-1"]
    assert sb.select(items, difficulty="<15") == items


def test_the_working_directory_survives_a_cd():
    """Every docker exec is a fresh process, so a `cd` in one command would be
    invisible to the next -- which is not how any shell behaves."""
    out, cwd = sb._split_marker(f"some output\n{sb._PWD_MARKER}/testbed/astropy\n", "/testbed")

    assert out == "some output"
    assert cwd == "/testbed/astropy"


def test_no_marker_keeps_the_directory_it_had():
    out, cwd = sb._split_marker("plain output", "/testbed")

    assert (out, cwd) == ("plain output", "/testbed")


# --------------------------------------------------------------------------
# Two bugs this harness shipped with, each caught by a control run
# --------------------------------------------------------------------------

def test_a_coloured_summary_is_still_read():
    """astropy's own setup.cfg turns pytest colour on, so the status no longer
    starts the line and the node id has an escape in the middle of it. Both
    halves of the pattern miss, every test reads as not-passed, and the
    harness reported 0 of 141 passing on a repository where nothing was
    wrong -- which is exactly what two instances scored before the escapes
    were stripped."""
    coloured = (
        "\x1b[32mPASSED\x1b[0m astropy/io/fits/tests/test_connect.py::"
        "\x1b[1mTestSingleTable::test_simple\x1b[0m"
    )

    assert sb.parse_report(coloured) == {
        "astropy/io/fits/tests/test_connect.py::TestSingleTable::test_simple": "PASSED"
    }


def test_colour_is_also_asked_not_to_happen():
    """Stripping copes; --color=no avoids. Both, because a project can force
    colour back on from its own config and then only the stripping saves it."""
    assert "--color=no" in sb.TEST_COMMAND


def test_an_instance_with_no_arm64_build_falls_back(monkeypatch):
    """Upstream publishes arm64 for only part of the set -- of six instances
    tried at random, two had one and four did not. Without the fallback the
    harness reports "pull access denied" for two thirds of the benchmark and
    calls it a harness error."""
    monkeypatch.setattr(sb.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sb, "_manifest_exists", lambda image: False)

    image, emulated = sb.resolve_image("django__django-10097")
    assert ".x86_64." in image
    assert emulated


def test_an_arm64_build_is_preferred_when_it_exists(monkeypatch):
    monkeypatch.setattr(sb.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sb, "_manifest_exists", lambda image: True)

    image, emulated = sb.resolve_image("astropy__astropy-12907")
    assert ".arm64." in image
    assert not emulated


def test_an_x86_host_never_probes_for_arm64(monkeypatch):
    """The probe costs a registry round trip per instance. On x86 there is
    nothing to fall back FROM, so it is pure latency."""
    monkeypatch.setattr(sb.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sb, "_manifest_exists",
                        lambda image: pytest.fail("probed on an x86 host"))

    image, emulated = sb.resolve_image("django__django-10097")
    assert ".x86_64." in image and not emulated


def test_tests_are_named_by_file_not_by_node_id():
    """Given explicit node ids pytest is all-or-nothing: if one does not
    resolve -- a parametrisation that moved, an id recorded from a slightly
    different tree -- it prints "no tests ran" and every id in the batch
    scores as not-passed. Measured on astropy-13236: 644 PASS_TO_PASS tests,
    one bad id, 0/644 on an UNCHANGED repository."""
    assert sb.files_of([
        "astropy/table/tests/test_mixin.py::test_ndarray_mixin[True]",
        "astropy/table/tests/test_mixin.py::test_attributes",
        "astropy/table/tests/test_table.py::TestMeta::test_non_mapping_set[a, b]",
    ]) == ["astropy/table/tests/test_mixin.py", "astropy/table/tests/test_table.py"]


def test_a_file_that_will_not_import_costs_only_its_own_tests():
    assert "--continue-on-collection-errors" in sb.TEST_COMMAND


def test_a_stale_node_id_only_costs_itself(instance):
    """The property the file-level run buys. A name nothing reports is not a
    pass, and it takes nothing else down with it."""
    ran = _report(["t/test_a.py::test_one", "t/test_b.py::test_three"])
    verdict = sb.grade("c", instance, run=_runner(ran))

    assert verdict.fail_to_pass_passed == 1      # test_two never appeared
    assert verdict.pass_to_pass_passed == 1
    assert not verdict.resolved


def test_grading_does_not_run_on_the_agents_clock():
    """`run_instance` used one runner for both phases, so once the agent's
    budget was spent every grading command was refused -- astropy-13398 spent
    1649s, changed nothing, and was scored "test patch would not apply: the
    time budget for this instance is spent", which is not a verdict about the
    work at all."""
    import inspect

    source = inspect.getsource(sb.run_instance)
    assert "GRADING_BUDGET_S" in source
    assert "ungated" in source

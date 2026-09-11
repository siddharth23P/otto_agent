"""No directory in the repository root may share a name with an installed
third-party module.

This is here because it happened, silently, and cost a benchmark run to find.
Adding `docker/` for the desktop container image put an empty implicit
namespace package on `sys.path` ahead of site-packages, so Claw-Eval's
`import docker` succeeded and returned a package with nothing in it. The
symptom was `AttributeError: module 'docker' has no attribute 'from_env'` at
the moment a container was needed -- not an ImportError, not a missing
dependency message, and 169 of the 300 benchmark tasks need a container.

Implicit namespace packages are what make this quiet: a directory with no
`__init__.py` is still importable, and it shadows rather than errors. The
directory is now `containers/`, and this keeps the next one from landing.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Directories Otto owns. Everything else in the root is checked.
OURS = {"agent", "tests"}


def _site_packages_names() -> set[str]:
    names: set[str] = set()
    for entry in sys.path:
        path = Path(entry)
        if "site-packages" not in str(path) or not path.is_dir():
            continue
        for child in path.iterdir():
            if child.is_dir() and not child.name.endswith((".dist-info", ".egg-info")):
                names.add(child.name)
            elif child.suffix == ".py":
                names.add(child.stem)
    return names


def test_no_repository_directory_shadows_an_installed_module():
    installed = _site_packages_names()
    if not installed:
        pytest.skip("no site-packages on sys.path to compare against")

    collisions = sorted(
        entry.name
        for entry in REPO.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name not in OURS
        and entry.name in installed
    )

    assert not collisions, (
        f"{collisions} shadow installed module(s) of the same name -- an "
        f"implicit namespace package wins over site-packages and fails as an "
        f"AttributeError later, not an ImportError here"
    )

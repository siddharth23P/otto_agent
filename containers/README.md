# containers/

Images Otto drives through its command-runner seam. Otto ships no
screen-control or browser dependency of its own; a tool sends a driver
script into a machine that has what it needs.

| image | purpose |
| --- | --- |
| [otto-desktop/](otto-desktop/README.md) | a throwaway desktop (Xvfb, fluxbox, xdotool, ImageMagick) for the `look` and `look_act` tools |

The directory is named `containers/`, not `docker/`, on purpose: a `docker/`
directory at the repository root becomes an implicit namespace package that
shadows the `docker` site package, and `tests/test_no_import_shadowing.py`
keeps that from happening again.

import atexit
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from agent.router.router import Router


def _release() -> str:
    """Installed version, or 'dev' outside an installed environment."""
    try:
        return version("otto-cli-agent")
    except PackageNotFoundError:
        return "dev"


class AppContext:
    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self._router: Router | None = None
        self._handler: Any | None = None

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = Router(strict=self.strict)
        return self._router
    
    @property
    def handler(self):
        if self._handler is None:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            # Groups traces by build in the Langfuse UI. setdefault, so a
            # deploy or CI run can override with a commit SHA:
            #   LANGFUSE_RELEASE=$GITHUB_SHA otto chat
            os.environ.setdefault("LANGFUSE_RELEASE", _release())

            self._handler = CallbackHandler()
            atexit.register(get_client().flush)
        return self._handler

    @property
    def client(self):
        """The Langfuse client itself, for the things a callback cannot do.

        Scores are the main one: the handler reports what happened, but a
        score is a judgement about it, so it is always an explicit call.
        Reading `handler` first is deliberate -- that is what sets
        LANGFUSE_RELEASE and registers the flush at exit.
        """
        self.handler
        from langfuse import get_client

        return get_client()
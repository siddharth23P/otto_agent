from agent.router.router import Router


class AppContext:
    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self._router: Router | None = None

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = Router(strict=self.strict)
        return self._router
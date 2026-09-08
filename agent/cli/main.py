from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated
import typer

from agent.cli.context import AppContext
from agent.cli.errors import friendly
from agent.cli import doctor as doctor_cmd
from agent.cli import models as model_cmd
from agent.cli import route as route_cmd
from agent.cli import chat as chat_cmd

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
app = typer.Typer()

@app.callback()
def bootstrap(ctx: typer.Context, strict: Annotated[bool, typer.Option()] = False) -> None:
    load_dotenv(ENV_PATH)
    ctx.obj = AppContext(strict=strict)
    
# `friendly` is applied here, once, rather than as a decorator on each command
# module. One registration site means a new command cannot forget it, and the
# command modules stay free of CLI exit-code concerns.
for _name, _fn in (
    ("doctor", doctor_cmd.doctor),
    ("models", model_cmd.models),
    ("route", route_cmd.route),
    ("chat", chat_cmd.chat),
):
    app.command(_name)(friendly(_fn))

if __name__ == "__main__":
    app()

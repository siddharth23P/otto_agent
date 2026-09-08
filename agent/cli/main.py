from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated
import typer

from agent.cli.context import AppContext
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
    
app.command("doctor")(doctor_cmd.doctor)
app.command("models")(model_cmd.models)
app.command("route")(route_cmd.route)
app.command("chat")(chat_cmd.chat)

if __name__ == "__main__":
    app()

from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated
import typer

# Loaded here, at import time, before anything below it is imported -- not
# inside bootstrap(). agent.cli.chat now pulls in the whole pipeline
# (agent.pipeline.nodes), which constructs a module-level Router() the
# moment it's imported, which reads *_API_KEY from the environment
# immediately. bootstrap() only runs once Typer has already finished
# importing every command module, which is too late for that first
# Router() call -- it would always see an environment with no keys in it,
# whatever's actually in .env.
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

from agent.cli.context import AppContext
from agent.cli.errors import friendly
from agent.cli import doctor as doctor_cmd
from agent.cli import eval as eval_cmd
from agent.cli import eval_claw as eval_claw_cmd
from agent.cli import eval_compaction as eval_compaction_cmd
from agent.cli import eval_swe as eval_swe_cmd
from agent.cli import eval_hle as eval_hle_cmd
from agent.cli import eval_memory as eval_memory_cmd
from agent.cli import lessons as lessons_cmd
from agent.cli import models as model_cmd
from agent.cli import route as route_cmd
from agent.cli import chat as chat_cmd
from agent.cli import tui as tui_cmd

app = typer.Typer()

@app.callback()
def bootstrap(ctx: typer.Context, strict: Annotated[bool, typer.Option()] = False) -> None:
    ctx.obj = AppContext(strict=strict)

# `friendly` is applied here, once, rather than as a decorator on each command
# module. One registration site means a new command cannot forget it, and the
# command modules stay free of CLI exit-code concerns.
for _name, _fn in (
    ("doctor", doctor_cmd.doctor),
    ("models", model_cmd.models),
    ("route", route_cmd.route),
    ("chat", chat_cmd.chat),
    ("tui", tui_cmd.tui),
    ("eval", eval_cmd.eval_cmd),
    ("eval-memory", eval_memory_cmd.eval_memory_cmd),
    ("eval-hle", eval_hle_cmd.eval_hle_cmd),
    ("eval-claw", eval_claw_cmd.eval_claw_cmd),
    ("eval-compaction", eval_compaction_cmd.eval_compaction_cmd),
    ("eval-swe", eval_swe_cmd.eval_swe_cmd),
    ("lessons", lessons_cmd.lessons_cmd),
):
    app.command(_name)(friendly(_fn))

if __name__ == "__main__":
    app()

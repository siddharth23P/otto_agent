import typer

# import commands

app = typer.Typer()


@app.callback()
def bootstrap() -> None:
    """Otto — routed LLM agents."""
    # Phase 5.1 fills this in: load_dotenv(ENV_PATH), the Langfuse handler,
    # and a lazily-constructed Router on ctx.obj.


# add commands

if __name__ == "__main__":
    app()

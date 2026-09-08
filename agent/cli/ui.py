from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "ok":      "bold green",
    "warn":    "yellow",
    "bad":     "bold red",
    "muted":   "dim",
    "spec":    "cyan",
    "chosen":  "bold cyan",
})

out = Console(theme=THEME)
err = Console(theme=THEME, stderr=True)
"""The token panel down the right-hand side of the TUI (split out of
agent/cli/tui.py on 2026-09-12; unchanged in behaviour)."""
from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual.selection import Selection
from textual.widgets import Static

from agent.cli.clipboard import plain_text_of
from agent.pipeline.pricing import PRICES_AS_OF, format_cost
from agent.pipeline.usage import UsageLedger

__all__ = ["UsagePanel", "_thousands", "_short_model", "_ID_PREFIXES"]


def _thousands(n: int) -> str:
    """1234567 -> "1.23M". A token count is read for its ORDER, and a panel 28
    columns wide has no room for the digits that do not change the reading."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


#: Dot-separated segments that are routing, not identity. Named explicitly
#: rather than matched by shape: a rule like "drop everything before the last
#: dot" reads "gemini-2.5-flash" as "5-flash", which is how this was first
#: written and what its test caught.
_ID_PREFIXES = frozenset((
    "us", "eu", "apac", "global",
    "anthropic", "openai", "google", "meta", "mistral", "cohere",
    "amazon", "bedrock", "azure", "inception",
))


def _short_model(name: str) -> str:
    """The part of a model id a person reads.

    Vendor ids carry a region, a vendor, and a date stamp that are identical
    on every row of a 34-column panel -- width spent distinguishing nothing.
    "us.anthropic.claude-sonnet-4-20250514-v1:0" is "claude-sonnet-4" to
    anybody looking at this.
    """
    tail = str(name or "").split("/")[-1]
    segments = tail.split(".")
    while len(segments) > 1 and segments[0].lower() in _ID_PREFIXES:
        segments.pop(0)
    tail = ".".join(segments)

    parts = tail.split("-")
    while len(parts) > 2 and (
        (parts[-1].isdigit() and len(parts[-1]) >= 6)          # a date stamp
        or (parts[-1].startswith("v") and parts[-1][1:].isdigit())
        or parts[-1].endswith(":0")                            # a bedrock suffix
    ):
        parts.pop()
    return "-".join(parts) or tail or "unknown"


class UsagePanel(Static):
    """What this session has spent, per model, down the right-hand side.

    Reads a `UsageLedger` (agent/pipeline/usage.py) the app owns and hands to
    every turn, so it is cumulative across turns and across an ask_user pause
    without anything here having to add snapshots up.

    A Static holding a Rich Table rather than a DataTable: nothing here is
    selectable, sortable or scrollable, and Static sizes to its renderable
    instead of reserving rows it has not got (the same reasoning as tui.py's
    Static-vs-RichLog note).
    """

    # Height fixed to the row rather than `auto`. Textual measures an auto
    # height by asking the renderable for one, and a Rich renderable inside a
    # Static has no `get_height` -- which fails as an AttributeError deep in
    # the compositor rather than as a layout warning.
    DEFAULT_CSS = """
    UsagePanel { width: 34; height: 1fr; padding: 0 1; border: round $panel-lighten-2; }
    """

    def __init__(self, ledger: UsageLedger) -> None:
        super().__init__(id="usage")
        self._ledger = ledger

    def on_mount(self) -> None:
        # The widget's OWN border carries the title -- a Rich Panel inside it
        # would be a second frame drawn inside the first.
        self.border_title = "tokens"
        self.refresh_usage()

    def refresh_usage(self) -> None:
        """Redraw from the ledger. UI thread only -- a worker goes through
        `call_from_thread`, like everything else that touches the tree."""
        self.update(self._table())

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Drag-selectable like every other block (agent/cli/clipboard.py):
        the Group this shows renders through a RichVisual, which Textual's
        default cannot extract text from."""
        text = plain_text_of(self.content, self.content_size.width or self.size.width or 30)
        return (selection.extract(text), "\n") if text else None

    # NOT `_render`. `Widget._render` is Textual's own internal hook and it
    # returns a Visual; overriding it with a Rich renderable makes the
    # compositor call `render_strips` on a Rich object, which fails several
    # frames deep with no hint that a name was shadowed.
    def _table(self) -> RenderableType:
        snap = self._ledger.snapshot()
        if not snap["models"]:
            # Text, not a markup string: Static.update() with a bare str is
            # handed on as a Visual and fails in the compositor on this
            # Textual version. Everything else this returns is Rich.
            return Text("nothing yet", style="dim")

        # ONE left-aligned column, not a table. Four facts per model -- id,
        # requests, tokens, dollars -- do not fit across the ~30 usable
        # columns this panel has, and a Table.grid makes it worse rather than
        # better: the model id sets the first column's width, so every number
        # beside it truncates into uselessness ("225.…", "$0.0"). Tried both.
        # Stacking the facts under the name needs no column agreement at all.
        lines: list[RenderableType] = []
        for row in snap["models"]:
            # "--", not "0". A model that reported no usage, or that nothing
            # has a rate for, has to look different from one that genuinely
            # cost nothing -- usage.py's `reported`, pricing.py's absences.
            tokens = _thousands(row["total_tokens"]) if row["reported"] else "--"
            cost = format_cost(row["cost"])
            lines.append(Text(_short_model(row["model"]), style="bold"))
            lines.append(Text(f"  {row['calls']} req · {tokens} · {cost}", style="dim"))

        lines.append(Text(""))
        total = Text(f"total {snap['calls']} req · ", style="bold")
        total.append(_thousands(snap["total_tokens"]), style="bold")
        lines.append(total)

        # A total missing somebody's share says so, rather than presenting a
        # short number as the whole bill.
        money = Text(format_cost(snap["cost"]), style="bold")
        if not snap["fully_priced"]:
            money.append("+", style="bold yellow")
            money.append("  some models unpriced", style="yellow dim")
        lines.append(money)

        lines.append(Text(
            f"in {_thousands(snap['input_tokens'])} · "
            f"out {_thousands(snap['output_tokens'])}", style="dim"))
        if snap["cached_input_tokens"]:
            # Only when there is some. On a vendor with no prompt caching this
            # would be a permanent zero taking up a line.
            lines.append(Text(
                f"cached {_thousands(snap['cached_input_tokens'])}", style="dim"))

        # The date is not decoration. These are list prices read off a page on
        # one day and never re-checked (agent/pipeline/pricing.py), and a cost
        # with no date on it invites more trust than this can earn.
        lines.append(Text(f"est. at {PRICES_AS_OF} rates", style="dim"))
        return Group(*lines)

"""The setup wizard: providers and keys, the models they serve, and which
model answers each task. Opened from the palette ("Setup…"), f2, or by the
app itself when nothing is configured.

Three tabs, in the order a fresh install needs them:

  1 · Providers  one row per vendor (the four built-ins plus every custom
                 endpoint), a masked key column, and a live status once
                 probed. A password Input to add or replace a key; a small
                 form to add a named OpenAI-compatible endpoint.
  2 · Models     every model the configured providers list, with the
                 capabilities otto believes it has.
  3 · Mapping    one row per Task: what resolves today, what auto-map
                 proposes and why, and a Select of the models eligible for
                 that seat. Save writes pins to ~/.otto/routes.json.

Nothing here touches the environment, a file or the registry: every effect
goes through `SetupBackend`, a bag of callables the app fills from its own
module globals (agent/cli/tui.py `_setup_backend`) -- which is what lets the
tests drive this screen against fakes and no network.

The screen never displays a secret. A pasted key lives in one local for the
length of the handler that saves it, the Input is cleared before anything
else happens, and every message written to the status line is built from
names and masks (`********abcd`), never from the value.

Two Textual 8.2.8 facts shape the code (checked, not assumed): a thread
worker's default is `exit_on_error=True`, so an exception escaping one exits
the WHOLE app -- every worker here is `exit_on_error=False` and routes its
failures to the status line; and `DataTable.update_cell` keys columns by the
KEY given to `add_columns((label, key))`, not by the label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Select, Static, TabbedContent, TabPane
from textual.widgets._select import InvalidSelectValueError

from agent.cli import art
from agent.cli.art import animations_enabled
from agent.router.llm_provider.base import ModelInfo, ProviderStatus
from agent.router.mapping import Candidate, Task

__all__ = ["SetupBackend", "SetupScreen", "MappingRow", "pin_options", "NO_PIN"]

# pin_options and NO_PIN live in agent/router/setup.py, beside the rest of the
# setup data layer, so `otto serve` offers the same choices; re-exported here
# for the screen and everything that imported them from it.
from agent.router.setup import NO_PIN, pin_options  # noqa: E402
CUSTOM = "__custom__"

#: Status word -> primitive Rich style. agent/cli/doctor.py's map, without
#: THEME names, because a DataTable never sees the app's pushed theme.
_STATUS_STYLE = {
    ProviderStatus.OK: "bold green",
    ProviderStatus.NO_KEY: "dim",
    ProviderStatus.AUTH_FAILED: "bold red",
    ProviderStatus.UNREACHABLE: "yellow",
    ProviderStatus.ERROR: "bold red",
}


@dataclass
class SetupBackend:
    """Everything the screen may call. Names match agent/router/setup.py and
    agent/router/overrides.py; see tui.py `_setup_backend` for the wiring."""

    vendor_rows: Callable[[], list]
    probe: Callable[[str], Any]                       # -> setup.ProbeResult
    detected_pool: Callable[[], list[ModelInfo]]
    resolve: Callable[[Task], Any]                    # -> RoutingDecision, may raise
    routes: Callable[[], Mapping[Task, tuple[Candidate, ...]]]
    propose: Callable[[list[ModelInfo]], dict]        # -> {Task: automap.Proposal}
    pins: Callable[[], dict[Task, str]]
    set_pin: Callable[[Task, str], None]
    clear_pin: Callable[[Task], None]
    set_key: Callable[[str, str], Any]
    set_base_url: Callable[[str, str], None]
    add_endpoint: Callable[[str, str], None]
    reload: Callable[[], Any]


class MappingRow(Horizontal):
    """One task: its name, what resolves now, the proposal, and the pick."""

    DEFAULT_CSS = """
    MappingRow { height: 3; }
    MappingRow > Label { width: 14; padding-top: 1; }
    MappingRow > .current { width: 1fr; padding-top: 1; }
    MappingRow > .proposal { width: 1fr; padding-top: 1; color: $text-muted; }
    MappingRow > Select { width: 44; }
    """

    def __init__(self, task: Task) -> None:
        super().__init__(id=f"row-{task.value}")
        # `seat`, not `task`: Widget already has a `task` property (its message-pump task).
        self.seat = task

    def compose(self) -> ComposeResult:
        yield Label(self.seat.value)
        yield Static(Text("not probed yet", style="dim"), classes="current")
        yield Static(Text(""), classes="proposal")
        yield Select([NO_PIN], allow_blank=False, id=f"pin-{self.seat.value}")

    @property
    def select(self) -> Select:
        return self.query_one(Select)


class SetupScreen(ModalScreen[dict | None]):
    DEFAULT_CSS = """
    SetupScreen { align: center middle; }
    SetupScreen > Vertical { width: 96%; max-width: 120; height: 92%; border: round $accent; padding: 0 1; }
    SetupScreen #banner { height: auto; margin-bottom: 1; }
    SetupScreen TabbedContent { height: 1fr; }
    SetupScreen DataTable { height: 1fr; }
    SetupScreen .row { height: auto; }
    SetupScreen .row > Select { width: 28; }
    SetupScreen .row > Input { width: 1fr; }
    SetupScreen #custom { height: auto; display: none; }
    SetupScreen #custom.shown { display: block; }
    SetupScreen .buttons { height: auto; margin-top: 1; }
    SetupScreen .buttons Button { margin-right: 1; min-width: 8; }
    SetupScreen #mapping-rows { height: 1fr; }
    SetupScreen #setup-status { height: 1; color: $text-muted; }
    """

    def __init__(self, backend: SetupBackend) -> None:
        super().__init__()
        self.backend = backend
        self._rows: dict[str, Any] = {}          # name -> VendorRow
        self._probing: set[str] = set()
        self._ok: set[str] = set()
        self._models: list[ModelInfo] = []
        self._pending: dict[Task, str] = {}
        #: What each row's Select was last SET to by code. `Select.Changed`
        #: is a message, delivered after the setter returns, so a flag held
        #: around the assignment is already down by the time the echo lands
        #: -- the echo is told apart by its value instead.
        self._shown: dict[Task, str] = {}
        self._applying = False
        self._warned = False
        self._spin_timer = None
        self._spin = 0
        self._note = ""

    # ---- widgets ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical():
            banner = art.art_text(art.WORDMARK_SMALL, art.WORDMARK_STYLE)
            banner.append("   setup", style=art.TAGLINE_STYLE)
            yield Static(banner, id="banner")
            with TabbedContent(id="tabs"):
                with TabPane("1 · Providers", id="providers"):
                    yield DataTable(id="provider-table")
                    with Horizontal(classes="row"):
                        yield Select([("custom endpoint…", CUSTOM)], allow_blank=False, id="key-provider")
                        yield Input(password=True, placeholder="paste an API key, Enter saves it to .env", id="key-input")
                    with Vertical(id="custom"):
                        yield Input(placeholder="name (becomes NAME_API_KEY / NAME_BASE_URL), e.g. ollama",
                                    restrict=r"[A-Za-z0-9_-]*", id="custom-name")
                        yield Input(placeholder="base URL, e.g. http://localhost:11434/v1", id="custom-url")
                        yield Input(password=True, placeholder="API key (any non-empty value for a server that ignores keys)",
                                    id="custom-key")
                    with Horizontal(classes="buttons"):
                        yield Button("Probe providers", id="probe", variant="primary")
                        yield Button("Next: models ›", id="to-models")
                with TabPane("2 · Models", id="models"):
                    yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
                    with Horizontal(classes="buttons"):
                        yield Button("Refresh", id="refresh-models")
                        yield Button("Next: mapping ›", id="to-mapping")
                with TabPane("3 · Mapping", id="mapping"):
                    with VerticalScroll(id="mapping-rows"):
                        for task in Task:
                            yield MappingRow(task)
                    with Horizontal(classes="buttons"):
                        yield Button("Apply auto-map", id="automap")
                        yield Button("Clear pin", id="clear-pin")
                        yield Button("Save", id="save", variant="success")
                        yield Button("Close", id="close")
            yield Static("", id="setup-status")

    def on_mount(self) -> None:
        self._finish_mount()

    def _finish_mount(self) -> None:
        """Set the tables up and fill them once every widget exists.

        The tables, the key picker and the key box sit inside TabbedContent
        panes, and Textual mounts a pane's children a beat after the screen
        itself: on a slow runner (CI on Windows, once) the screen's mount
        fired before `#key-provider` was in the DOM and the fill crashed with
        NoMatches. So this waits a refresh when anything is missing, which
        costs nothing when everything is already there.
        """
        try:
            table = self.query_one("#provider-table", DataTable)
            models = self.query_one("#model-table", DataTable)
            self.query_one("#key-provider", Select)
            key_input = self.query_one("#key-input", Input)
        except NoMatches:
            self.call_after_refresh(self._finish_mount)
            return
        if not table.columns:
            table.add_columns(("provider", "provider"), ("key", "key"), ("status", "status"),
                              ("models", "models"), ("detail", "detail"))
        if not models.columns:
            models.add_columns(("provider", "provider"), ("model", "model"), ("context", "context"),
                               ("max out", "maxout"), ("capabilities", "caps"))
        self._fill_rows()
        self._status("probe to check keys, or paste one")
        key_input.focus()

    # ---- helpers ----------------------------------------------------------

    def _status(self, text: str, style: str = "dim") -> None:
        try:
            self.query_one("#setup-status", Static).update(Text(text, style=style))
        except NoMatches:
            pass

    def _ui(self, fn: Callable, *args) -> None:
        """From a worker: run `fn(*args)` on the UI thread, tolerating a
        screen that was dismissed while the worker was still going."""
        def safely() -> None:
            try:
                fn(*args)
            except NoMatches:
                pass
            except Exception as exc:  # never let a UI update kill the worker
                self._status(f"{type(exc).__name__}: {exc}", "red")
        self.app.call_from_thread(safely)

    def _fill_rows(self) -> None:
        """Provider rows from the environment -- no network."""
        try:
            rows = self.backend.vendor_rows()
        except Exception as exc:
            self._status(f"could not list providers: {exc}", "red")
            return
        table = self.query_one("#provider-table", DataTable)
        select = self.query_one("#key-provider", Select)
        previous = select.value if select.value is not Select.BLANK else None
        for row in rows:
            self._rows[row.name] = row
            key_cell = Text(row.masked_key, style="dim" if not row.key_present else "")
            if row.custom and not row.url_present:
                key_cell = Text(f"{row.masked_key} · {row.url_var} not set", style="yellow")
            if row.name in table.rows:
                table.update_cell(row.name, "key", key_cell)
            else:
                table.add_row(Text(row.label), key_cell, Text("not probed", style="dim"),
                              Text(""), Text(""), key=row.name)
        options = [(r.label, r.name) for r in rows] + [("custom endpoint…", CUSTOM)]
        self._applying = True
        try:
            select.set_options(options)
            # Keep what the person had chosen; on the first fill the only
            # option was the custom placeholder, so start on the first vendor.
            if previous in {v for _, v in options} and previous != CUSTOM:
                select.value = previous
            elif rows:
                select.value = rows[0].name
        finally:
            self._applying = False
        self.query_one("#custom").set_class(select.value == CUSTOM, "shown")

    def _set_report(self, name: str, result: Any) -> None:
        self._probing.discard(name)
        report = result.report
        table = self.query_one("#provider-table", DataTable)
        if name not in table.rows:
            self._fill_rows()
        if name not in table.rows:
            return
        table.update_cell(name, "status", Text(report.status.value, style=_STATUS_STYLE.get(report.status, "")))
        table.update_cell(name, "models", Text(str(len(result.models) or report.model_count or "")))
        table.update_cell(name, "detail", Text(report.detail or "", style="dim"))
        if report.ok:
            self._ok.add(name)
        else:
            self._ok.discard(name)

    def _set_models(self, models: list[ModelInfo]) -> None:
        self._models = list(models)
        table = self.query_one("#model-table", DataTable)
        table.clear()
        for m in models:
            table.add_row(
                m.provider, m.id,
                f"{m.context_window:,}" if m.context_window else "—",
                f"{m.max_output_tokens:,}" if m.max_output_tokens else "—",
                " ".join(sorted(c.value for c in m.capabilities)),
                key=m.spec,
            )

    def _set_mapping(self, current: dict[Task, str], proposals: dict, models: list[ModelInfo]) -> None:
        routes = self.backend.routes()
        try:
            pins = self.backend.pins()
        except Exception:
            pins = {}
        self._applying = True
        try:
            for row in self.query(MappingRow):
                task = row.seat
                row.query_one(".current", Static).update(Text(current.get(task, "none")))
                proposal = proposals.get(task)
                if proposal is None:
                    note = Text("")
                elif proposal.model is None:
                    note = Text(proposal.reason, style="yellow")
                else:
                    note = Text(f"{proposal.spec} — {proposal.reason}", style="dim")
                row.query_one(".proposal", Static).update(note)
                options = pin_options(task, models, routes)
                row.select.set_options(options)
                wanted = self._pending.get(task, pins.get(task, ""))
                try:
                    row.select.value = wanted
                except InvalidSelectValueError:
                    row.select.value = ""
                    if wanted:
                        row.query_one(".proposal", Static).update(
                            Text(f"pinned {wanted} is not in the catalogue", style="yellow"))
                self._shown[task] = "" if row.select.value is Select.BLANK else str(row.select.value)
        finally:
            self._applying = False
        self._proposals = proposals

    def _start_spinner(self) -> None:
        if self._spin_timer is None and animations_enabled(self.app):
            self._spin_timer = self.set_interval(0.1, self._advance_spinner)

    def _advance_spinner(self) -> None:
        if not self._probing:
            return
        self._spin += 1
        glyph = art.SPINNER[self._spin % len(art.SPINNER)]
        table = self.query_one("#provider-table", DataTable)
        for name in list(self._probing):
            if name in table.rows:
                table.update_cell(name, "status", Text(glyph, style="dim"))

    def _probe_done(self) -> None:
        self._probing.clear()
        note = f"{self._note} · " if self._note else ""
        self._status(f"{note}probe finished — see Models and Mapping")

    # ---- the probe: everything that touches the network ------------------

    def _begin_probe(self, names: list[str], note: str = "") -> None:
        """`note` is what just happened ("saved OPENAI_API_KEY to .env"); it
        stays on the status line through the probe it triggered."""
        self._note = note
        table = self.query_one("#provider-table", DataTable)
        for name in names:
            self._probing.add(name)
            if name in table.rows:
                table.update_cell(name, "status", Text("probing…" if not animations_enabled(self.app)
                                                       else art.SPINNER[0], style="dim"))
        self._status((f"{note} · " if note else "") + f"probing {', '.join(names)}…")
        self._start_spinner()
        self._probe(names)

    @work(thread=True, exit_on_error=False)
    def _probe(self, names: list[str]) -> None:
        for name in names:
            try:
                result = self.backend.probe(name)
            except Exception as exc:
                self._ui(self._status, f"probe of {name} failed: {type(exc).__name__}: {exc}", "red")
                self._probing.discard(name)
                continue
            self._ui(self._set_report, name, result)
        try:
            models = self.backend.detected_pool()
        except Exception as exc:
            self._ui(self._status, f"could not list models: {type(exc).__name__}: {exc}", "red")
            models = []
        self._ui(self._set_models, models)

        current: dict[Task, str] = {}
        for task in Task:
            try:
                d = self.backend.resolve(task)
                current[task] = f"{d.provider}:{d.model.id}" + (" (degraded)" if d.fell_back else "")
            except Exception:
                current[task] = "none"
        try:
            proposals = self.backend.propose(models)
        except Exception as exc:
            self._ui(self._status, f"auto-map failed: {type(exc).__name__}: {exc}", "red")
            proposals = {}
        self._ui(self._set_mapping, current, proposals, models)
        self._ui(self._probe_done)

    # ---- keys and endpoints ------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # FIRST -- or the App starts a pipeline turn on the key (modals.py).
        event.stop()
        iid = event.input.id
        if iid == "key-input":
            value = event.value
            event.input.value = ""
            self._save_key(value)
        elif iid in ("custom-name", "custom-url"):
            self.query_one("#custom-key", Input).focus()
        elif iid == "custom-key":
            self._save_custom()

    def _save_key(self, value: str) -> None:
        name = self.query_one("#key-provider", Select).value
        if name is Select.BLANK or name is None:
            self._status("choose a provider first", "yellow")
            return
        if name == CUSTOM:
            self._status("fill in the custom endpoint form below", "yellow")
            self.query_one("#custom-name", Input).focus()
            return
        if not value.strip():
            self._status("nothing pasted", "yellow")
            return
        try:
            self.backend.set_key(name, value)
        except Exception as exc:
            self._status(f"could not save the key: {type(exc).__name__}: {exc}", "red")
            return
        self._fill_rows()
        var = self._rows[name].key_var if name in self._rows else f"{name.upper()}_API_KEY"
        self._begin_probe([name], note=f"saved {var} to .env")

    def _save_custom(self) -> None:
        name = self.query_one("#custom-name", Input).value.strip().lower()
        url = self.query_one("#custom-url", Input).value.strip()
        key_input = self.query_one("#custom-key", Input)
        key = key_input.value
        key_input.value = ""
        if not name or not url or not key.strip():
            self._status("a custom endpoint needs a name, a base URL and a key", "yellow")
            return
        try:
            self.backend.add_endpoint(name, name)
            self.backend.set_base_url(name, url)
            self.backend.set_key(name, key)
        except Exception as exc:
            self._status(f"could not add {name}: {type(exc).__name__}: {exc}", "red")
            return
        self._fill_rows()
        self._begin_probe([name], note=f"added {name}: {name.upper().replace('-', '_')}_API_KEY and _BASE_URL saved to .env")

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        sid = event.select.id or ""
        if sid == "key-provider":
            self.query_one("#custom").set_class(event.value == CUSTOM, "shown")
            if event.value == CUSTOM:
                self.query_one("#custom-name", Input).focus()
            return
        if sid.startswith("pin-"):
            task = Task(sid[len("pin-"):])
            value = "" if event.value is Select.BLANK else str(event.value)
            if self._applying or value == self._shown.get(task, ""):
                return          # the echo of a programmatic set, not a pick
            self._shown[task] = value
            self._pending[task] = value
            self._warned = False

    # ---- buttons ----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        tabs = self.query_one("#tabs", TabbedContent)
        if bid in ("probe", "refresh-models"):
            self._begin_probe(list(self._rows) or [r.name for r in self.backend.vendor_rows()])
        elif bid == "to-models":
            tabs.active = "models"
        elif bid == "to-mapping":
            tabs.active = "mapping"
        elif bid == "automap":
            self._apply_automap()
        elif bid == "clear-pin":
            self._clear_focused_pin()
        elif bid == "save":
            self._save()
        elif bid == "close":
            self.action_close()

    def _apply_automap(self) -> None:
        proposals = getattr(self, "_proposals", None) or {}
        if not proposals:
            self._status("probe first — auto-map needs the catalogue", "yellow")
            return
        applied = 0
        self._applying = True
        try:
            for row in self.query(MappingRow):
                proposal = proposals.get(row.seat)
                if proposal is None or proposal.model is None or not proposal.needs_pin:
                    continue
                spec = proposal.spec
                if spec not in {v for _, v in row.select._options}:
                    continue
                row.select.value = spec
                self._shown[row.seat] = spec
                self._pending[row.seat] = spec
                applied += 1
        finally:
            self._applying = False
        self._status(f"auto-map set {applied} pin(s); Save to keep them")

    def _clear_focused_pin(self) -> None:
        node = self.focused
        while node is not None and not isinstance(node, MappingRow):
            node = node.parent
        if node is None:
            self._status("focus a task row first", "yellow")
            return
        self._applying = True
        try:
            node.select.value = ""
        finally:
            self._applying = False
        self._shown[node.seat] = ""
        self._pending[node.seat] = ""
        self._status(f"{node.seat.value}: pin cleared (Save to keep)")

    def _save(self) -> None:
        pinned: dict[str, str] = {}
        cleared: list[str] = []
        problems: list[str] = []
        for task, spec in self._pending.items():
            try:
                if spec:
                    self.backend.set_pin(task, spec)
                    pinned[task.value] = spec
                else:
                    self.backend.clear_pin(task)
                    cleared.append(task.value)
            except Exception as exc:
                problems.append(f"{task.value}: {exc}")
        try:
            problems.extend(list(self.backend.reload() or []))
        except Exception as exc:
            problems.append(f"reload: {type(exc).__name__}: {exc}")
        self.dismiss({"pinned": pinned, "cleared": cleared,
                      "providers_ok": sorted(self._ok), "problems": problems})

    def action_close(self) -> None:
        if self._pending and not self._warned:
            self._warned = True
            self._status("unsaved pin changes — Esc again to discard, or Save", "yellow")
            return
        self.dismiss(None)

    def key_escape(self) -> None:
        self.action_close()

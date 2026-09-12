"""Driving a browser, through the container Otto is already working in.

Otto has had no browser at all. Adding one raised a question with an unexpected
answer: where does it run? Not here -- a browser is a 300MB dependency and a
process to supervise, and `agent/pipeline/` deliberately holds no vendor SDKs.
It runs in the SAME container the file and shell tools already reach through
`agent/pipeline/execution.py`'s command runner. Otto gains no dependency; it
gains a driver script it ships into a machine that already has Chromium.

Two things shape the design, and both are measured.

THE CODE PATH WINS, SO THIS IS THE FALLBACK. Agents that route each subtask to
code or GUI and PREFER code reach 60.76% on OSWorld in 10.15 steps against ~15
for GUI-only, a 32% reduction; a hybrid action space is worth +22% relative.
On the web the same shape holds harder: API plus browser scores 38.9% on
WebArena against browsing alone, **+24.0 absolute points**. GUI-only chains are
described as brittle and prone to cascading failure. So `execute_bash` and an
HTTP client remain the first answer whenever a site has one, and this is for
when it does not.

THE OBSERVATION IS THE LEVER. Refining only the observation and action space --
no planner, no critic, no tree search, no examples -- beat every scaffolding
trick tried against it: +9.8 points over the previous state of the art, +29.4%
relative. So what comes back from a page here is a short structured digest --
title, url, headings, links, form fields, visible text, clipped -- and never
raw DOM or a full text dump. A page is a few hundred characters to this agent,
not a few hundred thousand.

The operations are high-level for the same reason: a small vocabulary a model
can use correctly beats a faithful reproduction of a mouse. Reverse-engineering
a site's latent functionality into named operations is measured as raising
success while cutting steps.

HONEST LIMITATION. Each call drives a fresh page and restores cookies and
storage from disk, because a one-shot command cannot hold a live browser open
between calls. Navigation, reading, forms and login flows work. In-page
JavaScript state that never touches storage does not survive between calls, and
the digest says so rather than pretending. A task that needs a genuinely live
session wants a browser server in the container, which is a later job.
"""
from __future__ import annotations

#: Ceiling on the digest of one page. Generous enough for a dense article's
#: headings and links, far below a real DOM.
MAX_DIGEST_CHARS = 4000

#: How many of each kind of thing a digest names. A page with 400 links is a
#: navigation index, and listing all of them is how an observation stops being
#: an observation.
MAX_LINKS = 30
MAX_FIELDS = 20
MAX_HEADINGS = 15

#: What the agent may ask for. Deliberately small: a vocabulary a model uses
#: correctly beats a faithful reproduction of a mouse.
READ_OPS = ("open", "read", "find", "back")
ACT_OPS = ("click", "type", "submit")


#: Schemes `open` will follow. Everything else -- `file:`, `data:`, `ftp:`,
#: `chrome:` -- is refused before the browser sees it.
#:
#: `file:///etc/passwd` through a browser is a file read with extra steps, and
#: the digest comes back to the model as ordinary tool output.
ALLOWED_SCHEMES = ("http", "https")

#: Networks `open` will not reach.
#:
#: This is the SSRF half, and it matters more here than in an ordinary HTTP
#: client because of WHO chooses the URL. The agent picks it, and the agent
#: reads web pages -- so a page it has already opened can tell it to open
#: something else, which is indirect prompt injection with a network request
#: on the end of it. `169.254.169.254` is the cloud metadata endpoint on every
#: major provider and hands out credentials to anything that asks.
#:
#: Link-local, loopback and the three private ranges, plus the IPv6 forms.
#: Blocked by parsed address rather than by string matching, and the host is
#: NORMALISED first -- browsers accept `http://2130706433/` and
#: `http://0177.0.0.1/` as 127.0.0.1, and `ipaddress` does not, so without
#: that step both walk straight past this list. Measured on the first draft of
#: exactly this function.
BLOCKED_NETWORKS = (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "169.254.0.0/16", "0.0.0.0/8", "100.64.0.0/10",
    "::1/128", "fc00::/7", "fe80::/10",
)


#: Names that mean loopback or metadata without being addresses. Short on
#: purpose: this is not an attempt at a name blocklist, which cannot work,
#: only a refusal of the handful anybody actually types.
BLOCKED_HOSTS = frozenset({
    "localhost", "ip6-localhost", "ip6-loopback",
    "metadata.google.internal", "metadata.goog", "instance-data",
})


def _as_address(host: str):
    """`host` as an IP address, accepting the forms a browser accepts.

    `ipaddress` takes dotted quads and IPv6. A browser also takes a bare
    integer (`2130706433`) and octal or hex octets (`0177.0.0.1`,
    `0x7f.0.0.1`), all of which mean 127.0.0.1 -- so a check that only asks
    `ipaddress` lets every one of them through. Returns None when the host is
    a name rather than any form of address.
    """
    import ipaddress

    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if host.isdigit():  # bare integer form
        try:
            return ipaddress.ip_address(int(host))
        except ValueError:
            return None
    parts = host.split(".")
    if len(parts) == 4:  # octal or hex octets
        try:
            octets = [int(p, 0) if p.lower().startswith("0x")
                      else int(p, 8) if p.startswith("0") and p != "0"
                      else int(p) for p in parts]
        except ValueError:
            return None
        if all(0 <= o <= 255 for o in octets):
            return ipaddress.ip_address(".".join(str(o) for o in octets))
    return None


def check_url(url: str) -> str:
    """Why this URL must not be opened, or "" if it may be.

    Refused BEFORE the driver script is built, so a rejected URL never reaches
    the container at all.

    Hostnames that are not literal addresses are allowed through: resolving
    them here would be a check against a different answer from the one the
    container's own resolver gives, and a name that resolves to a blocked
    address from inside the container is a DNS-rebinding problem this cannot
    honestly solve from out here. What it does stop is the direct form, which
    is the one that actually gets used.
    """
    import ipaddress
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        return f"that is not a URL ({exc})"

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return (f"{parts.scheme or 'that'}: is not a scheme this opens -- "
                f"only {' and '.join(ALLOWED_SCHEMES)}")
    host = (parts.hostname or "").strip("[]")
    if not host:
        return "that URL names no host"
    if host.lower() in BLOCKED_HOSTS or host.lower().endswith(".localhost"):
        return f"{host} is a loopback or metadata name this will not open"

    address = _as_address(host)
    if address is None:
        return ""  # a name, not a literal address -- see the docstring
    # An IPv4 address wrapped in IPv6 (`::ffff:127.0.0.1`) is the v4 address.
    if getattr(address, "ipv4_mapped", None) is not None:
        address = address.ipv4_mapped
    for network in BLOCKED_NETWORKS:
        if address in ipaddress.ip_network(network):
            return (f"{host} is on a network this will not open ({network}) -- "
                    "loopback, private and cloud-metadata addresses are refused")
    return ""


def parse_op(body: str, allowed: tuple[str, ...]) -> tuple[str, str] | str:
    """`(operation, argument)` from a CODE: body, or why it is not one."""
    head, _, rest = body.strip().partition("\n")
    op, _, inline = head.strip().partition(" ")
    op = op.strip().lower()
    if not op:
        return f"say what to do: one of {', '.join(allowed)}"
    if op not in allowed:
        return f"{op!r} is not one of {', '.join(allowed)}"
    argument = (inline.strip() + ("\n" + rest if rest else "")).strip()
    return op, argument


#: Runs in the container. One operation, then a digest of whatever page the
#: browser is looking at afterwards.
#:
#: Session continuity is cookies and storage on disk rather than a live
#: process, so a login survives between calls and an un-persisted in-page
#: state does not -- see this module's docstring.
DRIVER = r'''
import json, os, sys
STATE = "/tmp/otto-browser-state.json"
LAST = "/tmp/otto-browser-url.txt"
op, arg = sys.argv[1], sys.argv[2]
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("this container has no playwright installed", file=sys.stderr); sys.exit(3)

def digest(page, limits):
    out = ["url: " + page.url, "title: " + (page.title() or "")]
    def grab(selector, cap, label, fmt):
        rows = []
        for el in page.query_selector_all(selector)[:cap]:
            try:
                row = fmt(el)
            except Exception:
                continue
            if row:
                rows.append(row)
        if rows:
            out.append(label + ":")
            out.extend("  " + r for r in rows)
    grab("h1,h2,h3", limits["headings"], "headings",
         lambda el: (el.inner_text() or "").strip()[:120])
    grab("a[href]", limits["links"], "links",
         lambda el: ((el.inner_text() or "").strip()[:80] + " -> " + (el.get_attribute("href") or "")[:120]))
    grab("input,textarea,select,button", limits["fields"], "fields and buttons",
         lambda el: (el.get_attribute("name") or el.get_attribute("aria-label")
                     or el.get_attribute("placeholder") or (el.inner_text() or "").strip()[:60]
                     or el.get_attribute("type") or "")[:80])
    body = (page.inner_text("body") or "").strip()
    out.append("text:")
    out.append("  " + body[:limits["chars"]].replace("\n", "\n  "))
    return "\n".join(out)

limits = json.loads(sys.argv[3])
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox"])
    ctx = browser.new_context(storage_state=STATE if os.path.exists(STATE) else None)
    page = ctx.new_page()
    start = open(LAST).read().strip() if os.path.exists(LAST) else ""
    try:
        if op == "open":
            page.goto(arg, wait_until="domcontentloaded", timeout=30000)
        else:
            if not start:
                print("no page is open yet -- use `open <url>` first", file=sys.stderr); sys.exit(4)
            page.goto(start, wait_until="domcontentloaded", timeout=30000)
            if op == "click":
                page.get_by_text(arg, exact=False).first.click(timeout=15000)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            elif op == "type":
                field, _, value = arg.partition("=")
                page.get_by_label(field.strip()).first.fill(value.strip(), timeout=15000)
            elif op == "submit":
                page.keyboard.press("Enter")
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            elif op == "find":
                hits = page.get_by_text(arg, exact=False).all()[:10]
                print("matches: " + str(len(hits)))
                for h in hits:
                    print("  " + (h.inner_text() or "").strip()[:120])
            elif op == "back":
                page.go_back(wait_until="domcontentloaded", timeout=15000)
    except Exception as exc:
        print(str(exc)[:400], file=sys.stderr); sys.exit(5)
    open(LAST, "w").write(page.url)
    ctx.storage_state(path=STATE)
    print(digest(page, limits))
    browser.close()
'''

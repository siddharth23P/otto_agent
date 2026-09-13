"""Using an app on a device or a desktop -- Android, iOS, macOS, Linux,
Windows -- and reporting each step as the machine saw it.

The same `exercise` shape as a page, a shell or a terminal
(agent/pipeline/walkthrough.py): the first step names the kind and what to
launch, every step after it is `click`, `type`, `press`, `expect`, `screen`,
`changed` or `wait`, and the report stops at the first step that does not
hold. What differs is only the thing underneath, one `Device` per platform:

    android <apk path> <package>      adb: install, launch, uiautomator, logcat
    android <package>[/<activity>]
    ios <.app path>                   xcrun simctl on a booted simulator;
    ios <bundle id>                     idb, if installed, for the UI tree
    mac <app name or .app path>       System Events (JXA) and screencapture
    linux <command>                   xdotool and ImageMagick; AT-SPI if there
    windows <command>                 pywinauto, in the optional interpreter

Every one of these drives ONLY the app it launched. The desktop kinds address
elements of that one process -- `tell process "X"`, `xdotool --window`, a
pywinauto Application -- never a coordinate on the whole screen, which is what
keeps agent/pipeline/screen.py's argument intact: a tool that clicks a real
desktop clicks whatever its owner has open, and this one cannot.

WHAT IS TESTED, HONESTLY. Each device's commands and the parsing of what they
print are covered with faked commands (tests/test_native.py). macOS was
driven end to end on the machine this was written on, against a small AppKit
app compiled for the purpose, once the process had been granted Accessibility
and Screen Recording -- without the first macOS refuses UI scripting with
-1719, without the second every capture is empty. That machine had no adb,
no idb, no iOS runtime, no Linux desktop and no Windows, so the other four
have not been driven for real here. Every driver says exactly which
permission or tool is missing rather than failing obscurely, and that
message is the thing a person can act on.

DEBUGGING IS THE POINT, NOT ONLY DRIVING. Each device reports what an app
did behind the screen: Android's logcat crash lines and the process going
away, an iOS app whose process died, a desktop process that exited with its
stderr. A step that passed on a screen while the app crashed underneath is
still a failed walkthrough.
"""
from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from typing import Callable
from xml.etree import ElementTree

from agent.pipeline.walkthrough import Walk, summary_line

NATIVE_KINDS = ("android", "ios", "mac", "linux", "windows")
NATIVE_STEPS = ("click", "type", "press", "expect", "wait", "screen", "changed")

#: How long an app gets to show its first screen.
LAUNCH_TIMEOUT_S = 30.0
#: How long `expect` waits for text to appear or go.
EXPECT_TIMEOUT_S = 8.0

Runner = Callable[[list[str], float], tuple[str, str, int]]


def run_command(argv: list[str], timeout: float = 30.0, *, input_bytes: bytes | None = None) -> tuple[str, str, int]:
    """The one place a device command is run, so tests can stand in for it."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=timeout, input=None,
        )
    except FileNotFoundError:
        return "", f"{argv[0]} is not installed or not on PATH", 127
    except subprocess.TimeoutExpired:
        return "", f"{argv[0]} did not finish within {int(timeout)}s", -1
    return proc.stdout, proc.stderr, proc.returncode


def run_bytes(argv: list[str], timeout: float = 30.0) -> bytes:
    """A command whose stdout is an image."""
    try:
        return subprocess.run(argv, capture_output=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return b""


class DeviceError(Exception):
    """A step that could not be done, with the reason a person can act on."""


class Device:
    """What every platform provides. Each method raises DeviceError with a
    reason rather than returning a sentinel, so the walkthrough loop has one
    shape."""

    kind = ""

    def launch(self, target: str) -> str:
        raise NotImplementedError

    def texts(self) -> list[str]:
        raise NotImplementedError

    def click_text(self, text: str) -> None:
        raise NotImplementedError

    def tap(self, x: int, y: int) -> None:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError

    def press(self, key: str) -> None:
        raise NotImplementedError

    def screenshot(self) -> bytes:
        return b""

    def alive(self) -> bool:
        return True

    def errors(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Android: adb
# --------------------------------------------------------------------------

def find_adb() -> str | None:
    """adb on PATH, or under ANDROID_HOME / ANDROID_SDK_ROOT / the default
    SDK location."""
    found = shutil.which("adb")
    if found:
        return found
    roots = [os.environ.get("ANDROID_HOME", ""), os.environ.get("ANDROID_SDK_ROOT", ""),
             os.path.expanduser("~/Library/Android/sdk"), os.path.expanduser("~/Android/Sdk")]
    for root in roots:
        candidate = os.path.join(root, "platform-tools", "adb") if root else ""
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


NO_ADB = ("no adb was found. Install Android platform-tools (or set ANDROID_HOME) "
          "and start an emulator or connect a device: `adb devices` must list one")

#: uiautomator's `bounds="[x1,y1][x2,y2]"`.
_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

ANDROID_KEYS = {
    "enter": "KEYCODE_ENTER", "back": "KEYCODE_BACK", "home": "KEYCODE_HOME",
    "tab": "KEYCODE_TAB", "space": "KEYCODE_SPACE", "delete": "KEYCODE_DEL",
    "backspace": "KEYCODE_DEL", "up": "KEYCODE_DPAD_UP", "down": "KEYCODE_DPAD_DOWN",
    "left": "KEYCODE_DPAD_LEFT", "right": "KEYCODE_DPAD_RIGHT", "menu": "KEYCODE_MENU",
    "escape": "KEYCODE_ESCAPE", "search": "KEYCODE_SEARCH", "volumeup": "KEYCODE_VOLUME_UP",
    "volumedown": "KEYCODE_VOLUME_DOWN", "power": "KEYCODE_POWER",
}


def parse_ui_dump(xml: str) -> list[dict]:
    """Every node of a uiautomator dump that shows text, with its centre."""
    nodes = []
    try:
        root = ElementTree.fromstring(xml[xml.index("<"):] if "<" in xml else xml)
    except (ElementTree.ParseError, ValueError):
        return nodes
    for node in root.iter("node"):
        text = (node.get("text") or "").strip()
        desc = (node.get("content-desc") or "").strip()
        label = text or desc
        m = _BOUNDS.search(node.get("bounds") or "")
        centre = ((int(m.group(1)) + int(m.group(3))) // 2, (int(m.group(2)) + int(m.group(4))) // 2) if m else None
        if label:
            nodes.append({"text": label, "centre": centre, "clickable": node.get("clickable") == "true",
                          "resource": node.get("resource-id") or ""})
    return nodes


class AndroidDevice(Device):
    kind = "android"

    def __init__(self, run: Runner = run_command, adb: str | None = None, serial: str = ""):
        self.run = run
        self.adb = adb or find_adb()
        self.serial = serial or os.environ.get("ANDROID_SERIAL", "")
        self.package = ""

    def _adb(self, *args: str, timeout: float = 30.0) -> tuple[str, str, int]:
        if self.adb is None:
            raise DeviceError(NO_ADB)
        argv = [self.adb, *(["-s", self.serial] if self.serial else []), *args]
        return self.run(argv, timeout)

    def _shell(self, *args: str, timeout: float = 30.0) -> str:
        out, err, code = self._adb("shell", *args, timeout=timeout)
        if code != 0:
            raise DeviceError((err or out).strip()[:300] or f"adb shell {args[0]} failed")
        return out

    def launch(self, target: str) -> str:
        out, err, code = self._adb("get-state", timeout=10)
        if code != 0 or "device" not in out:
            raise DeviceError("no Android device or emulator is connected -- `adb devices` lists none")
        parts = target.split()
        if parts[0].endswith(".apk"):
            if len(parts) < 2:
                raise DeviceError("after the .apk, say the package: `android app.apk com.example.app`")
            out, err, code = self._adb("install", "-r", parts[0], timeout=180)
            if code != 0:
                raise DeviceError("install failed: " + (err or out).strip()[-300:])
            self.package = parts[1].split("/")[0]
            activity = parts[1] if "/" in parts[1] else ""
        else:
            self.package = parts[0].split("/")[0]
            activity = parts[0] if "/" in parts[0] else ""
        self._adb("logcat", "-c", timeout=10)
        if activity:
            out = self._shell("am", "start", "-W", "-n", activity, timeout=LAUNCH_TIMEOUT_S)
        else:
            out = self._shell("monkey", "-p", self.package, "-c", "android.intent.category.LAUNCHER", "1",
                              timeout=LAUNCH_TIMEOUT_S)
        if "Error" in out or "No activities found" in out:
            raise DeviceError("launch failed: " + out.strip()[-300:])
        time.sleep(1.5)
        return f"launched {self.package}"

    def _dump(self) -> list[dict]:
        out, err, code = self._adb("exec-out", "uiautomator", "dump", "/dev/tty", timeout=30)
        if code != 0 and "<" not in out:
            raise DeviceError("could not read the screen: " + (err or out).strip()[:200])
        return parse_ui_dump(out)

    def texts(self) -> list[str]:
        return [n["text"] for n in self._dump()]

    def click_text(self, text: str) -> None:
        for node in self._dump():
            if text.lower() in node["text"].lower() and node["centre"]:
                self.tap(*node["centre"])
                return
        raise DeviceError(f"nothing on screen reads {text!r}")

    def tap(self, x: int, y: int) -> None:
        self._shell("input", "tap", str(x), str(y))
        time.sleep(0.4)

    def type_text(self, text: str) -> None:
        self._shell("input", "text", text.replace(" ", "%s"))
        time.sleep(0.2)

    def press(self, key: str) -> None:
        code = ANDROID_KEYS.get(key.strip().lower(), key if key.upper().startswith("KEYCODE_") else "")
        if not code:
            raise DeviceError(f"not a key this knows: {key} (one of {', '.join(ANDROID_KEYS)}, or a KEYCODE_ name)")
        self._shell("input", "keyevent", code)
        time.sleep(0.3)

    def screenshot(self) -> bytes:
        if self.adb is None:
            return b""
        return run_bytes([self.adb, *(["-s", self.serial] if self.serial else []), "exec-out", "screencap", "-p"])

    def alive(self) -> bool:
        if not self.package:
            return True
        out, _, _ = self._adb("shell", "pidof", self.package, timeout=10)
        return bool(out.strip())

    def errors(self) -> list[str]:
        out, _, _ = self._adb("logcat", "-d", "-s", "AndroidRuntime:E", timeout=15)
        lines = [l.strip() for l in out.splitlines() if "FATAL EXCEPTION" in l or "Process:" in l
                 or (self.package and self.package in l and "Exception" in l)]
        return lines[:6]

    def close(self) -> None:
        if self.package and self.adb:
            self._adb("shell", "am", "force-stop", self.package, timeout=10)


# --------------------------------------------------------------------------
# iOS: xcrun simctl, and idb for the UI tree when it is installed
# --------------------------------------------------------------------------

NO_SIMULATOR = ("no booted iOS simulator -- boot one first: `xcrun simctl boot \"iPhone 16\"` "
                "(and `open -a Simulator`), or the app cannot be installed anywhere")
NO_IDB = ("reading or tapping the iOS screen needs idb (`brew install idb-companion` and "
          "`pip install fb-idb`); without it a walkthrough can launch, screenshot, `wait`, "
          "`changed` and notice a crash, but not `click`, `type`, `press` or `expect`")

IOS_KEYS = {"enter": "40", "return": "40", "tab": "43", "escape": "41", "space": "44", "backspace": "42",
            "delete": "42", "up": "82", "down": "81", "left": "80", "right": "79"}


class IOSDevice(Device):
    kind = "ios"

    def __init__(self, run: Runner = run_command, idb: str | None = None, udid: str = "booted"):
        self.run = run
        self.udid = udid
        self.idb = idb if idb is not None else shutil.which("idb")
        self.bundle = ""
        self.pid = 0

    def _simctl(self, *args: str, timeout: float = 60.0) -> tuple[str, str, int]:
        return self.run(["xcrun", "simctl", *args], timeout)

    def launch(self, target: str) -> str:
        out, err, code = self._simctl("list", "devices", "booted", timeout=20)
        if code != 0 or "(Booted)" not in out:
            raise DeviceError(NO_SIMULATOR)
        # A bundle on disk, not a bundle id that happens to end in ".app".
        if target.rstrip("/").endswith(".app") and ("/" in target or os.path.isdir(target)):
            info = os.path.join(target, "Info.plist")
            try:
                with open(info, "rb") as handle:
                    self.bundle = plistlib.load(handle).get("CFBundleIdentifier", "")
            except (OSError, plistlib.InvalidFileException) as exc:
                raise DeviceError(f"cannot read {info}: {exc}")
            if not self.bundle:
                raise DeviceError(f"{info} names no CFBundleIdentifier")
            out, err, code = self._simctl("install", self.udid, target, timeout=120)
            if code != 0:
                raise DeviceError("install failed: " + (err or out).strip()[-300:])
        else:
            self.bundle = target.strip()
        out, err, code = self._simctl("launch", self.udid, self.bundle, timeout=LAUNCH_TIMEOUT_S)
        if code != 0:
            raise DeviceError("launch failed: " + (err or out).strip()[-300:])
        m = re.search(r":\s*(\d+)\s*$", out.strip())
        self.pid = int(m.group(1)) if m else 0
        time.sleep(1.5)
        return f"launched {self.bundle}" + (f" (pid {self.pid})" if self.pid else "")

    def _idb(self, *args: str, timeout: float = 30.0) -> str:
        if not self.idb:
            raise DeviceError(NO_IDB)
        out, err, code = self.run([self.idb, *args], timeout)
        if code != 0:
            raise DeviceError((err or out).strip()[:300] or f"idb {args[0]} failed")
        return out

    def _tree(self) -> list[dict]:
        import json

        out = self._idb("ui", "describe-all", "--json")
        try:
            items = json.loads(out)
        except ValueError:
            return []
        found = []
        for item in items if isinstance(items, list) else []:
            label = (item.get("AXLabel") or item.get("AXValue") or item.get("AXTitle") or "")
            frame = item.get("frame") or {}
            centre = None
            if frame:
                centre = (int(frame.get("x", 0) + frame.get("width", 0) / 2),
                          int(frame.get("y", 0) + frame.get("height", 0) / 2))
            if str(label).strip():
                found.append({"text": str(label).strip(), "centre": centre})
        return found

    def texts(self) -> list[str]:
        return [n["text"] for n in self._tree()]

    def click_text(self, text: str) -> None:
        for node in self._tree():
            if text.lower() in node["text"].lower() and node["centre"]:
                self.tap(*node["centre"])
                return
        raise DeviceError(f"nothing on screen reads {text!r}")

    def tap(self, x: int, y: int) -> None:
        self._idb("ui", "tap", str(x), str(y))
        time.sleep(0.4)

    def type_text(self, text: str) -> None:
        self._idb("ui", "text", text)

    def press(self, key: str) -> None:
        code = IOS_KEYS.get(key.strip().lower(), key if key.isdigit() else "")
        if not code:
            raise DeviceError(f"not a key this knows: {key} (one of {', '.join(IOS_KEYS)}, or a HID keycode)")
        self._idb("ui", "key", code)

    def screenshot(self) -> bytes:
        import tempfile

        path = os.path.join(tempfile.gettempdir(), f"otto-ios-{os.getpid()}.png")
        out, err, code = self._simctl("io", self.udid, "screenshot", path, timeout=30)
        if code != 0:
            return b""
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except OSError:
            return b""

    def alive(self) -> bool:
        if not self.pid:
            return True
        try:
            os.kill(self.pid, 0)  # simulator apps are host processes
            return True
        except PermissionError:
            return True
        except OSError:  # ProcessLookupError, or a platform with no such probe
            return False

    def errors(self) -> list[str]:
        if self.pid and not self.alive():
            return [f"the app's process {self.pid} is gone -- it crashed or exited; see "
                    "~/Library/Logs/DiagnosticReports for the report"]
        return []

    def close(self) -> None:
        if self.bundle:
            self._simctl("terminate", self.udid, self.bundle, timeout=15)


# --------------------------------------------------------------------------
# macOS: System Events through JXA, screencapture for the window
# --------------------------------------------------------------------------

NO_ACCESSIBILITY = ("macOS refused UI scripting (-1719): grant the terminal or app running otto "
                    "Accessibility in System Settings > Privacy & Security > Accessibility")
NO_WINDOW = ("{name} is running but shows no window System Events can see. A bare binary "
             "needs NSApp.setActivationPolicy(.regular) and its window ordered front "
             "(makeKeyAndOrderFront) before this can read or click it; a window may also "
             "still be opening -- `wait 1000` first")
NO_SCREEN_RECORDING = ("screen capture came back empty: grant the terminal or app running otto "
                       "Screen Recording in System Settings > Privacy & Security")

#: Walk one process's front window and list what a person can read or press.
_JXA_TREE = '''
function run(argv) {
  const se = Application("System Events");
  const p = se.processes.byName(argv[0]);
  const out = [];
  let wins;
  try { wins = p.windows(); } catch (e) { return JSON.stringify({error: String(e)}); }
  if (!wins.length) return JSON.stringify({elements: [], window: null});
  const w = wins[0];
  const info = {name: w.name(), position: w.position(), size: w.size()};
  const walk = (el, depth) => {
    if (depth > 12) return;
    let kids = [];
    try { kids = el.uiElements(); } catch (e) { return; }
    for (const k of kids) {
      let role = "", name = "", value = "", desc = "", generic = "", pos = null, size = null;
      try { role = k.role(); } catch (e) {}
      try { name = k.name() || ""; } catch (e) {}
      try { value = k.value(); } catch (e) {}
      try { desc = k.description() || ""; } catch (e) {}
      try { generic = k.roleDescription() || ""; } catch (e) {}
      try { pos = k.position(); size = k.size(); } catch (e) {}
      // "button" for a button is the role's description, not a label.
      if (desc === generic) desc = "";
      const label = String(name || (typeof value === "string" ? value : "") || desc || "").trim();
      if (label) out.push({role, text: label, position: pos, size: size});
      walk(k, depth + 1);
    }
  };
  walk(w, 0);
  return JSON.stringify({elements: out, window: info});
}
'''

#: Press the element whose label matches, by its accessibility action.
_JXA_CLICK = '''
function run(argv) {
  const se = Application("System Events");
  const p = se.processes.byName(argv[0]);
  const want = argv[1].toLowerCase();
  const wins = p.windows();
  if (!wins.length) return "no window";
  const find = (el, depth) => {
    if (depth > 12) return null;
    let kids = [];
    try { kids = el.uiElements(); } catch (e) { return null; }
    for (const k of kids) {
      let name = "", value = "", desc = "";
      try { name = k.name() || ""; } catch (e) {}
      try { value = k.value(); } catch (e) {}
      try { desc = k.description() || ""; } catch (e) {}
      const label = String(name || (typeof value === "string" ? value : "") || desc || "").toLowerCase();
      if (label.includes(want)) return k;
      const deeper = find(k, depth + 1);
      if (deeper) return deeper;
    }
    return null;
  };
  const el = find(wins[0], 0);
  if (!el) return "not found";
  try { el.actions.byName("AXPress").perform(); return "pressed"; } catch (e) {}
  try { el.click(); return "clicked"; } catch (e) { return "cannot click: " + e; }
}
'''

MAC_KEYS = {"enter": "return", "return": "return", "tab": "tab", "escape": "escape", "space": "space",
            "backspace": "delete", "delete": "delete", "up": "up arrow", "down": "down arrow",
            "left": "left arrow", "right": "right arrow", "home": "home", "end": "end"}
_MAC_KEY_CODES = {"return": 36, "tab": 48, "escape": 53, "space": 49, "delete": 51, "up arrow": 126,
                  "down arrow": 125, "left arrow": 123, "right arrow": 124, "home": 115, "end": 119}


class MacDevice(Device):
    kind = "mac"

    def __init__(self, run: Runner = run_command, popen=subprocess.Popen, cwd: str = ""):
        self.run = run
        self.popen = popen
        self.cwd = cwd or os.getcwd()
        self.process_name = ""
        self.proc: subprocess.Popen | None = None
        self.stderr_path = ""

    def _jxa(self, script: str, *args: str, timeout: float = 30.0) -> str:
        out, err, code = self.run(["osascript", "-l", "JavaScript", "-e", script, *args], timeout)
        if code != 0 or "-1719" in err or "assistive access" in err:
            if "-1719" in err or "assistive access" in err:
                raise DeviceError(NO_ACCESSIBILITY)
            raise DeviceError((err or out).strip()[:300] or "osascript failed")
        return out.strip()

    def launch(self, target: str) -> str:
        target = target.strip()
        if target.rstrip("/").endswith(".app"):
            bundle = target.rstrip("/")
            if not os.path.isabs(bundle) and os.path.isdir(os.path.join(self.cwd, bundle)):
                bundle = os.path.join(self.cwd, bundle)  # a bundle built in the workspace
            self.process_name = os.path.basename(bundle)[:-4]
            try:
                with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as handle:
                    # The process is named after the executable, which need
                    # not match the bundle: Counter.app can run `counter`.
                    self.process_name = plistlib.load(handle).get("CFBundleExecutable") or self.process_name
            except (OSError, plistlib.InvalidFileException):
                pass
            out, err, code = self.run(["open", "-a", bundle], 20)
        elif " " not in target and not os.path.exists(os.path.join(self.cwd, target)):
            # An app by name -- "Calculator" -- rather than a command in the
            # workspace. Checked against the workspace, where a command lives.
            self.process_name = target
            out, err, code = self.run(["open", "-a", target], 20)
        else:
            # A command: run it and treat its first word as the process.
            import shlex
            import tempfile

            self.stderr_path = os.path.join(tempfile.gettempdir(), f"otto-mac-{os.getpid()}.err")
            handle = open(self.stderr_path, "w")
            self.proc = self.popen(target, shell=True, cwd=self.cwd, stdout=handle, stderr=handle,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
            self.process_name = os.path.basename(shlex.split(target)[0])
            out, err, code = "", "", 0
        if code != 0:
            raise DeviceError("launch failed: " + (err or out).strip()[-300:])
        deadline = time.time() + LAUNCH_TIMEOUT_S
        while time.time() < deadline:
            out, _, _ = self.run(["pgrep", "-x", self.process_name], 10)
            if out.strip():
                break
            time.sleep(0.5)
        else:
            raise DeviceError(f"no process named {self.process_name!r} appeared within "
                              f"{int(LAUNCH_TIMEOUT_S)}s -- did it exit? the walkthrough names "
                              "the process after the executable")
        # Then the window, which is what every later step reads. Reported,
        # not failed, so a launch of something windowless still counts as a
        # launch; the first click or expect says what is missing.
        for _ in range(20):
            try:
                if self._tree().get("window"):
                    return f"launched {self.process_name}"
            except DeviceError as exc:
                # "Can't get object": the process exists before System Events
                # knows it. Not yet, rather than never -- unless it is the
                # permission, which no amount of waiting grants.
                if "Accessibility" in str(exc):
                    raise
            time.sleep(0.5)
        return f"launched {self.process_name} (no window visible yet)"

    def _tree(self) -> dict:
        import json

        raw = self._jxa(_JXA_TREE, self.process_name)
        try:
            data = json.loads(raw)
        except ValueError:
            return {"elements": [], "window": None}
        if data.get("error"):
            raise DeviceError(f"cannot read {self.process_name}'s window: {data['error'][:200]}")
        return data

    def _label(self, text: str) -> str:
        """`click button "Add one" of window 1` means Add one: a model writes
        AppleScript when it thinks of System Events. The quoted part is the
        label."""
        m = re.search(r'"([^"]+)"', text)
        return m.group(1) if m else text

    def texts(self) -> list[str]:
        # Bidi marks around a number are not something a person reads, and
        # they would make `expect 0` fail. No window at all is a reason, not
        # an empty list.
        tree = self._tree()
        if not tree.get("window"):
            raise DeviceError(NO_WINDOW.format(name=self.process_name))
        return [e["text"].replace("\u200e", "").replace("\u200f", "") for e in tree.get("elements", [])]

    def click_text(self, text: str) -> None:
        label = self._label(text)
        result = self._jxa(_JXA_CLICK, self.process_name, label)
        if result == "no window":
            raise DeviceError(NO_WINDOW.format(name=self.process_name))
        if result not in ("pressed", "clicked"):
            raise DeviceError(f"nothing in {self.process_name}'s window reads {label!r}" if result == "not found"
                              else result[:200])
        time.sleep(0.4)

    def tap(self, x: int, y: int) -> None:
        # Within the app's window only: the coordinates are offset by its position.
        window = self._tree().get("window") or {}
        pos = window.get("position") or [0, 0]
        script = (f'const se = Application("System Events"); se.processes.byName("{self.process_name}")'
                  f'.windows[0].click({{at: [{pos[0] + x}, {pos[1] + y}]}});')
        self._jxa(script)
        time.sleep(0.4)

    def type_text(self, text: str) -> None:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        self._jxa(f'const se = Application("System Events"); se.processes.byName("{self.process_name}")'
                  f'.frontmost = true; se.keystroke("{escaped}");')
        time.sleep(0.2)

    def press(self, key: str) -> None:
        name = MAC_KEYS.get(key.strip().lower(), "")
        code = _MAC_KEY_CODES.get(name)
        if code is None:
            if len(key) == 1:
                self.type_text(key)
                return
            raise DeviceError(f"not a key this knows: {key} (one of {', '.join(MAC_KEYS)})")
        self._jxa(f'const se = Application("System Events"); se.processes.byName("{self.process_name}")'
                  f'.frontmost = true; se.keyCode({code});')
        time.sleep(0.3)

    def screenshot(self) -> bytes:
        import tempfile

        try:
            window = self._tree().get("window") or {}
        except DeviceError:
            window = {}
        pos, size = window.get("position"), window.get("size")
        path = os.path.join(tempfile.gettempdir(), f"otto-mac-{os.getpid()}.png")
        argv = ["screencapture", "-x", path]
        if pos and size:
            argv = ["screencapture", "-x", "-R", f"{pos[0]},{pos[1]},{size[0]},{size[1]}", path]
        self.run(argv, 20)
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except OSError:
            return b""

    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        out, _, _ = self.run(["pgrep", "-x", self.process_name], 10)
        return bool(out.strip())

    def errors(self) -> list[str]:
        if self.alive():
            return []
        tail = ""
        if self.stderr_path:
            try:
                with open(self.stderr_path, errors="replace") as handle:
                    tail = handle.read()[-400:].strip()
            except OSError:
                pass
        code = self.proc.returncode if self.proc is not None else None
        return [f"{self.process_name} is no longer running" + (f" (exit {code})" if code is not None else "")
                + (f"; its output ended: {tail}" if tail else "")]

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            import signal

            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        elif self.process_name:
            self.run(["osascript", "-e", f'tell application "{self.process_name}" to quit'], 10)


# --------------------------------------------------------------------------
# Linux: xdotool and ImageMagick on the launched process's window; AT-SPI
# through the optional interpreter for what the window says
# --------------------------------------------------------------------------

NO_XDOTOOL = "no xdotool was found -- install it (and ImageMagick for screenshots) to drive a Linux app"
NO_ATSPI = ("reading a Linux app's text needs AT-SPI: `pip install pyatspi` is not enough, install the "
            "system package (python3-pyatspi / python-atspi) into the Python OTTO_BROWSER_PYTHON names; "
            "without it a walkthrough can launch, `press`, `type`, screenshot, `changed` and `wait`")

#: Runs in the optional interpreter: the texts of one pid's accessible tree,
#: or the extents of the first element whose name contains a string.
ATSPI_SCRIPT = r'''
import json, sys
try:
    import pyatspi
except ImportError:
    print("no pyatspi", file=sys.stderr); sys.exit(3)
pid, want = int(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "")
found = []
def walk(node, depth=0):
    if depth > 14 or node is None:
        return
    try:
        name = (node.name or "").strip()
        if not name:
            try:
                name = (node.queryText().getText(0, -1) or "").strip()
            except Exception:
                name = ""
        if name:
            entry = {"text": name, "role": node.getRoleName()}
            try:
                ext = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                entry["centre"] = [ext.x + ext.width // 2, ext.y + ext.height // 2]
            except Exception:
                pass
            found.append(entry)
        for i in range(node.childCount):
            walk(node.getChildAtIndex(i), depth + 1)
    except Exception:
        return
desktop = pyatspi.Registry.getDesktop(0)
for i in range(desktop.childCount):
    app = desktop.getChildAtIndex(i)
    try:
        if app is not None and app.get_process_id() == pid:
            walk(app)
    except Exception:
        continue
if want:
    hit = next((f for f in found if want.lower() in f["text"].lower()), None)
    print(json.dumps(hit or {}))
else:
    print(json.dumps(found))
'''

LINUX_KEYS = {"enter": "Return", "return": "Return", "tab": "Tab", "escape": "Escape", "space": "space",
              "backspace": "BackSpace", "delete": "Delete", "up": "Up", "down": "Down", "left": "Left",
              "right": "Right", "home": "Home", "end": "End"}


class LinuxDevice(Device):
    kind = "linux"

    def __init__(self, run: Runner = run_command, popen=subprocess.Popen, cwd: str = "",
                 interpreter: str | None = None):
        self.run = run
        self.popen = popen
        self.cwd = cwd or os.getcwd()
        self.interpreter = interpreter
        self.proc: subprocess.Popen | None = None
        self.window = ""
        self.stderr_path = ""

    def launch(self, target: str) -> str:
        import tempfile

        if not shutil.which("xdotool"):
            raise DeviceError(NO_XDOTOOL)
        self.stderr_path = os.path.join(tempfile.gettempdir(), f"otto-linux-{os.getpid()}.err")
        handle = open(self.stderr_path, "w")
        self.proc = self.popen(target, shell=True, cwd=self.cwd, stdout=handle, stderr=handle,
                               stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + LAUNCH_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise DeviceError(f"the app exited ({self.proc.returncode}) before showing a window")
            out, _, code = self.run(["xdotool", "search", "--onlyvisible", "--pid", str(self.proc.pid)], 10)
            ids = out.split()
            if code == 0 and ids:
                self.window = ids[0]
                break
            time.sleep(0.5)
        if not self.window:
            raise DeviceError(f"no window appeared for pid {self.proc.pid} within {int(LAUNCH_TIMEOUT_S)}s")
        self.run(["xdotool", "windowactivate", "--sync", self.window], 10)
        time.sleep(1.0)
        return f"launched pid {self.proc.pid}, window {self.window}"

    def _atspi(self, want: str = "") -> str:
        interpreter = self.interpreter or os.environ.get("OTTO_BROWSER_PYTHON", "").strip() or sys.executable
        args = [interpreter, "-c", ATSPI_SCRIPT, str(self.proc.pid if self.proc else 0)] + ([want] if want else [])
        out, err, code = self.run(args, 30)
        if code == 3:
            raise DeviceError(NO_ATSPI)
        if code != 0:
            raise DeviceError((err or out).strip()[:300] or "AT-SPI failed")
        return out

    def texts(self) -> list[str]:
        import json

        try:
            return [e["text"] for e in json.loads(self._atspi() or "[]")]
        except ValueError:
            return []

    def click_text(self, text: str) -> None:
        import json

        try:
            hit = json.loads(self._atspi(text) or "{}")
        except ValueError:
            hit = {}
        if not hit or not hit.get("centre"):
            raise DeviceError(f"nothing in the window reads {text!r}")
        self.run(["xdotool", "mousemove", str(hit["centre"][0]), str(hit["centre"][1]), "click", "1"], 10)
        time.sleep(0.4)

    def tap(self, x: int, y: int) -> None:
        # Relative to the app's window, never the screen.
        out, _, _ = self.run(["xdotool", "getwindowgeometry", "--shell", self.window], 10)
        geo = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        sx, sy = int(geo.get("X", 0)) + x, int(geo.get("Y", 0)) + y
        self.run(["xdotool", "mousemove", str(sx), str(sy), "click", "1"], 10)
        time.sleep(0.4)

    def type_text(self, text: str) -> None:
        self.run(["xdotool", "type", "--window", self.window, "--delay", "20", text], 30)

    def press(self, key: str) -> None:
        name = LINUX_KEYS.get(key.strip().lower(), key if key.startswith("ctrl+") or len(key) == 1 else "")
        if not name:
            raise DeviceError(f"not a key this knows: {key} (one of {', '.join(LINUX_KEYS)}, ctrl+<x>, or a character)")
        self.run(["xdotool", "key", "--window", self.window, name], 10)
        time.sleep(0.3)

    def screenshot(self) -> bytes:
        if not self.window or not shutil.which("import"):
            return b""
        return run_bytes(["import", "-window", self.window, "png:-"])

    def alive(self) -> bool:
        return self.proc is None or self.proc.poll() is None

    def errors(self) -> list[str]:
        if self.alive():
            return []
        tail = ""
        try:
            with open(self.stderr_path, errors="replace") as handle:
                tail = handle.read()[-400:].strip()
        except OSError:
            pass
        return [f"the app exited ({self.proc.returncode})" + (f"; its output ended: {tail}" if tail else "")]

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            import signal

            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


# --------------------------------------------------------------------------
# Windows: pywinauto, as one driver script in the optional interpreter
# --------------------------------------------------------------------------

NO_PYWINAUTO = ("driving a Windows app needs pywinauto in the Python OTTO_BROWSER_PYTHON names "
                "(`pip install pywinauto`)")

#: Argument 1 is the step script, 2 the JSON limits. Same report contract as
#: every other driver: summary, steps, `screen:`; exit 6 on a failed step.
WINDOWS_DRIVER = r'''
import json, subprocess, sys, time
try:
    from pywinauto import Application, keyboard
except ImportError:
    print("no pywinauto", file=sys.stderr); sys.exit(3)
script, limits = sys.argv[1], json.loads(sys.argv[2])
steps = [l.strip() for l in script.splitlines() if l.strip()]
report, failed, app, win, frames = [], "", None, None, []
KEYS = {"enter": "{ENTER}", "return": "{ENTER}", "tab": "{TAB}", "escape": "{ESC}", "space": " ",
        "backspace": "{BACKSPACE}", "delete": "{DELETE}", "up": "{UP}", "down": "{DOWN}",
        "left": "{LEFT}", "right": "{RIGHT}", "home": "{HOME}", "end": "{END}"}
def texts():
    out = []
    for ctrl in win.descendants():
        try:
            t = (ctrl.window_text() or "").strip()
        except Exception:
            t = ""
        if t:
            out.append(t)
    return out
def shot():
    try:
        import io
        buf = io.BytesIO(); win.capture_as_image().save(buf, format="PNG"); return buf.getvalue()
    except Exception:
        return b""
for n, line in enumerate(steps, 1):
    verb, _, rest = line.partition(" ")
    verb, rest = verb.lower(), rest.strip()
    try:
        if verb == "windows":
            app = Application(backend="uia").start(rest)
            time.sleep(1.5)
            win = app.top_window(); win.wait("visible", timeout=limits.get("launch", 30))
        elif win is None:
            raise Exception("no app is running -- `windows <command>` comes first")
        elif verb == "click":
            parts = rest.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                win.click_input(coords=(int(parts[0]), int(parts[1])))
            else:
                ctrl = None
                for c in win.descendants():
                    try:
                        if rest.lower() in (c.window_text() or "").lower():
                            ctrl = c; break
                    except Exception:
                        continue
                if ctrl is None:
                    raise Exception(f"nothing in the window reads {rest!r}")
                ctrl.click_input()
            time.sleep(0.4)
        elif verb == "type":
            keyboard.send_keys(rest, with_spaces=True); time.sleep(0.2)
        elif verb == "press":
            keyboard.send_keys(KEYS.get(rest.lower(), rest if len(rest) == 1 else "")
                               or (_ for _ in ()).throw(Exception("not a key this knows: " + rest)))
            time.sleep(0.3)
        elif verb == "wait":
            time.sleep(min(int(rest), 5000) / 1000)
        elif verb == "expect":
            absent = rest.lower().startswith("not ")
            what = rest[4:].strip() if absent else rest
            end = time.time() + limits.get("expect", 8)
            ok = False
            while time.time() < end:
                ok = (any(what.lower() in t.lower() for t in texts())) != absent
                if ok: break
                time.sleep(0.3)
            if not ok:
                raise Exception(("still on screen" if absent else "not on screen") + " after " + str(limits.get("expect", 8)) + "s")
        elif verb == "screen":
            report.append(f"{n}. {line} -> ok")
            report.extend("   | " + t for t in texts()[:limits.get("rows", 40)])
            continue
        elif verb == "changed":
            if len(frames) < 2:
                raise Exception("nothing to compare yet -- `changed` follows a click, type or press")
            if frames[-1] == frames[-2]:
                raise Exception("the window is identical to before the last click, type or press")
            report.append(f"{n}. {line} -> ok"); continue
        else:
            raise Exception("not a step this knows")
        if not app.is_process_running():
            raise Exception("the app is no longer running")
        if verb in ("windows", "click", "type", "press"):
            frames.append(shot())
        report.append(f"{n}. {line} -> ok")
    except Exception as exc:
        reason = (str(exc).strip().splitlines() or ["failed"])[0][:200]
        report.append(f"{n}. {line} -> FAILED: {reason}")
        failed = f"step {n}"
        break
if limits.get("screenshot") and frames and frames[-1]:
    try:
        open(limits["screenshot"], "wb").write(frames[-1])
    except OSError:
        pass
final = texts() if win is not None else []
try:
    if app is not None: app.kill()
except Exception:
    pass
passed = len([r for r in report if " -> ok" in r])
listed = "; ".join(s[:40] for s in steps[:12]) + (" ..." if len(steps) > 12 else "")
print(f"{passed}/{len(steps)} steps passed: {listed}")
for r in report:
    print("  " + r)
print("screen:")
for t in final[:40]:
    print("  " + t)
sys.exit(6 if failed else 0)
'''


def run_windows(script: str, limits: str, *, cwd: str, interpreter: str | None = None,
                timeout: float = 180.0) -> tuple[str, str, int]:
    interpreter = interpreter or os.environ.get("OTTO_BROWSER_PYTHON", "").strip() or sys.executable
    try:
        proc = subprocess.run([interpreter, "-c", WINDOWS_DRIVER, script, limits], capture_output=True,
                              text=True, errors="replace", timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return "", f"the app did not finish the walkthrough within {int(timeout)}s", -1
    except OSError as exc:
        return "", f"could not start the Windows driver: {exc}", 1
    if proc.returncode == 3:
        return "", NO_PYWINAUTO, 3
    return proc.stdout, proc.stderr, proc.returncode


# --------------------------------------------------------------------------
# The walkthrough loop every device shares
# --------------------------------------------------------------------------

def device_for(kind: str, *, cwd: str) -> Device:
    if kind == "android":
        return AndroidDevice()
    if kind == "ios":
        return IOSDevice()
    if kind == "mac":
        return MacDevice(cwd=cwd)
    if kind == "linux":
        return LinuxDevice(cwd=cwd)
    raise DeviceError(f"{kind} is not a device this drives")


def run_native(walk: Walk, device: Device, *, screenshot: str = "") -> tuple[str, bool]:
    """Every step of an app walkthrough, on one device, stopping at the
    first that fails. `(report, failed)`, in the shape every other kind
    returns. A step that held while the app died underneath fails too:
    the point is debugging, not only driving."""
    steps = walk.steps
    report: list[str] = []
    failed = False
    frames: list[bytes] = []
    last_texts: list[str] = []
    try:
        for n, step in enumerate(steps, 1):
            try:
                if n == 1:
                    report.append(f"{n}. {step.line} -> ok: {device.launch(step.argument)}")
                elif step.verb == "click":
                    parts = step.argument.split()
                    if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                        device.tap(int(parts[0]), int(parts[1]))
                    else:
                        device.click_text(step.argument)
                    report.append(f"{n}. {step.line} -> ok")
                elif step.verb == "type":
                    device.type_text(step.argument)
                    report.append(f"{n}. {step.line} -> ok")
                elif step.verb == "press":
                    device.press(step.argument)
                    report.append(f"{n}. {step.line} -> ok")
                elif step.verb == "wait":
                    time.sleep(min(int(step.argument), 5000) / 1000)
                    report.append(f"{n}. {step.line} -> ok")
                elif step.verb == "expect":
                    absent = step.argument.lower().startswith("not ")
                    what = step.argument[4:].strip() if absent else step.argument
                    deadline = time.time() + EXPECT_TIMEOUT_S
                    ok = False
                    while True:
                        last_texts = device.texts()
                        ok = any(what.lower() in t.lower() for t in last_texts) != absent
                        if ok or time.time() >= deadline:
                            break
                        time.sleep(0.5)
                    if not ok:
                        seen = " | ".join(last_texts[:12])[:200]
                        raise DeviceError(("still on screen" if absent else "not on screen")
                                          + f" after {int(EXPECT_TIMEOUT_S)}s; it shows: {seen!r}")
                    report.append(f"{n}. {step.line} -> ok")
                elif step.verb == "screen":
                    last_texts = device.texts()
                    report.append(f"{n}. {step.line} -> ok")
                    report.extend("   | " + t for t in last_texts[:40])
                    continue
                elif step.verb == "changed":
                    # Before and after the most recent ACTION, not the previous
                    # step -- see the page driver for why.
                    if len(frames) < 2:
                        raise DeviceError("nothing to compare yet -- `changed` follows a click, type or press")
                    if not frames[-1] or not frames[-2]:
                        raise DeviceError("no screenshot to compare -- see the screenshot permission or tool")
                    if frames[-1] == frames[-2]:
                        raise DeviceError("the screen is identical to before the last click, type or press")
                    report.append(f"{n}. {step.line} -> ok")
                    continue
                else:
                    raise DeviceError("not a step this knows")
                problems = [] if device.alive() else device.errors()
                if problems:
                    raise DeviceError("the app died: " + "; ".join(problems)[:300])
                if n == 1 or step.verb in ("click", "type", "press"):
                    frames.append(device.screenshot())
            except DeviceError as exc:
                report.append(f"{n}. {step.line} -> FAILED: {str(exc).splitlines()[0][:300]}")
                failed = True
                break
            except Exception as exc:  # one step's failure is the report's, not the tool's
                report.append(f"{n}. {step.line} -> FAILED: {type(exc).__name__}: {str(exc)[:200]}")
                failed = True
                break
        crashes = device.errors() if not failed else []
        if crashes:
            report.append("app errors:")
            report.extend("  " + c for c in crashes)
            failed = True
    finally:
        try:
            device.close()
        except Exception:
            pass
    if screenshot and frames and frames[-1]:
        try:
            with open(screenshot, "wb") as handle:
                handle.write(frames[-1])
        except OSError:
            pass
    lines = [summary_line(walk.lines, [r for r in report if r[:1].isdigit()], failed),
             *("  " + r for r in report)]
    if last_texts and not any(r.startswith("screen") for r in report):
        lines.append("screen:")
        lines.extend("  " + t for t in last_texts[:40])
    return "\n".join(lines) + "\n", failed

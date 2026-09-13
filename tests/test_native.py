"""Coverage for app walkthroughs on a device or a desktop -- Android, iOS,
macOS, Linux, Windows -- with every device command faked.

Same `exercise` shape as a page, a shell or a terminal; what differs is the
Device underneath (agent/pipeline/native.py). None of the five can be driven
end to end on the machine this was written on, and the module says so; what
is tested here is what each device SENDS, what it makes of what comes back,
and that a missing tool or permission is named rather than failing obscurely.
"""
import os
import sys

import pytest

import agent.pipeline.tools as pt
from agent.pipeline import native, walkthrough
from agent.pipeline.execution import bind_command_runner
from agent.pipeline.workspace import bind_workspace

_REAL_MAC_UI = os.environ.get("OTTO_MAC_UI", "")


class _Fake:
    """A command runner that answers by the first words of argv."""

    def __init__(self, answers: dict):
        self.answers, self.seen = answers, []

    def __call__(self, argv, timeout):
        self.seen.append(argv)
        for key, reply in self.answers.items():
            if " ".join(argv[:len(key.split())]) == key or key in " ".join(argv):
                return reply if isinstance(reply, tuple) else (reply, "", 0)
        return "", "", 0


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_a_device_kind_starts_a_walkthrough_and_has_its_own_steps():
    for first in ("android com.example.app", "ios build/App.app", "mac Calculator",
                  "linux ./app", "windows app.exe"):
        walk = walkthrough.parse_walk(f"{first}\nclick Add\nexpect 1\nscreen\nchanged")
        assert walk.backend == first.split()[0]
    assert "not an android step" in walkthrough.parse_walk("android com.x\nexit = 0")
    why = walkthrough.parse_walk("run swiftc counter.swift -o counter\nmac ./counter\nclick Add one")
    assert "build with execute_bash" in why and "`mac ./counter`" in why, why
    assert "walkthrough of its own" in walkthrough.parse_walk("mac A\nmac B")


# --------------------------------------------------------------------------
# Android: adb
# --------------------------------------------------------------------------

_DUMP = '''<?xml version='1.0' encoding='UTF-8'?><hierarchy rotation="0">
<node text="" bounds="[0,0][1080,1920]" clickable="false">
  <node text="Count: 0" resource-id="com.x:id/n" bounds="[100,200][500,260]" clickable="false"/>
  <node text="" content-desc="Add one" bounds="[100,300][300,400]" clickable="true"/>
</node></hierarchy>'''


def _android(extra=None):
    fake = _Fake({"adb get-state": "device\n", "adb exec-out uiautomator": _DUMP,
                  "adb shell monkey": "Events injected: 1\n", "adb shell pidof": "1234\n",
                  "adb logcat -d": "", **(extra or {})})
    return native.AndroidDevice(run=fake, adb="adb"), fake


def test_android_reads_the_screen_and_taps_the_centre_of_what_it_finds():
    device, fake = _android()
    walk = walkthrough.parse_walk("android com.x\nexpect Count: 0\nclick Add one\nscreen")
    report, failed = native.run_native(walk, device)

    assert not failed, report
    assert report.startswith("4/4 steps passed")
    assert "| Count: 0" in report
    taps = [a for a in fake.seen if "tap" in a]
    assert taps and taps[0][-2:] == ["200", "350"], taps
    assert ["adb", "shell", "am", "force-stop", "com.x"] in fake.seen, "the app was left running"


def test_android_installs_an_apk_first_and_launches_by_package():
    device, fake = _android({"adb install": "Success\n"})
    walk = walkthrough.parse_walk("android build/app.apk com.x/.MainActivity\nwait 10")
    report, failed = native.run_native(walk, device)

    assert not failed, report
    assert ["adb", "install", "-r", "build/app.apk"] in fake.seen
    assert any(a[-2:] == ["-n", "com.x/.MainActivity"] for a in fake.seen)


def test_android_reports_a_crash_even_when_the_steps_held():
    device, fake = _android({"adb shell pidof": "", "adb logcat -d": (
        "E AndroidRuntime: FATAL EXCEPTION: main\nE AndroidRuntime: Process: com.x, PID: 1234\n")})
    walk = walkthrough.parse_walk("android com.x\nclick Add one")
    report, failed = native.run_native(walk, device)

    assert failed
    assert "FATAL EXCEPTION" in report


def test_android_without_adb_or_a_device_says_so(monkeypatch):
    monkeypatch.setattr(native, "find_adb", lambda: None)  # CI runners ship one
    device = native.AndroidDevice(run=_Fake({}), adb=None)
    report, failed = native.run_native(walkthrough.parse_walk("android com.x"), device)
    assert failed and "platform-tools" in report

    device, _ = _android({"adb get-state": ("", "error: no devices/emulators found", 1)})
    report, failed = native.run_native(walkthrough.parse_walk("android com.x"), device)
    assert failed and "adb devices" in report


def test_the_ui_dump_parser_ignores_nodes_with_nothing_to_read():
    nodes = native.parse_ui_dump(_DUMP)
    assert [n["text"] for n in nodes] == ["Count: 0", "Add one"]
    assert native.parse_ui_dump("garbage") == []


# --------------------------------------------------------------------------
# iOS: simctl, and idb when it is there
# --------------------------------------------------------------------------

def test_ios_launches_on_the_booted_simulator_and_notices_a_dead_process(tmp_path):
    fake = _Fake({"xcrun simctl list": "iPhone 16 (ABC) (Booted)\n",
                  "xcrun simctl launch": "com.x.app: 999999\n"})
    device = native.IOSDevice(run=fake, idb=None)
    walk = walkthrough.parse_walk("ios com.x.app\nwait 10")
    report, failed = native.run_native(walk, device)

    assert failed, report
    assert "process 999999 is gone" in report
    assert ["xcrun", "simctl", "terminate", "booted", "com.x.app"] in fake.seen


def test_ios_installs_an_app_bundle_by_its_info_plist(tmp_path):
    import plistlib

    app = tmp_path / "Demo.app"
    app.mkdir()
    (app / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.demo"}))
    fake = _Fake({"xcrun simctl list": "(Booted)", "xcrun simctl install": "",
                  "xcrun simctl launch": f"com.demo: {os.getpid()}\n"})
    device = native.IOSDevice(run=fake, idb=None)
    report, failed = native.run_native(walkthrough.parse_walk(f"ios {app}\nwait 10"), device)

    assert not failed, report
    assert ["xcrun", "simctl", "install", "booted", str(app)] in fake.seen


def test_ios_without_idb_can_launch_but_names_what_it_cannot_do():
    fake = _Fake({"xcrun simctl list": "(Booted)", "xcrun simctl launch": f"com.x: {os.getpid()}\n"})
    device = native.IOSDevice(run=fake, idb=None)
    report, failed = native.run_native(walkthrough.parse_walk("ios com.x\nexpect Hello"), device)

    assert failed and "idb" in report


def test_ios_with_idb_reads_the_tree_and_taps():
    tree = '[{"AXLabel": "Add one", "frame": {"x": 10, "y": 20, "width": 100, "height": 40}}]'
    fake = _Fake({"xcrun simctl list": "(Booted)", "xcrun simctl launch": f"com.x: {os.getpid()}\n",
                  "idb ui describe-all": tree})
    device = native.IOSDevice(run=fake, idb="idb")
    report, failed = native.run_native(walkthrough.parse_walk("ios com.x\nclick Add one\nexpect Add"), device)

    assert not failed, report
    assert ["idb", "ui", "tap", "60", "40"] in fake.seen


def test_ios_without_a_booted_simulator_says_how_to_boot_one():
    device = native.IOSDevice(run=_Fake({"xcrun simctl list": ""}), idb=None)
    report, failed = native.run_native(walkthrough.parse_walk("ios com.x"), device)
    assert failed and "simctl boot" in report


# --------------------------------------------------------------------------
# macOS: System Events
# --------------------------------------------------------------------------

def test_mac_names_the_accessibility_permission_when_refused():
    fake = _Fake({"open -a": "", "pgrep": "42\n",
                  "osascript -l JavaScript": ("", "execution error: osascript is not allowed assistive access. (-1719)", 1)})
    device = native.MacDevice(run=fake)
    report, failed = native.run_native(walkthrough.parse_walk("mac Calculator\nexpect 0"), device)

    assert failed and "Accessibility" in report


def test_mac_reads_the_window_and_presses_by_label():
    tree = '{"elements": [{"role": "AXStaticText", "text": "15"}, {"role": "AXButton", "text": "equals"}], "window": {"name": "Calculator", "position": [10, 20], "size": [300, 400]}}'
    def answer(argv, timeout):
        fake.seen.append(argv)
        if argv[:1] == ["pgrep"]:
            return "42\n", "", 0
        if argv[:3] == ["osascript", "-l", "JavaScript"] and "AXPress" in argv[4]:
            return "pressed\n", "", 0
        if argv[:3] == ["osascript", "-l", "JavaScript"]:
            return tree, "", 0
        return "", "", 0
    fake = _Fake({}); fake.__call__ = answer
    device = native.MacDevice(run=answer)
    report, failed = native.run_native(walkthrough.parse_walk("mac Calculator\nclick equals\nexpect 15"), device)

    assert not failed, report
    assert any(a[-2:] == ["Calculator", "equals"] for a in fake.seen), "the click named the process and the label"


_APPKIT_COUNTER = """
import Cocoa
class Delegate: NSObject, NSApplicationDelegate {
    var count = 0
    let label = NSTextField(labelWithString: "Count: 0")
    var window: NSWindow!
    func applicationDidFinishLaunching(_ n: Notification) {
        window = NSWindow(contentRect: NSRect(x: 200, y: 200, width: 300, height: 160),
                          styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Counter"
        label.frame = NSRect(x: 20, y: 100, width: 260, height: 30)
        let add = NSButton(title: "Add one", target: self, action: #selector(add(_:)))
        add.frame = NSRect(x: 20, y: 40, width: 120, height: 32)
        window.contentView!.addSubview(label); window.contentView!.addSubview(add)
        window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true)
    }
    @objc func add(_ s: Any?) { count += 1; label.stringValue = "Count: \\(count)" }
}
let app = NSApplication.shared; let d = Delegate(); app.delegate = d
app.setActivationPolicy(.regular); app.run()
"""


@pytest.mark.skipif(sys.platform != "darwin" or not _REAL_MAC_UI or not __import__("shutil").which("swiftc"),
                    reason="set OTTO_MAC_UI=1 on a Mac with swiftc that granted Accessibility and Screen Recording")
def test_for_real_an_appkit_app_is_clicked_and_read(tmp_path):
    """Calculator's SwiftUI buttons expose no labels to System Events at all,
    so the real test builds its own AppKit app: a label and a button, which
    is what otto would be checking anyway."""
    import subprocess

    (tmp_path / "counter.swift").write_text(_APPKIT_COUNTER)
    built = subprocess.run(["swiftc", "-O", "counter.swift", "-o", "counter"], cwd=tmp_path,
                           capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr
    with bind_workspace(tmp_path):
        result = pt.exercise(
            "mac ./counter\nexpect Count: 0\nclick Add one\nexpect Count: 1\nchanged\n"
            "click Add one\nexpect Count: 2\nexpect not Count: 1\nscreen"
        )
        wrong = pt.exercise("mac ./counter\nclick Subtract one")

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.startswith("9/9 steps passed")
    assert "| Count: 2" in result.stdout
    assert wrong.returncode == 1 and "Subtract one" in wrong.stderr
    assert not subprocess.run(["pgrep", "-x", "counter"], capture_output=True).stdout, "the app was left running"


# --------------------------------------------------------------------------
# Linux and Windows: what is sent, since neither can run here
# --------------------------------------------------------------------------

def test_linux_without_xdotool_says_so(monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda name: None)
    device = native.LinuxDevice(run=_Fake({}))
    report, failed = native.run_native(walkthrough.parse_walk("linux ./app"), device)
    assert failed and "xdotool" in report


def test_the_windows_driver_knows_every_step_and_needs_pywinauto():
    for verb in walkthrough.DEVICE_STEPS:
        assert f'verb == "{verb}"' in native.WINDOWS_DRIVER, verb
    assert "from pywinauto import" in native.WINDOWS_DRIVER


def test_otto_ships_none_of_the_device_libraries():
    for name in ("pywinauto", "pyatspi", "idb"):
        with pytest.raises(ImportError):
            __import__(name)


# --------------------------------------------------------------------------
# Through the tool
# --------------------------------------------------------------------------

def test_a_device_walkthrough_is_local_only(tmp_path):
    with bind_workspace(tmp_path), bind_command_runner(lambda c, t: ("", "", 0)):
        result = pt.exercise("android com.x\nexpect hi")
    assert result.returncode == 1 and "container" in result.stderr


def test_a_device_walkthrough_reports_through_the_same_shape(tmp_path, monkeypatch):
    device, _ = _android()
    monkeypatch.setattr(native, "device_for", lambda kind, cwd: device)
    with bind_workspace(tmp_path):
        result = pt.exercise("android com.x\nexpect Count: 0\nclick Add one")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("3/3 steps passed")


def test_mac_says_when_the_app_shows_no_window_and_reads_applescript_labels(monkeypatch):
    """Live, a model's AppKit app never ordered its window front, and every
    click came back "no window" with nothing to act on; and it wrote the
    click as `button "Add one" of window 1`, as AppleScript would."""
    empty = '{"elements": [], "window": null}'
    seen = []

    def answer(argv, timeout):
        seen.append(argv)
        if argv[:1] == ["pgrep"]:
            return "42\n", "", 0
        if argv[:3] == ["osascript", "-l", "JavaScript"] and "AXPress" in argv[4]:
            return "no window\n", "", 0
        if argv[:3] == ["osascript", "-l", "JavaScript"]:
            return empty, "", 0
        return "", "", 0

    monkeypatch.setattr(native.time, "sleep", lambda s: None)
    device = native.MacDevice(run=answer)
    report, failed = native.run_native(
        walkthrough.parse_walk('mac Counter\nclick button "Add one" of window 1'), device)

    assert failed
    assert "no window visible yet" in report
    assert "setActivationPolicy" in report
    assert any(a[-1] == "Add one" for a in seen if a[:1] == ["osascript"]), "the label was not unquoted"


def test_mac_launches_a_bundle_built_in_the_workspace_by_its_executable_name(tmp_path, monkeypatch):
    import plistlib

    bundle = tmp_path / "Counter.app" / "Contents"
    (bundle / "MacOS").mkdir(parents=True)
    (bundle / "Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": "counter"}))
    tree = ('{"elements": [{"role": "AXStaticText", "text": "Count: 0"}], '
            '"window": {"name": "Counter", "position": [0, 0], "size": [10, 10]}}')
    seen = []

    def answer(argv, timeout):
        seen.append(argv)
        if argv[:1] == ["pgrep"]:
            return "42\n", "", 0
        if argv[:1] == ["osascript"]:
            return tree, "", 0
        return "", "", 0

    monkeypatch.setattr(native.time, "sleep", lambda s: None)
    device = native.MacDevice(run=answer, cwd=str(tmp_path))
    report, failed = native.run_native(walkthrough.parse_walk("mac Counter.app\nexpect Count: 0"), device)

    assert not failed, report
    assert ["open", "-a", str(tmp_path / "Counter.app")] in seen
    assert ["pgrep", "-x", "counter"] in seen, "the process is named after the executable, not the bundle"

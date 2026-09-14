"""The child half of agent/pipeline/python_session.py: one interpreter that
runs snippet after snippet in the same namespace and reports each one back.

Run as a script, never imported:

    python _python_session_shim.py <private dir> <memory limit bytes>

The parent talks to it over two channels that a snippet cannot reach by
accident. Requests arrive as JSON lines on a private copy of the original
stdin; results leave as JSON lines on a private copy of the original stdout.
Both are made by `os.dup()` at startup and the real descriptors 0, 1 and 2
are then pointed at /dev/null, so a snippet's `print()`, a `os.write(1, ..)`,
or a subprocess it starts can never land on the result channel -- during a
call, descriptors 1 and 2 point at capture files in the private dir, and
what those files hold when the call ends is what the call printed.

Every request carries an id the parent minted, and every result echoes it.
That is the framing: a snippet that prints something frame-shaped is
printing into its own capture file, not the channel, and even a snippet that
found the channel would have to guess a fresh uuid to be believed (see the
parent module for what happens to a frame with the wrong id).

A snippet's uncaught exception, `sys.exit()` and even a `KeyboardInterrupt`
end that call and nothing else; the traceback is printed WITHOUT the frames
of this file, because the model reads it to fix its own code. The parent
interrupts a call that has run too long by sending SIGINT (CTRL_BREAK on
Windows); the handler here raises KeyboardInterrupt only while a snippet is
executing, so an interrupt that lands between calls cannot corrupt a result
half-written to the channel. The handler is reinstalled after every call
because a snippet may replace it -- that is the one thing it cannot make
permanent.

The builtins are restored after every call. A snippet that rebinds
`builtins.len` has broken `json.loads`, which this file needs to read the
NEXT request -- found when exactly that turned one poisoned call into a
silent timeout on every call after it. So the builtins module is
snapshotted at startup and put back in `finally`, before anything here
runs again: a poisoned builtin lasts the call that poisoned it, never the
run. The same cannot be done for every module a snippet could patch, so the
handful of os/io functions the result path depends on are bound to local
names at import and used through those. Neither is a sandbox; both are
what keeps the plumbing answering.

A memory ceiling, when the parent asks for one, is `RLIMIT_AS` on this
process: past it an allocation raises MemoryError inside the snippet, which
is a failed call the parent reads like any other, rather than an OOM kill.
Linux enforces it; macOS ignores it and Windows has no equivalent, and the
parent says so in its docs rather than pretending otherwise.
"""
from __future__ import annotations

import builtins
import io
import json
import os
import signal
import sys
import traceback

#: The result path through these names only -- see the module docstring.
_BUILTINS = dict(builtins.__dict__)
_open = builtins.open
_os_open = os.open
_os_close = os.close
_os_dup2 = os.dup2
_os_remove = os.remove
_os_getsize = os.path.getsize
_FileIO = io.FileIO
_TextIOWrapper = io.TextIOWrapper


def _restore_builtins() -> None:
    table = builtins.__dict__
    table.update(_BUILTINS)
    for name in [key for key in table if key not in _BUILTINS]:
        del table[name]

#: Kept per stream from a call's output: the head and the tail, with the cut
#: marked. The parent clips further; this bound only stops one runaway print
#: loop from turning into a gigabyte-sized frame.
_KEEP = 64 * 1024

_executing = False


def _on_interrupt(signum, frame):
    if _executing:
        raise KeyboardInterrupt


def _arm_interrupt() -> None:
    signal.signal(signal.SIGINT, _on_interrupt)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_interrupt)


def _set_std_handles() -> None:
    """Windows keeps the console's standard handles apart from the C runtime's
    descriptors, and a subprocess with no explicit stdio inherits the former.
    After a dup2 on 1 and 2, point the handles at the same place so a
    subprocess started by a snippet writes into the capture file too."""
    if os.name != "nt":
        return
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        for fd, which in ((0, 0xFFFFFFF6), (1, 0xFFFFFFF5), (2, 0xFFFFFFF4)):
            kernel32.SetStdHandle(which, msvcrt.get_osfhandle(fd))
    except Exception:
        pass


def _fresh_text_stream(fd: int):
    """A new text wrapper over `fd`, so a snippet that closed or replaced the
    previous `sys.stdout` cannot take the next call's output with it."""
    return _TextIOWrapper(
        _FileIO(fd, "w", closefd=False), encoding="utf-8", errors="replace",
        write_through=True,
    )


def _point_std_streams(devnull: int) -> None:
    """After a call: descriptors back to /dev/null, fresh stream objects."""
    _quiet_flush()
    _os_dup2(devnull, 1)
    _os_dup2(devnull, 2)
    _set_std_handles()
    sys.stdout = sys.__stdout__ = _fresh_text_stream(1)
    sys.stderr = sys.__stderr__ = _fresh_text_stream(2)


def _quiet_flush() -> None:
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            stream.flush()
        except Exception:
            pass


def _read_capture(path: str) -> str:
    try:
        size = _os_getsize(path)
        with _open(path, "rb") as handle:
            if size <= 2 * _KEEP:
                data = handle.read()
            else:
                head = handle.read(_KEEP)
                handle.seek(size - _KEEP)
                tail = handle.read()
                data = head + b"\n... [output cut here] ...\n" + tail
    except OSError:
        return ""
    finally:
        try:
            _os_remove(path)
        except OSError:
            pass
    return data.decode("utf-8", errors="replace")


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _run_one(code: str, namespace: dict, private: str, devnull: int) -> dict:
    global _executing
    out_path = os.path.join(private, "stdout")
    err_path = os.path.join(private, "stderr")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    out_fd = _os_open(out_path, flags, 0o600)
    err_fd = _os_open(err_path, flags, 0o600)
    _quiet_flush()
    _os_dup2(out_fd, 1)
    _os_dup2(err_fd, 2)
    _os_close(out_fd)
    _os_close(err_fd)
    _set_std_handles()
    sys.stdout = sys.__stdout__ = _fresh_text_stream(1)
    sys.stderr = sys.__stderr__ = _fresh_text_stream(2)

    returncode = 0
    timed_out = False
    try:
        try:
            _executing = True
            exec(compile(code, "<session>", "exec"), namespace)
        finally:
            _executing = False
    except SystemExit as exc:
        returncode = _exit_code(exc)
    except KeyboardInterrupt:
        timed_out = True
        returncode = -1
    except BaseException as exc:  # noqa: BLE001 -- a snippet may raise anything
        # Drop the one frame that is this function's own `exec` call; what is
        # left starts at `<session>`, which is the snippet.
        tb = exc.__traceback__.tb_next if exc.__traceback__ else None
        try:
            traceback.print_exception(type(exc), exc, tb, file=sys.stderr)
        except Exception:
            pass
        returncode = 1
    finally:
        _restore_builtins()
        _arm_interrupt()
        _point_std_streams(devnull)

    return {
        "stdout": _read_capture(out_path),
        "stderr": _read_capture(err_path),
        "returncode": returncode,
        "timed_out": timed_out,
    }


def main() -> None:
    private = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    # This file's directory is what `python <script>` put first on sys.path;
    # the snippets want the working directory there, exactly as a script run
    # from it would have (and as the fresh-process path arranged through
    # PYTHONPATH).
    sys.path[0] = os.getcwd()

    if limit > 0 and os.name == "posix":
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            pass

    requests = os.fdopen(os.dup(0), "rb")
    results = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)      # input() gets EOF, never a hang on the pipe
    _point_std_streams(devnull)
    _arm_interrupt()

    namespace: dict = {"__name__": "__main__"}
    results.write(json.dumps({"ready": True}) + "\n")
    results.flush()

    for line in requests:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        outcome = _run_one(str(request.get("code", "")), namespace, private, devnull)
        outcome["id"] = request.get("id")
        results.write(json.dumps(outcome) + "\n")
        results.flush()


if __name__ == "__main__":
    main()

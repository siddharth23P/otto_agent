"""The child half of agent/pipeline/python_session.py: one interpreter that
runs snippet after snippet in the same namespace and reports each one back.

Run as a script, never imported:

    python _python_session_shim.py <private dir> <memory limit bytes>

The parent talks to it over two channels that a snippet cannot reach by
accident. Requests arrive as lines `<id> <base64 code>` on a private copy of
the original stdin; results leave as lines `R <id> <returncode> <0|1 timed
out> <base64 stdout> <base64 stderr>` on a private copy of the original
stdout, after one `READY` line at startup.
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
`builtins.len` had broken `json.loads`, which this file then needed to read
the NEXT request -- found when exactly that turned one poisoned call into a
silent timeout on every call after it. So the builtins module is
snapshotted at startup and put back the moment the snippet returns, before
anything here runs again: a poisoned builtin lasts the call that poisoned
it, never the run.

The same cannot be done for every module a snippet could patch, so the rule
for the wire path is: no pure-Python module at call time. The protocol above
is base64 through `binascii` rather than JSON, because `json.dumps` reads
the json module's own globals when called and a snippet that patched them
(review finding on the first cut) would have made every later result
unparseable -- self-healing through the parent's timeout-and-restart path,
but a whole session's state lost to one line. Every os/io/binascii/signal/
traceback function the plumbing uses is bound to a local name at import and
used through that, so rebinding the module attribute changes nothing here.
What remains patchable -- `sys.modules`, the methods of `bytes` and `str`
through something like forbiddenfruit, or `os.dup2` at the C level -- is
inside the "not a sandbox" line this module and tools.py both draw: the aim
is that the plumbing survives what a snippet does by accident or by habit,
not that it survives a snippet written to break it.

A memory ceiling, when the parent asks for one, is `RLIMIT_AS` on this
process: past it an allocation raises MemoryError inside the snippet, which
is a failed call the parent reads like any other, rather than an OOM kill.
Linux enforces it; macOS ignores it and Windows has no equivalent, and the
parent says so in its docs rather than pretending otherwise.
"""
from __future__ import annotations

import binascii
import builtins
import io
import os
import signal
import sys
import traceback

#: The wire and result path through these names only -- see the module
#: docstring. Bound at import, before any snippet runs.
_BUILTINS = dict(builtins.__dict__)
_open = builtins.open
_os_open = os.open
_os_close = os.close
_os_dup2 = os.dup2
_os_remove = os.remove
_os_getsize = os.path.getsize
_os_path_join = os.path.join
_O_CAPTURE = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
_FileIO = io.FileIO
_BytesIO = io.BytesIO
_TextIOWrapper = io.TextIOWrapper
_b2a = binascii.b2a_base64
_a2b = binascii.a2b_base64
_signal_signal = signal.signal
_SIGINT = signal.SIGINT
_SIGBREAK = getattr(signal, "SIGBREAK", None)
_print_exception = traceback.print_exception


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
    _signal_signal(_SIGINT, _on_interrupt)
    if _SIGBREAK is not None:
        _signal_signal(_SIGBREAK, _on_interrupt)


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
        write_through=True, newline="\n",
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
    # Universal newlines, exactly as `subprocess.run(text=True)` decoded the
    # fresh-process path: on Windows a print() or a child process writes
    # `\r\n`, and the model must read the same `\n` it always did.
    return _TextIOWrapper(_BytesIO(data), encoding="utf-8", errors="replace").read()


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _run_one(code: str, namespace: dict, out_path: str, err_path: str,
             devnull: int) -> tuple[int, bool, str, str]:
    """Run one snippet; (returncode, timed_out, stdout, stderr)."""
    global _executing
    out_fd = _os_open(out_path, _O_CAPTURE, 0o600)
    err_fd = _os_open(err_path, _O_CAPTURE, 0o600)
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
            # First, before any handler below runs: they use builtins.
            _executing = False
            _restore_builtins()
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
            _print_exception(type(exc), exc, tb, file=sys.stderr)
        except Exception:
            pass
        returncode = 1
    finally:
        _arm_interrupt()
        _point_std_streams(devnull)

    return returncode, timed_out, _read_capture(out_path), _read_capture(err_path)


def _field(text: str) -> bytes:
    return _b2a(text.encode("utf-8", errors="replace"), newline=False)


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
    results = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)      # input() gets EOF, never a hang on the pipe
    _point_std_streams(devnull)
    _arm_interrupt()
    out_path = _os_path_join(private, "stdout")
    err_path = _os_path_join(private, "stderr")

    namespace: dict = {"__name__": "__main__"}
    results.write(b"READY\n")
    results.flush()

    for line in requests:
        request_id, _, payload = line.rstrip(b"\r\n").partition(b" ")
        if not request_id:
            continue
        try:
            code = _a2b(payload).decode("utf-8", errors="replace")
        except ValueError:  # binascii.Error is one, and named without the module
            continue
        returncode, timed_out, out, err = _run_one(code, namespace, out_path, err_path, devnull)
        results.write(b" ".join((
            b"R", request_id, str(returncode).encode("ascii"),
            b"1" if timed_out else b"0", _field(out), _field(err),
        )) + b"\n")
        results.flush()


if __name__ == "__main__":
    main()
